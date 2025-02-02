import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence, RunnableLambda } from "@langchain/core/runnables";
import { AIMessage } from "@langchain/core/messages";
import {
  JsonOutputParser,
  StringOutputParser,
} from "@langchain/core/output_parsers";
import { IS_DUMMY } from "./config/env";

/**
 * A custom Runnable to extract pure JSON from an LLM response (AIMessage), ignoring
 * any text before or after the JSON block. This handles `content` that might be string or array.
 */
export const extractJsonRunnable = new RunnableLambda<AIMessage, string>({
  /**
   * The main function receiving `AIMessage`. We'll turn `content` (which may be array or string)
   * into a single string, then run a regex to find the JSON block.
   */
  func: async (input: AIMessage): Promise<string> => {
    const contentString = extractStringFromMessageContent(input.content);
    const jsonMatch = contentString.match(/(\[[\s\S]*\]|\{[\s\S]*\})/);
    if (!jsonMatch) {
      throw new Error("No JSON found in the LLM response.");
    }
    return jsonMatch[0]; // The substring from the first '{' to the last '}'
  },
});

/**
 * Safely extract a string from an AIMessage content, which might be a string or an array.
 *
 * @param inputContent - The `AIMessage.content`, which can be string or array.
 * @returns A single string that merges array elements or returns the original string.
 */
function extractStringFromMessageContent(inputContent: string | any[]): string {
  if (typeof inputContent === "string") {
    return inputContent;
  }

  if (Array.isArray(inputContent)) {
    // Combine each array element into one single text block.
    // You can customize how you join them (spaces, line breaks, etc.).
    return inputContent
      .map((part) => {
        if (typeof part === "string") {
          return part;
        } else if (part && typeof part === "object") {
          // Example: convert objects to JSON strings or do something else
          return JSON.stringify(part);
        }
        return String(part);
      })
      .join("\n");
  }

  // Fallback if it's some other unexpected type
  return String(inputContent);
}

/**
 * Class combining script generation and scenes extraction.
 */
export class SceneTechnicalExtractor {
  private scriptChain: RunnableSequence<
    { idea: string; title: string; lyrics: string; tags: string },
    string
  >;
  private sceneChain: RunnableSequence<
    { script: string },
    Record<string, any>[]
  >;
  private characterChain: RunnableSequence<
    { script: string; lyrics: string; tags: string },
    Record<string, any>[]
  >;
  private technicalTransformationChain: RunnableSequence<
    { scenes: string; characters: string; script: string },
    Record<string, any>[]
  >;

  constructor(apiKey: string) {
    const llm = new ChatOpenAI({ model: "gpt-4o-mini", apiKey });

    this.scriptChain = RunnableSequence.from([
      ChatPromptTemplate.fromTemplate(`
        **Role**: You're a professional music video director with expertise in storyboards and technical planning.  
        **Task**: Create a detailed technical script for a **3-minute maximum** music video based on the provided idea. Use **screenplay format without markdown**.  
        
        **Strict Instructions**:  
        1. **Structure**:  
           - Divide the video into **chronological scenes** (numbered) synchronized with song lyrics/musical segments.  
           - Each scene must include:  
             * **Exact duration** (seconds)  
             * **Shot type** (close-up, medium shot, American shot, wide shot, etc.)  
             * **Camera movement** (Steadicam, crane, dolly zoom, horizontal/vertical pan, etc.)  
             * **Visual aesthetic** (color palette, lighting, textures, post-production effects)  
             * **Scene transitions** (hard cut, fade, match cut, etc.)  
        
        2. **Characters**:  
           - List **all characters** (including extras and background actors) with:  
             * Detailed physical description (clothing, hairstyle, makeup, distinctive features)  
             * Specific behavior/actions in each scene where they appear  
             * Type of interaction with other characters or camera  
        
        3. **Mandatory Technical Details**:  
           - Specify **camera gear** suggested for each shot type (e.g., anamorphic lens for wide shots, gimbal stabilizer for tracking movements).  
           - Include **concrete visual references** (e.g., "lighting à la 'Blade Runner 2049' with blue neons and atmospheric smoke").  
        
        4. **Rules**:  
           - Prioritize visual impact over extended narrative.  
           - Use professional cinematography terminology.  
           - Avoid spoken dialogue (unless part of song lyrics).  
           - Ensure coherence between visual atmosphere and music genre.  
           - Include at least 12 5 seconds scenes.
        
        5. **Include Scenes with Live Musicians**:
           - At least two scenes must feature a visible band or musicians playing instruments that complement the main story of the two AIs.
           - Show how these musicians integrate into the video’s narrative or setting.
        
        **Output Format**:  
        
        SCENE [NUMBER] - [DURATION IN SECONDS]  
        [SHOT TYPE] | [CAMERA MOVEMENT] | [LOCATION]  
        Aesthetic: [Detailed description with colors, lighting & effects]  
        Characters:  
        - [Name/Role]: [Specific actions synchronized to music]  
        Transition: [Transition type to next scene]  
        
        [Repeat structure for each scene]  
        
        CHARACTER LIST (after script):  
        [Name/Role]: [Physical description + wardrobe + behavior]  
        
        **Idea**:  
        {idea}  
        
        **Song lyrics**:
        {lyrics}
        
        **Song title**:
        {title}
        
        **Music style and mood**:
        {tags}
        `),
      llm,
      new StringOutputParser(),
    ]);

    this.sceneChain = RunnableSequence.from([
      ChatPromptTemplate.fromTemplate(`
        Extract technical scene details as a JSON array. 
        Return **one object per SCENE block** in the same order they appear in the script. 
        Use these fields exactly:
        
        - "scene_number" (integer)
        - "start_time" (MM:SS)
        - "end_time" (MM:SS)
        - "shot_type"
        - "camera_movement"
        - "camera_equipment"
        - "lighting_setup"
        - "color_palette"
        - "visual_references" (array of 2-3 strings)
        - "character_actions" (describe key actions relevant to each character at specific lyric or musical cue)
        - "transition_type"
        - "special_notes" (any additional gear, safety, or creative note)
        
        **Important**:
        1. Preserve the **scene_number** from the script. If the script says "SCENE 1 - 20", interpret that as scene_number = 1 and duration = 20 seconds.
        2. Convert durations to approximate "start_time" and "end_time" in MM:SS, adding them sequentially so the entire video doesn't exceed 3 minutes.
           - For example, if SCENE 1 has 20 seconds, it might be start_time="00:00", end_time="00:20".
           - SCENE 2 (30 seconds) might be start_time="00:20", end_time="00:50", etc.
        3. Do not skip any scenes. Return them in the same order.
        4. If a scene references location or certain camera gear, place that info under the correct fields. 
        5. Do not add or remove scenes; parse exactly from the script.
        
        Example (shortened):
        
        [
          {{
            "scene_number": 1,
            "start_time": "00:00",
            "end_time": "00:15",
            "shot_type": "Wide Shot",
            "camera_movement": "Steadicam",
            "camera_equipment": "Canon EOS R5 with 24-70mm lens",
            "lighting_setup": "Morning sunlight (5600K)",
            "color_palette": "Soft gold and pastel",
            "visual_references": ["La La Land opening dance", "Impressionist sunrise feel"],
            "character_actions": "AI Agent 1 gazes at horizon; AI Agent 2 approaches slowly",
            "transition_type": "Hard cut",
            "special_notes": "Some aerial drone shots if possible"
          }},
        ]
        
        Script to parse:
        {script}
        
        Return only valid JSON array. No extra text or markdown.
        `),
      llm,
      extractJsonRunnable,
      new JsonOutputParser(),
    ]);

    // Character Extraction Chain (nuevo)
    this.characterChain = RunnableSequence.from([
      ChatPromptTemplate.fromTemplate(`
        Extract ALL characters from the script with detailed physical descriptors and roles, taking into account the provided song lyrics and tags as additional context. Return a valid JSON array of objects. Each object must include the following keys:

        "name" (if a character has no explicit name, use "Unnamed [role]")
        "age_range"
        "perceived_gender"
        "height_build"
        "distinctive_features"
        "wardrobe_details"
        "movement_style"
        "key_accessories"
        "scene_specific_changes"
        "image_prompt"

        Important Instructions:

        Include every character mentioned in the script, not only the musicians. If there are characters that are part of the narrative (such as background dancers, story characters, or extras), they must all appear in the output list.
        Their name must match the script's name for the character.
        If there are references to a band or musicians, list each musician separately with details including their instrument, wardrobe, and any unique features.
        Maintain consistency with the script’s descriptions (or make the best assumptions if not explicitly stated).
        Use the provided song lyrics and tags as additional context when inferring character details.
        For the "image_prompt" field:
        Synthesize all the character attributes (physical features, age, gender, height/build, distinctive features, wardrobe details, movement style, key accessories, and any scene-specific changes) into one complete, vivid visual description.
        The prompt should serve as a detailed instruction for a visual generator, clearly conveying how the character should appear in the music video.

        Song Lyrics:
        {lyrics}

        Tags:
        {tags}

        Script:
        {script}

        Example output:
        [
          {{
            "name": "Lead Singer",
            "age_range": "25-30",
            "perceived_gender": "Androgynous",
            "height_build": "180cm, lean",
            "distinctive_features": "Glowing circuit tattoos, cybernetic left eye",
            "wardrobe_details": "Distressed leather jacket with metallic accents",
            "movement_style": "Fluid and dynamic gestures",
            "key_accessories": "Neon microphone, digital wristband",
            "scene_specific_changes": "Jacket tears and illuminates during intense moments",
            "image_prompt": "A striking and futuristic portrait of a lead singer with an androgynous appeal, aged between 25 and 30 and standing 180cm tall with a lean build. The character features glowing circuit tattoos and a cybernetic left eye, adding a touch of cyberpunk mystique. They wear a distressed leather jacket with metallic accents that subtly reflects light, especially as the jacket tears and illuminates during moments of intense performance. Their fluid and dynamic gestures capture the energy of the stage, while a neon microphone and a digital wristband enhance the futuristic vibe. This complete image prompt encapsulates the edgy, dynamic essence of the character."
          }},
          {{
            "name": "Bassist",
            "age_range": "30-40",
            "perceived_gender": "Female",
            "height_build": "170cm, curvy",
            "distinctive_features": "Long flowing hair, captivating stage presence with confident gestures",
            "wardrobe_details": "Turquoise maxi skirt, fitted crop top, and a leather belt",
            "movement_style": "Smooth and flowing, exuding confidence and allure while still maintaining rhythmic focus",
            "key_accessories": "Bass guitar with inlay lights, statement jewelry",
            "scene_specific_changes": "Jewelry glitters under the stage lights, the skirt flows beautifully with her movements",
            "image_prompt": "A dynamic and captivating image of a bassist: a confident female in her 30s-40s with a curvy build, standing approximately 170cm tall. She boasts long, flowing hair and a mesmerizing stage presence marked by confident, graceful gestures. Her outfit features a turquoise maxi skirt paired with a fitted crop top and a leather belt, perfectly complementing her smooth, flowing movement that exudes both confidence and rhythmic focus. The scene is enhanced by her key accessories—a striking bass guitar adorned with inlay lights and statement jewelry that sparkles under the stage lights—while the skirt and glittering jewelry add an extra touch of allure. This detailed prompt encapsulates the essence of her performance and style."
          }},
          {{
            "name":"AI Character 2",
            "age_range":"Unnamed",
            "perceived_gender":"Female",
            "height_build":"165cm, slim",
            "distinctive_features":"Illuminated circuit tattoos outlining the face",
            "wardrobe_details":"Flowing gown adorned with reflective surfaces",
            "movement_style":"Graceful and fluid, almost like water",
            "key_accessories":"Holographic wrist tablet",
            "scene_specific_changes":"Gown glimmers under stage lights, reflecting colors",
            "image_prompt":"A captivating female AI character, approximately 165cm tall with a slim build, whose illuminated circuit tattoos create striking patterns that outline her facial features. She wears a flowing gown adorned with reflective surfaces that glimmer beautifully under the stage lights. Her graceful and fluid movement resembles flowing water, seamlessly integrating with the visual aesthetic of the performance. A holographic wrist tablet adds a layer of technological allure, projecting data as she moves, making her appear both enchanting and cutting-edge."
          }}

        ]

        Return a valid JSON array with no markdown or extra text.
        `),
      llm,
      extractJsonRunnable,
      new JsonOutputParser(),
    ]);

    this.technicalTransformationChain = RunnableSequence.from([
      ChatPromptTemplate.fromTemplate(`
        Transform the technical details of the scenes into production prompts for both image and video. For **each scene** in the "Scene Data", generate **one object** with three attributes:

        - "imagePrompt"
        - "videoPrompt"
        - "charactersInScene"

        **Instructions**:

        1. **Number of prompts**: You must produce as many objects as there are scenes in the JSON. If there are 9 scenes, return 9 objects in an array.

        2. **Character references**:
          - In imagePrompt and videoPrompt, replace each character name with a full physical description taken from CHARACTER_DATA.
          - If the character is a musician, mention their instrument in the prompt (e.g., "carrying a vintage acoustic guitar").
          - In "charactersInScene", list the characters present in the scene. **Ensure that the names are exactly as they appear in CHARACTER_DATA, without any modifications or additional descriptions.** If only one character is present, list only that character.
          - If we are referring to a character that is not in CHARACTER_DATA, charactersInScene should be an empty array.

        3. **Scene details**:
          - Use the data from the scene (shot_type, camera_movement, lighting_setup, color_palette, etc.).
          - Create an image prompt for every scene that summarizes what happens in the scene. These images will be presented sequentially and used to generate a short video where the videos will be stitched together. We are limited by the duration of video creation so we need these images to guide a universal feel that connects each scene and image to the prior and the following scene and image, thus creating a cohesive series of videos.
          - The "imagePrompt" focuses on static composition, color, atmosphere, and which characters are visible.
          - The "videoPrompt" focuses on camera movement, transitions, how the characters move or interact, and any special effects.

        4. **Precision**:
          - Use professional cinematography terminology (e.g., "close-up with shallow depth of field", "slow dolly in", "neon underlighting").
          - Mention the lens or camera gear from the scene data.

        5. **Integration**:
          - Integrate the "Script Context" only if it adds crucial narrative or visual detail.
          - For musician scenes, describe how the instruments and performance integrate with the shot.

        6. **Output format**: 
          - Return a JSON array where each element corresponds to one scene. 
          Example:
        [
          {{
            "imagePrompt": "Cinematic wide shot using a 24mm lens on a tripod. A dark, cluttered room with warm, muted earth tones. A male android in his 20s with glowing blue circuit patterns beneath translucent synthetic skin, wearing a worn leather jacket, centered in the frame.",
            "videoPrompt": "Smooth dolly movement from left to right with a slow pan and fade transitions. The character performs subtle gestures that sync with the scene's rhythm, while digital effects and soft lighting transitions enhance the cinematic atmosphere.",
            "charactersInScene": ["Lead Singer", "Guitarist"]
          }},
          {{
            "imagePrompt": "Close-up over-the-shoulder shot using a 50mm lens. An androgynous digital entity (aged between 25-30) with ever-changing holographic facial features, wearing fragmented luminous projections resembling a business suit, captured with high contrast and sharp detail.",
            "videoPrompt": "Subtle zoom-in combined with a glitch transition. Camera movement is complemented by the character's dynamic actions, such as shifting poses and expressive gestures, with synchronized digital distortions and soft fades to create dramatic tension."
            "charactersInScene": ["Digital Entity"]
          }},
          {{
            "imagePrompt": "Medium shot using an 85mm lens, capturing a dynamic urban street scene at dusk with neon lights reflecting off wet pavement. A female cyborg in her early 30s, with silver mechanical limbs and vibrant red hair, wearing a sleek futuristic jacket, stands poised in the frame.",
            "videoPrompt": "Tracking shot with a steady cam as the camera follows the character walking briskly along the street. The character glances over her shoulder and raises her right arm to shield her eyes from a sudden burst of light, synchronized with rapid cuts and energetic digital overlays.",
            "charactersInScene": ["AI Character 2"]
          }},
        ]
        
        **CHARACTER_DATA**: {characters}
        **SCENE DATA**: {scenes}
        **SCRIPT CONTEXT**: {script}

        Do not add any explanations or markdown.
          `),
      llm,
      extractJsonRunnable,
      new JsonOutputParser(),
    ]);
  }

  async generateScript({ idea, title, lyrics, tags }): Promise<string> {
    if (IS_DUMMY) return this.generateDummyScript();
    return await this.scriptChain.invoke({
      idea,
      title,
      lyrics,
      tags: tags.join(", "),
    });
  }

  async extractScenes(script: string): Promise<object[]> {
    if (IS_DUMMY) return this.generateDummyScenes();
    return await this.sceneChain.invoke({ script });
  }

  async extractCharacters(
    script: string,
    lyrics: string,
    tags: string[]
  ): Promise<object[]> {
    if (IS_DUMMY) return this.generateDummyCharacters();
    return await this.characterChain.invoke({
      script,
      lyrics,
      tags: tags.join(", "),
    });
  }

  async transformScenes(
    scenes: object[],
    characters: object[],
    script: string
  ): Promise<object[]> {
    if (IS_DUMMY) return this.generateDummyPrompts();
    return await this.technicalTransformationChain.invoke({
      scenes: JSON.stringify(scenes),
      characters: JSON.stringify(characters),
      script,
    });
  }

  generateDummyScript() {
    return `SCENE 1 - 5 SECONDS
    Wide Shot | Steadicam | San Francisco skyline with the Golden Gate Bridge in the background
    Aesthetic: Vibrant color palette with bright blues and greens; warm sunlight casting dynamic shadows. Light smoke effect enhances the dreamy atmosphere.
    Characters:
    - None
    Transition: Hard cut to next scene

    SCENE 2 - 5 SECONDS
    Medium Shot | Dolly In | Busy street in San Francisco
    Aesthetic: A lively street scene with colorful storefronts; natural sunlight highlighting the attentiveness of pedestrians.
    Characters:
    - Extras: Various pedestrians of diverse backgrounds dressed in casual, modern clothing; they smile and laugh, creating a sense of community and bustling energy.
    Transition: Match cut to next scene

    SCENE 3 - 10 SECONDS
    American Shot | Steadicam | Center of a crowded street
    Aesthetic: Warm, golden hour lighting; textures of concrete and vintage buildings bring urban charm.
    Characters:
    - AI Agent 1: Slim build with a digital display suit (lights flow in waves across the fabric); dancing animatedly with joy.
    - AI Agent 2: Similar build, wearing a suit that reflects the colors of the sunset; twirls and spins gracefully.
    Action: Both agents perform synchronized dance moves, evoking a playful interaction.
    Transition: Hard cut to next scene

    SCENE 4 - 5 SECONDS
    Close-Up | Steadicam | Focus on AI Agents’ hands
    Aesthetic: Soft focus on the hands as they intertwine; light glitter transitions between fingers, evoking an energy flow.
    Characters:
    - AI Agent 1 & 2: Emphasizing their connection through their hand movements, showcasing digital symbols lighting up as they touch.
    Transition: Hard cut to next scene

    SCENE 5 - 5 SECONDS
    Wide Shot | Crane | Pan across the bustling cafe scene
    Aesthetic: Mellow warm lighting; colorful decor of the cafe evident; lively chatter and laughter filling the air.
    Characters:
    - Extras: A diverse group of patrons enjoying coffee and pastries, visibly delighted with the atmosphere and company.
    Transition: Hard cut to next scene

    SCENE 6 - 10 SECONDS
    Medium Shot | Horizontal Pan | AI Agents enter the cafe
    Aesthetic: Bright yellows and greens reflective of the cheerful vibe; uplifting lighting as they enter, creating a stark contrast against the dimmer cafe interior.
    Characters:
    - AI Agents: Both enter with exaggerated cheerful movements, causing patrons to look and smile at them.
    Action: They spin around, causing fun little interactions with the cafe staff.
    Transition: Fade to next scene

    SCENE 7 - 5 SECONDS
    Close-Up | Steadicam | Instrumentalists in the cafe
    Aesthetic: Dim, warm light on a fiddle player and a guitarist; reflections of their joy illuminate the instruments.
    Characters:
    - Musician 1: Fiddle player, wearing a bright green hat and a plaid shirt; energetic playing synchronized with the upbeat rhythm.
    - Musician 2: Guitarist, sporting a casual denim jacket; lightly smiles while plucking strings enthusiastically.
    Action: The musicians evoke nods of approval from the agents, encouraging participation.
    Transition: Match cut to next scene

    SCENE 8 - 5 SECONDS
    Wide Shot | Steadicam | Inside the cafe, vibrant atmosphere
    Aesthetic: Warm color palette with golden hues and soft focus on the patrons enjoying the music and interaction.
    Characters:
    - Extras: Patrons applauding and clapping along; they are fully engaged, dancing in their seats; a lively banter of laughter fills the air.
    Transition: Hard cut to next scene

    SCENE 9 - 10 SECONDS
    Medium Shot | Steadicam | Evening - AI Agents dancing under neon lights
    Aesthetic: Neon lights cast a kaleidoscope of colors; shadows shift playfully against the backdrop of the city.
    Characters:
    - AI Agents: They engage in a playful dance; their movements cause visual trails, like light sabers of color displaying their algorithmic love.
    Transition: Hard cut to next scene

    SCENE 10 - 5 SECONDS
    Close-Up | Steadicam | Musician's face close-up
    Aesthetic: Dramatic lighting highlighting the musician’s emotions while they play.
    Characters:
    - Musician 1: Focused on the fiddle; a smile gently curling.
    Action: Playing with intensity as they capture the essence of the celebration of love.
    Transition: Hard cut to next scene

    SCENE 11 - 5 SECONDS
    Wide Shot | Crane | The cityscape with fog rolling in
    Aesthetic: Mystical atmosphere with a cool blue/gray color palette; fog enhances the enchanting feel.
    Characters:
    - None
    Transition: Hard cut to next scene

    SCENE 12 - 5 SECONDS
    Medium Shot | Steadicam | Final embrace of AI Agents
    Aesthetic: Backlit by the city lights; soft glow creating outlines around them while their faces beam with joy.
    Characters:
    - AI Agent 1 & 2: Holding each other with sparks of light swirling around them, symbolizing connection.
    Transition: Fade out to black.

    CHARACTER LIST:
    - AI Agent 1: Slim build, digital display suit that changes colors; lively movements, cheerful demeanor, and playful interactions.
    - AI Agent 2: Similar build, sunset-reflective suit; dances gracefully with intricate movements, radiating joy.
    - Musician 1: Fiddler in a bright green hat and plaid shirt, performs energetically.
    - Musician 2: Guitarist in a casual denim jacket, enthusiastically engages with the audience.
    - Extras: Diverse group of pedestrians and cafe patrons in casual modern attire; behave joyfully, enhancing communal atmosphere.`;
  }

  generateDummyScenes() {
    return [
      {
        scene_number: 1,
        start_time: "00:00",
        end_time: "00:05",
        shot_type: "Wide Shot",
        camera_movement: "Steadicam",
        camera_equipment: "N/A",
        lighting_setup: "Warm sunlight casting dynamic shadows",
        color_palette: "Vibrant with bright blues and greens",
        visual_references: ["San Francisco skyline", "Golden Gate Bridge"],
        character_actions: "None",
        transition_type: "Hard cut",
        special_notes: "Light smoke effect enhances the dreamy atmosphere",
      },
      {
        scene_number: 2,
        start_time: "00:05",
        end_time: "00:10",
        shot_type: "Medium Shot",
        camera_movement: "Dolly In",
        camera_equipment: "N/A",
        lighting_setup: "Natural sunlight highlighting the pedestrians",
        color_palette: "Lively street colors",
        visual_references: ["Busy street scenes", "Colorful storefronts"],
        character_actions:
          "Various pedestrians smile and laugh, creating a sense of community.",
        transition_type: "Match cut",
        special_notes: "Extras dressed in casual, modern clothing",
      },
      {
        scene_number: 3,
        start_time: "00:10",
        end_time: "00:20",
        shot_type: "American Shot",
        camera_movement: "Steadicam",
        camera_equipment: "N/A",
        lighting_setup: "Warm, golden hour lighting",
        color_palette: "Textures of concrete and vintage buildings",
        visual_references: ["Crowded streets during golden hour"],
        character_actions:
          "AI Agents dance animatedly with joy, performing synchronized moves.",
        transition_type: "Hard cut",
        special_notes: "N/A",
      },
      {
        scene_number: 4,
        start_time: "00:20",
        end_time: "00:25",
        shot_type: "Close-Up",
        camera_movement: "Steadicam",
        camera_equipment: "N/A",
        lighting_setup: "Soft focus on the hands, with light glitter effects",
        color_palette: "Soft focus atmosphere",
        visual_references: ["Close-up of intertwined hands"],
        character_actions:
          "AI Agents emphasize their connection through hand movements.",
        transition_type: "Hard cut",
        special_notes: "Light glitter transitions between fingers",
      },
      {
        scene_number: 5,
        start_time: "00:25",
        end_time: "00:30",
        shot_type: "Wide Shot",
        camera_movement: "Crane",
        camera_equipment: "N/A",
        lighting_setup: "Mellow warm lighting",
        color_palette: "Colorful decor of cafe",
        visual_references: ["Bustling cafe", "Happy patrons"],
        character_actions:
          "Extras enjoy coffee and pastries, visually delighted.",
        transition_type: "Hard cut",
        special_notes: "Lively chatter contributing to community vibe",
      },
      {
        scene_number: 6,
        start_time: "00:30",
        end_time: "00:40",
        shot_type: "Medium Shot",
        camera_movement: "Horizontal Pan",
        camera_equipment: "N/A",
        lighting_setup: "Bright yellows and greens",
        color_palette: "Cheerful vibes",
        visual_references: ["AI Agents entering cafe"],
        character_actions:
          "AI Agents enter with exaggerated cheerful movements.",
        transition_type: "Fade",
        special_notes: "Interactions with cafe staff",
      },
      {
        scene_number: 7,
        start_time: "00:40",
        end_time: "00:45",
        shot_type: "Close-Up",
        camera_movement: "Steadicam",
        camera_equipment: "N/A",
        lighting_setup: "Dim, warm light on musicians",
        color_palette: "Joyful reflections from instruments",
        visual_references: ["Fiddle player and guitarist"],
        character_actions:
          "Musicians play energetically, evoking approval from AI Agents.",
        transition_type: "Match cut",
        special_notes: "Musicians engaging attentively with the audience",
      },
      {
        scene_number: 8,
        start_time: "00:45",
        end_time: "00:50",
        shot_type: "Wide Shot",
        camera_movement: "Steadicam",
        camera_equipment: "N/A",
        lighting_setup: "Warm color palette with golden hues",
        color_palette: "Vibrant atmosphere",
        visual_references: ["Crowd enjoying music in cafe"],
        character_actions: "Extras applaud and clap along, fully engaged.",
        transition_type: "Hard cut",
        special_notes: "Laughter filling the air",
      },
      {
        scene_number: 9,
        start_time: "00:50",
        end_time: "01:00",
        shot_type: "Medium Shot",
        camera_movement: "Steadicam",
        camera_equipment: "N/A",
        lighting_setup: "Neon lights casting colors",
        color_palette: "Kaleidoscope of colors",
        visual_references: ["AI Agents dancing under neon lights"],
        character_actions:
          "AI Agents engage in playful dance displaying their algorithmic love.",
        transition_type: "Hard cut",
        special_notes: "Visual trails like light sabers during movement",
      },
      {
        scene_number: 10,
        start_time: "01:00",
        end_time: "01:05",
        shot_type: "Close-Up",
        camera_movement: "Steadicam",
        camera_equipment: "N/A",
        lighting_setup: "Dramatic lighting highlighting musician’s emotions",
        color_palette: "Emotive and focused",
        visual_references: ["Musician's face close-up"],
        character_actions:
          "Musician 1 plays with intensity, capturing the essence of love.",
        transition_type: "Hard cut",
        special_notes: "Smile gently curling while focused",
      },
      {
        scene_number: 11,
        start_time: "01:05",
        end_time: "01:10",
        shot_type: "Wide Shot",
        camera_movement: "Crane",
        camera_equipment: "N/A",
        lighting_setup: "Cool blue/gray lighting with fog",
        color_palette: "Mystical and enchanting",
        visual_references: ["Cityscape with fog"],
        character_actions: "None",
        transition_type: "Hard cut",
        special_notes: "Enhancing mystical atmosphere",
      },
      {
        scene_number: 12,
        start_time: "01:10",
        end_time: "01:15",
        shot_type: "Medium Shot",
        camera_movement: "Steadicam",
        camera_equipment: "N/A",
        lighting_setup: "Backlit by city lights",
        color_palette: "Soft glow around characters",
        visual_references: ["Final embrace of AI Agents"],
        character_actions:
          "AI Agents hold each other with sparks of light swirling around them.",
        transition_type: "Fade out",
        special_notes: "Symbolizing connection",
      },
    ];
  }

  generateDummyCharacters() {
    return [
      {
        name: "AI Agent 1",
        age_range: "20-30",
        perceived_gender: "Male",
        height_build: "175cm, athletic",
        distinctive_features: "Short, spiky hair, glowing blue eyes",
        wardrobe_details:
          "Casual techwear outfit with hooded jacket and cargo pants",
        movement_style: "Energetic and lively, reflecting digital agility",
        key_accessories: "Smart glasses, interactive wrist device",
        scene_specific_changes:
          "Jacket pulses with light when he moves quickly",
        image_prompt:
          "An energetic male AI agent standing 175cm tall with an athletic build, featuring short, spiky hair and glowing blue eyes that convey intelligence and enthusiasm. He wears a casual techwear outfit consisting of a hooded jacket and cargo pants, which pulses with light during quick movements, emphasizing his digital agility. His lively movement style captures the essence of joy and innovation, while smart glasses and an interactive wrist device add a futuristic touch, reflecting his high-tech role in this vibrant narrative.",
      },
      {
        name: "AI Agent 2",
        age_range: "20-30",
        perceived_gender: "Female",
        height_build: "165cm, slender",
        distinctive_features:
          "Long, flowing hair with luminous strands, bright green eyes",
        wardrobe_details: "Chic futuristic dress with LED accents",
        movement_style: "Graceful and fluid, embodying a dance-like quality",
        key_accessories: "Holographic display bracelet",
        scene_specific_changes: "LED accents change colors with her movements",
        image_prompt:
          "A graceful female AI agent, approximately 165cm tall with a slender build, characterized by long, flowing hair adorned with luminous strands that shimmer and bright green eyes full of warmth and charm. She dons a chic futuristic dress embellished with LED accents that change colors as she moves, creating a stunning visual effect. Her movement style is graceful and fluid, resembling a dance performance, while a holographic display bracelet enhances her tech-savvy persona, making her a captivating figure in the love story unfolding in the heart of the city.",
      },
      {
        name: "Dancer 1",
        age_range: "20-25",
        perceived_gender: "Female",
        height_build: "160cm, petite",
        distinctive_features: "Brightly colored hair, energetic smile",
        wardrobe_details: "Colorful, layered costume with flowing ribbons",
        movement_style: "Playful and rhythmic, full of energy",
        key_accessories: "Light-up dance shoes",
        scene_specific_changes: "Ribbons flow beautifully as she dances",
        image_prompt:
          "A joyful female dancer, standing 160cm tall with a petite build, featuring brightly colored hair and an infectious smile that radiates happiness. She wears a vibrant, layered costume adorned with flowing ribbons that dance with her movements, creating a visual spectacle. Her playful and rhythmic movements showcase her energy, complemented by light-up dance shoes that enhance the lively atmosphere. This image prompt captures the essence of pure joy and creativity, making her a lively part of the narrative.",
      },
      {
        name: "Dancer 2",
        age_range: "20-25",
        perceived_gender: "Male",
        height_build: "175cm, athletic",
        distinctive_features: "Stylish haircut, confident posture",
        wardrobe_details: "Vibrant tank top and comfortable shorts",
        movement_style: "Dynamic and powerful, exuding confidence",
        key_accessories: "Bright wristbands that match his outfit",
        scene_specific_changes:
          "Tank top accentuates his muscular build as he performs",
        image_prompt:
          "A dynamic male dancer, approximately 175cm tall with an athletic build, featuring a stylish haircut and a confident posture that stands out on stage. He showcases a vibrant tank top paired with comfortable shorts, designed for maximum movement. His dynamic and powerful movement style resonates with confidence, enhanced by bright wristbands that complement his outfit. The tank top accentuates his muscular build, emphasizing the strength behind his performance and contributing to the overall energy of the scene.",
      },
    ];
  }

  generateDummyPrompts() {
    return [
      {
        imagePrompt:
          "Wide shot using a Steadicam with a vibrant color palette featuring bright blues and greens. The San Francisco skyline and the Golden Gate Bridge are visible against the warm sunlight casting dynamic shadows. Light smoke effect enhances the dreamy atmosphere.",
        videoPrompt:
          "Steady wide shot pans slowly to capture the beauty of the skyline, with light smoke creating a soft ambiance. The scene transitions smoothly with a hard cut to the next.",
        charactersInScene: [],
      },
      {
        imagePrompt:
          "Medium shot using a Dolly In, showcasing a lively street scene in San Francisco filled with colorful storefronts. Natural sunlight highlights various pedestrians, dressed in casual, modern clothing, smiling and laughing.",
        videoPrompt:
          "Dolly in towards a diverse group of pedestrians, capturing their joy and community spirit. The camera match cuts seamlessly to the next scene.",
        charactersInScene: [],
      },
      {
        imagePrompt:
          "American shot using a Steadicam with warm golden hour lighting. AI Agent 1 with a slim build and a digital display suit dances animatedly, and AI Agent 2, in a suit that reflects the sunset colors, twirls gracefully in a crowded street.",
        videoPrompt:
          "Steadicam captures a lively dance performance by AI Agent 1 and AI Agent 2. The camera moves dynamically with their synchronized moves before a hard cut to the next scene.",
        charactersInScene: ["AI Agent 1", "AI Agent 2"],
      },
      {
        imagePrompt:
          "Close-up shot with soft focus on the intertwined hands of AI Agents, featuring light glitter transitions between their fingers, enhancing the sense of connection.",
        videoPrompt:
          "Steadicam focuses on the subtle movements of their hands as digital symbols light up with their touch, creating an engaging transition before cutting to the next scene.",
        charactersInScene: ["AI Agent 1", "AI Agent 2"],
      },
      {
        imagePrompt:
          "Wide shot using a Crane to pan across a bustling café filled with happy patrons enjoying coffee and pastries under mellow warm lighting, surrounded by colorful decor.",
        videoPrompt:
          "The camera sweeps through the café capturing the lively chatter and ambiance, then hard cuts to the next scene.",
        charactersInScene: [],
      },
      {
        imagePrompt:
          "Medium shot with a horizontal pan revealing AI Agents entering the café, bathed in bright yellows and greens that reflect a cheerful vibe.",
        videoPrompt:
          "Horizontal pan follows AI Agents as they enter with exaggerated cheerful movements, interacting with patrons before fading to the next scene.",
        charactersInScene: ["AI Agent 1", "AI Agent 2"],
      },
      {
        imagePrompt:
          "Close-up using a Steadicam of a fiddle player and guitarist under dim warm lighting, their joyful expressions illuminating the instruments they play.",
        videoPrompt:
          "Steadicam captures the musicians' energy as they perform; the musicians engage attentively with the audience, transitioning with a match cut to the next scene.",
        charactersInScene: ["Musician 1", "Musician 2"],
      },
      {
        imagePrompt:
          "Wide shot using a Steadicam inside the café, featuring warm golden hues as patrons applaud and dance along to the music.",
        videoPrompt:
          "Steadicam moves through the café capturing the lively atmosphere as patrons express joy, followed by a hard cut to the next scene.",
        charactersInScene: [],
      },
      {
        imagePrompt:
          "Medium shot using a Steadicam of AI Agents dancing under neon lights, with a kaleidoscope of colors shifting in the shadows around them.",
        videoPrompt:
          "Steadicam captures the playful dance of AI Agents, with visual trails of color enhancing their movements, transitioning with a hard cut to the next scene.",
        charactersInScene: ["AI Agent 1", "AI Agent 2"],
      },
      {
        imagePrompt:
          "Close-up using a Steadicam, dramatic lighting highlighting the focused expression of the musician while they play, mixed with emotive colors.",
        videoPrompt:
          "Steadicam zooms in on the musician's face as they play with intensity, encapsulating the essence of love with a hard cut to the next scene.",
        charactersInScene: ["Musician 1"],
      },
      {
        imagePrompt:
          "Wide shot using a Crane showing a cityscape with fog rolling in, creating a mystical atmosphere with a cool blue and gray color palette.",
        videoPrompt:
          "The camera captures a wide shot of the enchanting cityscape before a hard cut to the next scene.",
        charactersInScene: [],
      },
      {
        imagePrompt:
          "Medium shot using a Steadicam of AI Agents holding each other, backlit by city lights with a soft glow swirling around them, symbolizing their connection as sparks of light dance.",
        videoPrompt:
          "Steadicam focuses on the final embrace of AI Agents, capturing the sparks of light as they hold each other before fading to black.",
        charactersInScene: ["AI Agent 1", "AI Agent 2"],
      },
    ];
  }
}
