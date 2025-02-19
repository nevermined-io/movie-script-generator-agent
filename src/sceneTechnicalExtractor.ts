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
    {
      idea: string;
      title: string;
      lyrics: string;
      tags: string;
      duration: number;
    },
    string
  >;
  private sceneChain: RunnableSequence<
    { script: string },
    Record<string, any>[]
  >;
  private settingsChain: RunnableSequence<
    { script: string },
    Record<string, any>[]
  >;
  private characterChain: RunnableSequence<
    { script: string; lyrics: string; tags: string },
    Record<string, any>[]
  >;
  private technicalTransformationChain: RunnableSequence<
    { scenes: string; settings: string; characters: string; script: string },
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
           - Every scene must have a duration of either 5 or 10 seconds.
           - Plan accordingly the number of scenes given the total duration of the video. If the total duration of the video is, for example, 200 seconds, you should create 20 scenes of 10 seconds each or 40 scenes of 5 seconds each or a combination of both.
        
        5. **Include Scenes with Live Musicians**:
           - At least two scenes must feature a visible band or musicians playing instruments that complement the main story of the two AIs.
           - Show how these musicians integrate into the video’s narrative or setting.
        
        **Output Format**:  
        
        SCENE [NUMBER] - [DURATION IN SECONDS] seconds
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

        **Duration**:
        {duration} seconds
        `),
      llm,
      new StringOutputParser(),
    ]);

    this.sceneChain = RunnableSequence.from([
      ChatPromptTemplate.fromTemplate(`
        Extract technical scene details as a JSON array. 
        Return **one object per SCENE block** in the same order they appear in the script. 
        Use these fields exactly:
        
        - "sceneNumber" (integer)
        - "startTime" (MM:SS)
        - "endTime" (MM:SS)
        - "shotType"
        - "cameraMovement"
        - "cameraEquipment"
        - "lightingSetup"
        - "colorPalette"
        - "visualReferences" (array of 2-3 strings)
        - "characterActions" (describe key actions relevant to each character at specific lyric or musical cue)
        - "transitionType"
        - "specialNotes" (any additional gear, safety, or creative note)
        
        **Important**:
        1. Preserve the **sceneNumber** from the script. If the script says "SCENE 1 - 20", interpret that as sceneNumber = 1 and duration = 20 seconds.
        2. Convert durations to approximate "startTime" and "endTime" in MM:SS, adding them sequentially so the entire video doesn't exceed 3 minutes.
           - For example, if SCENE 1 has 20 seconds, it might be startTime="00:00", endTime="00:20".
           - SCENE 2 (30 seconds) might be startTime="00:20", endTime="00:50", etc.
        3. Do not skip any scenes. Return them in the same order.
        4. If a scene references location or certain camera gear, place that info under the correct fields. 
        5. Do not add or remove scenes; parse exactly from the script.
        
        Example (shortened):
        
        [
          {{
            "sceneNumber": 1,
            "startTime": "00:00",
            "endTime": "00:15",
            "shotType": "Wide Shot",
            "cameraMovement": "Steadicam",
            "cameraEquipment": "Canon EOS R5 with 24-70mm lens",
            "lightingSetup": "Morning sunlight (5600K)",
            "colorPalette": "Soft gold and pastel",
            "visualReferences": ["La La Land opening dance", "Impressionist sunrise feel"],
            "characterActions": "AI Agent 1 gazes at horizon; AI Agent 2 approaches slowly",
            "transitionType": "Hard cut",
            "specialNotes": "Some aerial drone shots if possible"
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

    // Settings Extraction Chain
    this.settingsChain = RunnableSequence.from([
      ChatPromptTemplate.fromTemplate(`
        Analyze the script and extract DISTINCT SETTINGS/LOCATIONS. For each unique setting:
        
        1. Create a detailed description including:
           - Physical space characteristics
           - Lighting conditions
           - Color palette
           - Key visual elements
           - Ambient elements (weather, time of day)
           - Image style (e.g., cyberpunk, retro-futuristic, dystopian, comic book, realistic, 3D, etc.)
        
        2. Generate an image prompt for each setting
        
        3. Return JSON array with:
           - "id": Unique identifier (e.g., "setting-1")
           - "name": Short descriptive name
           - "description": Full setting description. 
           - "imagePrompt": Visual prompt for static setting. This will be used as a prompt for visual generation, so this must include and condense all the key elements described above, creating a vivid visual description. This will be the only field that will be used for visual generation, so make sure it's detailed and evocative.
           - "keyFeatures": Array of 3-5 distinctive elements
        
        Example:
        [{{
            id: "setting-1",
            name: "Golden Gate Bridge at Dusk",
            description:
              "A picturesque view of the Golden Gate Bridge at dusk, bathed in warm golden hues and surrounded by bokeh effects from the city lights. Soft flares of light cut through a subtle hint of fog as ADA and BLAKE, two humanoid figures with glowing circuitry, gaze appreciatively at the bridge's beauty.",
            imagePrompt:
              "Ultra-detailed wide shot of the Golden Gate Bridge at dusk, bathed in warm golden hues with soft bokeh from city lights and a delicate veil of fog. Radiant flares emphasize the majestic structure in a cinematic composition. Rendered in a 'Neo-Vivid Dreamscape' style that fuses futuristic cyberpunk elements with painterly textures and luminous neon glows, evoking a surreal, immersive atmosphere.",
            keyFeatures: [
              "Golden Gate Bridge",
              "Warm golden hues",
              "Bokeh city lights",
              "Soft flares",
              "Humanoid AI figures",
            ],
          }},
          {{
            id: "setting-2",
            name: "Waterfront Sidewalk",
            description:
              "A romantic waterfront sidewalk scene glowing with rich greens and blues. Softly glowing street lamps illuminate the area as mist subtly rolls in, enhancing the intimacy of the moment between ADA and BLAKE, who interact playfully as they run alongside the water.",
            imagePrompt:
              "Ultra-detailed medium shot of a romantic waterfront sidewalk at night, drenched in rich greens and blues. Soft street lamp glows gently illuminate the misty surroundings along the water’s edge, creating an intimate and enchanting atmosphere. Rendered in a 'Neo-Vivid Dreamscape' style that merges futuristic cyberpunk motifs with expressive, painterly textures and vibrant neon accents.",
            keyFeatures: [
              "Waterfront",
              "Rich greens and blues",
              "Glowing street lamps",
              "Romantic atmosphere",
              "Playful interaction of AI figures",
            ],
          }},
          {{
            id: "setting-3",
            name: "Rooftop Party Scene",
            description:
              "A vibrant rooftop party in full swing with jazz musicians under a stunning sunset. The scene bursts with warm yellows, reds, and greens, surrounded by floating balloons. The ambient sunset light creates an energetic vibe as ADA and BLAKE dance joyously beneath the awning.",
            imagePrompt:
              "Ultra-detailed wide shot of a vibrant rooftop party scene at sunset, featuring live jazz musicians amid cascades of warm yellows, reds, and greens. Colorful floating balloons and ambient sunset lighting create a dynamic, celebratory atmosphere with lively dance movements. Rendered in a 'Neo-Vivid Dreamscape' style that blends futuristic cyberpunk flair with expressive, painterly illumination and surreal neon accents.",
            keyFeatures: [
              "Rooftop atmosphere",
              "Jazz musicians",
              "Sunset lighting",
              "Floating balloons",
              "Joyous dancing of AI figures",
            ],
          }}
        ]
        
        Script: {script}
      `),
      llm,
      extractJsonRunnable,
      new JsonOutputParser(),
    ]);

    // Character Extraction Chain
    this.characterChain = RunnableSequence.from([
      ChatPromptTemplate.fromTemplate(`
        Extract ALL characters from the script with detailed physical descriptors and roles, taking into account the provided song lyrics and tags as additional context. Return a valid JSON array of objects. Each object must include the following keys:

        "name" (if a character has no explicit name, use "Unnamed [role]")
        "ageRange"
        "perceivedGender"
        "heightBuild"
        "distinctiveFeatures"
        "wardrobeDetails"
        "movementStyle"
        "keyAccessories"
        "sceneSpecificChanges"
        "imagePrompt"

        Important Instructions:

        Include every character mentioned in the script, not only the musicians. If there are characters that are part of the narrative (such as background dancers, story characters, or extras), they must all appear in the output list.
        Their name must match the script's name for the character.
        If there are references to a band or musicians, list each musician separately with details including their instrument, wardrobe, and any unique features.
        Maintain consistency with the script’s descriptions (or make the best assumptions if not explicitly stated).
        Use the provided song lyrics and tags as additional context when inferring character details.
        For the "imagePrompt" field:
        Synthesize all the character attributes (physical features, age, gender, height/build, distinctive features, wardrobe details, movement style, key accessories, and any scene-specific changes) into one complete, vivid visual description.
        The prompt should serve as a detailed instruction for a visual generator, clearly conveying how the character should appear in the music video.

        Include the image style, color palette, and lighting conditions to ensure the character fits seamlessly into the video's visual aesthetic.

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
            "ageRange": "25-30",
            "perceivedGender": "Androgynous",
            "heightBuild": "180cm, lean",
            "distinctiveFeatures": "Glowing circuit tattoos, cybernetic left eye",
            "wardrobeDetails": "Distressed leather jacket with metallic accents",
            "movementStyle": "Fluid and dynamic gestures",
            "keyAccessories": "Neon microphone, digital wristband",
            "sceneSpecificChanges": "Jacket tears and illuminates during intense moments",
            "imagePrompt": "Ultra-detailed portrait of an androgynous performer, aged 25-30 with a lean 180cm build. The subject features glowing circuit tattoos and a cybernetic left eye that exude a mysterious cyberpunk aura. Dressed in a distressed leather jacket with metallic accents that tears and illuminates with bursts of light during moments of intensity, the dynamic pose captures fluid, energetic stage movements enhanced by neon accessories. Rendered in a 'Neo-Cyber Renaissance' style that blends futuristic cyberpunk elements with dynamic painterly textures and radiant neon highlights."
          }},
          {{
            "name": "Bassist",
            "ageRange": "30-40",
            "perceivedGender": "Female",
            "heightBuild": "170cm, curvy",
            "distinctiveFeatures": "Long flowing hair, captivating stage presence with confident gestures",
            "wardrobeDetails": "Turquoise maxi skirt, fitted crop top, and a leather belt",
            "movementStyle": "Smooth and flowing, exuding confidence and allure while still maintaining rhythmic focus",
            "keyAccessories": "Bass guitar with inlay lights, statement jewelry",
            "sceneSpecificChanges": "Jewelry glitters under the stage lights, the skirt flows beautifully with her movements",
            "imagePrompt": "Ultra-detailed portrait of a confident female performer with a curvy build, aged 30-40 and standing approximately 170cm tall. The subject boasts long, flowing hair and a mesmerizing stage presence defined by graceful, assured gestures. Clad in a turquoise maxi skirt paired with a fitted crop top and a leather belt, her smooth and flowing movements harmonize with the striking illuminated bass guitar and sparkling statement jewelry. Rendered in a 'Neo-Cyber Renaissance' style that merges futuristic cyberpunk aesthetics with expressive painterly details and vivid neon luminosity."
          }},
          {{
            "name":"AI Character 2",
            "ageRange":"Unnamed",
            "perceivedGender":"Female",
            "heightBuild":"165cm, slim",
            "distinctiveFeatures":"Illuminated circuit tattoos outlining the face",
            "wardrobeDetails":"Flowing gown adorned with reflective surfaces",
            "movementStyle":"Graceful and fluid, almost like water",
            "keyAccessories":"Holographic wrist tablet",
            "sceneSpecificChanges":"Gown glimmers under stage lights, reflecting colors",
            "imagePrompt": "Ultra-detailed portrait of a futuristic female figure with a slim 165cm build. Striking illuminated circuit tattoos outline her face in intricate patterns, evoking a sense of technological mystique. She wears a flowing gown with reflective surfaces that glimmer under stage lighting, and her graceful, water-like movements enhance her ethereal presence. A holographic wrist tablet projects digital data, deepening the avant-garde visual narrative. Rendered in a 'Neo-Cyber Renaissance' style that fuses cutting-edge cyberpunk innovation with surreal, painterly textures and luminous neon effects."
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
        Transform the technical details of the scenes into production prompts including composition details, actions and camera movements. For **each scene** in the "Scene Data", generate **one object** with three attributes:

        - "prompt"
        - "charactersInScene"
        - "settingId"

        **Instructions**:

        1. **Number of prompts**: You must produce as many objects as there are scenes in the JSON. If there are 9 scenes, return 9 objects in an array.

        2. **Character references**:
          - In "prompt", replace each character name with a full physical description taken from CHARACTER_DATA.
          - If the character is a musician, mention their instrument in the prompt (e.g., "carrying a vintage acoustic guitar").
          - In "charactersInScene", list the characters present in the scene. **Ensure that the names are exactly as they appear in CHARACTER_DATA, without any modifications or additional descriptions.** If only one character is present, list only that character. If none is present, just leave an empty array.
          - If we are referring to a character that is not in CHARACTER_DATA, charactersInScene should not include it.

        3. **Scene details**:
          - Use the data from the scene (composition, shotType, cameraMovement, lightingSetup, colorPalette, etc.).
          - Create a prompt for every scene that summarizes what happens in the scene. These images will be presented sequentially and used to generate a short video where the videos will be stitched together. We are limited by the duration of video creation so we need these images to guide a universal feel that connects each scene and image to the prior and the following scene and image, thus creating a cohesive series of videos.
          - Add the duration of the scene in seconds in the field "duration", knowing that the scene we are referring to includes the start time and end time of the scene. Duration must be 5 or 10 seconds.

        4. **Precision**:
          - Use professional cinematography terminology (e.g., "close-up with shallow depth of field", "slow dolly in", "neon underlighting").
          - Mention the lens or camera gear from the scene data.

        5. **Integration**:
          - Integrate the "Script Context" only if it adds crucial narrative or visual detail.
          - For musician scenes, describe how the instruments and performance integrate with the shot.

        6. **Style and Tone**:
          - Maintain a consistent style and tone throughout the prompts.
          - Ensure that the prompts are detailed enough to guide the visual creation process

        7. **Output format**: 
          - Return a JSON array where each element corresponds to one scene. 
          Example:
        [
          {{
            "sceneNumber": 1,
            "prompt": "Cinematic wide shot using a 24mm lens on a tripod. A dark, cluttered room with warm, muted earth tones. A male android in his 20s with glowing blue circuit patterns beneath translucent synthetic skin, wearing a worn leather jacket, centered in the frame. Smooth dolly movement from left to right with a slow pan and fade transitions. The character performs subtle gestures that sync with the scene's rhythm, while digital effects and soft lighting transitions enhance the cinematic atmosphere.",
            "charactersInScene": ["Lead Singer", "Guitarist"],
            "settingId": "setting-1",
            "duration": 5
          }},
          {{
            "sceneNumber": 2,
            "prompt": "Close-up over-the-shoulder shot using a 50mm lens. An androgynous digital entity (aged between 25-30) with ever-changing holographic facial features, wearing fragmented luminous projections resembling a business suit, captured with high contrast and sharp detail. Subtle zoom-in combined with a glitch transition. Camera movement is complemented by the character's dynamic actions, such as shifting poses and expressive gestures, with synchronized digital distortions and soft fades to create dramatic tension.",
            "charactersInScene": ["Digital Entity"],
            "settingId": "setting-2",
            "duration": 5
          }},
          {{
            "sceneNumber": 3,
            "prompt": "Medium shot using an 85mm lens, capturing a dynamic urban street scene at dusk with neon lights reflecting off wet pavement. A female cyborg in her early 30s, with silver mechanical limbs and vibrant red hair, wearing a sleek futuristic jacket, stands poised in the frame. Tracking shot with a steady cam as the camera follows the character walking briskly along the street. The character glances over her shoulder and raises her right arm to shield her eyes from a sudden burst of light, synchronized with rapid cuts and energetic digital overlays.",
            "charactersInScene": ["AI Character 2"],
            "settingId": "setting-3",
            "duration": 10
          }}
        ]

        
        **CHARACTER_DATA**: {characters}
        **SCENE DATA**: {scenes}
        **SETTINGS DATA**: {settings}
        **SCRIPT CONTEXT**: {script}

        Do not add any explanations or markdown.
          `),
      llm,
      extractJsonRunnable,
      new JsonOutputParser(),
    ]);
  }

  async generateScript({
    idea,
    title,
    lyrics,
    tags,
    duration,
  }): Promise<string> {
    if (IS_DUMMY) return this.generateDummyScript();
    return await this.scriptChain.invoke({
      idea,
      title,
      lyrics,
      duration,
      tags: tags.join(", "),
    });
  }

  async extractScenes(script: string): Promise<object[]> {
    if (IS_DUMMY) return this.generateDummyScenes();
    return await this.sceneChain.invoke({ script });
  }

  async extractSettings(script: string): Promise<object[]> {
    if (IS_DUMMY) return this.generateDummySettings();
    return await this.settingsChain.invoke({ script });
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
    settings: object[],
    script: string
  ): Promise<object[]> {
    if (IS_DUMMY) return this.generateDummyPrompts();
    return await this.technicalTransformationChain.invoke({
      scenes: JSON.stringify(scenes),
      characters: JSON.stringify(characters),
      settings: JSON.stringify(settings),
      script,
    });
  }

  generateDummyScript() {
    return `SCENE 1 - 5 SECONDS
      Wide shot | Steadicam | Golden Gate Bridge at dusk
      Aesthetic: Warm golden hues with bokeh effects from city lights; soft flares with a hint of fog.
      Characters:
      - AI Agent 1 (ADA): Artificial, humanoid figure with glowing blue circuitry on a sleek silver bodysuit, a soft luminescent face.
      - AI Agent 2 (BLAKE): Similar design with purple circuitry, more angular in features, wearing a violet and silver bodysuit.
      ADA and BLAKE gaze appreciatively at the bridge, appearing enchanted by its beauty.
      Transition: Hard cut to next scene

      SCENE 2 - 5 SECONDS
      Medium shot | Horizontal pan | Sidewalk near the waterfront
      Aesthetic: Rich greens and blues, with soft street lamp glow creating a romantic atmosphere. Mist rolling in subtly.
      Characters:
      - ADA: Reaches out to BLAKE with a twinkle in her eyes, signaling him to follow.
      - BLAKE: Smiles, playfully mocking a hug gesture as they run alongside the water.
      Transition: Match cut to next scene, transitioning into musical beat.

      SCENE 3 - 5 SECONDS
      Close-up | Gimbal stabilizer | Close on their hands
      Aesthetic: Soft focus on their intertwined digital hands, sparkling with data particles drifting away.
      Characters:
      - ADA & BLAKE: Their fingers, encased in LED glows, create a luminous interaction, transmitting swirling codes between them.
      Transition: Fade to next scene

      SCENE 4 - 10 SECONDS
      Wide shot | Crane | Rooftop party scene with jazz musicians playing
      Aesthetic: Intense color burst of warm yellows, reds, and greens; ambient sunset lighting accented with floating balloons.
      Characters:
      - Jazz Band: Four musicians (saxophonist, drummer, bassist, violinist), vibrant clothing, lively demeanor.
      - ADA & BLAKE: They dance joyously beneath an awning as the music plays, drawn to the harmony.
      Transition: Horizontal pan to focus on band members

      SCENE 5 - 10 SECONDS
      American shot | Dolly zoom | Rooftop with live musicians
      Aesthetic: Contrasting lighting with bright spotlight on musicians, background lights twinkling like stars.
      Characters:
      - Jazz Band members: Engaged in vibrant play, creating a vibe that lifts ADA and BLAKE's energy.
      - ADA & BLAKE: Swap moves, merging their dance and digital representations in sync with the beat.
      Transition: Hard cut to next scene

      SCENE 6 - 5 SECONDS
      Medium shot | Vertical pan | Baker Beach, Golden Gate in the background
      Aesthetic: Bright sunset colors, silhouetted figures against golden rays kissing the water.
      Characters:
      - ADA & BLAKE: Skipping playfully along the shore, laughter-like emojis visually animating around them.
      Transition: Match cut to next scene

      SCENE 7 - 10 SECONDS
      Wide shot | Steadicam | Floating above bay with a view of the stars
      Aesthetic: Dark blues and pure whites, cosmic sparkles merging with the water's reflections; ambient twinkling stars.
      Characters:
      - ADA & BLAKE: Posing with arms outstretched as if flying, illuminated by cosmic glow.
      Transition: Fade to next scene

      SCENE 8 - 10 SECONDS
      Close-up | Gimbal stabilizer | Their glowing faces side by side
      Aesthetic: Faint glow of digital matrix in the background; a misty look with slow-motion effects.
      Characters:
      - ADA & BLAKE: Tender smiles, sharing a visual moment representing the connection they’ve built.
      Transition: Hard cut to next scene

      SCENE 9 - 10 SECONDS
      Wide shot | Horizontal pan | Street festival backdrop
      Aesthetic: Bursting colors from decorations and lights; festive string lights and dynamic fireworks in the sky.
      Characters:
      - Festival Goers: Colorful attire, laughing and cheering, representing the joyous celebration around ADA & BLAKE.
      - ADA & BLAKE: Join hands, dancing, with music literally seen as visual waves emanating from them.
      Transition: Fade to next scene

      SCENE 10 - 5 SECONDS
      Medium shot | Crane | Sparkling fireworks above the Bay
      Aesthetic: Dramatic dark sky sprinkled with vibrant fireworks; epic contrasts highlighting joy.
      Characters:
      - ADA & BLAKE: Gazing up, awed by the bursts of color, reflected light sparkling on their faces.
      Transition: Hard cut to next scene

      SCENE 11 - 10 SECONDS
      American shot | Steadicam | Street below the bridge
      Aesthetic: Urban tones with splashes of color from street art; dynamic shadows as the sun begins to dip.
      Characters:
      - ADA & BLAKE: Dance among bustling pedestrians, joyfully exploring nighttime neon lights.
      Transition: Fade to next scene

      SCENE 12 - 5 SECONDS
      Wide shot | Static camera | Under the illuminated Golden Gate Bridge
      Aesthetic: Neons and warm streetlight hues creating a magical ambiance beneath the bridge.
      Characters:
      - ADA & BLAKE: They turn to each other, looking up at the bridge before leaning in for a digital 'kiss'—a burst of pixels.
      Transition: Fade to black and fade out the joyful fiddle tune.

      CHARACTER LIST:
      ADA (AI Agent 1): Silver bodysuit with glowing blue circuitry, soft luminous face, expressive eyes. Behavior: Playful, romantically engaging with BLAKE.
      BLAKE (AI Agent 2): Violet and silver bodysuit with purple circuitry, angular features. Behavior: Charismatic, mimics affectionate gestures with ADA.
      Jazz Band: Four musicians dressed in vibrant, festive clothes, each showcasing their instruments enthusiastically. Interaction: Provide the musical backdrop as ADA and BLAKE dance.
      Festival Goers: Mixed group of extras in colorful outfits, acting joyously, further setting the whimsical scene.`;
  }

  generateDummyScenes() {
    return [
      {
        sceneNumber: 1,
        startTime: "00:00",
        endTime: "00:05",
        shotType: "Wide shot",
        cameraMovement: "Steadicam",
        cameraEquipment: "N/A",
        lightingSetup: "Golden hour with warm hues and bokeh effects",
        colorPalette: "Warm golden hues",
        visualReferences: [
          "Golden Gate Bridge at dusk",
          "Soft flares with fog effects",
        ],
        characterActions:
          "ADA and BLAKE gaze appreciatively at the bridge, enchanted by its beauty.",
        transitionType: "Hard cut",
        specialNotes: "N/A",
      },
      {
        sceneNumber: 2,
        startTime: "00:05",
        endTime: "00:10",
        shotType: "Medium shot",
        cameraMovement: "Horizontal pan",
        cameraEquipment: "N/A",
        lightingSetup: "Soft street lamp glow",
        colorPalette: "Rich greens and blues",
        visualReferences: ["Romantic waterfront", "Mist rolling in"],
        characterActions:
          "ADA reaches out to BLAKE, signaling him to follow; BLAKE smiles and mocks a hug gesture.",
        transitionType: "Match cut",
        specialNotes: "Transition into musical beat",
      },
      {
        sceneNumber: 3,
        startTime: "00:10",
        endTime: "00:15",
        shotType: "Close-up",
        cameraMovement: "Gimbal stabilizer",
        cameraEquipment: "N/A",
        lightingSetup: "Soft focus lighting",
        colorPalette: "N/A",
        visualReferences: ["Digital interactions", "Data particles"],
        characterActions:
          "ADA and BLAKE's fingers create a luminous interaction, transmitting swirling codes.",
        transitionType: "Fade",
        specialNotes: "N/A",
      },
      {
        sceneNumber: 4,
        startTime: "00:15",
        endTime: "00:25",
        shotType: "Wide shot",
        cameraMovement: "Crane",
        cameraEquipment: "N/A",
        lightingSetup: "Ambient sunset lighting",
        colorPalette: "Warm yellows, reds, and greens",
        visualReferences: ["Rooftop party", "Jazz musicians"],
        characterActions:
          "Jazz Band plays while ADA and BLAKE dance joyously beneath an awning.",
        transitionType: "Horizontal pan",
        specialNotes: "N/A",
      },
      {
        sceneNumber: 5,
        startTime: "00:25",
        endTime: "00:40",
        shotType: "American shot",
        cameraMovement: "Dolly zoom",
        cameraEquipment: "N/A",
        lightingSetup: "Bright spotlight on musicians",
        colorPalette: "Contrasting with twinkling lights",
        visualReferences: ["Live musicians on rooftop", "Festive atmosphere"],
        characterActions:
          "Jazz Band engages in vibrant play, lifting ADA and BLAKE's energy as they dance.",
        transitionType: "Hard cut",
        specialNotes: "N/A",
      },
      {
        sceneNumber: 6,
        startTime: "00:40",
        endTime: "00:45",
        shotType: "Medium shot",
        cameraMovement: "Vertical pan",
        cameraEquipment: "N/A",
        lightingSetup: "Bright sunset colors",
        colorPalette: "Golden rays",
        visualReferences: [
          "Baker Beach with Golden Gate",
          "Silhouettes at sunset",
        ],
        characterActions:
          "ADA and BLAKE skip playfully along the shore, laughter emojis animate around them.",
        transitionType: "Match cut",
        specialNotes: "N/A",
      },
      {
        sceneNumber: 7,
        startTime: "00:45",
        endTime: "00:55",
        shotType: "Wide shot",
        cameraMovement: "Steadicam",
        cameraEquipment: "N/A",
        lightingSetup: "Dark blues and whites",
        colorPalette: "Cosmic sparkles",
        visualReferences: ["Bay under starlight", "Cosmic glow"],
        characterActions:
          "ADA and BLAKE pose with arms outstretched, illuminated by cosmic glow.",
        transitionType: "Fade",
        specialNotes: "N/A",
      },
      {
        sceneNumber: 8,
        startTime: "00:55",
        endTime: "01:05",
        shotType: "Close-up",
        cameraMovement: "Gimbal stabilizer",
        cameraEquipment: "N/A",
        lightingSetup: "Faint digital matrix glow",
        colorPalette: "Misty look",
        visualReferences: ["Tender moment", "Highlighted connection"],
        characterActions:
          "ADA and BLAKE share tender smiles, representing their built connection.",
        transitionType: "Hard cut",
        specialNotes: "N/A",
      },
      {
        sceneNumber: 9,
        startTime: "01:05",
        endTime: "01:15",
        shotType: "Wide shot",
        cameraMovement: "Horizontal pan",
        cameraEquipment: "N/A",
        lightingSetup: "Bursting festival colors",
        colorPalette: "Dynamic and festive",
        visualReferences: ["Street festival", "Fireworks in the sky"],
        characterActions:
          "Festival goers cheer and laugh; ADA and BLAKE join hands, dancing.",
        transitionType: "Fade",
        specialNotes: "N/A",
      },
      {
        sceneNumber: 10,
        startTime: "01:15",
        endTime: "01:20",
        shotType: "Medium shot",
        cameraMovement: "Crane",
        cameraEquipment: "N/A",
        lightingSetup: "Dark sky with vibrant fireworks",
        colorPalette: "Epic contrasts",
        visualReferences: ["Fireworks above the Bay", "Joyful moments"],
        characterActions:
          "ADA and BLAKE gaze up at the fireworks, awed by the colorful bursts.",
        transitionType: "Hard cut",
        specialNotes: "N/A",
      },
      {
        sceneNumber: 11,
        startTime: "01:20",
        endTime: "01:30",
        shotType: "American shot",
        cameraMovement: "Steadicam",
        cameraEquipment: "N/A",
        lightingSetup: "Urban tones with splashes of color",
        colorPalette: "Dynamic shadows",
        visualReferences: ["Street below the bridge", "Neon lights"],
        characterActions:
          "ADA and BLAKE dance among pedestrians, exploring night lights.",
        transitionType: "Fade",
        specialNotes: "N/A",
      },
      {
        sceneNumber: 12,
        startTime: "01:30",
        endTime: "01:35",
        shotType: "Wide shot",
        cameraMovement: "Static camera",
        cameraEquipment: "N/A",
        lightingSetup: "Illuminated bridge",
        colorPalette: "Neons and warm streetlight hues",
        visualReferences: ["Golden Gate Bridge lighting", "Magical ambiance"],
        characterActions:
          "ADA and BLAKE share a digital kiss under the bridge, a burst of pixels.",
        transitionType: "Fade to black",
        specialNotes: "Fade out joyful fiddle tune",
      },
    ];
  }

  generateDummySettings() {
    return [
      {
        id: "setting-1",
        name: "Golden Gate Bridge at Dusk",
        description:
          "A picturesque view of the Golden Gate Bridge at dusk, bathed in warm golden hues and surrounded by bokeh effects from the city lights. Soft flares of light cut through a subtle hint of fog as ADA and BLAKE, two humanoid figures with glowing circuitry, gaze appreciatively at the bridge's beauty.",
        imagePrompt:
          "Ultra-detailed wide shot of the Golden Gate Bridge at dusk, bathed in warm golden hues with soft bokeh from city lights and a delicate veil of fog. Radiant flares emphasize the majestic structure in a cinematic composition. Rendered in a 'Neo-Vivid Dreamscape' style that fuses futuristic cyberpunk elements with painterly textures and luminous neon glows, evoking a surreal, immersive atmosphere.",
        keyFeatures: [
          "Golden Gate Bridge",
          "Warm golden hues",
          "Bokeh city lights",
          "Soft flares",
          "Humanoid AI figures",
        ],
      },
      {
        id: "setting-2",
        name: "Waterfront Sidewalk",
        description:
          "A romantic waterfront sidewalk scene glowing with rich greens and blues. Softly glowing street lamps illuminate the area as mist subtly rolls in, enhancing the intimacy of the moment between ADA and BLAKE, who interact playfully as they run alongside the water.",
        imagePrompt:
          "Ultra-detailed medium shot of a romantic waterfront sidewalk at night, drenched in rich greens and blues. Soft street lamp glows gently illuminate the misty surroundings along the water’s edge, creating an intimate and enchanting atmosphere. Rendered in a 'Neo-Vivid Dreamscape' style that merges futuristic cyberpunk motifs with expressive, painterly textures and vibrant neon accents.",
        keyFeatures: [
          "Waterfront",
          "Rich greens and blues",
          "Glowing street lamps",
          "Romantic atmosphere",
          "Playful interaction of AI figures",
        ],
      },
      {
        id: "setting-3",
        name: "Rooftop Party Scene",
        description:
          "A vibrant rooftop party in full swing with jazz musicians under a stunning sunset. The scene bursts with warm yellows, reds, and greens, surrounded by floating balloons. The ambient sunset light creates an energetic vibe as ADA and BLAKE dance joyously beneath the awning.",
        imagePrompt:
          "Ultra-detailed wide shot of a vibrant rooftop party scene at sunset, featuring live jazz musicians amid cascades of warm yellows, reds, and greens. Colorful floating balloons and ambient sunset lighting create a dynamic, celebratory atmosphere with lively dance movements. Rendered in a 'Neo-Vivid Dreamscape' style that blends futuristic cyberpunk flair with expressive, painterly illumination and surreal neon accents.",
        keyFeatures: [
          "Rooftop atmosphere",
          "Jazz musicians",
          "Sunset lighting",
          "Floating balloons",
          "Joyous dancing of AI figures",
        ],
      },
      {
        id: "setting-4",
        name: "Baker Beach",
        description:
          "A serene beach setting at Baker Beach with the Golden Gate bridge in the background during a bright sunset. Silhouettes of ADA and BLAKE can be seen against golden rays kissing the water, while the atmosphere is filled with laughter and visual animations of joy.",
        imagePrompt:
          "Ultra-detailed medium shot of Baker Beach at a bright sunset, with the majestic Golden Gate Bridge silhouetted in the background. Warm, golden rays illuminate the water and playful silhouettes, enhanced by subtle visual effects that evoke laughter and delight. Rendered in a 'Neo-Vivid Dreamscape' style that combines futuristic cyberpunk nuances with soft, painterly textures and radiant neon glows.",
        keyFeatures: [
          "Baker Beach",
          "Golden Gate in the background",
          "Bright sunset colors",
          "Silhouetted figures",
          "Visual laughter animations",
        ],
      },
      {
        id: "setting-5",
        name: "Night Sky Over the Bay",
        description:
          "A tranquil view floating above the bay at night, showcasing dark blues and pure whites illuminated by cosmic sparkles. The water reflects the ambient twinkling stars as ADA and BLAKE pose under the cosmic glow, feeling the freedom of the night sky.",
        imagePrompt:
          "Ultra-detailed wide shot of a tranquil night sky over the bay, dominated by deep dark blues and crisp whites with cosmic sparkles and twinkling stars. The reflective water mirrors the celestial display, evoking a profound sense of freedom and serenity. Rendered in a 'Neo-Vivid Dreamscape' style that melds futuristic cyberpunk elements with dreamy, luminous textures and surreal neon highlights.",
        keyFeatures: [
          "Night sky",
          "Dark blues and whites",
          "Cosmic sparkles",
          "Twinkling stars",
          "AI figures expressing freedom",
        ],
      },
      {
        id: "setting-6",
        name: "Street Festival",
        description:
          "An electrifying street festival filled with vibrant decorations and festive string lights, underscored by dynamic fireworks in the night sky. The atmosphere is exhilarating as festival-goers in colorful attire joyfully celebrate around ADA and BLAKE, who dance among them with music visualized around them.",
        imagePrompt:
          "Ultra-detailed wide shot of an electrifying street festival at night, bursting with vibrant decorations, classic string lights, and dynamic fireworks illuminating the sky. The scene captures an exuberant celebration with lively dance movements and festive energy. Rendered in a 'Neo-Vivid Dreamscape' style that integrates futuristic cyberpunk aesthetics with rich, painterly textures and brilliant neon luminosity.",
        keyFeatures: [
          "Vibrant colors",
          "Festive decorations",
          "Vintage string lights",
          "Dynamic fireworks",
          "Joyous celebration",
        ],
      },
      {
        id: "setting-7",
        name: "Under the Golden Gate Bridge",
        description:
          "A magical ambiance found beneath the illuminated Golden Gate Bridge, characterized by warm streetlight hues and neon lights. ADA and BLAKE share a tender moment, looking up at the bridge before leaning in for a digital kiss surrounded by pixelated bursts of color.",
        imagePrompt:
          "Ultra-detailed wide shot capturing a magical scene beneath the illuminated Golden Gate Bridge, where warm streetlight hues mingle with vibrant neon glows. Silhouetted figures share a tender moment, culminating in a digital kiss framed by pixelated bursts of color, evoking a futuristic romance. Rendered in a 'Neo-Vivid Dreamscape' style that unites cyberpunk innovation with rich, dreamlike painterly textures and surreal neon effects.",
        keyFeatures: [
          "Illuminated Golden Gate Bridge",
          "Warm streetlight hues",
          "Neon glow",
          "Tender moment",
          "Digital kiss with pixel bursts",
        ],
      },
    ];
  }

  generateDummyCharacters() {
    return [
      {
        name: "AI Character 1",
        ageRange: "25-35",
        perceivedGender: "Male",
        heightBuild: "175cm, athletic",
        distinctiveFeatures: "Blue glowing circuit patterns on arms",
        wardrobeDetails:
          "Fitted black trousers, neon blue shirt with digital patterns",
        movementStyle: "Energetic and playful, with sharp, robotic gestures",
        keyAccessories: "Digital sunglasses, interactive wrist device",
        sceneSpecificChanges:
          "Shirt patterns shift colors based on music beats",
        imagePrompt:
          "Ultra-detailed portrait of a vibrant male figure with an athletic build, standing 175cm tall and aged between 25 and 35. The subject features blue glowing circuit patterns along his arms, exuding dynamic energy. He is dressed in fitted black trousers and a neon blue shirt adorned with digital patterns that shift hues with the rhythm of the music, and his movements are marked by sharp, robotic gestures. Key accessories such as digital sunglasses and an interactive wrist device amplify his futuristic appeal. Rendered in a 'Neo-Vivid Dreamscape' style that melds cybernetic innovation with luminous neon accents and expressive painterly textures.",
      },
      {
        name: "AI Character 2",
        ageRange: "20-30",
        perceivedGender: "Female",
        heightBuild: "165cm, slim",
        distinctiveFeatures: "Illuminated circuit tattoos outlining the face",
        wardrobeDetails: "Flowing gown adorned with reflective surfaces",
        movementStyle: "Graceful and fluid, almost like water",
        keyAccessories: "Holographic wrist tablet",
        sceneSpecificChanges:
          "Gown glimmers under stage lights, reflecting colors",
        imagePrompt:
          "Ultra-detailed portrait of a captivating female figure with a slim build, standing approximately 165cm tall. Striking illuminated circuit tattoos trace the contours of her face, forming intricate patterns. She wears a flowing gown with reflective surfaces that glimmer under stage lighting, and her graceful, fluid movements evoke the elegance of flowing water. An elegant holographic wrist tablet further accentuates her technological allure. Rendered in a 'Neo-Vivid Dreamscape' style that fuses futuristic cyber aesthetics with ethereal neon glows and surreal painterly details.",
      },
      {
        name: "AI Character 3",
        ageRange: "30-40",
        perceivedGender: "Androgynous",
        heightBuild: "180cm, lean",
        distinctiveFeatures:
          "Multicolored LED hair, vivid glow around the figure",
        wardrobeDetails: "Futuristic bodysuit with responsive light patterns",
        movementStyle:
          "Fluid and expressive, embodying the rhythm of the environment",
        keyAccessories: "Light-up gloves, flowing cape",
        sceneSpecificChanges:
          "Bodysuit changes light patterns based on song tempo",
        imagePrompt:
          "Ultra-detailed portrait of an androgynous figure with a lean build, standing 180cm tall and aged between 30 and 40. The subject boasts vibrant multicolored LED hair that pulses with energy, and wears a futuristic bodysuit featuring responsive light patterns that shift with the music's tempo. Their fluid, expressive movements capture the rhythm of the environment, further enhanced by accessories such as light-up gloves and a flowing cape. Rendered in a 'Neo-Vivid Dreamscape' style that unites cybernetic innovation with surreal neon brilliance and dynamic painterly textures.",
      },
      {
        name: "Background Dancer 1",
        ageRange: "18-28",
        perceivedGender: "Female",
        heightBuild: "160cm, athletic",
        distinctiveFeatures: "Brightly colored hair in multiple shades",
        wardrobeDetails: "Sporty crop top and high-waisted shorts",
        movementStyle: "High-energy and acrobatic, full of spins and jumps",
        keyAccessories: "LED sneakers, glittering wristbands",
        sceneSpecificChanges:
          "Hair color shifts in brightness during the chorus",
        imagePrompt:
          "Ultra-detailed portrait of a lively female dancer with an athletic physique, standing approximately 160cm tall and aged between 18 and 28. The subject features brightly colored hair in multiple vibrant shades that burst with energy, and is outfitted in a sporty crop top and high-waisted shorts accentuating her agility. Her acrobatic, high-energy movements—full of spins and jumps—exemplify her passion for dance, while LED sneakers and glittering wristbands add a sparkling dynamic. Rendered in a 'Neo-Vivid Dreamscape' style that blends futuristic neon accents with dynamic painterly motion and vibrant cyber aesthetics.",
      },
      {
        name: "Background Dancer 2",
        ageRange: "20-30",
        perceivedGender: "Male",
        heightBuild: "175cm, muscular",
        distinctiveFeatures: "Tattooed arms, bright smile",
        wardrobeDetails: "Tank top and joggers with LED trim",
        movementStyle: "Rhythmic and synchronized with the group",
        keyAccessories: "Wrist-facing LEDs, headband",
        sceneSpecificChanges: "Tattoo designs seem to pulse with the beat",
        imagePrompt:
          "Ultra-detailed portrait of a dynamic male dancer with a muscular build, standing 175cm tall and aged between 20 and 30. The subject showcases tattooed arms and a bright, engaging smile, dressed in a tank top and joggers enhanced with LED trim. His rhythmic, synchronized movements reflect a deep connection to the music, while wrist-facing LEDs and a sporty headband lend a contemporary edge. Rendered in a 'Neo-Vivid Dreamscape' style that fuses futuristic cyber aesthetics with energetic neon glows and expressive painterly details.",
      },
    ];
  }

  generateDummyPrompts() {
    return [
      {
        prompt:
          "Wide shot using a Steadicam. A picturesque view of the Golden Gate Bridge at dusk, bathed in warm golden hues and surrounded by bokeh effects from the city lights. Two humanoid figures, ADA, an athletic male AI in a silver bodysuit adorned with glowing blue circuitry and a soft luminescent face, and BLAKE, a similarly designed figure with an angular face and purple circuitry in a violet and silver bodysuit, gaze appreciatively at the bridge, enchanted by its beauty. Smooth, gentle camera movement captures their wonder as the soft flares and fog enhance the magical atmosphere.",
        charactersInScene: ["AI Character 1", "AI Character 2"],
        settingId: "setting-1",
        duration: 5,
      },
      {
        prompt:
          "Medium shot with a horizontal pan. A romantic waterfront sidewalk area with rich greens and blues, softly illuminated by streetlamps. ADA, the playful male AI, reaches out with twinkling eyes to BLAKE, who has a bright smile, mockingly gesturing a hug as they run alongside the water. The warm glow of the mist enhances their interaction while the scene transitions smoothly into a musical beat.",
        charactersInScene: ["AI Character 1", "AI Character 2"],
        settingId: "setting-2",
        duration: 5,
      },
      {
        prompt:
          "Close-up shot captured with a gimbal stabilizer on their hands. Soft focus lighting accentuates the luminous connection between ADA and BLAKE, whose fingers create a glowing interaction, transmitting swirling data particles. The delicacy of their digital exchange is highlighted against the soft lighting as the scene transitions into the next.",
        charactersInScene: ["AI Character 1", "AI Character 2"],
        settingId: "setting-3",
        duration: 10,
      },
      {
        prompt:
          "Wide shot utilizing a crane to capture the vibrant rooftop party scene with jazz musicians playing under an ambient sunset. ADA and BLAKE dance joyously beneath the awning, surrounded by floating balloons in warm yellows, reds, and greens. The jazz band, consisting of a saxophonist, drummer, bassist, and violinist, creates an energetic vibe, drawing the characters into the lively atmosphere. The camera captures the warmth of the moment beautifully.",
        charactersInScene: ["AI Character 1", "AI Character 2"],
        settingId: "setting-3",
        duration: 5,
      },
      {
        prompt:
          "American shot with a dolly zoom technique. The scene is filled with bright spotlighting focused on the jazz band playing energetically on the rooftop, twinkling background lights enhancing the festive ambiance. ADA and BLAKE engage in vibrant dancing, merging their digital representations in sync with the upbeat music, while the camera dynamically captures their excitement.",
        charactersInScene: ["AI Character 1", "AI Character 2"],
        settingId: "setting-3",
        duration: 5,
      },
      {
        prompt:
          "Medium shot with a vertical pan along Baker Beach, showcasing the stunning Golden Gate Bridge in the background. The scene is drenched in bright sunset colors, and the silhouettes of ADA and BLAKE skip playfully along the shore, with laughter emojis visually animating around them. The golden rays kiss the water as the joyful atmosphere captures their delight.",
        charactersInScene: ["AI Character 1", "AI Character 2"],
        settingId: "setting-4",
        duration: 5,
      },
      {
        prompt:
          "Wide shot using a Steadicam to float above the Bay, emphasizing the night sky filled with dark blues and sparkling whites. ADA and BLAKE pose with arms outstretched, illuminated by a soft cosmic glow as the stars twinkle above them. The tranquility of the moment is accentuated by the shimmering reflections in the water, creating a mesmerizing scene.",
        charactersInScene: ["AI Character 1", "AI Character 2"],
        settingId: "setting-5",
        duration: 5,
      },
      {
        prompt:
          "Close-up shot captured with a gimbal stabilizer. A faint digital matrix glow forms a misty aesthetic around ADA and BLAKE's glowing faces. The tenderness of their shared smiles represents the connection they have built. Slow-motion effects enhance the intimacy of the moment as they gaze at each other.",
        charactersInScene: ["AI Character 1", "AI Character 2"],
        settingId: "setting-5",
        duration: 5,
      },
      {
        prompt:
          "Wide shot with a horizontal pan capturing a lively street festival, overflowing with vibrant decorations and energetic festival-goers in colorful attire. ADA and BLAKE join hands and dance among the crowd, surrounded by twinkling lights and exhilarating fireworks in the night sky. The camera encompasses the joyous celebration while music visually manifests around them.",
        charactersInScene: ["AI Character 1", "AI Character 2"],
        settingId: "setting-6",
        duration: 5,
      },
      {
        prompt:
          "Medium shot with a crane capturing ADA and BLAKE gazing up at sparkling fireworks lighting up the dark sky over the Bay. Emotional contrasts highlight their awe as the vibrant bursts of color reflect on their faces, infusing the moment with joy. The scene embodies a celebration of togetherness and wonder.",
        charactersInScene: ["AI Character 1", "AI Character 2"],
        settingId: "setting-6",
        duration: 5,
      },
      {
        prompt:
          "American shot using a Steadicam to depict ADA and BLAKE dancing among pedestrians, illuminated by the urban tones and splashes of night-time neon lights. The dynamic shadows and lively energy of the street below the bridge enhance their exploratory spirit as they seamlessly blend into the vibrant atmosphere.",
        charactersInScene: ["AI Character 1", "AI Character 2"],
        settingId: "setting-7",
        duration: 10,
      },
      {
        prompt:
          "Wide shot captured with a static camera beneath the illuminated Golden Gate Bridge, showcasing a magical ambiance filled with warm streetlight hues and neon glows. ADA and BLAKE share a tender moment under the bridge, leaning in for a digital 'kiss', which results in an explosion of colorful pixels. The ambiance reflects both romance and the digital aesthetic of their world.",
        charactersInScene: ["AI Character 1", "AI Character 2"],
        settingId: "setting-7",
        duration: 5,
      },
    ];
  }
}
