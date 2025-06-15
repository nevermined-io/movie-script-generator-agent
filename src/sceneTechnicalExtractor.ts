import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence, RunnableLambda } from "@langchain/core/runnables";
import { AIMessage } from "@langchain/core/messages";
import {
  JsonOutputParser,
  StringOutputParser,
} from "@langchain/core/output_parsers";
import { IS_DUMMY, HELICONE_API_KEY } from "./config/env";
import { Scene } from "./types";
import { v4 as uuidv4 } from "uuid";
import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";

// Generate deterministic agent ID based on class name
const generateDeterministicAgentId = (className: string): string => {
  const hash = crypto.createHash('sha256').update(className).digest('hex').substring(0, 32);
  // Format as UUID: 8-4-4-4-12
  return `${hash.substring(0, 8)}-${hash.substring(8, 12)}-${hash.substring(12, 16)}-${hash.substring(16, 20)}-${hash.substring(20, 32)}`;
};

// Generate random session ID
const generateSessionId = (): string => {
  return uuidv4();
};

// Log session information
const logSessionInfo = (agentId: string, sessionId: string, agentName: string = 'SceneTechnicalExtractor'): void => {
  const timestamp = new Date().toISOString();
  const logsDir = path.join(__dirname, 'logs');
  
  // Ensure logs directory exists
  if (!fs.existsSync(logsDir)) {
    fs.mkdirSync(logsDir, { recursive: true });
  }
  
  // Create session-specific log file with timestamp format (YYYYMMDD_HHMMSS)
  const now = new Date();
  const timestampStr = now.toISOString()
    .replace(/[-:]/g, '')  // Remove dashes and colons
    .replace(/T/, '_')     // Replace T with underscore
    .substring(0, 15);     // Take YYYYMMDD_HHMMSS format
  
  const sessionLogFile = path.join(logsDir, `session_${timestampStr}.txt`);
  
  // Check if session file already exists to avoid duplicating session ID
  let sessionExists = false;
  if (fs.existsSync(sessionLogFile)) {
    sessionExists = true;
  }
  
  // If session file doesn't exist, create it with session ID header
  if (!sessionExists) {
    const sessionHeader = `Session ID: ${sessionId}\n`;
    fs.writeFileSync(sessionLogFile, sessionHeader);
  }
  
  // Append agent information in the expected format
  const agentEntry = `${agentName}: ${agentId}\n`;
  fs.appendFileSync(sessionLogFile, agentEntry);
  
  console.log(`Session logged: Timestamp: ${timestamp}, Agent Name: ${agentName}, Agent ID: ${agentId}, Session ID: ${sessionId}`);
};

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
  private readonly agentId: string;
  private readonly sessionId: string;
  
  private scriptChain: RunnableSequence<
    {
      idea: string;
      title: string;
      lyrics: string;
      tags: string;
      duration: number;
      meanScenes: number;
    },
    string
  >;
  private sceneChain: RunnableSequence<
    { script: string; duration: number },
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
    Scene[]
  >;

  constructor(apiKey: string) {
    // Generate deterministic agent ID and random session ID
    this.agentId = generateDeterministicAgentId('SceneTechnicalExtractor');
    this.sessionId = generateSessionId();
    
    // Log session information
    logSessionInfo(this.agentId, this.sessionId, 'SceneTechnicalExtractor');
    
    const llm = new ChatOpenAI({
      model: "gpt-4o-mini",
      apiKey,
      configuration: {
        baseURL: "https://oai.helicone.ai/v1",
        defaultHeaders: {
          "Helicone-Auth": `Bearer ${HELICONE_API_KEY}`,
          "Helicone-Property-AgentId": this.agentId,
          "Helicone-Property-SessionId": this.sessionId,
        }
      }
    });

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
           - Plan accordingly the number of scenes given the total duration of the video. 
           - Optimal number of scenes: {meanScenes}.
        
        5. **Include Scenes with Live Musicians**:
           - At least two scenes must feature a visible band or musicians playing instruments that complement the main story.
           - Show how these musicians integrate into the video's narrative or setting.
        
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
        1. Preserve the **sceneNumber** from the script. If the script says "SCENE 1 - 10 seconds", interpret that as sceneNumber = 1 and duration = 10 seconds.
        2. Convert durations to approximate "startTime" and "endTime" in MM:SS, adding them sequentially so the entire video doesn't exceed {duration} seconds.
           - For example, if SCENE 1 has 10 seconds, it might be startTime="00:00", endTime="00:10".
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

        Duration: {duration} seconds
        
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
              "Ultra-detailed medium shot of a romantic waterfront sidewalk at night, drenched in rich greens and blues. Soft street lamp glows gently illuminate the misty surroundings along the water's edge, creating an intimate and enchanting atmosphere. Rendered in a 'Neo-Vivid Dreamscape' style that merges futuristic cyberpunk motifs with expressive, painterly textures and vibrant neon accents.",
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
        Maintain consistency with the script's descriptions (or make the best assumptions if not explicitly stated).
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

    const minScenes = Math.floor(duration / 10);
    const maxScenes = Math.floor(duration / 5);
    const meanScenes = Math.floor((minScenes + maxScenes) / 2);

    return await this.scriptChain.invoke({
      idea,
      title,
      lyrics,
      duration,
      tags: (tags || []).join(", "),
      meanScenes,
    });
  }

  async extractScenes(script: string, duration: number): Promise<object[]> {
    if (IS_DUMMY) return this.generateDummyScenes();
    return await this.sceneChain.invoke({ script, duration });
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
      tags: (tags || []).join(", "),
    });
  }

  async transformScenes(
    scenes: object[],
    characters: object[],
    settings: object[],
    script: string
  ): Promise<Scene[]> {
    if (IS_DUMMY) return this.generateDummyPrompts();
    return await this.technicalTransformationChain.invoke({
      scenes: JSON.stringify(scenes),
      characters: JSON.stringify(characters),
      settings: JSON.stringify(settings),
      script,
    });
  }

  generateDummyScript() {
    return `SCENE 1 - 5 seconds  
      Close-up | Steadicam | Urban Street Corner  
      Aesthetic: Neon colors pulsating softly; blue, pink, and purple lights reflecting off wet pavement. A low fog drifts in the air, reminiscent of a lively city night.  
      Characters:  
      - Dancer 1: A young man with curly hair, wearing a vibrant red tank top and black joggers; he smiles as he grooves to the music, establishing an energetic vibe.  
      Transition: Hard cut to next scene.  
      
      SCENE 2 - 5 seconds  
      Medium shot | Horizontal pan | Skate Park  
      Aesthetic: Bright colors against a graffiti-covered wall; daylight reflects off the ramps, creating a warm atmosphere. Soft sunlight adds dimension.  
      Characters:  
      - Dancer 2: A woman with short green hair in oversized streetwear with bold patterns, twirls while skillfully navigating around skateboards.  
      Transition: Hard cut to next scene.  
      
      SCENE 3 - 10 seconds  
      Wide shot | Crane shot | Rooftop with City Skyline  
      Aesthetic: Golden hour lighting bathes the dancers in a warm glow; cityscape glimmering in the background. Dynamic lens flares are added in post-production.  
      Characters:  
      - Group of 5 Dancers: All in colorful outfits reflecting urban street fashion, engaging in synchronized group choreography against the skyline.  
      Transition: Match cut to next scene (a dancer jumps off the rooftop into the next shot).  
      
      SCENE 4 - 10 seconds  
      American shot | Gimbal | Neon-lit Alley  
      Aesthetic: Electric blues and fiery oranges dominate; lights blink and flicker as a smoke machine enhances the atmosphere.  
      Characters:  
      - Dancer 3: A tall man in a white tracksuit with bold accessories; effortlessly transitions into intricate footwork, surrounded by 3 background dancers who mirror his movements.  
      - Background Dancers: Displaying variations of dance styles, dressed in complementary streetwear.  
      Transition: Hard cut to next scene.  
      
      SCENE 5 - 10 seconds  
      Close-up | Steadicam | Close on Hands  
      Aesthetic: Bright colors of nail polish; hands raised high, adorned with colorful bracelets reflecting light. The focus is on movement and quick cuts to create rhythm.  
      Characters:  
      - Various Dancers: Hands from different ethnic backgrounds show diversity and unity; synchronized movements celebrate connection.  
      Transition: Crossfade to next scene.  
      
      SCENE 6 - 5 seconds  
      Medium shot | Dolly zoom | City Park  
      Aesthetic: Daylight filters through trees, casting shadows on the ground; cheerful colors of clothing contrast with natural greens.  
      Characters:  
      - Dancer 4: A woman in a flowing yellow dress twirls, her joy radiating; she beckons kids and teens around her to join the dance.  
      Transition: Hard cut to next scene.  
      
      SCENE 7 - 10 seconds  
      Wide shot | Horizontal pan | Street Festival  
      Aesthetic: A joyful scene bursting with life, bright lights strung overhead; confetti and streamers fill the air, enhancing the joyful riot of colors.  
      Characters:  
      - Live Musicians: A small band playing energetic dance music; one electric guitarist with spiked hair and a drummer with face paint, setting the infectious beat.  
      - Crowd Dancers: Various ethnicity and ages dancing freely, creating an atmosphere of celebration and unity.  
      Transition: Match cut as a singer joyfully points toward the next scene.  
      
      SCENE 8 - 10 seconds  
      Close-up | Steadicam | Celebratory Montage  
      Aesthetic: Quick cuts of diverse faces laughing and smiling; thrilled expressions catch the colorful reflections around them.  
      Characters:  
      - Crowd of 10-15 dancers: Each showing a unique expression of joy; diverse in age and ethnicity, connecting through movement.  
      Transition: Hard cut to next scene.  
      
      SCENE 9 - 5 seconds  
      Wide shot | Crane pull-away | City Skyline at Night  
      Aesthetic: Neon city lights shine against a starry sky; the ensemble casts final poses with hands raised—the essence of unity celebrated.  
      Characters:  
      - All Dancers: In a final formation on a rooftop, unified and celebrating, silhouetted against the vibrant city lights, embodying unity in motion.  
      Transition: Fade out as beats drop and crowd noise accompanies the fading visuals.  
      
      CHARACTER LIST:  
      
      Dancer 1: Male, curly hair, vibrant red tank top, black joggers; energetic and welcoming movements.  
      Dancer 2: Female, short green hair, oversized patterned clothes; dynamic in movement through skateboarding.  
      Group of 5 Dancers: Diverse, in colorful urban fashion, synchronized choreography against the skyline.  
      Dancer 3: Male, tall, in a white tracksuit with bold accessories; intricate footwork.  
      Background Dancers: Various ethnicities in matching streetwear, showcasing different dance styles.  
      Dancer 4: Female, in a flowing yellow dress; twirling, interactive with surrounding children.  
      Live Musicians: Band with an electric guitarist and drummer; energetic presence, enhancing the vibe.  
      Crowd Dancers: Diverse, including various ages reflecting joy and unity in their movements.`;
  }

  generateDummyScenes() {
    return [
      {
        endTime: "00:05",
        shotType: "Close-up",
        startTime: "00:00",
        sceneNumber: 1,
        colorPalette: "Neon colors",
        specialNotes:
          "Low fog drifts in the air, reminiscent of a lively city night.",
        lightingSetup:
          "Neon colors pulsating softly; blue, pink, and purple lights",
        cameraMovement: "Steadicam",
        transitionType: "Hard cut",
        cameraEquipment: "Urban Street Corner",
        characterActions:
          "Dancer 1 smiles and grooves to the music, establishing an energetic vibe.",
        visualReferences: ["Urban nightlife", "Neon reflections"],
      },
      {
        endTime: "00:10",
        shotType: "Medium shot",
        startTime: "00:05",
        sceneNumber: 2,
        colorPalette: "Warm colors",
        specialNotes: "Atmosphere created by graffiti-covered walls.",
        lightingSetup: "Bright daylight",
        cameraMovement: "Horizontal pan",
        transitionType: "Hard cut",
        cameraEquipment: "Skate Park",
        characterActions:
          "Dancer 2 twirls skillfully while navigating around skateboards.",
        visualReferences: ["Skateboarding action", "Graffiti art"],
      },
      {
        endTime: "00:20",
        shotType: "Wide shot",
        startTime: "00:10",
        sceneNumber: 3,
        colorPalette: "Warm glows",
        specialNotes: "Lens flares enhanced in post-production.",
        lightingSetup: "Golden hour lighting",
        cameraMovement: "Crane shot",
        transitionType: "Match cut",
        cameraEquipment: "Rooftop with City Skyline",
        characterActions:
          "Group of 5 Dancers engages in synchronized choreography against the skyline.",
        visualReferences: ["Cityscape", "Dynamic lens flares"],
      },
      {
        endTime: "00:30",
        shotType: "American shot",
        startTime: "00:20",
        sceneNumber: 4,
        colorPalette: "Vibrant night colors",
        specialNotes: "Flickering lights enhance the mood.",
        lightingSetup: "Electric blues and fiery oranges",
        cameraMovement: "Gimbal",
        transitionType: "Hard cut",
        cameraEquipment: "Neon-lit Alley",
        characterActions:
          "Dancer 3 transitions into intricate footwork surrounded by background dancers.",
        visualReferences: ["Neon signage", "Smoke machine effects"],
      },
      {
        endTime: "00:40",
        shotType: "Close-up",
        startTime: "00:30",
        sceneNumber: 5,
        colorPalette: "Colorful and bold",
        specialNotes: "Focus on quick cuts to create rhythm.",
        lightingSetup: "Bright colors of nail polish",
        cameraMovement: "Steadicam",
        transitionType: "Crossfade",
        cameraEquipment: "Close on Hands",
        characterActions:
          "Various Dancers display hands from different ethnic backgrounds in unison.",
        visualReferences: ["Close-up details", "Synchronized movements"],
      },
      {
        endTime: "00:45",
        shotType: "Medium shot",
        startTime: "00:40",
        sceneNumber: 6,
        colorPalette: "Cheerful colors",
        specialNotes: "Natural greens contrast with clothing colors.",
        lightingSetup: "Daylight filtering through trees",
        cameraMovement: "Dolly zoom",
        transitionType: "Hard cut",
        cameraEquipment: "City Park",
        characterActions:
          "Dancer 4 twirls in a yellow dress, inviting kids to join her.",
        visualReferences: ["Nature and dance", "Celebratory scene"],
      },
      {
        endTime: "00:55",
        shotType: "Wide shot",
        startTime: "00:45",
        sceneNumber: 7,
        colorPalette: "Joyful riot of colors",
        specialNotes: "Atmosphere enhanced by confetti and streamers.",
        lightingSetup: "Bright festival lights",
        cameraMovement: "Horizontal pan",
        transitionType: "Match cut",
        cameraEquipment: "Street Festival",
        characterActions:
          "Live Musicians play energetic dance music, inspiring crowd dancers.",
        visualReferences: ["Festival atmosphere", "Diverse crowd"],
      },
      {
        endTime: "01:05",
        shotType: "Close-up",
        startTime: "00:55",
        sceneNumber: 8,
        colorPalette: "Joyful and vibrant",
        specialNotes: "Quick edits capture the excitement.",
        lightingSetup: "Colorful reflections",
        cameraMovement: "Steadicam",
        transitionType: "Hard cut",
        cameraEquipment: "Celebratory Montage",
        characterActions:
          "Crowd of 10-15 dancers express joy uniquely through movement.",
        visualReferences: ["Diverse faces", "Energetic expressions"],
      },
      {
        endTime: "01:10",
        shotType: "Wide shot",
        startTime: "01:05",
        sceneNumber: 9,
        colorPalette: "Vibrant night hues",
        specialNotes: "Final silhouettes reflect the bright city lights.",
        lightingSetup: "Neon city lights against a starry sky",
        cameraMovement: "Crane pull-away",
        transitionType: "Fade out",
        cameraEquipment: "City Skyline at Night",
        characterActions: "All Dancers strike final poses, celebrating unity.",
        visualReferences: ["Silhouetted figures", "Urban beauty"],
      },
    ];
  }

  generateDummySettings() {
    return [
      {
        id: "setting-1",
        name: "Urban Street Corner at Night",
        description:
          "An urban street corner perched in a lively city at night, alive with pulsating neon colors reflecting off the slick, wet pavement. Blue, pink, and purple hues dance in the air as low-hanging fog adds a mystic touch. The atmosphere is vibrant and energetic, perfect for showcasing the movements of a dancer dressed in a vibrant red tank top who grooves to the rhythm, embodying the energy of the nightlife.",
        imagePrompt:
          "Close-up shot of an urban street corner at night, glowing with neon colors of blue, pink, and purple reflected on the wet pavement with low fog drifting through. A young male dancer in a red tank top is captured in dynamic motion, exuding energy and excitement against a bustling city backdrop. The atmosphere is electric and lively, rendered in a modern, vibrant urban style, highlighting the nightlife.",
        keyFeatures: [
          "Pulsating neon colors",
          "Wet pavement reflections",
          "Low fog",
          "Energetic dancer",
          "Lively city atmosphere",
        ],
      },
      {
        id: "setting-2",
        name: "Skate Park in Daylight",
        description:
          "A bright and colorful skate park, filled with an array of graffiti-covered walls that offer a canvas of artistic expression. Under the warm daylight, skateboard ramps glimmer, amplifying the inviting atmosphere. A female dancer with green hair glides effortlessly through the space, twirling playfully in oversized streetwear, embodying the spirit of creativity and movement.",
        imagePrompt:
          "Medium shot of a vibrant skate park bathed in daylight, showcasing colorful graffiti and shiny skateboard ramps. A young woman with short green hair in bold patterned oversized streetwear twirls and skillfully navigates around skateboards, creating a lively and welcoming scene filled with warmth and energy. Rendered in a vivid, energetic style that captures the essence of urban youth culture.",
        keyFeatures: [
          "Graffiti-covered walls",
          "Daylight reflections",
          "Skateboard ramps",
          "Colorful oversized streetwear",
          "Dynamic twirling movements",
        ],
      },
      {
        id: "setting-3",
        name: "Rooftop with City Skyline at Golden Hour",
        description:
          "A stunning rooftop scene at golden hour, where the dancers bask in warm, golden sunlight while the city skyline glimmers in the background. The ambiance is rich with vibrant colors as dynamic lens flares create a magical atmosphere. A group of five dancers in an array of colorful outfits synchronously perform against the breathtaking backdrop of the city, showcasing urban street fashion and unity.",
        imagePrompt:
          "Wide shot of a rooftop at golden hour, featuring a group of five dancers in colorful street fashion performing synchronized choreography against a dazzling city skyline. The warm golden light bathes the performers as lens flares add an ethereal quality to the scene, rendered in a vivid, uplifting style that transports viewers to a moment of celebration and creativity.",
        keyFeatures: [
          "Golden hour lighting",
          "City skyline backdrop",
          "Synchronized group choreography",
          "Colorful urban outfits",
          "Dynamic lens flares",
        ],
      },
      {
        id: "setting-4",
        name: "Neon-lit Alley",
        description:
          "An electrifying alley illuminated by vibrant electric blues and fiery oranges, where the lights flicker and flash amid a haze from a smoke machine. The atmosphere is intense and immersive, accompanying the intricate footwork of a tall male dancer in a bold white tracksuit. The alley reflects a sense of urban excitement as background dancers mirror his movements in complementary streetwear, creating a captivating visual.",
        imagePrompt:
          "American shot of a neon-lit alley filled with electric blues and oranges. The lights blink and flicker through a subtle haze of smoke, highlighting a tall male dancer in a bold white tracksuit performing intricate footwork, surrounded by background dancers in matching streetwear. The atmosphere is vibrant and intense, rendered in a striking urban style that captures the energy of the nightlife.",
        keyFeatures: [
          "Electric blue and orange hues",
          "Flickering lights",
          "Smoke machine ambiance",
          "Intricate dance footwork",
          "Complementary streetwear of background dancers",
        ],
      },
      {
        id: "setting-5",
        name: "City Park in Daylight",
        description:
          "A serene city park bathed in cheerful daylight, where sunbeams filter through lush greenery, casting playful shadows on the earth. The colors of various clothing worn by a female dancer in a flowing yellow dress radiate joy. She twirls gracefully, encouraging the surrounding children and teens to join in the dance, creating a lively and interactive atmosphere amidst nature.",
        imagePrompt:
          "Medium shot of a city park in daylight, filled with sunlight filtering through trees and casting dynamic shadows. A woman in a flowing yellow dress twirls joyfully among children and teens, inviting them to join the dance. The vibrant greens of the park contrast beautifully with the cheerful colors of the dancers, rendered in a light, uplifting style that encapsulates community and nature.",
        keyFeatures: [
          "Sunlight filtering through trees",
          "Vibrant colors of clothing",
          "Twirling dancer",
          "Interactive atmosphere",
          "Contrast with natural greens",
        ],
      },
      {
        id: "setting-6",
        name: "Street Festival Celebration",
        description:
          "A lively street festival bursting with joy and life, characterized by bright lights strung overhead and a plethora of festive confetti fluttering in the air. The scene is colorful and exuberant, enhanced by live musicians engaging the crowd, while individuals of varied ethnicities and ages dance freely, celebrating unity and togetherness amidst the festive background.",
        imagePrompt:
          "Wide shot of a vibrant street festival overflowing with life, colorful string lights overhead illuminating the scene and confetti cascading through the air. A small band plays upbeat dance music while various crowd dancers, representing different ethnicities and ages, move joyfully in a celebration of unity. Rendered in a vivid, festive style that captures the energizing essence of community celebration.",
        keyFeatures: [
          "Bright festival lights",
          "Confetti filling the air",
          "Live band playing",
          "Diverse crowd dancing",
          "Atmosphere of celebration and unity",
        ],
      },
    ];
  }

  generateDummyCharacters() {
    return [
      {
        name: "Lead Singer",
        ageRange: "25-30",
        heightBuild: "180cm, lean",
        imagePrompt:
          "Ultra-detailed portrait of an androgynous performer, aged 25-30 with a lean 180cm build. The subject features glowing circuit tattoos and a cybernetic left eye that exude a mysterious cyberpunk aura. Dressed in a distressed leather jacket with metallic accents that tears and illuminates with bursts of light during moments of intensity, the dynamic pose captures fluid, energetic stage movements enhanced by neon accessories. Rendered in a 'Neo-Cyber Renaissance' style that blends futuristic cyberpunk elements with dynamic painterly textures and radiant neon highlights.",
        movementStyle: "Fluid and dynamic gestures",
        keyAccessories: "Neon microphone, digital wristband",
        perceivedGender: "Androgynous",
        wardrobeDetails: "Distressed leather jacket with metallic accents",
        distinctiveFeatures: "Glowing circuit tattoos, cybernetic left eye",
        sceneSpecificChanges:
          "Jacket tears and illuminates during intense moments",
      },
      {
        name: "Bassist",
        ageRange: "30-40",
        heightBuild: "170cm, curvy",
        imagePrompt:
          "Ultra-detailed portrait of a confident female performer with a curvy build, aged 30-40 and standing approximately 170cm tall. The subject boasts long, flowing hair and a mesmerizing stage presence defined by graceful, assured gestures. Clad in a turquoise maxi skirt paired with a fitted crop top and a leather belt, her smooth and flowing movements harmonize with the striking illuminated bass guitar and sparkling statement jewelry. Rendered in a 'Neo-Cyber Renaissance' style that merges futuristic cyberpunk aesthetics with expressive painterly details and vivid neon luminosity.",
        movementStyle:
          "Smooth and flowing, exuding confidence and allure while still maintaining rhythmic focus",
        keyAccessories: "Bass guitar with inlay lights, statement jewelry",
        perceivedGender: "Female",
        wardrobeDetails:
          "Turquoise maxi skirt, fitted crop top, and a leather belt",
        distinctiveFeatures:
          "Long flowing hair, captivating stage presence with confident gestures",
        sceneSpecificChanges:
          "Jewelry glitters under the stage lights, the skirt flows beautifully with her movements",
      },
      {
        name: "Drummer",
        ageRange: "20-30",
        heightBuild: "175cm, athletic",
        imagePrompt:
          "Ultra-detailed portrait of a male performer aged 20-30 with an athletic build standing about 175cm tall. The subject features a buzz cut and an infectious, energetic smile that radiates enthusiasm. He wears a black tank top with ripped jeans and sneakers, exuding a rebellious vibe. His high-energy drumming style showcases explosive gestures that sync with vibrant sparks flying from the drum hits, enhanced by dynamic, colorful lighting changes that pulse with the music. Rendered in a 'Neo-Cyber Renaissance' style that combines intense kinetic energy with bright, vivid neon colors.",
        movementStyle: "High-energy, rhythmic drumming with explosive gestures",
        keyAccessories: "Drumsticks with LED lights, sweatband",
        perceivedGender: "Male",
        wardrobeDetails: "Black tank top, ripped jeans, and sneakers",
        distinctiveFeatures:
          "Buzz cut hairstyle, energetic and infectious smile",
        sceneSpecificChanges:
          "Sparks fly from drum hits, colorful lighting changes with rhythms",
      },
      {
        name: "Background Dancer 1",
        ageRange: "18-25",
        heightBuild: "165cm, slender",
        imagePrompt:
          "Ultra-detailed portrait of a slender female dancer aged 18-25, standing approximately 165cm tall. The dancer showcases short, spiky hair and vibrant face paint that enhances her lively demeanor. Clothed in a neon bodysuit adorned with holographic patterns, her fast-paced, agile movements create a captivating visual. The bodysuit shimmers and changes color with each gesture, her LED bracelets and ankle bells producing an engaging light show during the performance. Rendered in a 'Neo-Cyber Renaissance' style that integrates vivid dance-pop aesthetics with surreal, colorful textures.",
        movementStyle: "Fast-paced, agile movements filled with creativity",
        keyAccessories: "LED bracelets, ankle bells",
        perceivedGender: "Female",
        wardrobeDetails: "Neon bodysuit with holographic patterns",
        distinctiveFeatures: "Short, spiky hair, vibrant face paint",
        sceneSpecificChanges:
          "Bodysuit shimmers and changes color with movement",
      },
      {
        name: "Background Dancer 2",
        ageRange: "18-25",
        heightBuild: "178cm, athletic",
        imagePrompt:
          "Ultra-detailed portrait of an athletic male dancer aged 18-25, standing approximately 178cm tall. The subject has medium-length tousled hair and a strong jawline that exudes confidence. He is dressed in a bright tank top and cargo pants, complemented by high-top sneakers that emphasize his dynamic dance moves. His expressive style blends contemporary and street dance, with the tank top changing patterns during beat drops and glow sticks illuminating his movements in rhythm. Rendered in a 'Neo-Cyber Renaissance' style that enriches the narrative with exciting, colorful elements.",
        movementStyle:
          "Dynamic and expressive, blending contemporary and street dance",
        keyAccessories: "Colored sunglasses, glow sticks",
        perceivedGender: "Male",
        wardrobeDetails: "Bright tank top, cargo pants, and high-top sneakers",
        distinctiveFeatures: "Medium-length tousled hair, strong jawline",
        sceneSpecificChanges:
          "Tank top changes pattern with beat drops, glow sticks illuminate",
      },
    ];
  }

  generateDummyPrompts() {
    return [
      {
        prompt:
          "Close-up shot using a Steadicam on an urban street corner at night, vibrant neon colors pulsating softly. A young male dancer with curly hair, wearing a vibrant red tank top and black joggers, smiles and grooves to the music amidst glowing blue, pink, and purple lights reflecting off the wet pavement, as a low fog drifts in the air, establishing an energetic vibe. The atmosphere captures the essence of nightlife, enhanced by a hard cut to the next scene.",
        charactersInScene: ["Dancer 1"],
        settingId: "setting-1",
        duration: 5,
      },
      {
        prompt:
          "Medium shot using a horizontal pan at a bright and colorful skate park during daylight. A female dancer with short green hair in oversized patterned streetwear twirls skillfully while negotiating around skateboards, under the warm atmosphere created by graffiti-covered walls and soft sunlight that reflects beautifully off skateboard ramps. The scene captures dynamic movement and energy, transitioning hard to the subsequent scene.",
        charactersInScene: ["Dancer 2"],
        settingId: "setting-2",
        duration: 5,
      },
      {
        prompt:
          "Wide shot using a crane shot on a rooftop at golden hour, capturing a group of five dancers in colorful urban street fashion performing synchronized choreography against a stunning city skyline. The golden hour lighting bathes the dancers in warmth while dynamic lens flares create an ethereal ambiance. The scene celebrates unity and creativity, transitioning with a match cut to the next sequence.",
        charactersInScene: ["Group of 5 Dancers"],
        settingId: "setting-3",
        duration: 10,
      },
      {
        prompt:
          "American shot using a gimbal in a neon-lit alley, highlighted by electric blues and fiery oranges, where a tall male dancer in a white tracksuit performs intricate footwork amidst flickering lights and a haze from a smoke machine. Surrounding background dancers mirror his movements in complementary streetwear, enhancing the visual intensity of the dance. The pulse of the rhythm invites a hard cut to the next scene.",
        charactersInScene: ["Dancer 3", "Background Dancers"],
        settingId: "setting-4",
        duration: 10,
      },
      {
        prompt:
          "Close-up shot using a Steadicam, focusing on a chorus of hands from various dancers displaying vibrant nail polish and colorful bracelets in unison. The quick cuts accentuate the synchronized movements celebrating diversity and unity in a burst of bright colors. The rhythm culminates in a crossfade to the following scene.",
        charactersInScene: ["Various Dancers"],
        settingId: "setting-5",
        duration: 10,
      },
      {
        prompt:
          "Medium shot using a dolly zoom in a cheerful city park, where daylight filters through the trees casting playful shadows. A female dancer in a flowing yellow dress twirls joyfully, inviting kids and teens around her to join the dance, creating a lively and interactive atmosphere enriched by the contrast of vibrant clothing against natural greens. The scene transitions hard to the next.",
        charactersInScene: ["Dancer 4"],
        settingId: "setting-6",
        duration: 5,
      },
      {
        prompt:
          "Wide shot using a horizontal pan at a lively street festival, bursting with bright lights overhead and a joyful riot of colors. Live musicians, including an electric guitarist with spiked hair and a drummer with face paint, engage the crowd while diverse dancers, representing various ethnicities and ages, move freely in celebration around them, creating an atmosphere of unity and festivity. The vibrant scene transitions with a match cut to the next segment.",
        charactersInScene: ["Live Musicians", "Crowd Dancers"],
        settingId: "setting-6",
        duration: 10,
      },
      {
        prompt:
          "Close-up shot using a Steadicam in a celebratory montage filled with quick cuts of diverse faces laughing and moving joyfully. 10-15 dancers, representing various ages and ethnic backgrounds, express their unique joy through movement against a backdrop of colorful reflections. This vibrant and joyful scene leads to a hard cut to the final moment.",
        charactersInScene: ["Crowd of 10-15 dancers"],
        settingId: "setting-6",
        duration: 10,
      },
      {
        prompt:
          "Wide shot using a crane pull-away at a city skyline at night, where neon city lights shine against a starry sky. All dancers, unified in a final pose with hands raised high, create silhouettes that embody celebration and unity. The atmosphere is vibrant, culminating in a gentle fade-out as beats drop and crowd noise accompanies the visuals.",
        charactersInScene: ["All Dancers"],
        settingId: "setting-6",
        duration: 5,
      },
    ];
  }
}
