import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence, RunnableLambda } from "@langchain/core/runnables";
import { AIMessage } from "@langchain/core/messages";
import {
  JsonOutputParser,
  StringOutputParser,
} from "@langchain/core/output_parsers";

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
           - This is a music video, 
        
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
        Extract ALL characters from the script with detailed physical descriptors and roles, considering the song lyrics and tags as additional context. Return a JSON array of objects, each with:

        - "name" (or "Unnamed [role]")
        - "age_range"
        - "perceived_gender"
        - "height_build"
        - "distinctive_features"
        - "wardrobe_details"
        - "movement_style"
        - "key_accessories"
        - "scene_specific_changes"

        **Important**:
        1. Include every character or extra mentioned in the script, such as background dancers, band members, or instrumentalists.
        2. If there are references to a band or musicians, list each musician separately with their instrument, wardrobe, and any unique features.
        3. Maintain consistency with the script's descriptions (or best assumptions if not explicit).
        4. Use the provided song lyrics and tags for additional context when inferring character details.

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
            "scene_specific_changes": "Jacket tears and illuminates during intense moments"
          }},
          {{
            "name": "Guitarist",
            "age_range": "30-35",
            "perceived_gender": "Male",
            "height_build": "175cm, stocky",
            "distinctive_features": "Sleeve tattoo of mythical creatures, scar above right eyebrow",
            "wardrobe_details": "Vintage band T-shirt, ripped jeans",
            "movement_style": "Energetic and rhythmic headbanging",
            "key_accessories": "Electric guitar with flame decals",
            "scene_specific_changes": "Hair gets windblown in outdoor scenes"
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
        Transform the technical details of the scenes into production prompts for both image and video. 
        For **each scene** in the "Scene Data", generate **one object** with two attributes:
        - "imagePrompt"
        - "videoPrompt"

        **Instructions**:
        1. **Number of prompts**: You must produce as many objects as there are scenes in the JSON. 
          If there are 9 scenes, return 9 objects in an array.

        2. **Character references**:
          - Replace each character name with a full physical description taken from CHARACTER_DATA.
          - If the character is a musician, mention their instrument in the prompt (e.g., "carrying a vintage acoustic guitar").

        3. **Scene details**:
          - Use the data from the scene (shot_type, camera_movement, lighting_setup, color_palette, etc.).
          - The "imagePrompt" focuses on static composition, color, atmosphere, and which characters are visible.
          - The "videoPrompt" focuses on camera movement, transitions, how the characters move or interact, and any special effects.

        4. **Precision**:
          - Use professional cinematography terminology (e.g., 'close-up with shallow depth of field', 'slow dolly in', 'neon underlighting').
          - Mention the lens or camera gear from the scene data.

        5. **Integration**:
          - Integrate the "Script Context" only if it adds crucial narrative or visual detail. 
          - For musician scenes, describe how the instruments and performance integrate with the shot.

        6. **Output format**: 
          Return a JSON array where each element corresponds to one scene. 
          Example:
        [
          {{
            "imagePrompt": "Cinematic wide shot using a 24mm lens on a tripod. A dark, cluttered room with warm, muted earth tones. A male android in his 20s with glowing blue circuit patterns beneath translucent synthetic skin, wearing a worn leather jacket, centered in the frame.",
            "videoPrompt": "Smooth dolly movement from left to right with a slow pan and fade transitions. The character performs subtle gestures that sync with the scene's rhythm, while digital effects and soft lighting transitions enhance the cinematic atmosphere."
          }},
          {{
            "imagePrompt": "Close-up over-the-shoulder shot using a 50mm lens. An androgynous digital entity (aged between 25-30) with ever-changing holographic facial features, wearing fragmented luminous projections resembling a business suit, captured with high contrast and sharp detail.",
            "videoPrompt": "Subtle zoom-in combined with a glitch transition. Camera movement is complemented by the character's dynamic actions, such as shifting poses and expressive gestures, with synchronized digital distortions and soft fades to create dramatic tension."
          }},
          {{
            "imagePrompt": "Medium shot using an 85mm lens, capturing a dynamic urban street scene at dusk with neon lights reflecting off wet pavement. A female cyborg in her early 30s, with silver mechanical limbs and vibrant red hair, wearing a sleek futuristic jacket, stands poised in the frame.",
            "videoPrompt": "Tracking shot with a steady cam as the camera follows the character walking briskly along the street. The character glances over her shoulder and raises her right arm to shield her eyes from a sudden burst of light, synchronized with rapid cuts and energetic digital overlays."
          }},
        ]
        
        **CHARACTER_DATA**: {characters}
        **SCENE DATA**: {scenes}
        **SCRIPT CONTEXT**: {script}

        Return only a valid JSON array of objects using double quotes, with the same length as the SCENE DATA. 
        Do not add any explanations or markdown.
          `),
      llm,
      extractJsonRunnable,
      new JsonOutputParser(),
    ]);
  }

  async generateScript({ idea, title, lyrics, tags }): Promise<string> {
    return await this.scriptChain.invoke({
      idea,
      title,
      lyrics,
      tags: tags.join(", "),
    });
  }

  async extractScenes(script: string): Promise<object[]> {
    return await this.sceneChain.invoke({ script });
  }

  async extractCharacters(
    script: string,
    lyrics: string,
    tags: string[]
  ): Promise<object[]> {
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
    return await this.technicalTransformationChain.invoke({
      scenes: JSON.stringify(scenes),
      characters: JSON.stringify(characters),
      script,
    });
  }
}
