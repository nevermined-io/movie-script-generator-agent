import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import {
  JsonOutputParser,
  StringOutputParser,
} from "@langchain/core/output_parsers";

/**
 * Class combining script generation and scenes extraction.
 */
export class SceneTechnicalExtractor {
  private scriptChain: RunnableSequence<{ idea: string }, string>;
  private sceneChain: RunnableSequence<
    { script: string },
    Record<string, any>[]
  >;
  private characterChain: RunnableSequence<
    { script: string },
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
      You are a professional music video director. Generate a technical script for a 3-minute music video based on this idea. Include:
      1. Scene breakdown with exact timings (seconds)
      2. Shot types (close-up, wide, etc.) and camera movements
      3. Color palette and lighting for each scene
      4. Equipment suggestions (lenses, stabilizers)
      5. Visual references (cinematic comparisons)
      6. Transition types between scenes
      7. Character actions synchronized with music
      
      Follow this structure:
      SCENE [N] - [DURATION]
      SHOT TYPE | CAMERA MOVEMENT | LOCATION
      AESTHETIC: [description]
      CHARACTER ACTIONS: [specific movements]
      TRANSITION: [type]
      
      Idea: {idea}
      Script:
      `),
      llm,
      new StringOutputParser(),
    ]);

    this.sceneChain = RunnableSequence.from([
      ChatPromptTemplate.fromTemplate(`
      Extract technical scene details as JSON array with these fields:
      - scene_number
      - start_time (MM:SS)
      - end_time (MM:SS)
      - shot_type (include both vertical/horizontal framing)
      - camera_movement
      - camera_equipment (specific model references)
      - lighting_setup (type + color temp + position)
      - color_palette (main + accent colors)
      - visual_references (2-3 comparable films/scenes)
      - character_actions (timing-linked to music)
      - transition_type
      - special_notes (equipment needs, safety considerations)

      Example:
      {{
        "scene_number": 1,
        "start_time": "00:12",
        "end_time": "00:24",
        "shot_type": "Medium close-up/wide",
        "camera_movement": "Slow dolly zoom",
        "camera_equipment": "Sony FE 24-70mm f/2.8 GM II on Ronin 4D",
        "lighting_setup": "Softbox key light (5600K) at 45°, LED fill (3200K)",
        "color_palette": "Desaturated teal base with neon pink accents",
        "visual_references": ["Blade Runner 2049 rainy scenes", "Euphoria club lighting"],
        "character_actions": "Lead singer: Aggressive mic stand spin at 00:18 (sync with drum hit)",
        "transition_type": "RGB split glitch transition",
        "special_notes": "Need rain machine and lens waterproofing"
      }}

      Script: {script}
      Return only valid JSON array. Use double quotes.
      `),
      llm,
      new JsonOutputParser(),
    ]);

    // Character Extraction Chain (nuevo)
    this.characterChain = RunnableSequence.from([
      ChatPromptTemplate.fromTemplate(`
      Extract ALL characters with detailed physical descriptors. For each, include:
      - name (or "Unnamed [role]")
      - age_range
      - perceived_gender
      - height/build
      - distinctive_features (tattoos, scars, cybernetics)
      - wardrobe_details (include brand references if relevant)
      - movement_style
      - key_accessories
      - scene_specific_changes (wardrobe/makeup evolution)

      Example output:
      {{
        "name": "Lead Singer",
        "age_range": "25-30",
        "perceived_gender": "Androgynous",
        "height_build": "5'10\", lean muscular",
        "distinctive_features": "Glowing circuit tattoos on neck, cybernetic left eye",
        "wardrobe_details": "Distressed Balmain leather jacket, Chrome Hearts belt",
        "movement_style": "Jagged, robotic gestures",
        "key_accessories": "Neon microphone with smoke effects",
        "scene_specific_changes": "Jacket tears gradually throughout video"
      }}

      Script: {script}
      Return JSON array. No markdown.
      `),
      llm,
      new JsonOutputParser(),
    ]);

    this.technicalTransformationChain = RunnableSequence.from([
      ChatPromptTemplate.fromTemplate(`
Transform scene technical details into self-contained video production prompts. For each character reference, replace names with full physical descriptions using this format:
      "[Gender] [Age] [Height] [Species] with [Physical Features], wearing [Attire]".
      
      Example transformation:
      Before: "Larry gestures wildly"
      After: "A male humanoid AI in late 20s appearance, 6'1" with glowing blue circuit patterns under translucent synthetic skin, wearing a distressed leather jacket, gestures wildly"
      
      Each prompt must include:
      1. Shot composition details (framing, movement)
      2. Camera specifications (lens, stabilizer)
      3. Lighting setup (type, placement, color)
      4. Color grading notes
      5. Special effects requirements
      6. Transition execution instructions
      7. Character descriptions and actions
      
      Character Requirements:
      1. Always use character descriptors instead of names
      2. Include key physical features from CHARACTER_DATA
      3. Maintain cinematic terminology from SCENE_DATA
      4. Merge scene and character details seamlessly
            
      CHARACTER_DATA: {characters}
      Scene Data: {scenes}
      Script Context: {script}

      Example output format:
      [
        "Cinematic wide shot using 24mm lens on tripod. Dimly-lit cluttered room with warm, muted earth tones. Male humanoid AI (late 20s appearance, 6'1\") with glowing blue circuit patterns under translucent synthetic skin, wearing a faded graphic tee depicting old film reels, demonstrates frustration while pacing. Color grading with warm earth tones. Transition: Fade to black with soft hum.",
        "Close-up over-the-shoulder shot using 50mm lens. Androgynous digital entity (apparent age 25-30, 5'11\") with constantly shifting facial features rendered in holographic pixels, wearing fragmented light projections resembling a business suit, rejects script pitches. Cool blue tone grading. Transition: Glitch effect to next scene.",
        ...
      ]
      
      Return JSON array of strings. No explanations.
      `),
      llm,
      new JsonOutputParser(),
    ]);
  }

  async generateScript(idea: string): Promise<string> {
    return await this.scriptChain.invoke({ idea });
  }

  async extractScenes(script: string): Promise<object[]> {
    return await this.sceneChain.invoke({ script });
  }

  async extractCharacters(script: string): Promise<object[]> {
    return await this.characterChain.invoke({ script });
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
