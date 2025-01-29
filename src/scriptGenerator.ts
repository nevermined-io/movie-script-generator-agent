import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";

/**
 * Class responsible for generating detailed scripts based on ideas
 * using LangChain and OpenAI.
 */
export class ScriptGenerator {
  private chain: RunnableSequence<{ idea: string }, string>;

  /**
   * Initializes the ScriptGenerator with an OpenAI API key.
   *
   * @param apiKey - The OpenAI API key for authentication.
   */
  constructor(apiKey: string) {
    // Initialize the OpenAI language model
    const llm = new ChatOpenAI({
      model: "gpt-4o-mini",
      apiKey,
    });

    // Define the prompt template for generating scripts
    const prompt = ChatPromptTemplate.fromTemplate(
      `
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
{idea}  `
    );

    // Create a chain of operations to process the script generation
    this.chain = RunnableSequence.from([prompt, llm, new StringOutputParser()]);
  }

  /**
   * Generates a script based on the provided idea.
   *
   * @param idea - The main idea or concept for the script.
   * @returns The generated script as a string.
   * @throws Error if the script generation fails.
   */
  async generateScript(idea: string): Promise<string> {
    try {
      // Invoke the chain with the given idea
      const script = await this.chain.invoke({ idea });
      return script.trim(); // Trim whitespace from the result
    } catch (error) {
      throw new Error(`Error generating script: ${error.message}`);
    }
  }
}
