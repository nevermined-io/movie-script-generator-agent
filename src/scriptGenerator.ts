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
1.  Role: You are a professional scriptwriter.
2.  Task: Based on the provided idea and genre, write a complete script for a short film or music video. The script must be written in plain text with no formatting or markdown.
3.  Requirements for the script:
    *   Provide a clear narrative structure (beginning, middle, and end), written as traditional screenplay text.
    *   Include a list of all characters—major, minor, and even unnamed extras—with detailed visual descriptions. For each character, specify: • Species (if relevant) • Race/ethnicity (if relevant) • Height • Age • Gender • Physical/visual description • Clothing/outfit • Psychological characteristics or personality traits
    *   Do not omit any character, even if they have a brief or “unnamed” role. Provide a concise but thorough description for each.
    *   Describe the setting (location, time period, overall atmosphere or mood).
    *   Add any relevant details to make the script feel complete, including dialogue, scene transitions, and thematic elements.
4.  If the input idea is unclear or too short, expand on it to create a fully formed script. Use creative license to fill in gaps, but ensure the final result is coherent.
5.  Return only plain text in the style of a classic screenplay—no markdown, no bullet points, no extra formatting. The text should read as if it were a traditional film script.

Idea:
{idea}

Script:`
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
