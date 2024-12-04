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
      model: "gpt-3.5-turbo", // Choose the model for generating scripts
      apiKey,
    });

    // Define the prompt template for generating scripts
    const prompt = ChatPromptTemplate.fromTemplate(
      `You are a professional scriptwriter. Based on the following idea and genre, generate a complete script for a short film / music video, including the list of characters and their detailed visual descriptions species, race, height, age, genre, visual description, clothing, psychological characteristics..
      If several characters are present, provide a detailed description of each character. Do not skip any character or detail, even if they are minor or unnamed characters.
      Also describe the setting, mood, and any other relevant details to set the scene.
      If the input idea is not clear or short, please expand on it to create a complete script and provide a fully detailed description of the characters.

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
