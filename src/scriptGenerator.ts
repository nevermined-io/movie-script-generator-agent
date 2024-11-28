// src/scriptGenerator.ts

import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";

/**
 * Class responsible for generating a film script based on an idea and genre using LangChain and OpenAI.
 */
export class ScriptGenerator {
  private chain: RunnableSequence<{ idea: string }, string>;

  /**
   * Initializes the ScriptGenerator with the OpenAI API key.
   * @param apiKey - The OpenAI API key.
   */
  constructor(apiKey: string) {
    // Initialize the OpenAI language model
    const llm = new ChatOpenAI({
      model: "gpt-3.5-turbo", // or "gpt-4" if you have access
      apiKey,
    });

    // Define a prompt template for the script generation task
    const prompt = ChatPromptTemplate.fromTemplate(
      `You are a professional scriptwriter. Based on the following idea and genre, generate a complete script for a short film / music video, including the list of characters and their detailed visual descriptions species, race, height, age, genre, visual description, clothing, psychological characteristics..
      If several characters are present, provide a detailed description of each character. Do not skip any character or detail, even if they are minor or unnamed characters.
      Also describe the setting, mood, and any other relevant details to set the scene.
      If the input idea is not clear or short, please expand on it to create a complete script and provide a fully detailed description of the characters.

Idea:
{idea}

Script:`
    );

    // Create a script generation chain using LangChain's RunnableSequence
    this.chain = RunnableSequence.from([prompt, llm, new StringOutputParser()]);
  }

  /**
   * Generates a film script based on the given idea and genre.
   * @param idea - The idea for the script.
   * @param genre - The genre of the film.
   * @returns The generated script.
   */
  async generateScript(idea: string): Promise<string> {
    try {
      // Execute the script generation chain with the input idea and genre
      const script = await this.chain.invoke({ idea });
      return script.trim();
    } catch (error) {
      console.error(`Error during script generation: ${error}`);
      throw error;
    }
  }
}
