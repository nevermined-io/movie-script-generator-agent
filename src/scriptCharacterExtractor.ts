import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import {
  JsonOutputParser,
  StringOutputParser,
} from "@langchain/core/output_parsers";

/**
 * Class combining script generation and character extraction.
 */
export class ScriptCharacterExtractor {
  private scriptChain: RunnableSequence<{ idea: string }, string>;
  private characterChain: RunnableSequence<
    { script: string },
    Record<string, any>[]
  >;

  constructor(apiKey: string) {
    const llm = new ChatOpenAI({ model: "gpt-3.5-turbo", apiKey });

    this.scriptChain = RunnableSequence.from([
      ChatPromptTemplate.fromTemplate(`
        You are a professional screenwriter. Based on the idea below, generate a detailed script:
        Idea: {idea}
        Script:
      `),
      llm,
      new StringOutputParser(),
    ]);

    this.characterChain = RunnableSequence.from([
      ChatPromptTemplate.fromTemplate(`
        Extract character details from the following script. Provide a JSON array where each object contains:
        - name
        - age
        - gender
        - physical_description
        - attire
        - personality_traits
        Script: {script}
        JSON:
      `),
      llm,
      new JsonOutputParser(),
    ]);
  }

  async generateScript(idea: string): Promise<string> {
    return await this.scriptChain.invoke({ idea });
  }

  async extractCharacters(script: string): Promise<object[]> {
    return await this.characterChain.invoke({ script });
  }
}
