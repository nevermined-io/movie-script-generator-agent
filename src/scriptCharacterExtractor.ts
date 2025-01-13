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
      You are a professional scriptwriter. Based on the following idea and genre, generate a complete script for a short film / music video, including the list of characters and their detailed visual descriptions species, race, height, age, genre, visual description, clothing, psychological characteristics..
      If several characters are present, provide a detailed description of each character. Do not skip any character or detail, even if they are minor or unnamed characters.
      Also describe the setting, mood, and any other relevant details to set the scene.
      If the input idea is not clear or short, please expand on it to create a complete script and provide a fully detailed description of the characters.
Idea:
{idea}
Script:
      `),
      llm,
      new StringOutputParser(),
    ]);

    this.characterChain = RunnableSequence.from([
      ChatPromptTemplate.fromTemplate(`
        You are an expert at analyzing film scripts. Extract a list of characters from the following script. 

For each character, provide the following details as a JSON object:
- name: (string) The name of the character or a placeholder like "Unnamed Character" if no name is given.
- age: (string) A description of the character's age, e.g., "30s" or "child".
- gender: (string) The gender of the character, e.g., "male", "female", or "non-binary".
- species: (string) The species of the character, e.g., "human", "alien", or "animal".
- physical_description: (string) A description of the character's physical appearance.
- attire: (string) A description of the character's clothing or outfit.
- personality_traits: (string) A summary of the character's personality.
- role: (string) The role of the character in the story, e.g., "protagonist", "villain", or "side character".
- scene_description: (string) A description of the scene or context where the character appears.
- additional_notes: (string) Any additional relevant details about the character.

The JSON output should be an array of objects, with one object for each character. Example:

[
  {{
    "name": "Jane Doe",
    "age": "30s",
    "gender": "female",
    "species": "human",
    "physical_description": "long brown hair, blue eyes, 5'6\"",
    "attire": "casual attire with a red scarf",
    "personality_traits": "outgoing and caring",
    "role": "protagonist",
    "scene_description": "Outdoor scene in a park, sunny day",
    "additional_notes": "Wears a bracelet with sentimental value"
    }},
  {{
    "name": "Unnamed Alien",
    "age": "100s",
    "gender": "unknown",
    "species": "extraterrestrial",
    "physical_description": "green slimy skin, tall with tentacles",
    "attire": "formal attire with a regal crown",
    "personality_traits": "wise and mysterious",
    "role": "council member",
    "scene_description": "Meeting in a grand hall with other aliens",
    "additional_notes": "Speaks with a deep, resonant voice"
    }}
]

Script:
{script}

Output an array of JSON objects only.
Details from each character should not reference other characters or the script itself, as they will be processed independently.
All the characters provided in the script should be extracted, even if they are minor or unnamed characters.
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
