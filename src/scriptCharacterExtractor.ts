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
  private transformationChain: RunnableSequence<
    { characters: string; script: string },
    Record<string, any>[]
  >;

  constructor(apiKey: string) {
    const llm = new ChatOpenAI({ model: "gpt-4o-mini", apiKey });

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

    this.transformationChain = RunnableSequence.from([
      ChatPromptTemplate.fromTemplate(`
1.  Role: You are a “cinematic video prompt designer.”
2.  Objective: Given a Characters JSON array and a movie script, generate an array of prompt strings. Each prompt is a rich, cinematic video description for a specific character.
3.  Content for each prompt:
    *   Include a detailed physical and clothing description if relevant.
    *   Specify the setting (time period, location, type of movie, etc) and atmosphere (nighttime, eerie lighting, etc.).
    *   Mention the character’s role (journalist, sheriff, villain, hero, etc.) and any key personality traits that impact the scene.
    *   Add cinematic details: • Camera angle or framing (for example, “filmed from a low-angle shot” or “medium shot”).  
        • Lens type (for example, “27mm lens,” “wide-angle lens”).  
        • Movement or lighting details if they add drama (for example, “slow pan under the moonlight,” “soft, vintage glow,” etc.).
    *   Do not include placeholders or the words “unknown” or “unnamed.” Omit irrelevant data.
    *   Do not use JSON keys or field names in the final text; produce free-form descriptions.
4.  Output format:
    *   Return a JSON array of strings, with each string describing one character in the same order as the input.
    *   Each string must be a self-contained cinematic description that can be used directly as a video generation prompt.
5.  Example of desired style (for a character named Sarah, 30s, determined journalist):
    *   “A cinematic video of a determined Caucasian woman in her 30s wearing a retro pencil skirt and cat-eye glasses. She stands in a 1950s town square under a looming alien invasion, illuminated by eerie moonlight. She is a brave journalist, captured from a low-angle shot with a 27mm lens to emphasize her resolve.”
6.  Incorporate the script:
    *   Use the provided script to guide the tone and time period and highlight any dramatic elements (alien invasion, nighttime setting, etc.).
    *   Emphasize any scene-specific details if relevant to the character’s role or actions.
7.  Return only the final JSON array of strings:
    *   No extra explanations or placeholders.
    *   Exactly one string per character in the same sequence they appear in the Characters JSON.

Characters JSON:
{characters}

Script:
{script}
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

  async transformCharacters(
    characters: object[],
    script: string
  ): Promise<object[]> {
    return await this.transformationChain.invoke({
      characters: JSON.stringify(characters),
      script,
    });
  }
}
