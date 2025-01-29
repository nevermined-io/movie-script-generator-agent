[![banner](https://raw.githubusercontent.com/nevermined-io/assets/main/images/logo/banner_logo.png)](https://nevermined.io)

Music Video Technical Script Generator Agent (TypeScript)
=========================================================

> A **TypeScript-based agent** that generates detailed technical scripts for music videos using OpenAI's API via LangChain. Integrates with **Nevermined's Payments API** for structured task handling and billing management.

* * *

Table of Contents
-----------------

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
    * [Installation](#installation)
    * [Running the Agent](#running-the-agent)
3. [Project Structure](#project-structure)
4. [Integration with Nevermined Payments API](#integration-with-nevermined-payments-api)
5. [Key Features](#key-features)
6. [License](#license)

* * *

Introduction
------------

The **Music Video Technical Script Generator** is a specialized application designed to create production-ready music video scripts with precise technical specifications. It leverages **LangChain** and **OpenAI API** to generate:

- Scene-by-scene breakdowns with exact timings
- Camera shot types and movements (dolly, crane, Steadicam)
- Lighting setups and color palettes
- Equipment recommendations (lenses, stabilizers)
- Visual references and transition types

Integrated with **Nevermined's Payments API**, the agent provides:

- **Task lifecycle management**: Handle script generation requests efficiently
- **Event-driven architecture**: React to task updates in real-time
- **Seamless billing integration**: Manage payments through Nevermined ecosystem

The agent processes input ideas into complete technical scripts including:

- Shot compositions with duration markers
- Character actions synchronized to music
- Post-production effects planning
- Cinematic style references

* * *

Getting Started
---------------

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/nevermined-io/music-video-script-generator-agent.git
    cd music-video-script-generator-agent
    ```

2. **Install dependencies**:
    ```bash
    npm install
    ```

3. **Configure environment variables**:
    - Copy `.env.example` to `.env`:
        ```bash
        cp .env.example .env
        ```
    - Populate `.env`:
        ```bash
        NVM_API_KEY=YOUR_NVM_API_KEY
        OPENAI_API_KEY=YOUR_OPENAI_API_KEY
        NVM_ENVIRONMENT=testing # or staging/production
        AGENT_DID=YOUR_AGENT_DID
        ```

4. **Build the project**:
    ```bash
    npm run build
    ```

* * *

### Running the Agent

Start the agent:
```bash
npm start
```

The agent will connect to Nevermined's task system and await music video script generation requests.

* * *

Project Structure
-----------------

```plaintext
music-video-script-generator-agent/
├── src/
│   ├── main.ts                       # Agent entry point
│   ├── scriptGenerator.ts            # Core script generation logic
│   ├── sceneTechnicalExtractor.ts    # Technical scene breakdowns
├── config/
│   ├── env.ts                        # Environment configuration
├── logger/
│   ├── logger.ts                     # Logging system
├── .env.example                      # Environment template
├── package.json                      # Dependencies
├── tsconfig.json                     # TypeScript config
```

### Key Components:

1.  **`main.ts`**: Handles task lifecycle, from receiving steps to sending back results.
2.  **`scriptGenerator.ts`**:  Implements music video script generation pipeline
2.  **`sceneTechnicalExtractor.ts`**: Handles technical scene breakdowns and production prompts

* * *

Integration with Nevermined Payments API
----------------------------------------

This agent is designed to work within the **Nevermined ecosystem**, leveraging the **Payments API** for managing tasks and billing. Below is a step-by-step guide to integrate an agent with Nevermined:

1.  **Initialize the Payments Instance**: The Payments instance provides access to the Nevermined task system.
    
    ```typescript
    import { Payments, EnvironmentName } from "@nevermined-io/payments";
    
    const payments = Payments.getInstance({
      nvmApiKey: process.env.NVM_API_KEY!,
      environment: process.env.NVM_ENVIRONMENT as EnvironmentName,
    });
    
    if (!payments.isLoggedIn) {
      throw new Error("Failed to authenticate with Nevermined.");
    }
    ```
    
2.  **Subscribe to Task Updates**: The agent subscribes to task updates using the `subscribe` method:
    
    ```typescript
    const subscriptionOpts = {
      joinAccountRoom: false,
      joinAgentRooms: [process.env.AGENT_DID!],
      subscribeEventTypes: ["step-updated"],
      getPendingEventsOnSubscribe: false,
    };
    
    await payments.query.subscribe(run, subscriptionOpts);
    ```
    
3.  **Task Lifecycle**:
    
    *   Fetch the step details:
        
        ```typescript
        const step = await payments.query.getStep(stepId);
        ```
        
    *   Update the step status:
        
        ```typescript
        await payments.query.updateStep(step.did, {
          step_status: "Completed",
          output: "Generated script content",
        });
        ```

   Once a task is received:
   - The step details are fetched using the `getStep` method.
   - The agent processes the task, such as generating a script.
   - The status is updated using the `updateStep` method.
        

For more details, visit the [official documentation](https://docs.nevermined.app/docs/tutorials/integration/agent-integration#4b-using-the-nevermined-payment-libraries-to-integrate-the-agent).

* * *

How to create your own agent
--------
### 1\. Subscribing to Task Requests

The core functionality begins with subscribing to events using the `subscribe` method in `main.ts`:
```typescript
await payments.query.subscribe(run, {
  joinAccountRoom: false,
  joinAgentRooms: [process.env.AGENT_DID!],
  subscribeEventTypes: ["step-updated"],
  getPendingEventsOnSubscribe: false,
});
```

#### How It Works:
- **Event Subscription**: The `subscribe` method ensures the agent listens for task updates.
- **Callback Execution**: The `run` function is executed each time a task request is made.

### 2. Handling Task Lifecycle

The agent processes incoming tasks through the `run` function in `main.ts`:

```typescript
async function run(data: any) {
  const step = await payments.query.getStep(data.step_id);
  if (step.step_status !== "Pending") return;

  const idea = step.input_query; // Extract the idea
  const script = await scriptGenerator.generateScript(idea); // Generate the script

  await payments.query.updateStep(step.did, {
    ...step,
    step_status: "Completed",
    output: script,
  });
}
```

#### Key Points:

*   **Fetching step details**: Retrieve all task-related metadata.
*   **Generating output**: Use the `scriptGenerator` instance to generate content.
*   **Updating the step**: Mark the task as completed and provide the generated script.

* * *

### 3\. Generating Film Scripts with LangChain

The core script generation logic is in `scriptGenerator.ts`:

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const prompt = ChatPromptTemplate.fromTemplate(`
 **Role**: You're a professional music video director with expertise in storyboards and technical planning.  
**Task**: Create a detailed technical script for a **3-minute maximum** music video based on the provided idea. Use **screenplay format without markdown**.  
[...]
  Idea: {idea}
`);

const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  apiKey: process.env.OPENAI_API_KEY!,
});

export class ScriptGenerator {
  async generateScript(idea: string): Promise<string> {
    const script = await prompt.invoke({ idea });
    return script.trim();
  }
}
```

#### Key Features:
- **Custom Prompt**: Guides OpenAI to generate detailed scripts, including scenes, characters and settings.
- **LangChain Integration**: Utilizes LangChain for structured prompt handling and output parsing.

* * *

### 4\. Validating Steps and Sending Logs

The agent uses `logMessage` to send logs to the Nevermined system. This ensures task status updates are tracked:

The `logMessage` object should follow this structure:
```typescript
{
  task_id: string;   // Unique identifier for the task this log belongs to.
  level: "info" | "warning" | "error" | "debug";  // Log severity level.
  message: string;   // The log message content.
}
```

Example:
```typescript
logMessage(payments, {
  task_id: "task123",
  level: "info",
  message: "Processing step X successfully."
});
```

Use this to track progress, errors, or task completion.

* * *

License
-------

```
Copyright 2025 Nevermined AG

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```