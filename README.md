[![banner](https://raw.githubusercontent.com/nevermined-io/assets/main/images/logo/banner_logo.png)](https://nevermined.io)

Script Generator Agent using Nevermined's Payments API (TypeScript)
===================================================================

> A **TypeScript-based agent** that generates detailed film or music video scripts using OpenAI's API via LangChain. This agent integrates seamlessly with **Nevermined's Payments API**, enabling structured task handling and efficient billing through the Nevermined ecosystem.

* * *

Related Projects
----------------

This project is part of a larger workflow that explores the interconnection between agents and how can they communicate and work together. Please, refer to these projects in order to have a full view of the whole process

1.  [Movie Orchestrator Agent](https://github.com/nevermined-io/movie-orchestrator-agent):
    
    *   Coordinates the entire workflow, ensuring smooth task execution across agents.
    
2.  [Movie Script Generator Agent](https://github.com/nevermined-io/movie-script-generator-agent):
    
    *   Generates movie scripts and characters descriptions based on input ideas.

3.  [Video Generator Agent](https://github.com/nevermined-io/video-generator-agent):
    
    *   Generates realistic character videos based on their descriptions.

#### Workflow Diagram:

![[Init Step] --> [generateScript] --> [generateImagesForCharacters]](https://github.com/nevermined-io/movie-orchestrator-agent/blob/main/flow_img.png?raw=true)

* * *

Table of Contents
-----------------

1.  [Introduction](#introduction)
2.  [Getting Started](#getting-started)
    *   [Installation](#installation)
    *   [Running the Agent](#running-the-agent)
3.  [Project Structure](#project-structure)
4.  [Integration with Nevermined Payments API](#integration-with-nevermined-payments-api)
5.  [How to create your own agent](#how-to-create-your-own-agent)
    *   [1. Subscribing to Task Requests](#1-subscribing-to-task-requests)
    *   [2. Handling Task Lifecycle](#2-handling-task-lifecycle)
    *   [3. Generating Film Scripts with LangChain](#3-generating-film-scripts-with-langchain)
    *   [4. Validating Steps and Sending Logs](#4-validating-steps-and-sending-logs)

* * *

Introduction
------------

The **Script Generator Agent** is a powerful application designed to generate highly detailed film or music video scripts. It leverages the **LangChain library** and **OpenAI API** to create scripts, including character descriptions, settings, and mood.

This agent is integrated with **Nevermined's Payments API**, which provides:

*   **Task lifecycle management**: Efficiently handle task creation, updates, and completion.
*   **Billing integration**: Utilize subscription plans (DIDs) to manage balance and execute tasks.
*   **Event subscription**: React to events like task updates in real-time.

The agent receives input ideas as tasks, processes them into fully detailed scripts, and then updates the Nevermined task system with the results.

One of the standout features of this integration is the **absence of server-side complexity**:
- **No custom server setup**: The Nevermined platform handles all API calls and client requests.
- **Event-driven architecture**: The agent listens for task updates and processes them automatically.
- **Seamless integration**: Tasks and billing are managed via the Payments API.

By subscribing to events with the `subscribe` method, the agent is notified of task requests, processes them through a callback, and sends updates back to the platform.

* * *

Getting Started
---------------

### Installation

1.  **Clone the repository**:
    
    ```bash
    git clone https://github.com/nevermined-io/movie-script-generator-agent.git
    cd script-generator-agent
    ```
    
2.  **Install dependencies**:
    
    ```bash
    npm install
    ```
    
3.  **Configure environment variables**:
    
    *   Copy the `.env.example` file to `.env`:
        
        ```bash
        cp .env.example .env
        ```
        
    *   Populate the `.env` file with the following details:
        
        ```bash
        NVM_API_KEY=YOUR_NVM_API_KEY
        OPENAI_API_KEY=YOUR_OPENAI_API_KEY
        NVM_ENVIRONMENT=testing # or staging/production
        AGENT_DID=YOUR_AGENT_DID
        ```
        
4.  **Build the project**:
    
    ```bash
    npm run build
    ```
    

* * *

### Running the Agent

Run the agent with the following command:

```bash
npm start
```

The agent will subscribe to the Nevermined task system and begin listening for task updates.

* * *

Project Structure
-----------------

```plaintext
script-generator-agent/
├── src/
│   ├── main.ts               # Main entry point for the agent
│   ├── scriptGenerator.ts    # Script generation logic using LangChain
├── .env.example              # Example environment variables file
├── package.json              # Project dependencies and scripts
├── tsconfig.json             # TypeScript configuration
```

### Key Components:

1.  **`main.ts`**: Handles task lifecycle, from receiving steps to sending back results.
2.  **`scriptGenerator.ts`**: Implements the logic for generating scripts using LangChain and OpenAI.
3.  **`.env`**: Stores sensitive configuration details like API keys and environment.

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
  Based on the following idea, generate a short film script including characters, their visual descriptions, and the setting:
  Idea: {idea}
`);

const llm = new ChatOpenAI({
  model: "gpt-3.5-turbo",
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
- **Custom Prompt**: Guides OpenAI to generate detailed scripts, including characters and settings.
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
Copyright 2024 Nevermined AG

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