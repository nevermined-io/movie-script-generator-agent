[![banner](https://raw.githubusercontent.com/nevermined-io/assets/main/images/logo/banner_logo.png)](https://nevermined.io)

Music Video Technical Script Generator Agent (TypeScript)
=========================================================

> A **TypeScript-based agent** that generates **detailed technical scripts** for music videos using **LangChain** + **OpenAI**, integrated with the **Nevermined Payments** API for orchestrating multi-step tasks and managing subscriptions or billing.

* * *

**Description**
---------------

The **Music Video Technical Script Generator Agent** automates the process of creating complex, production-ready scripts for music videos. Drawing on AI-powered text generation, the agent:

1.  **Generates** a script outline (scene breakdowns, camera movements, lighting, etc.).
2.  **Extracts** scene data (start/end times, shot type, transitions).
3.  **Identifies** the **settings** or environments in each scene with rich descriptors.
4.  **Extracts** all **characters** mentioned (including musicians, background dancers, extras).
5.  **Transforms** the final scene data into a set of prompts suitable for subsequent AI-based image/video generation.

All steps are **event-driven**, with tasks managed and billed through **Nevermined**. By subscribing to `step-updated` events, this agent listens for instructions and updates each step’s status as it progresses (e.g., from `Pending` to `Completed`).

* * *
**Related Projects**
--------------------

This **Video Generator Agent** is part of a larger ecosystem of AI-driven media creation. For a complete view of how multiple agents work together, see:

1.  [Music Orchestrator Agent](https://github.com/nevermined-io/music-video-orchestrator)
    
    *   Coordinates end-to-end workflows: collects user prompts, splits them into tasks, pays agents in multiple tokens, merges final output.
2.  [Song Generator Agent](https://github.com/nevermined-io/song-generation-agent)
    
    *   Produces lyrics, titles, and final audio tracks using LangChain + OpenAI and a chosen music generation API.
3.  [Image / Video Generator Agent](https://github.com/nevermined-io/video-generator-agent)
    
    *   Produces Images / Video using 3rd party wrapper APIs (Fal.ai and TTapi, wrapping Flux and Kling.ai)

**Workflow Example**:

```
[ User Prompt ] --> [Music Orchestrator] --> [Song Generation] --> [Script Generation] --> [Image/Video Generation] --> [Final Compilation]
```

* * *

**Table of Contents**
---------------------

1.  [Features](#features)
2.  [Prerequisites](#prerequisites)
3.  [Installation](#installation)
4.  [Environment Variables](#environment-variables)
5.  [Project Structure](#project-structure)
6.  [Architecture & Workflow](#architecture--workflow)
7.  [Usage](#usage)
8.  [Detailed Guide: Creating & Managing Tasks](#detailed-guide-creating--managing-tasks)
9.  [Development & Testing](#development--testing)
10.  [License](#license)

* * *

**Features**
------------

*   **Nevermined Integration**: Subscribes to tasks via `step-updated` events, updates the workflow’s steps, and logs progress remotely.
*   **LangChain + OpenAI**: Uses advanced prompt templates to produce detailed cinematic descriptions.
*   **Scene Extraction & Settings**: Splits an existing script into discrete scenes with recommended camera gear, lighting setups, and transitions. Also identifies distinct locations/settings.
*   **Character Extraction**: Provides names, descriptions, wardrobe details, and AI-friendly prompts for each character.
*   **Transform Scenes**: Creates final prompts summarizing each scene in a single JSON array (duration, setting, character references).
*   **Modular Design**: Extend or replace steps with minimal disruption.
*   **Logging & Error Handling**: Comprehensive logs at every step, with robust fallback in case of failures.

* * *

**Prerequisites**
-----------------

*   **Node.js** (>= 18.0.0 recommended)
*   **TypeScript** (project developed on ^5.x)
*   **Nevermined** credentials (API key, environment config, `AGENT_DID`)
*   **OpenAI API Key** for text generation

* * *

**Installation**
----------------

1.  **Clone** this repository:
    
    ```bash
    git clone https://github.com/nevermined-io/music-video-script-generator-agent.git
    cd music-video-script-generator-agent
    ```
    
2.  **Install** dependencies:
    
    ```bash
    npm install
    ```
    
3.  **Build** the project (optional for production):
    
    ```bash
    npm run build
    ```
    

* * *

**Environment Variables**
-------------------------

Rename `.env.example` to `.env` and set the required keys:

```env
NVM_API_KEY=your_nevermined_api_key
NVM_ENVIRONMENT=testing
AGENT_DID=did:nv:your_agent_did
OPENAI_API_KEY=your_openai_api_key
IS_DUMMY=false
```

*   `NVM_API_KEY` and `NVM_ENVIRONMENT` configure the connection to **Nevermined**.
*   `AGENT_DID` sets the unique identifier for this agent.
*   `OPENAI_API_KEY` grants access to OpenAI’s text generation models.
*   `IS_DUMMY` controls whether to use a “dummy” mode (returning static data instead of calling real services).

* * *

**Project Structure**
---------------------

```plaintext
music-video-script-generator-agent/
├── main.ts                      # Main entry (initializes Payments, subscribes to steps)
├── config/
│   └── env.ts                   # Environment configuration
├── steps/
│   └── stepHandlers.ts          # Core logic for handling each step
├── sceneTechnicalExtractor.ts   # Class for script creation, scene extraction, etc.
├── types.ts                     # Shared TypeScript interfaces (e.g., Scene)
├── logger/
│   └── logger.ts               # Logging system (local + remote)
├── .env.example                 # Environment template
├── package.json
├── tsconfig.json
└── README.md                    # This file
```

### Key Components:

1.  **`main.ts`**: Handles task lifecycle, from receiving steps to sending back results.
2.  **`sceneTechnicalExtractor.ts`**: Implements the multi-step logic for generating the initial script, extracting scenes, identifying settings, and extracting characters.
3.  **`stepHandlers.ts`**: Coordinates the different steps (`init`, `generateScript`, `extractScenes`, `generateSettings`, `extractCharacters`, and `transformScenes`) that shape the final output.
4.  **`logger/logger.ts`**: A logging system that logs both locally and through the **Nevermined Payments** API.

* * *

**Architecture & Workflow**
---------------------------

1.  **Initialization** (`init` step)
    
    *   When a new task is created for this agent, it starts with `init`.
    *   We create subsequent steps (e.g., `generateScript`, `extractScenes`, `generateSettings`, `extractCharacters`, `transformScenes`).
2.  **Script Generation** (`generateScript`)
    
    *   Takes the user’s “idea” (plus optional lyrics/tags/duration) and generates a full technical script.
    *   The script typically includes scene descriptions, durations, camera gear, transitions, and other cinematic details.
3.  **Scene Extraction** (`extractScenes`)
    
    *   Parses the newly generated script to produce structured data: start/end times for each scene, shot types, transitions, etc.
4.  **Settings Generation** (`generateSettings`)
    
    *   Identifies each unique environment (e.g., rooftop party, city street, beach) and creates a JSON array describing them.
5.  **Characters Extraction** (`extractCharacters`)
    
    *   Finds all references to characters (lead singer, extras, musicians) in the script.
    *   Produces a list with each character’s physical descriptors, wardrobe details, movement style, etc.
6.  **Scene Transformation** (`transformScenes`)
    
    *   Merges all extracted data to produce final “prompts,” each describing a scene’s visuals, camera movements, participating characters, and links to the correct setting.
    *   These prompts can be used directly for downstream AI-based image or video generation tasks.

Throughout these steps, the agent **updates** each step’s status (from `Pending` to `Completed` or `Failed`) in the **Nevermined** system. If a step fails, it logs the error and halts.

* * *

**Usage**
---------

After installing and configuring `.env`:

```bash
npm start
```

1.  The agent logs into **Nevermined** with your `NVM_API_KEY` and listens for `step-updated` events for `AGENT_DID`.
2.  When an orchestrator (or higher-level process) triggers a new “init” step for this agent, the agent spawns the subsequent steps and processes them one by one.
3.  The final step yields a JSON with scene prompts, character data, and settings.

* * *

**Detailed Guide: Creating & Managing Tasks**
---------------------------------------------

### Introduction

The **Music Video Technical Script Generator** leverages **LangChain** and **OpenAI** to create production-ready scripts with precise technical specs—scene-by-scene breakdowns, camera movements, lighting setups, color palettes, lens/stabilizer recommendations, visual references, transitions, and more. All of this is integrated with **Nevermined's Payments API** for structured task handling and billing.

### Installation & Running the Agent

1.  **Clone** and **install** as described above (`npm install`, configure `.env`).
2.  **Build** (optional) with `npm run build`.
3.  **Start** via `npm start`. The agent will connect to **Nevermined** and await new tasks.

### Project Structure (Extended)

```plaintext
music-video-script-generator-agent/
├── src/
│   ├── main.ts                       # Agent entry point for step subscription
│   ├── steps/
│   │   └── stepHandlers.ts           # Detailed step logic (init, generateScript, etc.)
│   ├── sceneTechnicalExtractor.ts    # Core logic for generating / extracting script data
│   ├── logger/
│   │   └── logger.ts                 # Logging system
│   └── config/
│       └── env.ts                    # Environment variable load
├── .env.example                      # Environment template
├── package.json                      # Dependencies
├── tsconfig.json                     # TypeScript config
└── README.md
```

#### Notable Files

*   **`sceneTechnicalExtractor.ts`**: Contains the logic to generate an initial script (via OpenAI), then parse out scenes, settings, and characters, returning structured JSON data.
*   **`stepHandlers.ts`**: Defines the workflow steps, updating the task in the **Nevermined** system accordingly.

### Integration with Nevermined Payments API

Below is a step-by-step outline of how **Nevermined** orchestrates tasks in this agent:

1.  **Initialize the Payments Instance**
    
    ```ts
    import { Payments, EnvironmentName } from "@nevermined-io/payments";
    
    const payments = Payments.getInstance({
      nvmApiKey: process.env.NVM_API_KEY!,
      environment: process.env.NVM_ENVIRONMENT as EnvironmentName,
    });
    
    if (!payments.isLoggedIn) {
      throw new Error("Failed to authenticate with Nevermined.");
    }
    ```
    
    This connects to **Nevermined** using the API key and environment.
    
2.  **Subscribe to Task Updates**
    
    ```ts
    await payments.query.subscribe(run, {
      joinAccountRoom: false,
      joinAgentRooms: [process.env.AGENT_DID!],
      subscribeEventTypes: ["step-updated"],
      getPendingEventsOnSubscribe: false,
    });
    ```
    
    Here, `run` is a callback that processes each `step-updated` event.
    
3.  **Task Lifecycle**
    
    *   **Fetch** a step’s details using `payments.query.getStep(stepId)`.
    *   **Process** the step (e.g., generate a script or extract scenes).
    *   **Update** the step in the system using `payments.query.updateStep(step.did, { ... })`.

### 1\. Subscribing to Task Requests

In `main.ts`, you’ll typically see:

```ts
await payments.query.subscribe(processSteps(payments), {
  joinAccountRoom: false,
  joinAgentRooms: [AGENT_DID],
  subscribeEventTypes: ["step-updated"],
  getPendingEventsOnSubscribe: false,
});
```

*   **Event Subscription**: The agent will now receive an event each time a relevant step is updated for its DID.
*   **Callback Execution**: The function `processSteps(payments)` is invoked with the event data.

### 2\. Handling Task Lifecycle

In `stepHandlers.ts`, you might see something like:

```ts
export function processSteps(payments: Payments) {
  return async (data: any) => {
    const eventData = JSON.parse(data);
    const step = await payments.query.getStep(eventData.step_id);

    if (step.name === "init") {
      // create sub-steps (generateScript, extractScenes, etc.)
    } else if (step.name === "generateScript") {
      // call handleScriptGeneration() ...
    }
    // ...
  };
}
```

The agent checks the step’s name and dispatches it to the right logic (e.g., `generateScript`, `extractScenes`). Each step runs an AI-driven process or data extraction, then updates the step status accordingly.

### 3\. Generating Film Scripts with LangChain

**LangChain** manages prompts and output parsers. For example, in `sceneTechnicalExtractor.ts`, we might see:

```ts
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";

export class SceneTechnicalExtractor {
  private scriptChain = /* Sequence definition with ChatOpenAI... */

  async generateScript({ idea, title, lyrics, tags, duration }) {
    // builds a prompt with specialized instructions
    return await this.scriptChain.invoke({ idea, title, lyrics, tags, duration });
  }
}
```

This approach instructs the AI to produce a screenplay-like text with scenes, durations, camera specs, etc.

### 4\. Validating Steps and Sending Logs

To record events:

```ts
import { logMessage } from "../logger/logger";

logMessage(payments, {
  task_id: step.task_id,
  level: "info",
  message: `Processing step: ${step.name}`,
});
```

This logs both locally and in **Nevermined**, ensuring a full audit trail for each sub-step.

* * *

**Development & Testing**
-------------------------

1.  **Development Server**
    
    ```bash
    npm run dev
    ```
    
    *   In some setups, this watches for file changes and restarts automatically.
2.  **Build**
    
    ```bash
    npm run build
    ```
    
3.  **Testing**
    
    ```bash
    npm test
    ```
    
    *   Runs any available unit or integration tests.

* * *

**License**
-----------

```
Apache License 2.0

(C) 2025 Nevermined AG

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions
and limitations under the License.
```