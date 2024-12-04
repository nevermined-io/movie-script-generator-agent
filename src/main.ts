import {
  AgentExecutionStatus,
  Payments,
  TaskLogMessage,
  EnvironmentName,
} from "@nevermined-io/payments";
import { ScriptGenerator } from "./scriptGenerator";
import dotenv from "dotenv";
import pino from "pino";

// Load environment variables from the .env file
dotenv.config();

// Retrieve environment variables and validate required fields
const NVM_ENVIRONMENT = process.env.NVM_ENVIRONMENT || "staging";
const NVM_API_KEY = process.env.NVM_API_KEY!;
const AGENT_DID = process.env.AGENT_DID!;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY!;

// Create a logger instance for structured and readable logs
const logger = pino({
  transport: { target: "pino-pretty" },
  level: "info",
});

// Initialize global variables for the Payments and ScriptGenerator instances
let payments: Payments;
let scriptGenerator: ScriptGenerator;

/**
 * Processes incoming steps and executes the script generation task.
 * This function is triggered when a new task is received from the Nevermined subscription.
 *
 * @param data - The serialized task data received from the subscription.
 */
async function run(data: any) {
  try {
    // Deserialize the incoming task data
    const eventData = JSON.parse(data);
    logger.info(`Received event: ${JSON.stringify(eventData)}`);

    // Fetch detailed step information using the step ID from the event
    const step = await payments.query.getStep(eventData.step_id);
    logger.info(
      `Processing Step ${step.task_id} - ${step.step_id} [${step.step_status}]`
    );

    // Skip processing if the step is not in the 'Pending' state
    if (step.step_status !== AgentExecutionStatus.Pending) {
      logger.warn(`Step ${step.step_id} is not pending. Skipping...`);
      return;
    }

    // Log the start of the script generation task
    await logMessage({
      task_id: step.task_id,
      level: "info",
      message: `Starting script generation...`,
    });

    const idea = step.input_query;
    logger.info(`Input Idea: ${idea}`);

    try {
      // Use the ScriptGenerator to generate a script based on the input idea
      const script = await scriptGenerator.generateScript(idea);

      logger.info(`Generated Script: ${script}`);

      // Update the task step with the generated script and mark it as completed
      const updateResult = await payments.query.updateStep(step.did, {
        ...step,
        step_status: AgentExecutionStatus.Completed,
        is_last: true, // Indicates that this is the final step in the process
        output: script,
      });

      // Log the result of the update operation
      if (updateResult.status === 201) {
        await logMessage({
          task_id: step.task_id,
          message: `Script generation completed successfully.`,
          level: "info",
          task_status: AgentExecutionStatus.Completed,
        });
      } else {
        // Log an error if updating the step fails
        await logMessage({
          task_id: step.task_id,
          message: `Error updating step ${step.step_id}: ${JSON.stringify(
            updateResult.data
          )}`,
          level: "error",
          task_status: AgentExecutionStatus.Failed,
        });
      }
    } catch (e) {
      // Handle errors that occur during script generation
      logger.error(`Error during script generation: ${e}`);
      await logMessage({
        task_id: step.task_id,
        message: `Error during script generation: ${e}`,
        level: "error",
        task_status: AgentExecutionStatus.Failed,
      });
    }
  } catch (error) {
    // Handle errors that occur while processing the step
    logger.error(`Error processing step: ${error}`);
  }
}

/**
 * Logs messages and sends them to the Nevermined Payments API.
 * This function ensures consistent logging across the agent and sends logs
 * to the central Nevermined task log system.
 *
 * @param logMessage - The log message containing task details, status, and message content.
 */
async function logMessage(logMessage: TaskLogMessage) {
  // Log the message locally using the appropriate log level
  if (logMessage.level === "error") logger.error(logMessage.message);
  else if (logMessage.level === "warning") logger.warn(logMessage.message);
  else if (logMessage.level === "debug") logger.debug(logMessage.message);
  else logger.info(logMessage.message);

  // Send the log message to the Nevermined Payments API
  await payments.query.logTask(logMessage);
}

/**
 * Initializes the Nevermined Payments instance for task management.
 * This function authenticates the agent with the Nevermined network and
 * provides a Payments instance for interacting with the task API.
 *
 * @param nvmApiKey - The API key for authenticating with the Nevermined network.
 * @param environment - The target environment ('staging', 'production', etc.).
 * @returns An authenticated Payments instance.
 */
function initializePayments(nvmApiKey: string, environment: string): Payments {
  const paymentsInstance = Payments.getInstance({
    nvmApiKey,
    environment: environment as EnvironmentName,
  });

  if (!paymentsInstance.isLoggedIn) {
    throw new Error("Failed to login to Nevermined Payments Library");
  }
  return paymentsInstance;
}

/**
 * The main entry point of the Script Generator Agent.
 * This function initializes the agent by:
 * 1. Authenticating with the Nevermined network.
 * 2. Creating the ScriptGenerator instance for script generation.
 * 3. Subscribing to the Nevermined task system for step updates.
 */
async function main() {
  try {
    // Initialize the Nevermined Payments instance
    payments = initializePayments(NVM_API_KEY, NVM_ENVIRONMENT);
    logger.info(`Connected to Nevermined Network: ${NVM_ENVIRONMENT}`);

    // Initialize the ScriptGenerator with the OpenAI API key
    scriptGenerator = new ScriptGenerator(OPENAI_API_KEY);

    // Define subscription options for receiving task updates
    const opts = {
      joinAccountRoom: false, // Exclude general account-wide events
      joinAgentRooms: [AGENT_DID], // Listen for events assigned to this agent
      subscribeEventTypes: ["step-updated"], // Trigger on 'step-updated' events
      getPendingEventsOnSubscribe: false, // Avoid fetching historical events
    };

    // Subscribe to task updates and handle them with the `run` function
    await payments.query.subscribe(run, opts);

    logger.info("Agent is now listening for task updates...");
  } catch (error) {
    logger.error(`Error initializing agent: ${error}`);
    process.exit(1); // Terminate the process if initialization fails
  }
}

logger.info("Starting Script Generator Agent...");
main();
