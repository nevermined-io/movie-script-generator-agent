import { processSteps } from "./steps/stepHandlers";
import {
  NVM_API_KEY,
  NVM_ENVIRONMENT,
  AGENT_DID,
  IS_DUMMY,
} from "./config/env";
import { logger } from "./logger/logger";
import { Payments, EnvironmentName } from "@nevermined-io/payments";

/**
 * Initializes the Nevermined Payments instance.
 */
export function initializePayments(): Payments {
  const payments = Payments.getInstance({
    nvmApiKey: NVM_API_KEY,
    environment: NVM_ENVIRONMENT as EnvironmentName,
  });

  if (!payments.isLoggedIn) {
    throw new Error("Failed to login to Nevermined Payments.");
  }

  logger.info(`Connected to Nevermined Network: ${NVM_ENVIRONMENT}`);

  if (IS_DUMMY) {
    logger.warn("Running in dummy mode. No transactions will be made.");
  }

  return payments;
}

/**
 * Main entry point for the combined Music Video Script and Scene Extractor Agent.
 */
async function main() {
  try {
    const payments = initializePayments();

    await payments.query.subscribe(processSteps(payments), {
      joinAccountRoom: false,
      joinAgentRooms: [AGENT_DID],
      subscribeEventTypes: ["step-updated"],
      getPendingEventsOnSubscribe: false,
    });

    logger.info("Music Video Script and Scene Extractor Agent is running.");
  } catch (error) {
    logger.error(`Error initializing agent: ${error.message}`);
    process.exit(1);
  }
}

main();
