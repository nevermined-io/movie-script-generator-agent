import {
  AgentExecutionStatus,
  Payments,
  generateStepId,
} from "@nevermined-io/payments";
import { logMessage, logger } from "../logger/logger";
import { ScriptCharacterExtractor } from "../scriptCharacterExtractor";

/**
 * Processes incoming steps based on their type.
 * Handles the "init" step by creating the steps for "generateScript" and "extractCharacters".
 */
export function processSteps(payments: Payments) {
  const extractor = new ScriptCharacterExtractor(process.env.OPENAI_API_KEY!);

  return async (data: any) => {
    const eventData = JSON.parse(data);
    const step = await payments.query.getStep(eventData.step_id);

    logMessage(payments, {
      task_id: step.task_id,
      level: "info",
      message: `Processing step: ${step.name}`,
    });

    switch (step.name) {
      case "init":
        await handleInitStep(step, payments);
        break;
      case "generateScript":
        await handleScriptGeneration(step, payments, extractor);
        break;
      case "extractCharacters":
        await handleCharacterExtraction(step, payments, extractor);
        break;
      default:
        logger.warn(`Unrecognized step name: ${step.name}. Skipping...`);
    }
  };
}

/**
 * Handles the "init" step by defining the workflow steps:
 * 1. "generateScript"
 * 2. "extractCharacters"
 */
async function handleInitStep(step: any, payments: Payments) {
  const scriptStepId = generateStepId();
  const characterStepId = generateStepId();

  const steps = [
    {
      step_id: scriptStepId,
      task_id: step.task_id,
      predecessor: step.step_id, // "generateScript" follows "init"
      name: "generateScript",
      is_last: false,
    },
    {
      step_id: characterStepId,
      task_id: step.task_id,
      predecessor: scriptStepId, // "extractCharacters" follows "generateScript"
      name: "extractCharacters",
      is_last: true,
    },
  ];

  const createResult = await payments.query.createSteps(
    step.did,
    step.task_id,
    { steps }
  );

  logMessage(payments, {
    task_id: step.task_id,
    level: createResult.status === 201 ? "info" : "error",
    message:
      createResult.status === 201
        ? "Steps created successfully."
        : `Error creating steps: ${JSON.stringify(createResult.data)}`,
  });

  await payments.query.updateStep(step.did, {
    ...step,
    step_status: AgentExecutionStatus.Completed,
    output: "Workflow initialized successfully.",
  });
}

/**
 * Handles the "generateScript" step, using the `ScriptCharacterExtractor` to generate a script.
 */
async function handleScriptGeneration(
  step: any,
  payments: Payments,
  extractor: ScriptCharacterExtractor
) {
  try {
    const script = await extractor.generateScript(step.input_query);

    logger.info(`Generated script: ${script}`);
    await payments.query.updateStep(step.did, {
      ...step,
      step_status: AgentExecutionStatus.Completed,
      output: script,
    });

    logMessage(payments, {
      task_id: step.task_id,
      level: "info",
      message: `Script generation completed.`,
    });
  } catch (error) {
    logger.error(`Error during script generation: ${error.message}`);
    await payments.query.updateStep(step.did, {
      ...step,
      step_status: AgentExecutionStatus.Failed,
      output: "Failed to generate script.",
    });
  }
}

/**
 * Handles the "extractCharacters" step, extracting characters from the generated script.
 */
async function handleCharacterExtraction(
  step: any,
  payments: Payments,
  extractor: ScriptCharacterExtractor
) {
  try {
    const characters = await extractor.extractCharacters(step.input_query);

    logger.info(`Extracted characters: ${JSON.stringify(characters)}`);

    await payments.query.updateStep(step.did, {
      ...step,
      step_status: AgentExecutionStatus.Completed,
      output: step.input_query,
      output_artifacts: [characters],
    });

    logMessage(payments, {
      task_id: step.task_id,
      level: "info",
      message: `Character extraction completed successfully.`,
    });
  } catch (error) {
    logger.error(`Error during character extraction: ${error.message}`);
    await payments.query.updateStep(step.did, {
      ...step,
      step_status: AgentExecutionStatus.Failed,
      output: "Failed to extract characters.",
    });
  }
}
