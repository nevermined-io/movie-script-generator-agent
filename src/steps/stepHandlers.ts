import {
  AgentExecutionStatus,
  Payments,
  generateStepId,
} from "@nevermined-io/payments";
import { logMessage, logger } from "../logger/logger";
import { SceneTechnicalExtractor } from "../sceneTechnicalExtractor";

/**
 * Processes incoming steps based on their type.
 * Handles the "init" step by creating the steps for "generateScript" and "extractScenes".
 */
export function processSteps(payments: Payments) {
  const extractor = new SceneTechnicalExtractor(process.env.OPENAI_API_KEY!);

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
      case "extractScenes":
        await handleScenesExtraction(step, payments, extractor);
        break;
      case "extractCharacters":
        await handleCharactersExtraction(step, payments, extractor);
        break;
      case "transformScenes":
        await handleScenesTransformation(step, payments, extractor);
        break;
      default:
        logger.warn(`Unrecognized step name: ${step.name}. Skipping...`);
    }
  };
}

/**
 * Handles the "init" step by defining the workflow steps:
 * 1. "generateScript"
 * 2. "extractScenes"
 */
async function handleInitStep(step: any, payments: Payments) {
  const scriptStepId = generateStepId();
  const scenestepId = generateStepId();
  const characterStepId = generateStepId();
  const transformStepId = generateStepId();

  const steps = [
    {
      step_id: scriptStepId,
      task_id: step.task_id,
      predecessor: step.step_id, // "generateScript" follows "init"
      name: "generateScript",
      is_last: false,
    },
    {
      step_id: scenestepId,
      task_id: step.task_id,
      predecessor: scriptStepId, // "extractScenes" follows "generateScript"
      name: "extractScenes",
      is_last: false,
    },
    {
      step_id: characterStepId,
      task_id: step.task_id,
      predecessor: scenestepId, // "extractCharacters" follows "extractScenes"
      name: "extractCharacters",
      is_last: false,
    },
    {
      step_id: transformStepId,
      task_id: step.task_id,
      predecessor: characterStepId, // "transformScenes" follows "extractCharacters"
      name: "transformScenes",
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
    output: step.input_query,
  });
}

/**
 * Handles the "generateScript" step, using the `SceneTechnicalExtractor` to generate a script.
 */
async function handleScriptGeneration(
  step: any,
  payments: Payments,
  extractor: SceneTechnicalExtractor
) {
  try {
    const script = await extractor.generateScript(step.input_query);

    logger.info(`Generated script: ${script}`);
    await payments.query.updateStep(step.did, {
      ...step,
      step_status: AgentExecutionStatus.Completed,
      output: "Script generation completed.",
      output_artifacts: [script],
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
 * Handles the "extractScenes" step, extracting scenes from the generated script.
 */
async function handleScenesExtraction(
  step: any,
  payments: Payments,
  extractor: SceneTechnicalExtractor
) {
  try {
    const scenes = await extractor.extractScenes(step.input_query);
    const script = JSON.parse(step.input_artifacts);

    logger.info(`Extracted scenes: ${JSON.stringify(scenes)}`);

    await payments.query.updateStep(step.did, {
      ...step,
      step_status: AgentExecutionStatus.Completed,
      output: "Scenes extraction completed.",
      output_artifacts: [script, scenes],
    });

    logMessage(payments, {
      task_id: step.task_id,
      level: "info",
      message: `Scenes extraction completed successfully.`,
    });
  } catch (error) {
    logger.error(`Error during scenes extraction: ${error.message}`);
    await payments.query.updateStep(step.did, {
      ...step,
      step_status: AgentExecutionStatus.Failed,
      output: "Failed to extract scenes.",
    });

    logMessage(payments, {
      task_id: step.task_id,
      level: "info",
      message: `Scenes extraction completed successfully.`,
      task_status: AgentExecutionStatus.Failed,
    });
  }
}

/**
 * Handles the "extractCharacters" step, extracting characters from the generated script.
 */
async function handleCharactersExtraction(
  step: any,
  payments: Payments,
  extractor: SceneTechnicalExtractor
) {
  try {
    const characters = await extractor.extractCharacters(step.input_query);
    const [script, scenes] = JSON.parse(step.input_artifacts);

    logger.info(`Extracted characters: ${JSON.stringify(characters)}`);

    await payments.query.updateStep(step.did, {
      ...step,
      step_status: AgentExecutionStatus.Completed,
      output: "Characters extraction completed.",
      output_artifacts: [script, scenes, characters],
    });

    logMessage(payments, {
      task_id: step.task_id,
      level: "info",
      message: `Characters extraction completed successfully.`,
    });
  } catch (error) {
    logger.error(`Error during characters extraction: ${error.message}`);
    await payments.query.updateStep(step.did, {
      ...step,
      step_status: AgentExecutionStatus.Failed,
      output: "Failed to extract characters.",
    });
  }
}

/**
 * Handles the "transformScenes" step, transforming the extracted scenes.
 */
async function handleScenesTransformation(
  step: any,
  payments: Payments,
  extractor: SceneTechnicalExtractor
) {
  try {
    const [script, scenes, characters] = JSON.parse(step.input_artifacts);
    const transformedScenes = await extractor.transformScenes(
      scenes,
      characters,
      script
    );

    try {
      JSON.stringify(transformedScenes);
    } catch (error) {
      logger.info(transformedScenes);
      logger.info(`Error transforming scenes: ${error.message}`);
      throw new Error("Failed to transform scenes.");
    }

    logger.info(`Transformed scenes: ${JSON.stringify(transformedScenes)}`);

    await payments.query.updateStep(step.did, {
      ...step,
      step_status: AgentExecutionStatus.Completed,
      output: step.input_query,
      output_artifacts: [
        JSON.stringify({
          scenes: JSON.parse(step.input_artifacts),
          prompts: transformedScenes,
        }),
      ],
    });

    logMessage(payments, {
      task_id: step.task_id,
      level: "info",
      message: `Scenes transformation completed successfully.`,
      task_status: AgentExecutionStatus.Completed,
    });
  } catch (error) {
    logger.error(`Error during scenes transformation: ${error.message}`);
    await payments.query.updateStep(step.did, {
      ...step,
      step_status: AgentExecutionStatus.Failed,
      output: "Failed to transform scenes.",
    });
  }
}
