import {
  AgentExecutionStatus,
  Payments,
  generateStepId,
} from "@nevermined-io/payments";
import { logMessage, logger } from "../logger/logger";
import { SceneTechnicalExtractor } from "../sceneTechnicalExtractor";
import { Scene } from "../types";

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
      case "generateSettings":
        await handleSettingsGeneration(step, payments, extractor);
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
  const sceneStepId = generateStepId();
  const settingsStepId = generateStepId();
  const characterStepId = generateStepId();
  const transformStepId = generateStepId();

  const steps = [
    {
      step_id: scriptStepId,
      task_id: step.task_id,
      predecessor: step.step_id,
      name: "generateScript",
      is_last: false,
    },
    {
      step_id: sceneStepId,
      task_id: step.task_id,
      predecessor: scriptStepId,
      name: "extractScenes",
      is_last: false,
    },
    {
      step_id: settingsStepId,
      task_id: step.task_id,
      predecessor: sceneStepId,
      name: "generateSettings",
      is_last: false,
    },
    {
      step_id: characterStepId,
      task_id: step.task_id,
      predecessor: settingsStepId,
      name: "extractCharacters",
      is_last: false,
    },
    {
      step_id: transformStepId,
      task_id: step.task_id,
      predecessor: characterStepId,
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
    output_artifacts: [JSON.parse(step.input_artifacts)],
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
    const [{ title, tags, lyrics, idea, duration }] = JSON.parse(
      step.input_artifacts
    );
    const script = await extractor.generateScript({
      title,
      tags,
      lyrics,
      idea,
      duration,
    });

    logger.info(`Generated script: ${script}`);
    await payments.query.updateStep(step.did, {
      ...step,
      step_status: AgentExecutionStatus.Completed,
      output: "Script generation completed.",
      output_artifacts: [
        {
          script,
          lyrics,
          tags,
          duration,
        },
      ],
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
    return;
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
    const [{ script, lyrics, tags, duration }] = JSON.parse(
      step.input_artifacts
    );
    const scenes = await extractor.extractScenes(script, duration);

    logger.info(`Extracted scenes: ${JSON.stringify(scenes)}`);

    await payments.query.updateStep(step.did, {
      ...step,
      step_status: AgentExecutionStatus.Completed,
      output: "Scenes extraction completed.",
      output_artifacts: [{ script, scenes, lyrics, tags, duration }],
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
 * Handles the "generateSettings" step, extracting settings from the generated script.
 *
 * @param step
 * @param payments
 * @param extractor
 */
async function handleSettingsGeneration(
  step: any,
  payments: Payments,
  extractor: SceneTechnicalExtractor
) {
  try {
    const [{ script, scenes, lyrics, tags, duration }] = JSON.parse(
      step.input_artifacts
    );
    const settings = await extractor.extractSettings(script);

    await payments.query.updateStep(step.did, {
      ...step,
      step_status: AgentExecutionStatus.Completed,
      output: "Settings generation completed.",
      output_artifacts: [{ script, scenes, settings, lyrics, tags, duration }],
    });
  } catch (error) {
    logger.error(`Error during settings generation: ${error.message}`);
    await payments.query.updateStep(step.did, {
      ...step,
      step_status: AgentExecutionStatus.Failed,
      output: "Failed to generate settings.",
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
    const [{ script, scenes, settings, lyrics, tags, duration }] = JSON.parse(
      step.input_artifacts
    );
    const characters = await extractor.extractCharacters(
      step.input_query,
      lyrics,
      tags
    );

    logger.info(`Extracted characters: ${JSON.stringify(characters)}`);

    await payments.query.updateStep(step.did, {
      ...step,
      step_status: AgentExecutionStatus.Completed,
      output: "Characters extraction completed.",
      output_artifacts: [{ script, settings, scenes, characters, duration }],
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
    const [{ script, settings, scenes, characters, duration }] = JSON.parse(
      step.input_artifacts
    );
    const prompts: Scene[] = await extractor.transformScenes(
      scenes,
      characters,
      settings,
      script
    );
    logger.info(`prompts: ${JSON.stringify(prompts)}`);
    logger.info("Sending data:");
    logger.info(
      JSON.stringify([{ prompts, settings, characters, script, scenes }])
    );

    const transformedScenes = adjustSceneDurations(prompts, duration);

    await payments.query.updateStep(step.did, {
      ...step,
      step_status: AgentExecutionStatus.Completed,
      output: step.input_query,
      output_artifacts: [
        { transformedScenes, settings, characters, script, scenes },
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

function adjustSceneDurations(
  scenes: Scene[],
  targetDuration: number
): Scene[] {
  // Calculate the current total duration of all scenes
  let currentSum = scenes.reduce((acc, scene) => acc + scene.duration, 0);

  // While the current total is less than the target duration,
  // randomly select a scene with a duration of 5 seconds and update it to 10 seconds.
  while (currentSum < targetDuration) {
    const candidates = scenes.filter((scene) => scene.duration === 5);
    if (candidates.length === 0) {
      console.warn("No scenes with duration 5 remain to increase.");
      break;
    }
    // Randomly select a candidate
    const randomIndex = Math.floor(Math.random() * candidates.length);
    const selectedScene = candidates[randomIndex];
    selectedScene.duration = 10;
    currentSum += 5; // Increase the total duration by 5 seconds
  }

  // While the current total is greater than the target duration,
  // randomly select a scene with a duration of 10 seconds and update it to 5 seconds.
  while (currentSum > targetDuration + 5) {
    const candidates = scenes.filter((scene) => scene.duration === 10);
    if (candidates.length === 0) {
      console.warn("No scenes with duration 10 remain to decrease.");
      break;
    }
    // Randomly select a candidate
    const randomIndex = Math.floor(Math.random() * candidates.length);
    const selectedScene = candidates[randomIndex];
    selectedScene.duration = 5;
    currentSum -= 5; // Decrease the total duration by 5 seconds
  }

  return scenes;
}
