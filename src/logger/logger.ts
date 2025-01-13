import pino from "pino";

/**
 * Logger instance for structured logs.
 */
export const logger = pino({
  transport: { target: "pino-pretty" },
  level: "info",
});

/**
 * Logs messages to both the console and the Nevermined Payments API.
 */
export async function logMessage(
  payments: any,
  log: { task_id: string; level: string; message: string }
) {
  logger.info(log.message);
  await payments.query.logTask({
    task_id: log.task_id,
    message: log.message,
    level: log.level,
  });
}
