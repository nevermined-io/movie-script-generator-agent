import pino from "pino";
import { TaskLogMessage } from "@nevermined-io/payments";

/**
 * Logger instance for structured logs.
 */
export const logger = pino({
  transport: { target: "pino-pretty" },
  level: "info",
});

/**
 * Logs a message both locally and remotely using the Nevermined Payments API.
 *
 * This function provides a unified logging mechanism to:
 * - Log messages locally using the `pino` logger.
 * - Send log messages to the Nevermined Payments API for task-level tracking.
 *
 * @param payments - The initialized Payments API instance used to interact with the Nevermined network.
 * @param logMessage - The log message object containing details such as the task ID, log level, and message content.
 *
 * The `logMessage` object should follow this structure:
 * ```typescript
 * {
 *   task_id: string;   // Unique identifier for the task this log belongs to.
 *   level: "info" | "warning" | "error" | "debug";  // Log severity level.
 *   message: string;   // The log message content.
 * }
 * ```
 *
 * Example:
 * ```typescript
 * logMessage(payments, {
 *   task_id: "task123",
 *   level: "info",
 *   message: "Processing step X successfully."
 * });
 * ```
 */
export async function logMessage(payments, logMessage: TaskLogMessage) {
  const message = `${logMessage.task_id} :: ${logMessage.message}`;

  if (logMessage.level === "error") logger.error(message);
  else if (logMessage.level === "warning") logger.warn(message);
  else if (logMessage.level === "debug") logger.debug(message);
  else logger.info(message);

  /**
   * Log remotely to the Nevermined Payments API.
   *
   * This allows external systems and users to track task progress and status in real-time.
   */
  await payments.query.logTask(logMessage);
}
