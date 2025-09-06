import dotenv from "dotenv";

dotenv.config();

export const NVM_API_KEY = process.env.NVM_API_KEY!;
export const NVM_ENVIRONMENT = process.env.NVM_ENVIRONMENT || "testing";
export const AGENT_DID = process.env.AGENT_DID!;
export const HELICONE_API_KEY = process.env.HELICONE_API_KEY!;
export const HELICONE_BASE_LOGGING_URL = process.env.HELICONE_BASE_LOGGING_URL!;
export const HELICONE_MANUAL_LOGGING_URL = process.env.HELICONE_MANUAL_LOGGING_URL!;
export const IS_DUMMY = process.env.IS_DUMMY === "true";
