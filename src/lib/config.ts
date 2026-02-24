// Centralized configuration — values can be overridden via environment variables

export const config = {
  /** OpenAI model name */
  aiModel: process.env.AI_MODEL || "gpt-4o",

  /** Risk classification thresholds (in yuan) */
  riskHighBalanceThreshold: Number(process.env.RISK_HIGH_BALANCE) || 1_000_000,
  riskMediumBalanceThreshold: Number(process.env.RISK_MEDIUM_BALANCE) || 100_000,
  riskHighAgingRatio: Number(process.env.RISK_HIGH_AGING_RATIO) || 0.3,
  riskMediumAgingRatio: Number(process.env.RISK_MEDIUM_AGING_RATIO) || 0.1,

  /** Difference tolerance — absolute amount below which differences are zeroed */
  differenceTolerance: Number(process.env.DIFFERENCE_TOLERANCE) || 0.01,

  /** Default confirmation due days from sent date */
  confirmationDueDays: Number(process.env.CONFIRMATION_DUE_DAYS) || 30,

  /** Max import file size in bytes */
  maxImportFileSize: Number(process.env.MAX_IMPORT_FILE_SIZE) || 10 * 1024 * 1024,
} as const;
