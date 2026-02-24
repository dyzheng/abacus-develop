import { sqliteTable, text, integer, real, index, uniqueIndex } from "drizzle-orm/sqlite-core";

export const projects = sqliteTable("projects", {
  id: text("id").primaryKey(),
  name: text("name").notNull(),
  clientCompany: text("client_company").notNull(),
  auditFirm: text("audit_firm").notNull(),
  balanceDate: text("balance_date").notNull(),
  status: text("status", { enum: ["active", "completed", "archived"] }).notNull().default("active"),
  createdAt: text("created_at").notNull(),
  updatedAt: text("updated_at").notNull(),
});

export const arRecords = sqliteTable("ar_records", {
  id: text("id").primaryKey(),
  projectId: text("project_id").notNull().references(() => projects.id),
  importBatchId: text("import_batch_id").references(() => importBatches.id),
  customerName: text("customer_name").notNull(),
  customerCode: text("customer_code"),
  totalBalance: real("total_balance").notNull(),
  within1Year: real("within_1_year").default(0),
  year1to2: real("year_1_to_2").default(0),
  year2to3: real("year_2_to_3").default(0),
  year3to4: real("year_3_to_4").default(0),
  year4to5: real("year_4_to_5").default(0),
  over5Years: real("over_5_years").default(0),
  riskLevel: text("risk_level", { enum: ["low", "medium", "high"] }).default("low"),
  isRelatedParty: integer("is_related_party", { mode: "boolean" }).default(false),
  selectionStatus: text("selection_status", { enum: ["unselected", "ai_suggested", "confirmed", "excluded"] }).default("unselected"),
  selectionReason: text("selection_reason"),
  contactPerson: text("contact_person"),
  contactPhone: text("contact_phone"),
  contactEmail: text("contact_email"),
  address: text("address"),
  city: text("city"),
  province: text("province"),
  postalCode: text("postal_code"),
  verificationStatus: text("verification_status", { enum: ["unverified", "verified", "suspicious", "flagged"] }).default("unverified"),
  verificationDetail: text("verification_detail"),
  verificationScore: real("verification_score"),
  createdAt: text("created_at").notNull(),
}, (table) => [
  index("idx_ar_records_project_id").on(table.projectId),
  index("idx_ar_records_import_batch_id").on(table.importBatchId),
  index("idx_ar_records_customer_name").on(table.customerName),
  index("idx_ar_records_selection_status").on(table.selectionStatus),
  index("idx_ar_records_verification_status").on(table.verificationStatus),
]);

export const confirmations = sqliteTable("confirmations", {
  id: text("id").primaryKey(),
  projectId: text("project_id").notNull().references(() => projects.id),
  arRecordId: text("ar_record_id").notNull().references(() => arRecords.id),
  confirmationNumber: text("confirmation_number").notNull(),
  type: text("type", { enum: ["positive", "blank"] }).notNull().default("positive"),
  status: text("status", {
    enum: ["draft", "generated", "sent", "received", "reconciled", "alternative_procedure"],
  }).notNull().default("draft"),
  letterContent: text("letter_content"),
  letterGeneratedAt: text("letter_generated_at"),
  pdfPath: text("pdf_path"),
  sentDate: text("sent_date"),
  dueDate: text("due_date"),
  receivedDate: text("received_date"),
  createdAt: text("created_at").notNull(),
  updatedAt: text("updated_at").notNull(),
}, (table) => [
  index("idx_confirmations_project_id").on(table.projectId),
  index("idx_confirmations_ar_record_id").on(table.arRecordId),
  index("idx_confirmations_status").on(table.status),
  uniqueIndex("idx_confirmations_number_project").on(table.projectId, table.confirmationNumber),
]);

export const responses = sqliteTable("responses", {
  id: text("id").primaryKey(),
  confirmationId: text("confirmation_id").notNull().references(() => confirmations.id),
  projectId: text("project_id").notNull().references(() => projects.id),
  responseType: text("response_type", {
    enum: ["agree", "disagree", "partial", "no_response"],
  }).notNull(),
  respondedAmount: real("responded_amount"),
  differenceAmount: real("difference_amount"),
  respondentName: text("respondent_name"),
  respondentTitle: text("respondent_title"),
  responseDate: text("response_date"),
  notes: text("notes"),
  createdAt: text("created_at").notNull(),
}, (table) => [
  index("idx_responses_confirmation_id").on(table.confirmationId),
  index("idx_responses_project_id").on(table.projectId),
]);

export const differences = sqliteTable("differences", {
  id: text("id").primaryKey(),
  responseId: text("response_id").notNull().references(() => responses.id),
  confirmationId: text("confirmation_id").notNull().references(() => confirmations.id),
  projectId: text("project_id").notNull().references(() => projects.id),
  bookAmount: real("book_amount").notNull(),
  confirmedAmount: real("confirmed_amount").notNull(),
  differenceAmount: real("difference_amount").notNull(),
  aiSuggestedCause: text("ai_suggested_cause"),
  aiAnalysisDetail: text("ai_analysis_detail"),
  aiConfidenceScore: real("ai_confidence_score"),
  auditorResolution: text("auditor_resolution"),
  status: text("status", { enum: ["pending", "analyzing", "analyzed", "resolved"] }).default("pending"),
  createdAt: text("created_at").notNull(),
  updatedAt: text("updated_at").notNull(),
}, (table) => [
  index("idx_differences_response_id").on(table.responseId),
  index("idx_differences_confirmation_id").on(table.confirmationId),
  index("idx_differences_project_id").on(table.projectId),
  index("idx_differences_status").on(table.status),
]);

export const aiLogs = sqliteTable("ai_logs", {
  id: text("id").primaryKey(),
  projectId: text("project_id").references(() => projects.id),
  action: text("action").notNull(),
  prompt: text("prompt").notNull(),
  response: text("response"),
  tokensUsed: integer("tokens_used"),
  model: text("model"),
  status: text("status", { enum: ["success", "error"] }),
  errorMessage: text("error_message"),
  createdAt: text("created_at").notNull(),
}, (table) => [
  index("idx_ai_logs_project_id").on(table.projectId),
]);

export const importBatches = sqliteTable("import_batches", {
  id: text("id").primaryKey(),
  projectId: text("project_id").notNull().references(() => projects.id),
  fileName: text("file_name").notNull(),
  recordCount: integer("record_count").default(0),
  columnMapping: text("column_mapping"),
  status: text("status", { enum: ["pending", "processing", "completed", "failed"] }).default("pending"),
  errorMessage: text("error_message"),
  createdAt: text("created_at").notNull(),
}, (table) => [
  index("idx_import_batches_project_id").on(table.projectId),
]);
