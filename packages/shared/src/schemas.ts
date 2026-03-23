import { z } from "zod";

export const modalitySchema = z.enum(["brain_mri"]);

export const supportStatusSchema = z.enum([
  "supported",
  "unsupported",
  "needs_human_target_selection"
]);

export const reviewReadinessSchema = z.enum([
  "reviewable",
  "blocked",
  "needs_manual_review"
]);

export const workflowStatusSchema = z.enum([
  "queued",
  "parsing_report",
  "checking_support",
  "preparing_study",
  "segmenting",
  "qa",
  "rendering",
  "reviewable",
  "approved",
  "rejected",
  "blocked",
  "failed"
]);

export const reviewActionSchema = z.enum(["approve", "reject", "rerun"]);

export const lateralitySchema = z.enum([
  "left",
  "right",
  "bilateral",
  "midline",
  "unknown"
]);

export const certaintySchema = z.enum(["low", "medium", "high"]);

export const segmentabilitySchema = z.enum([
  "likely_segmentable",
  "uncertain",
  "not_segmentable"
]);

export const structuredTargetSchema = z.object({
  finding: z.string(),
  anatomy: z.string(),
  subAnatomy: z.string().nullable(),
  laterality: lateralitySchema,
  sizeText: z.string().nullable(),
  sizeMm: z.array(z.number()),
  certainty: certaintySchema,
  priority: z.literal("primary"),
  segmentability: segmentabilitySchema,
  supportStatus: supportStatusSchema,
  supportReason: z.string(),
  confidence: z.number().min(0).max(1)
});

export const supportDecisionSchema = z.object({
  status: supportStatusSchema,
  reason: z.string(),
  confidence: z.number().min(0).max(1)
});

export const segmentationResultSchema = z.object({
  maskAssetId: z.string(),
  backend: z.string(),
  backendVersion: z.string(),
  inferenceConfidence: z.number().min(0).max(1),
  warnings: z.array(z.string()),
  generatedAt: z.string()
});

export const qaResultSchema = z.object({
  status: reviewReadinessSchema,
  score: z.number().min(0).max(1),
  flags: z.array(z.string()),
  blockingReasons: z.array(z.string())
});

export const assetSchema = z.object({
  id: z.string(),
  kind: z.enum(["slice_overlay", "render_3d", "summary_card"]),
  label: z.string(),
  url: z.string()
});

export const reviewDecisionSchema = z.object({
  action: reviewActionSchema,
  reviewerId: z.string(),
  notes: z.string().nullable(),
  createdAt: z.string()
});

export const caseRecordSchema = z.object({
  id: z.string(),
  studyId: z.string(),
  patientLabel: z.string(),
  modality: modalitySchema,
  sourceDatasetId: z.string().nullable(),
  sourceCaseId: z.string().nullable(),
  reportText: z.string(),
  workflowStatus: workflowStatusSchema,
  createdAt: z.string(),
  updatedAt: z.string(),
  structuredTarget: structuredTargetSchema.nullable(),
  supportDecision: supportDecisionSchema.nullable(),
  segmentationResult: segmentationResultSchema.nullable(),
  qaResult: qaResultSchema.nullable(),
  patientSummary: z.string().nullable(),
  assets: z.array(assetSchema),
  reviewDecision: reviewDecisionSchema.nullable()
});

export const createCaseInputSchema = z.object({
  studyId: z.string().min(1),
  patientLabel: z.string().min(1),
  reportText: z.string().min(20)
});

export const reviewActionInputSchema = z.object({
  action: reviewActionSchema,
  reviewerId: z.string().min(1),
  notes: z.string().trim().optional()
});

export const caseListSchema = z.array(caseRecordSchema);

export type Modality = z.infer<typeof modalitySchema>;
export type SupportStatus = z.infer<typeof supportStatusSchema>;
export type ReviewReadiness = z.infer<typeof reviewReadinessSchema>;
export type WorkflowStatus = z.infer<typeof workflowStatusSchema>;
export type ReviewAction = z.infer<typeof reviewActionSchema>;
export type Laterality = z.infer<typeof lateralitySchema>;
export type StructuredTarget = z.infer<typeof structuredTargetSchema>;
export type SupportDecision = z.infer<typeof supportDecisionSchema>;
export type SegmentationResult = z.infer<typeof segmentationResultSchema>;
export type QAResult = z.infer<typeof qaResultSchema>;
export type Asset = z.infer<typeof assetSchema>;
export type ReviewDecision = z.infer<typeof reviewDecisionSchema>;
export type CaseRecord = z.infer<typeof caseRecordSchema>;
export type CreateCaseInput = z.infer<typeof createCaseInputSchema>;
export type ReviewActionInput = z.infer<typeof reviewActionInputSchema>;
