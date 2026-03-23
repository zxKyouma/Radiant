import { z } from "zod";

import { mriSequenceTypeSchema } from "./manifests";

export const preprocessingStatusSchema = z.enum(["prepared", "invalid"]);

export const preparedSeriesSchema = z.object({
  sequenceType: mriSequenceTypeSchema,
  path: z.string().min(1),
  sizeBytes: z.number().int().nonnegative()
});

export const preparedMaskSchema = z.object({
  label: z.string().min(1),
  path: z.string().min(1),
  sizeBytes: z.number().int().nonnegative()
});

export const preparedStudySchema = z.object({
  caseId: z.string().min(1),
  datasetId: z.string().min(1),
  studyId: z.string().min(1),
  status: preprocessingStatusSchema,
  preparedAt: z.string().min(1),
  preparedSeries: z.array(preparedSeriesSchema),
  masks: z.array(preparedMaskSchema),
  notes: z.array(z.string())
});

export type PreprocessingStatus = z.infer<typeof preprocessingStatusSchema>;
export type PreparedSeries = z.infer<typeof preparedSeriesSchema>;
export type PreparedMask = z.infer<typeof preparedMaskSchema>;
export type PreparedStudy = z.infer<typeof preparedStudySchema>;
