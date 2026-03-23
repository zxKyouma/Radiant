import { z } from "zod";

import {
  lateralitySchema,
  modalitySchema,
  supportStatusSchema
} from "./schemas";

export const manifestVersionSchema = z.literal("1.0");

export const datasetSourceSchema = z.enum(["demo", "public", "internal"]);

export const datasetSensitivitySchema = z.enum([
  "demo",
  "deidentified",
  "restricted"
]);

export const datasetTaskSchema = z.enum([
  "segmentation_training",
  "report_grounding",
  "routing_negative",
  "evaluation",
  "pilot_review"
]);

export const lesionCategorySchema = z.enum([
  "meningioma",
  "glioma",
  "brain_metastasis",
  "unspecified_intracranial_mass",
  "unsupported_other"
]);

export const artifactFormatSchema = z.enum([
  "txt",
  "dicom_dir",
  "nifti",
  "nrrd",
  "json",
  "png",
  "other"
]);

export const mriSequenceTypeSchema = z.enum([
  "t1",
  "t1_post",
  "t2",
  "flair",
  "dwi",
  "adc",
  "other"
]);

export const annotationSourceSchema = z.enum([
  "manual",
  "semi_automatic",
  "automatic",
  "none"
]);

export const curationTierSchema = z.enum([
  "gold",
  "silver",
  "bronze",
  "unreviewed"
]);

export const splitKindSchema = z.enum([
  "train",
  "validation",
  "holdout",
  "pilot_review",
  "unsupported_eval",
  "demo"
]);

export const assetRefSchema = z.object({
  relativePath: z.string().min(1),
  format: artifactFormatSchema,
  description: z.string().nullable(),
  sha256: z.string().nullable()
});

export const reportAssetSchema = z.object({
  status: z.enum(["finalized", "preliminary", "unknown"]),
  file: assetRefSchema
});

export const imagingSeriesSchema = z.object({
  seriesId: z.string().min(1),
  seriesDescription: z.string().nullable(),
  sequenceType: mriSequenceTypeSchema,
  file: assetRefSchema
});

export const maskAnnotationSchema = z.object({
  label: z.string().min(1),
  source: annotationSourceSchema,
  file: assetRefSchema,
  approvedBy: z.string().nullable()
});

export const caseGroundTruthSchema = z.object({
  supportLabel: supportStatusSchema,
  lesionCategory: lesionCategorySchema,
  dominantFindingText: z.string().nullable(),
  dominantLaterality: lateralitySchema,
  curationTier: curationTierSchema,
  notes: z.string().nullable()
});

export const provenanceSchema = z.object({
  sourceInstitution: z.string().nullable(),
  sourceCaseId: z.string().nullable(),
  deidentified: z.boolean()
});

export const caseManifestSchema = z.object({
  manifestVersion: manifestVersionSchema,
  caseId: z.string().min(1),
  datasetId: z.string().min(1),
  studyId: z.string().min(1),
  patientLabel: z.string().min(1),
  modality: modalitySchema,
  report: reportAssetSchema,
  imagingSeries: z.array(imagingSeriesSchema).min(1),
  masks: z.array(maskAnnotationSchema),
  groundTruth: caseGroundTruthSchema,
  provenance: provenanceSchema,
  tags: z.array(z.string())
});

export const splitManifestSchema = z.object({
  manifestVersion: manifestVersionSchema,
  datasetId: z.string().min(1),
  splitId: z.string().min(1),
  kind: splitKindSchema,
  description: z.string(),
  caseIds: z.array(z.string())
});

export const datasetManifestSchema = z.object({
  manifestVersion: manifestVersionSchema,
  datasetId: z.string().min(1),
  name: z.string().min(1),
  description: z.string().min(1),
  source: datasetSourceSchema,
  sensitivity: datasetSensitivitySchema,
  modality: modalitySchema,
  tasks: z.array(datasetTaskSchema).min(1),
  lesionCategories: z.array(lesionCategorySchema).min(1),
  assetRoot: z.string().min(1),
  caseManifestPaths: z.array(z.string().min(1)).min(1),
  splitManifestPaths: z.array(z.string().min(1)),
  createdAt: z.string().min(1),
  notes: z.string().nullable()
});

export type DatasetSource = z.infer<typeof datasetSourceSchema>;
export type DatasetSensitivity = z.infer<typeof datasetSensitivitySchema>;
export type DatasetTask = z.infer<typeof datasetTaskSchema>;
export type LesionCategory = z.infer<typeof lesionCategorySchema>;
export type ArtifactFormat = z.infer<typeof artifactFormatSchema>;
export type MRISequenceType = z.infer<typeof mriSequenceTypeSchema>;
export type AnnotationSource = z.infer<typeof annotationSourceSchema>;
export type CurationTier = z.infer<typeof curationTierSchema>;
export type SplitKind = z.infer<typeof splitKindSchema>;
export type AssetRef = z.infer<typeof assetRefSchema>;
export type ReportAsset = z.infer<typeof reportAssetSchema>;
export type ImagingSeries = z.infer<typeof imagingSeriesSchema>;
export type MaskAnnotation = z.infer<typeof maskAnnotationSchema>;
export type CaseGroundTruth = z.infer<typeof caseGroundTruthSchema>;
export type Provenance = z.infer<typeof provenanceSchema>;
export type CaseManifest = z.infer<typeof caseManifestSchema>;
export type SplitManifest = z.infer<typeof splitManifestSchema>;
export type DatasetManifest = z.infer<typeof datasetManifestSchema>;

