import type {
  Asset,
  CaseRecord,
  CreateCaseInput,
  QAResult,
  ReviewDecision,
  SegmentationResult,
  StructuredTarget,
  SupportDecision
} from "./schemas";

export function makeId(prefix: string): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`;
}

export function nowIso(): string {
  return new Date().toISOString();
}

export function createQueuedCase(input: CreateCaseInput): CaseRecord {
  const timestamp = nowIso();

  return {
    id: makeId("case"),
    studyId: input.studyId,
    patientLabel: input.patientLabel,
    modality: "brain_mri",
    sourceDatasetId: null,
    sourceCaseId: null,
    reportText: input.reportText,
    workflowStatus: "queued",
    createdAt: timestamp,
    updatedAt: timestamp,
    structuredTarget: null,
    supportDecision: null,
    segmentationResult: null,
    qaResult: null,
    patientSummary: null,
    assets: [],
    reviewDecision: null
  };
}

export function createDemoIntake(): CreateCaseInput {
  return {
    studyId: "study_demo_001",
    patientLabel: "Demo Patient",
    reportText:
      "MRI brain demonstrates a 2.8 x 2.1 x 2.4 cm right frontal extra-axial mass with mild surrounding edema. Findings are most consistent with a dominant right frontal mass. No additional intracranial lesions are described."
  };
}

export function createDemoSupportDecision(): SupportDecision {
  return {
    status: "supported",
    reason: "Single dominant intracranial mass-like lesion with clear laterality.",
    confidence: 0.93
  };
}

export function createDemoTarget(): StructuredTarget {
  return {
    finding: "right frontal extra-axial mass",
    anatomy: "brain",
    subAnatomy: "right frontal region",
    laterality: "right",
    sizeText: "2.8 x 2.1 x 2.4 cm",
    sizeMm: [28, 21, 24],
    certainty: "high",
    priority: "primary",
    segmentability: "likely_segmentable",
    supportStatus: "supported",
    supportReason:
      "Single dominant intracranial mass-like lesion with clear location.",
    confidence: 0.93
  };
}

export function createDemoSegmentation(caseId: string): SegmentationResult {
  return {
    maskAssetId: makeId("mask"),
    backend: "medsam3",
    backendVersion: "demo-v1",
    inferenceConfidence: 0.82,
    warnings: [],
    generatedAt: nowIso()
  };
}

export function createDemoQA(): QAResult {
  return {
    status: "reviewable",
    score: 0.88,
    flags: [],
    blockingReasons: []
  };
}

export function createDemoAssets(caseId: string): Asset[] {
  return [
    {
      id: makeId("asset"),
      kind: "slice_overlay",
      label: "Axial overlay",
      url: `/cases/${caseId}/assets/axial-overlay`
    },
    {
      id: makeId("asset"),
      kind: "render_3d",
      label: "3D contextual render",
      url: `/cases/${caseId}/assets/render-3d`
    },
    {
      id: makeId("asset"),
      kind: "summary_card",
      label: "Patient summary card",
      url: `/cases/${caseId}/assets/summary-card`
    }
  ];
}

export function createReviewDecision(
  action: ReviewDecision["action"],
  reviewerId: string,
  notes?: string
): ReviewDecision {
  return {
    action,
    reviewerId,
    notes: notes?.trim() || null,
    createdAt: nowIso()
  };
}
