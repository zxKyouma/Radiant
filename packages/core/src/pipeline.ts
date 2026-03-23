import {
  createDemoAssets,
  createDemoQA,
  createDemoSegmentation,
  nowIso,
  type CaseRecord,
  type Laterality,
  type StructuredTarget,
  type SupportDecision
} from "@radiant/shared";

import { LocalCaseStore } from "./file-store";

export async function runDemoPipeline(
  store: LocalCaseStore,
  caseId: string
): Promise<CaseRecord | null> {
  const current = await store.getCase(caseId);

  if (!current) {
    return null;
  }

  const structuredTarget = inferStructuredTarget(current.reportText);
  const supportDecision = inferSupportDecision(structuredTarget);

  await store.saveCase({
    ...current,
    structuredTarget,
    supportDecision,
    workflowStatus: "checking_support",
    updatedAt: nowIso()
  });

  if (supportDecision.status !== "supported") {
    return store.saveCase({
      ...current,
      structuredTarget,
      supportDecision,
      qaResult: {
        status: "blocked",
        score: 0.25,
        flags: ["unsupported_case"],
        blockingReasons: [supportDecision.reason]
      },
      workflowStatus: "blocked",
      updatedAt: nowIso()
    });
  }

  const segmentationResult = createDemoSegmentation(caseId);
  const qaResult = createDemoQA();
  const patientSummary = buildPatientSummary(structuredTarget);
  const assets = createDemoAssets(caseId);

  return store.saveCase({
    ...current,
    structuredTarget,
    supportDecision,
    segmentationResult,
    qaResult,
    patientSummary,
    assets,
    workflowStatus: "reviewable",
    updatedAt: nowIso()
  });
}

export async function processNextQueuedCase(
  store: LocalCaseStore
): Promise<CaseRecord | null> {
  const cases = await store.listCases();
  const nextCase = cases.find((entry) => entry.workflowStatus === "queued");

  if (!nextCase) {
    return null;
  }

  await store.saveCase({
    ...nextCase,
    workflowStatus: "parsing_report",
    updatedAt: nowIso()
  });

  return runDemoPipeline(store, nextCase.id);
}

function inferStructuredTarget(reportText: string): StructuredTarget {
  const normalized = reportText.toLowerCase();
  const laterality = inferLaterality(normalized);
  const subAnatomy = inferSubAnatomy(normalized);
  const size = inferSize(reportText);
  const findingCore =
    normalized.includes("extra-axial")
      ? "extra-axial mass"
      : normalized.includes("mass")
        ? "mass"
        : normalized.includes("lesion")
          ? "lesion"
          : "intracranial finding";

  const findingParts = [laterality !== "unknown" ? laterality : null, subAnatomy, findingCore]
    .filter(Boolean)
    .join(" ");

  const supportStatus = inferSupportStatus(normalized);

  return {
    finding: findingParts || "intracranial mass",
    anatomy: "brain",
    subAnatomy,
    laterality,
    sizeText: size.sizeText,
    sizeMm: size.sizeMm,
    certainty: normalized.includes("likely") ? "medium" : "high",
    priority: "primary",
    segmentability:
      supportStatus === "supported" ? "likely_segmentable" : "uncertain",
    supportStatus,
    supportReason:
      supportStatus === "supported"
        ? "Single dominant intracranial mass-like lesion with clear location."
        : "Report content is ambiguous or outside the supported single-lesion workflow.",
    confidence: supportStatus === "supported" ? 0.9 : 0.42
  };
}

function inferSupportDecision(
  structuredTarget: StructuredTarget
): SupportDecision {
  return {
    status: structuredTarget.supportStatus,
    reason: structuredTarget.supportReason,
    confidence: structuredTarget.confidence
  };
}

function inferSupportStatus(reportText: string): SupportDecision["status"] {
  const unsupportedMarkers = [
    "multiple lesions",
    "multiple masses",
    "metastases",
    "diffuse",
    "postoperative",
    "post-op",
    "postsurgical"
  ];

  if (unsupportedMarkers.some((marker) => reportText.includes(marker))) {
    return "unsupported";
  }

  if (reportText.includes("mass") || reportText.includes("lesion")) {
    return "supported";
  }

  return "needs_human_target_selection";
}

function inferLaterality(reportText: string): Laterality {
  if (reportText.includes("left")) {
    return "left";
  }

  if (reportText.includes("right")) {
    return "right";
  }

  if (reportText.includes("bilateral")) {
    return "bilateral";
  }

  if (reportText.includes("midline")) {
    return "midline";
  }

  return "unknown";
}

function inferSubAnatomy(reportText: string): string | null {
  const regions = ["frontal", "parietal", "temporal", "occipital", "cerebellar"];

  return regions.find((region) => reportText.includes(region)) ?? null;
}

function inferSize(reportText: string): { sizeText: string | null; sizeMm: number[] } {
  const match = reportText.match(
    /(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*(cm|mm)/i
  );

  if (!match) {
    return {
      sizeText: null,
      sizeMm: []
    };
  }

  const unit = match[4].toLowerCase();
  const values = match.slice(1, 4).map(Number);
  const scale = unit === "cm" ? 10 : 1;

  return {
    sizeText: match[0],
    sizeMm: values.map((value) => Math.round(value * scale))
  };
}

function buildPatientSummary(structuredTarget: StructuredTarget): string {
  const location = [structuredTarget.laterality, structuredTarget.subAnatomy]
    .filter((value) => value && value !== "unknown")
    .join(" ");

  return `This review package highlights the main lesion described in the report${location ? ` in the ${location} brain` : ""}. It is intended for patient education and should be interpreted alongside the finalized radiology report and clinician review.`;
}

