import { mkdir, stat, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

import {
  preparedStudySchema,
  type CaseManifest,
  type MRISequenceType,
  type PreparedMask,
  type PreparedSeries,
  type PreparedStudy
} from "@radiant/shared";

import {
  loadDatasetBundle,
  resolveDatasetAssetPath,
  type LoadedDatasetBundle
} from "./manifests";

const REQUIRED_SEQUENCE_TYPES: MRISequenceType[] = ["t1", "t1_post", "t2", "flair"];

export type PreprocessStudyOptions = {
  outputDir: string;
};

export async function preprocessStudyFromBundle(
  datasetManifestPath: string,
  caseId: string,
  options: PreprocessStudyOptions
): Promise<PreparedStudy> {
  const bundle = await loadDatasetBundle(datasetManifestPath);
  const caseManifest = bundle.cases.find((entry) => entry.caseId == caseId);

  if (!caseManifest) {
    throw new Error(`Case ${caseId} was not found in dataset ${bundle.dataset.datasetId}.`);
  }

  return preprocessCaseManifest(bundle, caseManifest, options);
}

export async function preprocessCaseManifest(
  bundle: LoadedDatasetBundle,
  caseManifest: CaseManifest,
  options: PreprocessStudyOptions
): Promise<PreparedStudy> {
  const notes: string[] = [];
  const preparedSeries: PreparedSeries[] = [];
  const masks: PreparedMask[] = [];
  const seenSequenceTypes = new Set<MRISequenceType>();

  for (const sequenceType of REQUIRED_SEQUENCE_TYPES) {
    const series = caseManifest.imagingSeries.find((entry) => entry.sequenceType === sequenceType);

    if (!series) {
      throw new Error(`Case ${caseManifest.caseId} is missing required sequence ${sequenceType}.`);
    }

    const path = resolveDatasetAssetPath(
      bundle.datasetManifestPath,
      bundle.dataset,
      series.file.relativePath
    );
    const info = await stat(path);

    preparedSeries.push({
      sequenceType,
      path,
      sizeBytes: info.size
    });
    seenSequenceTypes.add(sequenceType);
  }

  for (const series of caseManifest.imagingSeries) {
    if (seenSequenceTypes.has(series.sequenceType)) {
      continue;
    }

    const path = resolveDatasetAssetPath(
      bundle.datasetManifestPath,
      bundle.dataset,
      series.file.relativePath
    );
    const info = await stat(path);

    preparedSeries.push({
      sequenceType: series.sequenceType,
      path,
      sizeBytes: info.size
    });
    notes.push(`Included optional sequence ${series.sequenceType}.`);
  }

  for (const mask of caseManifest.masks) {
    const path = resolveDatasetAssetPath(
      bundle.datasetManifestPath,
      bundle.dataset,
      mask.file.relativePath
    );
    const info = await stat(path);

    masks.push({
      label: mask.label,
      path,
      sizeBytes: info.size
    });
  }

  const preparedStudy = preparedStudySchema.parse({
    caseId: caseManifest.caseId,
    datasetId: caseManifest.datasetId,
    studyId: caseManifest.studyId,
    status: "prepared",
    preparedAt: new Date().toISOString(),
    preparedSeries,
    masks,
    notes
  });

  await persistPreparedStudy(preparedStudy, options.outputDir);

  return preparedStudy;
}

async function persistPreparedStudy(
  preparedStudy: PreparedStudy,
  outputDir: string
): Promise<void> {
  const outputPath = resolve(outputDir, `${preparedStudy.caseId}.prepared.json`);
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, JSON.stringify(preparedStudy, null, 2) + "\n");
}
