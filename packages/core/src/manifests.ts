import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

import {
  caseManifestSchema,
  createCaseInputSchema,
  datasetManifestSchema,
  splitManifestSchema,
  type CaseManifest,
  type CreateCaseInput,
  type DatasetManifest,
  type SplitManifest
} from "@radiant/shared";
import { z } from "zod";

export type LoadedDatasetBundle = {
  datasetManifestPath: string;
  dataset: DatasetManifest;
  cases: CaseManifest[];
  splits: SplitManifest[];
};

export async function loadDatasetManifest(
  filePath: string
): Promise<DatasetManifest> {
  return readJsonFile(filePath, datasetManifestSchema);
}

export async function loadCaseManifest(filePath: string): Promise<CaseManifest> {
  return readJsonFile(filePath, caseManifestSchema);
}

export async function loadSplitManifest(
  filePath: string
): Promise<SplitManifest> {
  return readJsonFile(filePath, splitManifestSchema);
}

export async function loadDatasetBundle(
  datasetManifestPath: string
): Promise<LoadedDatasetBundle> {
  const dataset = await loadDatasetManifest(datasetManifestPath);
  const baseDir = dirname(datasetManifestPath);
  const cases = await Promise.all(
    dataset.caseManifestPaths.map((relativePath) =>
      loadCaseManifest(resolve(baseDir, relativePath))
    )
  );
  const splits = await Promise.all(
    dataset.splitManifestPaths.map((relativePath) =>
      loadSplitManifest(resolve(baseDir, relativePath))
    )
  );

  const caseIds = new Set(cases.map((entry) => entry.caseId));

  for (const caseManifest of cases) {
    if (caseManifest.datasetId !== dataset.datasetId) {
      throw new Error(
        `Case ${caseManifest.caseId} does not belong to dataset ${dataset.datasetId}.`
      );
    }
  }

  for (const split of splits) {
    if (split.datasetId !== dataset.datasetId) {
      throw new Error(
        `Split ${split.splitId} does not belong to dataset ${dataset.datasetId}.`
      );
    }

    for (const caseId of split.caseIds) {
      if (!caseIds.has(caseId)) {
        throw new Error(
          `Split ${split.splitId} references unknown case ${caseId}.`
        );
      }
    }
  }

  return {
    datasetManifestPath,
    dataset,
    cases,
    splits
  };
}

export function resolveDatasetAssetPath(
  datasetManifestPath: string,
  dataset: DatasetManifest,
  relativePath: string
): string {
  return resolve(dirname(datasetManifestPath), dataset.assetRoot, relativePath);
}

export function resolveCaseReportPath(
  datasetManifestPath: string,
  dataset: DatasetManifest,
  caseManifest: CaseManifest
): string {
  return resolveDatasetAssetPath(
    datasetManifestPath,
    dataset,
    caseManifest.report.file.relativePath
  );
}

export async function buildCreateCaseInputFromManifest(
  datasetManifestPath: string,
  dataset: DatasetManifest,
  caseManifest: CaseManifest
): Promise<CreateCaseInput> {
  const reportPath = resolveCaseReportPath(
    datasetManifestPath,
    dataset,
    caseManifest
  );
  const reportText = await readFile(reportPath, "utf8");

  return createCaseInputSchema.parse({
    studyId: caseManifest.studyId,
    patientLabel: caseManifest.patientLabel,
    reportText
  });
}

async function readJsonFile<T>(
  filePath: string,
  schema: z.ZodSchema<T>
): Promise<T> {
  const raw = await readFile(filePath, "utf8");
  return schema.parse(JSON.parse(raw));
}

