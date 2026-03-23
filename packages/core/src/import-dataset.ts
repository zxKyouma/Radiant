import {
  createQueuedCase,
  type CaseRecord
} from "@radiant/shared";

import { LocalCaseStore } from "./file-store";
import {
  buildCreateCaseInputFromManifest,
  loadDatasetBundle
} from "./manifests";

export type ImportDatasetOptions = {
  splitId?: string;
};

export type ImportDatasetResult = {
  datasetId: string;
  importedCount: number;
  skippedCaseIds: string[];
  importedCases: CaseRecord[];
};

export async function importDatasetBundle(
  store: LocalCaseStore,
  datasetManifestPath: string,
  options: ImportDatasetOptions = {}
): Promise<ImportDatasetResult> {
  const bundle = await loadDatasetBundle(datasetManifestPath);
  const allowedCaseIds = getAllowedCaseIds(bundle, options.splitId);
  const existingCases = await store.listCases();
  const importedCases: CaseRecord[] = [];
  const skippedCaseIds: string[] = [];

  for (const caseManifest of bundle.cases) {
    if (allowedCaseIds && !allowedCaseIds.has(caseManifest.caseId)) {
      continue;
    }

    const duplicate = existingCases.find(
      (entry) =>
        (entry.sourceDatasetId === bundle.dataset.datasetId &&
          entry.sourceCaseId === caseManifest.caseId) ||
        entry.studyId === caseManifest.studyId
    );

    if (duplicate) {
      skippedCaseIds.push(caseManifest.caseId);
      continue;
    }

    const input = await buildCreateCaseInputFromManifest(
      datasetManifestPath,
      bundle.dataset,
      caseManifest
    );
    const caseRecord = createQueuedCase(input);
    const importedCase: CaseRecord = {
      ...caseRecord,
      sourceDatasetId: bundle.dataset.datasetId,
      sourceCaseId: caseManifest.caseId
    };

    await store.saveCase(importedCase);
    existingCases.push(importedCase);
    importedCases.push(importedCase);
  }

  return {
    datasetId: bundle.dataset.datasetId,
    importedCount: importedCases.length,
    skippedCaseIds,
    importedCases
  };
}

function getAllowedCaseIds(
  bundle: Awaited<ReturnType<typeof loadDatasetBundle>>,
  splitId?: string
): Set<string> | null {
  if (!splitId) {
    return null;
  }

  const split = bundle.splits.find((entry) => entry.splitId === splitId);

  if (!split) {
    throw new Error(
      `Split ${splitId} was not found in dataset ${bundle.dataset.datasetId}.`
    );
  }

  return new Set(split.caseIds);
}
