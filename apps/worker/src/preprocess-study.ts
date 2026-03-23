import { dirname, isAbsolute, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { preprocessStudyFromBundle } from "@radiant/core";

type CliArgs = {
  manifestPath: string;
  caseId: string;
};

const args = parseArgs(process.argv.slice(2));
const invocationRoot = process.env.INIT_CWD ?? process.cwd();
const workerRoot = dirname(fileURLToPath(import.meta.url));
const outputDir = resolve(workerRoot, "../../../data/state/preprocessed-studies");
const manifestPath = isAbsolute(args.manifestPath)
  ? args.manifestPath
  : resolve(invocationRoot, args.manifestPath);

const result = await preprocessStudyFromBundle(manifestPath, args.caseId, {
  outputDir
});

console.log(
  JSON.stringify(
    {
      caseId: result.caseId,
      datasetId: result.datasetId,
      status: result.status,
      preparedSeries: result.preparedSeries.map((entry) => ({
        sequenceType: entry.sequenceType,
        sizeBytes: entry.sizeBytes
      })),
      masks: result.masks.map((entry) => ({
        label: entry.label,
        sizeBytes: entry.sizeBytes
      }))
    },
    null,
    2
  )
);

function parseArgs(argv: string[]): CliArgs {
  let manifestPath: string | undefined;
  let caseId: string | undefined;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === "--manifest") {
      manifestPath = argv[index + 1];
      index += 1;
      continue;
    }

    if (arg === "--case") {
      caseId = argv[index + 1];
      index += 1;
    }
  }

  if (!manifestPath || !caseId) {
    throw new Error(
      "Missing required args. Example: npm run preprocess:study -- --manifest data/manifests/brats-men-v1/dataset.manifest.json --case BraTS-MEN-00307-000"
    );
  }

  return {
    manifestPath,
    caseId
  };
}
