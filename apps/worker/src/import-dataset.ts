import { dirname, isAbsolute, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { importDatasetBundle, LocalCaseStore } from "@radiant/core";

type CliArgs = {
  manifestPath: string;
  splitId?: string;
};

const args = parseArgs(process.argv.slice(2));
const invocationRoot = process.env.INIT_CWD ?? process.cwd();
const dataFile = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../../data/state/cases.json"
);
const store = new LocalCaseStore(dataFile);

const manifestPath = isAbsolute(args.manifestPath)
  ? args.manifestPath
  : resolve(invocationRoot, args.manifestPath);
const result = await importDatasetBundle(store, manifestPath, {
  splitId: args.splitId
});

console.log(
  JSON.stringify(
    {
      datasetId: result.datasetId,
      importedCount: result.importedCount,
      skippedCaseIds: result.skippedCaseIds,
      importedCaseIds: result.importedCases.map((entry) => entry.id)
    },
    null,
    2
  )
);

function parseArgs(argv: string[]): CliArgs {
  let manifestPath: string | undefined;
  let splitId: string | undefined;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === "--manifest") {
      manifestPath = argv[index + 1];
      index += 1;
      continue;
    }

    if (arg === "--split") {
      splitId = argv[index + 1];
      index += 1;
    }
  }

  if (!manifestPath) {
    throw new Error(
      "Missing --manifest <path>. Example: npm run import:dataset -- --manifest data/manifests/demo-brain-mri-meningioma/dataset.manifest.json"
    );
  }

  return {
    manifestPath,
    splitId
  };
}
