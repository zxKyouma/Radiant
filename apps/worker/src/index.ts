import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { LocalCaseStore, processNextQueuedCase } from "@radiant/core";

const intervalMs = Number(process.env.WORKER_INTERVAL_MS ?? 5000);
const dataFile = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../../data/state/cases.json"
);
const store = new LocalCaseStore(dataFile);

console.log(`[worker] watching ${dataFile}`);
console.log(`[worker] polling every ${intervalMs}ms`);

await tick();
setInterval(() => {
  void tick();
}, intervalMs);

async function tick(): Promise<void> {
  const processed = await processNextQueuedCase(store);

  if (!processed) {
    return;
  }

  console.log(`[worker] processed case ${processed.id} -> ${processed.workflowStatus}`);
}
