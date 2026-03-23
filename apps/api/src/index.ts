import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import cors from "@fastify/cors";
import Fastify from "fastify";
import { LocalCaseStore, runDemoPipeline } from "@radiant/core";
import {
  createCaseInputSchema,
  createDemoIntake,
  createQueuedCase,
  createReviewDecision,
  reviewActionInputSchema,
  type CaseRecord
} from "@radiant/shared";

const app = Fastify({ logger: true });
const port = Number(process.env.PORT ?? 3001);
const dataFile = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../../data/state/cases.json"
);
const store = new LocalCaseStore(dataFile);

await app.register(cors, {
  origin: true
});

await seedDemoCase();

app.get("/health", async () => ({
  ok: true,
  service: "radiant-api"
}));

app.get("/cases", async () => {
  const cases = await store.listCases();
  return { cases };
});

app.post("/cases", async (request, reply) => {
  const input = createCaseInputSchema.parse(request.body);
  const caseRecord = createQueuedCase(input);

  await store.saveCase(caseRecord);
  return reply.code(201).send({ case: caseRecord });
});

app.get("/cases/:caseId", async (request, reply) => {
  const { caseId } = request.params as { caseId: string };
  const caseRecord = await store.getCase(caseId);

  if (!caseRecord) {
    return reply.code(404).send({ error: "Case not found." });
  }

  return { case: caseRecord };
});

app.post("/cases/:caseId/process", async (request, reply) => {
  const { caseId } = request.params as { caseId: string };
  const caseRecord = await runDemoPipeline(store, caseId);

  if (!caseRecord) {
    return reply.code(404).send({ error: "Case not found." });
  }

  return { case: caseRecord };
});

app.get("/cases/:caseId/assets", async (request, reply) => {
  const { caseId } = request.params as { caseId: string };
  const caseRecord = await store.getCase(caseId);

  if (!caseRecord) {
    return reply.code(404).send({ error: "Case not found." });
  }

  return { assets: caseRecord.assets };
});

app.post("/cases/:caseId/review", async (request, reply) => {
  const { caseId } = request.params as { caseId: string };
  const input = reviewActionInputSchema.parse(request.body);
  const current = await store.getCase(caseId);

  if (!current) {
    return reply.code(404).send({ error: "Case not found." });
  }

  if (input.action !== "rerun" && current.workflowStatus !== "reviewable") {
    return reply.code(409).send({
      error: "Only reviewable cases can be approved or rejected."
    });
  }

  const nextCase = buildReviewUpdate(current, input.action, input.reviewerId, input.notes);
  await store.saveCase(nextCase);

  return { case: nextCase };
});

app.listen({ port, host: "0.0.0.0" }).catch((error) => {
  app.log.error(error);
  process.exit(1);
});

function buildReviewUpdate(
  current: CaseRecord,
  action: "approve" | "reject" | "rerun",
  reviewerId: string,
  notes?: string
): CaseRecord {
  if (action === "rerun") {
    return {
      ...current,
      workflowStatus: "queued",
      structuredTarget: null,
      supportDecision: null,
      segmentationResult: null,
      qaResult: null,
      patientSummary: null,
      assets: [],
      reviewDecision: createReviewDecision(action, reviewerId, notes),
      updatedAt: new Date().toISOString()
    };
  }

  return {
    ...current,
    workflowStatus: action === "approve" ? "approved" : "rejected",
    reviewDecision: createReviewDecision(action, reviewerId, notes),
    updatedAt: new Date().toISOString()
  };
}

async function seedDemoCase(): Promise<void> {
  const cases = await store.listCases();

  if (cases.length > 0) {
    return;
  }

  await store.saveCase(createQueuedCase(createDemoIntake()));
}
