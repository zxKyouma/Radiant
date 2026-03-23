import { useEffect, useState } from "react";

import type {
  CaseRecord,
  CreateCaseInput,
  ReviewAction
} from "@radiant/shared";

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:3001";

const emptyCaseInput: CreateCaseInput = {
  studyId: "",
  patientLabel: "",
  reportText: ""
};

export function App() {
  const [cases, setCases] = useState<CaseRecord[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [caseInput, setCaseInput] = useState<CreateCaseInput>(emptyCaseInput);

  const selectedCase =
    cases.find((entry) => entry.id === selectedCaseId) ?? cases[0] ?? null;

  useEffect(() => {
    void refreshCases();
  }, []);

  async function refreshCases() {
    setIsLoading(true);
    const response = await fetch(`${apiBaseUrl}/cases`);
    const payload = (await response.json()) as { cases: CaseRecord[] };

    setCases(payload.cases);
    setSelectedCaseId((current) => current ?? payload.cases[0]?.id ?? null);
    setIsLoading(false);
  }

  async function createCase() {
    setIsSubmitting(true);

    const response = await fetch(`${apiBaseUrl}/cases`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(caseInput)
    });

    if (response.ok) {
      setCaseInput(emptyCaseInput);
      await refreshCases();
    }

    setIsSubmitting(false);
  }

  async function processCase(caseId: string) {
    setIsSubmitting(true);
    await fetch(`${apiBaseUrl}/cases/${caseId}/process`, {
      method: "POST"
    });
    await refreshCases();
    setIsSubmitting(false);
  }

  async function reviewCase(caseId: string, action: ReviewAction) {
    setIsSubmitting(true);

    await fetch(`${apiBaseUrl}/cases/${caseId}/review`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        action,
        reviewerId: "demo-reviewer"
      })
    });

    await refreshCases();
    setIsSubmitting(false);
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="panel">
          <p className="eyebrow">Radiant</p>
          <h1>Review Queue</h1>
          <p className="muted">
            Patient education workflow for supported brain MRI cases.
          </p>
        </div>

        <div className="panel">
          <h2>Create Case</h2>
          <label>
            Study ID
            <input
              value={caseInput.studyId}
              onChange={(event) =>
                setCaseInput((current) => ({
                  ...current,
                  studyId: event.target.value
                }))
              }
              placeholder="study_001"
            />
          </label>
          <label>
            Patient Label
            <input
              value={caseInput.patientLabel}
              onChange={(event) =>
                setCaseInput((current) => ({
                  ...current,
                  patientLabel: event.target.value
                }))
              }
              placeholder="Patient Alpha"
            />
          </label>
          <label>
            Finalized Report
            <textarea
              rows={8}
              value={caseInput.reportText}
              onChange={(event) =>
                setCaseInput((current) => ({
                  ...current,
                  reportText: event.target.value
                }))
              }
              placeholder="Paste finalized report text"
            />
          </label>
          <button
            className="primary"
            disabled={isSubmitting}
            onClick={() => void createCase()}
          >
            Create queued case
          </button>
        </div>

        <div className="panel">
          <div className="row">
            <h2>Cases</h2>
            <button className="ghost" onClick={() => void refreshCases()}>
              Refresh
            </button>
          </div>
          {isLoading ? <p className="muted">Loading…</p> : null}
          <div className="case-list">
            {cases.map((entry) => (
              <button
                key={entry.id}
                className={`case-card ${selectedCase?.id === entry.id ? "active" : ""}`}
                onClick={() => setSelectedCaseId(entry.id)}
              >
                <span>{entry.patientLabel}</span>
                <span className="status">{entry.workflowStatus}</span>
              </button>
            ))}
          </div>
        </div>
      </aside>

      <main className="content">
        {selectedCase ? (
          <>
            <section className="panel hero">
              <div>
                <p className="eyebrow">Selected Case</p>
                <h2>{selectedCase.patientLabel}</h2>
                <p className="muted">{selectedCase.studyId}</p>
              </div>
              <div className="actions">
                <button
                  className="primary"
                  disabled={isSubmitting}
                  onClick={() => void processCase(selectedCase.id)}
                >
                  Process demo pipeline
                </button>
                <button
                  className="ghost"
                  disabled={isSubmitting}
                  onClick={() => void reviewCase(selectedCase.id, "approve")}
                >
                  Approve
                </button>
                <button
                  className="ghost"
                  disabled={isSubmitting}
                  onClick={() => void reviewCase(selectedCase.id, "reject")}
                >
                  Reject
                </button>
                <button
                  className="ghost"
                  disabled={isSubmitting}
                  onClick={() => void reviewCase(selectedCase.id, "rerun")}
                >
                  Rerun
                </button>
              </div>
            </section>

            <section className="grid">
              <article className="panel">
                <h3>Status</h3>
                <dl className="facts">
                  <div>
                    <dt>Workflow</dt>
                    <dd>{selectedCase.workflowStatus}</dd>
                  </div>
                  <div>
                    <dt>Support</dt>
                    <dd>{selectedCase.supportDecision?.status ?? "pending"}</dd>
                  </div>
                  <div>
                    <dt>QA</dt>
                    <dd>{selectedCase.qaResult?.status ?? "pending"}</dd>
                  </div>
                </dl>
              </article>

              <article className="panel">
                <h3>Patient Summary</h3>
                <p>{selectedCase.patientSummary ?? "No generated summary yet."}</p>
              </article>

              <article className="panel span-2">
                <h3>Report Text</h3>
                <pre>{selectedCase.reportText}</pre>
              </article>

              <article className="panel">
                <h3>Structured Target</h3>
                <pre>{JSON.stringify(selectedCase.structuredTarget, null, 2)}</pre>
              </article>

              <article className="panel">
                <h3>Assets</h3>
                <ul className="asset-list">
                  {selectedCase.assets.map((asset) => (
                    <li key={asset.id}>
                      <span>{asset.label}</span>
                      <code>{asset.kind}</code>
                    </li>
                  ))}
                  {selectedCase.assets.length === 0 ? <li>No assets yet.</li> : null}
                </ul>
              </article>

              <article className="panel">
                <h3>QA Result</h3>
                <pre>{JSON.stringify(selectedCase.qaResult, null, 2)}</pre>
              </article>

              <article className="panel">
                <h3>Review Decision</h3>
                <pre>{JSON.stringify(selectedCase.reviewDecision, null, 2)}</pre>
              </article>
            </section>
          </>
        ) : (
          <section className="panel empty-state">
            <p className="eyebrow">No cases yet</p>
            <h2>Create or refresh to load a queued case.</h2>
          </section>
        )}
      </main>
    </div>
  );
}
