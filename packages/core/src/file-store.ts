import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

import {
  caseListSchema,
  type CaseRecord
} from "@radiant/shared";

export class LocalCaseStore {
  constructor(private readonly filePath: string) {}

  async listCases(): Promise<CaseRecord[]> {
    const cases = await this.readCases();

    return [...cases].sort((left, right) =>
      right.updatedAt.localeCompare(left.updatedAt)
    );
  }

  async getCase(caseId: string): Promise<CaseRecord | null> {
    const cases = await this.readCases();

    return cases.find((entry) => entry.id === caseId) ?? null;
  }

  async saveCase(caseRecord: CaseRecord): Promise<CaseRecord> {
    const cases = await this.readCases();
    const nextCases = cases.filter((entry) => entry.id !== caseRecord.id);

    nextCases.push(caseRecord);
    await this.writeCases(nextCases);

    return caseRecord;
  }

  async updateCase(
    caseId: string,
    update: (current: CaseRecord) => CaseRecord
  ): Promise<CaseRecord | null> {
    const current = await this.getCase(caseId);

    if (!current) {
      return null;
    }

    return this.saveCase(update(current));
  }

  private async readCases(): Promise<CaseRecord[]> {
    await mkdir(dirname(this.filePath), { recursive: true });

    try {
      const raw = await readFile(this.filePath, "utf8");
      return caseListSchema.parse(JSON.parse(raw));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        await this.writeCases([]);
        return [];
      }

      throw error;
    }
  }

  private async writeCases(cases: CaseRecord[]): Promise<void> {
    await mkdir(dirname(this.filePath), { recursive: true });
    await writeFile(this.filePath, JSON.stringify(cases, null, 2));
  }
}

