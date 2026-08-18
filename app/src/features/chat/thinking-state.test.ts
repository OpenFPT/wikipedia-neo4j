import { describe, expect, test } from "bun:test";
import type { RetrievalStage } from "./types";
import {
  buildInitialThinkingState,
  updateThinkingStageStatus,
} from "./thinking-state";

const stages: RetrievalStage[] = [
  { id: "understand", label: "Understand the question" },
  { id: "retrieve", label: "Retrieve evidence" },
];

describe("thinking state", () => {
  test("starts every live stage as pending", () => {
    const thinking = buildInitialThinkingState(stages);

    expect(thinking.stages).toEqual(stages);
    expect(thinking.stageStatuses).toEqual({
      retrieve: "pending",
      understand: "pending",
    });
  });

  test("updates one stage without losing the others", () => {
    const thinking = buildInitialThinkingState(stages);
    const updated = updateThinkingStageStatus(thinking, "retrieve", "running");

    expect(updated.stageStatuses).toEqual({
      retrieve: "running",
      understand: "pending",
    });
  });
});
