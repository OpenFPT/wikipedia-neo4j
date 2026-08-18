import type {
  ChatThinkingState,
  RetrievalStage,
  StageStatus,
} from "./types";

export function buildInitialThinkingState(
  stages: RetrievalStage[]
): ChatThinkingState {
  return {
    stages,
    stageStatuses: Object.fromEntries(
      stages.map((stage) => [stage.id, "pending"])
    ) as Record<string, StageStatus>,
  };
}

export function updateThinkingStageStatus(
  thinking: ChatThinkingState,
  stageId: string,
  status: StageStatus
): ChatThinkingState {
  return {
    stages: thinking.stages,
    stageStatuses: {
      ...thinking.stageStatuses,
      [stageId]: status,
    },
  };
}
