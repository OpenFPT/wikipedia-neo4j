import { describe, expect, test } from "bun:test";
import type { MockChatEvent } from "../types";
import { streamMockChat } from "./mock-chat";

const immediate = async () => undefined;

async function collect(request: Parameters<typeof streamMockChat>[0]) {
  const events: MockChatEvent[] = [];
  for await (const event of streamMockChat(request, { pause: immediate })) {
    events.push(event);
  }
  return events;
}

function sourcesFrom(events: MockChatEvent[]) {
  const event = events.find((candidate) => candidate.type === "sources");
  return event?.type === "sources" ? event.sources : [];
}

function answerFrom(events: MockChatEvent[]) {
  return events
    .filter((event) => event.type === "token")
    .map((event) => event.value)
    .join("");
}

describe("mock chat stream", () => {
  test("simulates retrieval across every knowledge base", async () => {
    const events = await collect({
      locale: "en",
      message: "How does graph retrieval work?",
      model: "recommended",
      scope: "all",
    });

    expect(events.filter((event) => event.type === "stage")).toHaveLength(10);
    expect(sourcesFrom(events)).toHaveLength(6);
    expect(answerFrom(events)).toContain("graph retrieval");
    expect(events.at(-1)?.type).toBe("done");
  });

  test("keeps a named knowledge-base query within that base", async () => {
    const events = await collect({
      knowledgeName: "Thesis research",
      locale: "en",
      message: "Summarize my notes",
      model: "local-qwen",
      scope: "kb:thesis",
    });

    expect(sourcesFrom(events)).toHaveLength(2);
    expect(
      sourcesFrom(events).every((source) => source.knowledgeBaseId === "thesis")
    ).toBe(true);
  });

  test("supports a plain conversation with no retrieved evidence", async () => {
    const events = await collect({
      locale: "en",
      message: "Help me outline a plan",
      model: "recommended",
      scope: "none",
    });

    expect(sourcesFrom(events)).toHaveLength(0);
    expect(answerFrom(events)).toContain("without retrieving");
  });

  test("includes a newly ready knowledge base in all-source retrieval", async () => {
    const events = await collect({
      locale: "en",
      message: "Search everything",
      model: "recommended",
      scope: "all",
      knowledgeBases: [
        {
          id: "handbook",
          name: "Team handbook",
          kind: "hosted",
          status: "ready",
          documentCount: 8,
          chunks: 120,
        },
      ],
    });

    expect(sourcesFrom(events)).toHaveLength(7);
    expect(
      sourcesFrom(events).some(
        (source) => source.knowledgeBaseId === "handbook"
      )
    ).toBe(true);
  });

  test("streams Vietnamese interface copy and answer content", async () => {
    const events = await collect({
      knowledgeName: "Wikipedia tiếng Việt",
      locale: "vi",
      message: "Neo4j hoạt động thế nào?",
      model: "recommended",
      scope: "kb:viwiki",
    });

    expect(answerFrom(events)).toContain("Hệ thống mô phỏng");
    expect(sourcesFrom(events)[0]?.title).toBe("Đồ thị tri thức");
  });

  test("rejects an empty prompt", async () => {
    await expect(
      collect({
        locale: "en",
        message: "  ",
        model: "recommended",
        scope: "none",
      })
    ).rejects.toThrow("message is required");
  });
});
