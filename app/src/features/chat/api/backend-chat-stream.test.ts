import { describe, expect, test } from "bun:test";
import { parseSseEventBlock } from "./backend-chat";

describe("backend chat stream parser", () => {
  test("parses one SSE event block", () => {
    const event = parseSseEventBlock(
      'event: cypher\ndata: {"query":"MATCH (n) RETURN n LIMIT 1"}\n\n'
    );

    expect(event).toEqual({
      data: { query: "MATCH (n) RETURN n LIMIT 1" },
      event: "cypher",
    });
  });
});
