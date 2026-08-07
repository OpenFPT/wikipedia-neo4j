import { describe, expect, test } from "bun:test";
import {
  clampZoom,
  graphPathLabel,
  graphRoute,
  isGraphEdgeActive,
} from "./graph-canvas";

describe("graph canvas zoom", () => {
  test("clamps and rounds the simulated canvas zoom", () => {
    expect(clampZoom(0.2)).toBe(0.7);
    expect(clampZoom(1.234)).toBe(1.23);
    expect(clampZoom(2)).toBe(1.5);
  });

  test("binds highlighted graph nodes and edges to the selected citation", () => {
    const path = ["Thesis", "mentions", "GraphRAG"];
    const localRoute = graphRoute({
      external: false,
      knowledgeBaseId: "thesis",
    });
    const hostedRoute = graphRoute({
      external: true,
      knowledgeBaseId: "viwiki",
    });

    expect(graphPathLabel(path, 1, "fallback")).toBe("mentions");
    expect(graphPathLabel(path, 4, "Answer")).toBe("Answer");
    expect(isGraphEdgeActive(localRoute, "neo4j", "graphrag")).toBe(true);
    expect(isGraphEdgeActive(hostedRoute, "neo4j", "graphrag")).toBe(false);
    expect(isGraphEdgeActive(hostedRoute, "vector", "graphrag")).toBe(true);
  });
});
