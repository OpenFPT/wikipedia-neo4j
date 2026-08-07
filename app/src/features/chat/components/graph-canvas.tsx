import { Minus, Plus, RotateCcw } from "lucide-react";
import {
  type ReactNode,
  type PointerEvent as ReactPointerEvent,
  useMemo,
  useRef,
  useState,
} from "react";
import type { EvidenceSource } from "@/features/chat/types";
import { type MessageKey, useI18n } from "@/features/i18n/i18n";
import { cn } from "@/lib/utils";

interface GraphNode {
  id: string;
  labelKey: MessageKey;
  x: number;
  y: number;
}

interface GraphEdge {
  from: string;
  to: string;
}

interface RenderNode {
  id: string;
  label: string;
  x: number;
  y: number;
}

const nodes: GraphNode[] = [
  { id: "neo4j", labelKey: "neo4j", x: 70, y: 78 },
  { id: "graphrag", labelKey: "graphRag", x: 193, y: 42 },
  { id: "vector", labelKey: "vectorSearch", x: 330, y: 82 },
  { id: "knowledge", labelKey: "knowledgeGraph", x: 193, y: 118 },
  { id: "entities", labelKey: "entities", x: 72, y: 157 },
  { id: "relations", labelKey: "relations", x: 330, y: 157 },
  { id: "source", labelKey: "sourcePassage", x: 70, y: 238 },
  { id: "context", labelKey: "contextNode", x: 193, y: 211 },
  { id: "evidence", labelKey: "evidenceNode", x: 330, y: 238 },
  { id: "answer", labelKey: "answer", x: 193, y: 300 },
];

const edges: GraphEdge[] = [
  { from: "neo4j", to: "graphrag" },
  { from: "graphrag", to: "knowledge" },
  { from: "vector", to: "graphrag" },
  { from: "neo4j", to: "entities" },
  { from: "vector", to: "relations" },
  { from: "entities", to: "knowledge" },
  { from: "relations", to: "knowledge" },
  { from: "knowledge", to: "context" },
  { from: "source", to: "context" },
  { from: "evidence", to: "context" },
  { from: "context", to: "answer" },
];

export function clampZoom(value: number) {
  return Math.min(1.5, Math.max(0.7, Number(value.toFixed(2))));
}

export function graphPathLabel(
  path: string[] | undefined,
  index: number,
  fallback: string
) {
  return path?.[index] || fallback;
}

export function graphRoute(
  source?: Pick<EvidenceSource, "external" | "knowledgeBaseId">
) {
  if (source?.external) {
    return ["vector", "graphrag", "knowledge", "context", "answer"];
  }
  if (source?.knowledgeBaseId === "product-notes") {
    return ["neo4j", "entities", "knowledge", "context", "answer"];
  }
  return ["neo4j", "graphrag", "knowledge", "context", "answer"];
}

export function isGraphEdgeActive(route: string[], from: string, to: string) {
  return route.some((node, index) => node === from && route[index + 1] === to);
}

export function GraphCanvas({
  className,
  source,
}: {
  className?: string;
  source?: EvidenceSource;
}) {
  const { t } = useI18n();
  const activeRoute = useMemo(() => graphRoute(source), [source]);
  const [selectedId, setSelectedId] = useState(() => activeRoute[0] ?? "neo4j");
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const drag = useRef<{
    originX: number;
    originY: number;
    pointerId: number;
    x: number;
    y: number;
  } | null>(null);
  const graphNodes = useMemo<RenderNode[]>(
    () =>
      nodes.map((node) => {
        const pathIndex = activeRoute.indexOf(node.id);
        return {
          ...node,
          label:
            pathIndex >= 0
              ? graphPathLabel(source?.path, pathIndex, t(node.labelKey))
              : t(node.labelKey),
        };
      }),
    [activeRoute, source, t]
  );
  const nodeById = useMemo(
    () => new Map(graphNodes.map((node) => [node.id, node])),
    [graphNodes]
  );
  const selected = nodeById.get(selectedId) ?? graphNodes[0];
  const connectedCount = edges.filter(
    (edge) => edge.from === selected.id || edge.to === selected.id
  ).length;

  function zoomBy(amount: number) {
    setZoom((current) => clampZoom(current + amount));
  }

  function reset() {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }

  function handlePointerDown(event: ReactPointerEvent<SVGSVGElement>) {
    if ((event.target as Element).closest("[data-graph-node]")) {
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      originX: pan.x,
      originY: pan.y,
    };
  }

  function handlePointerMove(event: ReactPointerEvent<SVGSVGElement>) {
    if (!drag.current || drag.current.pointerId !== event.pointerId) {
      return;
    }
    setPan({
      x: drag.current.originX + (event.clientX - drag.current.x) / zoom,
      y: drag.current.originY + (event.clientY - drag.current.y) / zoom,
    });
  }

  function stopDragging(event: ReactPointerEvent<SVGSVGElement>) {
    if (drag.current?.pointerId === event.pointerId) {
      drag.current = null;
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  return (
    <div className={cn("min-w-0", className)}>
      <div className="relative overflow-hidden rounded-xl border bg-muted/20">
        <div className="absolute top-2 right-2 z-10 flex items-center gap-1 rounded-lg border bg-background p-1 shadow-sm">
          <GraphControl label={t("zoomOut")} onClick={() => zoomBy(-0.1)}>
            <Minus className="size-3.5" />
          </GraphControl>
          <GraphControl label={t("zoomIn")} onClick={() => zoomBy(0.1)}>
            <Plus className="size-3.5" />
          </GraphControl>
          <GraphControl label={t("resetView")} onClick={reset}>
            <RotateCcw className="size-3.5" />
          </GraphControl>
        </div>
        <svg
          aria-label={t("graphCanvas")}
          className="h-[330px] w-full cursor-grab touch-none active:cursor-grabbing"
          onPointerCancel={stopDragging}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={stopDragging}
          viewBox="0 0 400 340"
        >
          <g transform={`translate(${pan.x} ${pan.y}) scale(${zoom})`}>
            {edges.map((edge) => {
              const from = nodeById.get(edge.from);
              const to = nodeById.get(edge.to);
              if (!(from && to)) {
                return null;
              }
              const highlighted = isGraphEdgeActive(
                activeRoute,
                edge.from,
                edge.to
              );
              return (
                <line
                  className={cn(
                    "stroke-border-strong",
                    highlighted && "stroke-primary"
                  )}
                  key={`${edge.from}-${edge.to}`}
                  strokeWidth={highlighted ? 1.8 : 1}
                  x1={from.x}
                  x2={to.x}
                  y1={from.y}
                  y2={to.y}
                />
              );
            })}
            {graphNodes.map((node) => {
              const selectedNode = selectedId === node.id;
              const highlighted = activeRoute.includes(node.id);
              return (
                <foreignObject
                  height="26"
                  key={node.id}
                  width="92"
                  x={node.x - 46}
                  y={node.y - 13}
                >
                  <button
                    aria-pressed={selectedNode}
                    className={cn(
                      "h-full w-full cursor-pointer rounded-full border border-border-strong bg-background px-1 text-[10px] text-foreground leading-none outline-none transition-colors hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring/50",
                      highlighted && "bg-primary/10",
                      selectedNode && "bg-primary/15 font-medium"
                    )}
                    data-graph-node
                    onClick={() => setSelectedId(node.id)}
                    type="button"
                  >
                    <span className="block truncate">{node.label}</span>
                  </button>
                </foreignObject>
              );
            })}
          </g>
        </svg>
        <p className="sr-only">{t("graphInstructions")}</p>
        <ul className="sr-only">
          {graphNodes.map((node) => (
            <li key={node.id}>
              <button onClick={() => setSelectedId(node.id)} type="button">
                {node.label}
              </button>
            </li>
          ))}
        </ul>
      </div>
      <div className="mt-3 border-t pt-3">
        <span className="text-muted-foreground text-xs">
          {t("selectedNode")}
        </span>
        <div className="mt-1.5 flex flex-wrap items-center gap-2">
          <span className="size-1.5 rounded-full bg-primary" />
          <strong className="min-w-0 break-words font-medium text-xs">
            {selected.label}
          </strong>
          <span className="ml-auto text-muted-foreground text-xs">
            {t("connectedNodes", { count: connectedCount })}
          </span>
        </div>
        <p className="mt-2 break-words text-muted-foreground text-xs leading-relaxed">
          {t("answerPath")}:{" "}
          {source?.path.join(" → ") ??
            `${t("graphRag")} → Neo4j → ${t("knowledgeGraph")} → ${t("answer")}`}
        </p>
      </div>
    </div>
  );
}

function GraphControl({
  label,
  onClick,
  children,
}: {
  children: ReactNode;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      aria-label={label}
      className="grid size-7 place-items-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground"
      onClick={onClick}
      title={label}
      type="button"
    >
      {children}
    </button>
  );
}
