import { useEffect, useState } from "react";
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  MarkerType,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import LayerNode from "./LayerNode";
import { fetchNetwork, SAMPLE_NETWORK, type NetworkSpec } from "./network";

const nodeTypes = { layer: LayerNode };

// NetworkSpec -> React Flow's { nodes, edges }. Pure. The graph shape and
// positions come straight from the spec; the "name" (Input / Hidden / Output)
// is derived from where each layer sits in the wiring.
function toGraph(spec: NetworkSpec): { nodes: Node[]; edges: Edge[] } {
  const hasIncoming = new Set(spec.links.map((l) => l.target));
  const hasOutgoing = new Set(spec.links.map((l) => l.source));
  const nameOf = (id: string) =>
    !hasIncoming.has(id) ? "Input" : !hasOutgoing.has(id) ? "Output" : "Hidden";

  const nodes: Node[] = spec.layers.map((layer) => ({
    id: layer.id,
    type: "layer",
    position: { x: layer.pos[0], y: layer.pos[1] },
    data: {
      name: nameOf(layer.id),
      layerId: layer.id,
      size: layer.size,
      activation: layer.activation,
    },
  }));

  const edges: Edge[] = spec.links.map((link) => ({
    id: link.id,
    source: link.source,
    target: link.target,
    markerEnd: { type: MarkerType.ArrowClosed, color: "#8f99a8", width: 18, height: 18 },
  }));

  return { nodes, edges };
}

export default function Flow() {
  const initial = toGraph(SAMPLE_NETWORK);
  const [nodes, setNodes, onNodesChange] = useNodesState(initial.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initial.edges);
  const [live, setLive] = useState(false);

  useEffect(() => {
    fetchNetwork()
      .then((spec) => {
        const { nodes, edges } = toGraph(spec);
        setNodes(nodes);
        setEdges(edges);
        setLive(true);
      })
      .catch(() => setLive(false)); // backend down -> keep the sample
  }, [setNodes, setEdges]);

  return (
    <div className="flow">
      <ReactFlow
        colorMode="light"
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#c1c8d1" />
      </ReactFlow>
      <span className={`flow__tag${live ? " flow__tag--live" : ""}`}>
        {live ? "live from backend" : "sample · backend offline"}
      </span>
    </div>
  );
}
