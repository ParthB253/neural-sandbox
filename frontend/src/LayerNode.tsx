import { Handle, Position, type NodeProps } from "@xyflow/react";

// One card per layer. Data flows left-to-right, so the incoming handle sits on
// the left edge and the outgoing one on the right.
export interface LayerNodeData {
  name: string; // role in the graph: Input / Hidden / Output
  layerId: string; // the spec id, shown distinctly
  size: number;
  activation: string;
  [key: string]: unknown;
}

export default function LayerNode({ data, selected }: NodeProps) {
  const { name, layerId, size, activation } = data as LayerNodeData;
  return (
    <div className={`layer-node${selected ? " layer-node--selected" : ""}`}>
      <Handle type="target" position={Position.Left} className="layer-node__handle" />

      <header className="layer-node__header">
        <span className="layer-node__name">{name}</span>
        <code className="layer-node__id">{layerId}</code>
      </header>

      <div className="layer-node__body">
        <div className="layer-node__row">
          <span className="layer-node__key">size</span>
          <span className="layer-node__val">{size}</span>
        </div>
        <div className="layer-node__row">
          <span className="layer-node__key">activation</span>
          <code className="layer-node__val layer-node__val--mono">{activation}</code>
        </div>
      </div>

      <Handle type="source" position={Position.Right} className="layer-node__handle" />
    </div>
  );
}
