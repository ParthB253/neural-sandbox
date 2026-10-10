import { useContext } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { EditorContext } from "./editorContext";
import type { ProjectAnalysis } from "./network";

export default function LayerNode({ id, data, selected }: NodeProps) {
  const { act, disabled } = useContext(EditorContext);
  const isSource = data.kind === "dataset";
  const schema = data.sourceSchema as ProjectAnalysis["datasets"][string] | undefined;
  const issues = data.issues as string[] | undefined;
  return (
    <div className={`layer-node${selected ? " layer-node--selected" : ""}`}>
      {!isSource && (
        <Handle
          type="target"
          position={Position.Left}
          className="layer-node__handle"
        />
      )}
      <header className="layer-node__header">
        <span className="layer-node__name">
          {isSource ? "Data source" : String(data.name)}
        </span>
        <span className="node-kind">
          {isSource ? String(data.format).toUpperCase() : "LAYER"}
        </span>
      </header>
      <div className="layer-node__body">
        <strong className="node-title" title={id}>
          {id}
        </strong>
        {isSource ? (
          <>
            <span className="node-file" title={String(data.filename)}>
              {String(data.filename)}
            </span>
            <div className="layer-node__row">
              <span className="layer-node__key">transforms</span>
              <span>{String(data.transforms)}</span>
            </div>
            {schema && <>
              <small>{schema.sample_count.toLocaleString()} examples</small>
              {Object.entries(schema.fields).slice(0, 4).map(([name, field]) =>
                <small key={name}>{name}: {field.sample_shape.join(" × ") || "scalar"} · {field.dtype}</small>)}
            </>}
          </>
        ) : (
          <>
            <div className="layer-node__row">
              <span className="layer-node__key">size</span>
              <span>{String(data.size)}{data.sizeMode === "auto" ? " (auto)" : ""}</span>
            </div>
            <div className="layer-node__row">
              <span className="layer-node__key">activation</span>
              <code>{String(data.activation)}</code>
            </div>
          </>
        )}
        {issues?.length ? <small className="execution-error" title={issues.join("; ")}>{issues[0]}</small> : null}
      </div>
      <Handle
        type="source"
        position={Position.Right}
        className="layer-node__handle"
      />
      <div className="node-actions nodrag nopan">
        <button
          className="node-actions__trigger"
          aria-label={`Actions for ${id}`}
          title="Node actions"
          disabled={disabled}
        >
          +
        </button>
        <div className="node-actions__menu" aria-label={`Actions for ${id}`}>
          <button disabled={disabled} onClick={() => act("add", id)}>
            + Add layer
          </button>
          <button disabled={disabled} onClick={() => act("connect", id)}>
            Connect to layer
          </button>
          <button disabled={disabled} onClick={() => act("edit", id)}>
            Edit
          </button>
          {isSource && (
            <button disabled={disabled} onClick={() => act("transforms", id)}>
              Set transforms
            </button>
          )}
          <button
            className="danger"
            disabled={disabled}
            onClick={() => act("delete", id)}
          >
            Delete node
          </button>
        </div>
      </div>
    </div>
  );
}
