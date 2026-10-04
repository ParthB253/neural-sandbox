import { useCallback, useEffect, useRef, useState } from "react";
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  MarkerType,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type ReactFlowInstance,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import LayerNode from "./LayerNode";
import EditorDialog, { type DialogState } from "./EditorDialog";
import NetworkHeader from "./NetworkHeader";
import { EditorContext, type NodeAction } from "./editorContext";
import {
  clearCanvas,
  connections,
  emptyProject,
  removeItems,
  request,
  type ProjectSpec,
  type StoredFile,
  type NetworkSummary,
} from "./network";

const nodeTypes = { layer: LayerNode };
function toGraph(project: ProjectSpec): { nodes: Node[]; edges: Edge[] } {
  const incoming = new Set(project.network.links.map((link) => link.target));
  const outgoing = new Set(project.network.links.map((link) => link.source));
  return {
    nodes: [
      ...project.datasets.map((dataset) => ({
        id: dataset.id,
        type: "layer",
        position: { x: dataset.pos[0], y: dataset.pos[1] },
        data: {
          kind: "dataset",
          filename: dataset.source.path.split(/[\\/]/).pop(),
          format: dataset.source.type,
          transforms: dataset.transforms.length,
        },
      })),
      ...project.network.layers.map((layer) => ({
        id: layer.id,
        type: "layer",
        position: { x: layer.pos[0], y: layer.pos[1] },
        data: {
          kind: "layer",
          name: !incoming.has(layer.id)
            ? "Input"
            : !outgoing.has(layer.id)
              ? "Output"
              : "Hidden",
          size: layer.size,
          activation: layer.activation,
        },
      })),
    ],
    edges: connections(project).map((link) => ({
      ...link,
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: "#8f99a8",
        width: 18,
        height: 18,
      },
    })),
  };
}
const errorMessage = (error: unknown) =>
  error instanceof Error ? error.message : "Something went wrong.";

export default function Flow() {
  const [project, setProject] = useState<ProjectSpec>(emptyProject);
  const [files, setFiles] = useState<StoredFile[]>([]);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [tab, setTab] = useState<"files" | "graph">("files");
  const [dialog, setDialog] = useState<DialogState | null>(null);
  const [busy, setBusy] = useState(true);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("Connecting…");
  const [savedSnapshot, setSavedSnapshot] = useState<string | null>(null);
  const flow = useRef<ReactFlowInstance<Node, Edge> | null>(null);
  const uploadInput = useRef<HTMLInputElement>(null);
  const locked = useRef(false);
  const disabled = busy || !ready;
  const dirty =
    savedSnapshot !== null
      ? JSON.stringify(project) !== savedSnapshot
      : project.network.title !== "Untitled network" ||
        project.network.layers.length > 0 ||
        project.datasets.length > 0;

  useEffect(() => {
    document.title = `${project.network.title || "Untitled network"} · NeuralSandbox`;
  }, [project.network.title]);
  useEffect(() => {
    if (!dirty) return;
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  const apply = useCallback(
    (next: ProjectSpec) => {
      setProject(next);
      const graph = toGraph(next);
      setNodes(graph.nodes);
      setEdges(graph.edges);
    },
    [setNodes, setEdges],
  );

  const load = useCallback(() => {
    if (locked.current) return;
    locked.current = true;
    return Promise.all([
      request<ProjectSpec>("/project"),
      request<StoredFile[]>("/files"),
      request<NetworkSummary[]>("/networks"),
    ])
      .then(async ([next, stored, networks]) => {
        const saved = networks.some((network) => network.id === next.network.id)
          ? await request<ProjectSpec>(`/networks/${next.network.id}`)
          : null;
        setSavedSnapshot(saved ? JSON.stringify(saved) : null);
        apply(next);
        setFiles(stored);
        setReady(true);
        setStatus("Connected");
      })
      .catch((err) => {
        setStatus("Backend unavailable");
        setError(
          `Could not load the project. Start the backend on port 8000 and retry. ${errorMessage(err)}`,
        );
      })
      .finally(() => {
        locked.current = false;
        setBusy(false);
      });
  }, [apply]);
  useEffect(() => {
    void load();
  }, [load]);

  async function save(next: ProjectSpec): Promise<boolean> {
    if (locked.current || !ready) return false;
    locked.current = true;
    setBusy(true);
    setError("");
    try {
      const saved = await request<ProjectSpec>("/project", "PUT", next);
      apply(saved);
      if (saved.network.layers.length + saved.datasets.length > nodes.length) {
        // Wait for the new node's dimensions before fitting it into the viewport.
        requestAnimationFrame(() =>
          requestAnimationFrame(() => {
            void flow.current?.fitView({
              padding: 0.3,
              maxZoom: 1,
              duration: 250,
            });
          }),
        );
      }
      setStatus("Draft updated");
      return true;
    } catch (err) {
      setError(errorMessage(err));
      apply(project);
      setStatus("Changes not saved");
      return false;
    } finally {
      locked.current = false;
      setBusy(false);
    }
  }
  async function persistNetwork(copyTitle?: string): Promise<boolean> {
    if (locked.current || !ready) return false;
    locked.current = true;
    setBusy(true);
    setError("");
    try {
      const next = {
        ...project,
        network: {
          ...project.network,
          title:
            (copyTitle ?? project.network.title).trim() || "Untitled network",
        },
      };
      const saved = await request<ProjectSpec>(
        copyTitle === undefined ? `/networks/${next.network.id}` : "/networks",
        copyTitle === undefined ? "PUT" : "POST",
        next,
      );
      apply(saved);
      setSavedSnapshot(JSON.stringify(saved));
      setStatus("Network saved");
      return true;
    } catch (err) {
      setError(errorMessage(err));
      return false;
    } finally {
      locked.current = false;
      setBusy(false);
    }
  }
  async function openNetwork(id: string): Promise<boolean> {
    if (locked.current || !ready) return false;
    locked.current = true;
    setBusy(true);
    setError("");
    try {
      const next = await request<ProjectSpec>(`/networks/${id}/open`, "POST");
      apply(next);
      setSavedSnapshot(JSON.stringify(next));
      setStatus("Network opened");
      requestAnimationFrame(() =>
        requestAnimationFrame(() => {
          void flow.current?.fitView({ padding: 0.3, maxZoom: 1 });
        }),
      );
      return true;
    } catch (err) {
      setError(errorMessage(err));
      return false;
    } finally {
      locked.current = false;
      setBusy(false);
    }
  }
  async function newNetwork(): Promise<boolean> {
    if (!(await save(emptyProject()))) return false;
    setSavedSnapshot(null);
    return true;
  }
  async function upload(selected: FileList): Promise<StoredFile[]> {
    if (locked.current) return [];
    locked.current = true;
    setBusy(true);
    setError("");
    const added: StoredFile[] = [];
    try {
      for (const file of Array.from(selected)) {
        const form = new FormData();
        form.append("file", file);
        const stored = await request<StoredFile>("/files", "POST", form);
        added.push(stored);
        setFiles((previous) => [...previous, stored]);
      }
      setStatus(`${added.length} file${added.length === 1 ? "" : "s"} stored`);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      locked.current = false;
      setBusy(false);
    }
    return added;
  }
  function open(next: DialogState) {
    setError("");
    setDialog(next);
  }
  function act(action: NodeAction, id: string) {
    if (disabled) return;
    if (action === "delete") {
      void save(removeItems(project, [id]));
      return;
    }
    if (action === "add") open({ kind: "layer", parent: id });
    if (action === "connect") open({ kind: "connection", source: id });
    if (action === "edit")
      open({
        kind: project.datasets.some((item) => item.id === id)
          ? "source"
          : "layer",
        id,
      });
    if (action === "transforms") open({ kind: "transforms", id });
  }
  function editConnection(id: string) {
    const edge = connections(project).find((item) => item.id === id);
    if (edge) open({ kind: "connection", ...edge });
  }
  function focusNode(id: string) {
    setNodes((items) =>
      items.map((node) => ({ ...node, selected: node.id === id })),
    );
    void flow.current?.fitView({
      nodes: [{ id }],
      duration: 250,
      maxZoom: 1.2,
      padding: 0.7,
    });
  }
  function savePositions() {
    const position = (
      id: string,
      fallback: [number, number],
    ): [number, number] => {
      const node = nodes.find((item) => item.id === id);
      return node ? [node.position.x, node.position.y] : fallback;
    };
    void save({
      ...project,
      network: {
        ...project.network,
        layers: project.network.layers.map((layer) => ({
          ...layer,
          pos: position(layer.id, layer.pos),
        })),
      },
      datasets: project.datasets.map((dataset) => ({
        ...dataset,
        pos: position(dataset.id, dataset.pos),
      })),
    });
  }
  function deleteFile(file: StoredFile) {
    open({
      kind: "confirm",
      title: `Delete ${file.name}?`,
      description:
        "This removes the stored file from the project. Upload it again if you need it later.",
      confirm: async () => {
        if (locked.current) return false;
        locked.current = true;
        setBusy(true);
        setError("");
        try {
          await request(`/files/${file.id}`, "DELETE");
          setFiles((previous) =>
            previous.filter((item) => item.id !== file.id),
          );
          setStatus("File deleted");
          return true;
        } catch (err) {
          setError(errorMessage(err));
          return false;
        } finally {
          locked.current = false;
          setBusy(false);
        }
      },
    });
  }
  const allConnections = connections(project);
  return (
    <EditorContext.Provider value={{ act, disabled }}>
      <div className="workspace">
        <NetworkHeader
          title={project.network.title}
          disabled={disabled}
          busy={busy}
          dirty={dirty}
          saved={savedSnapshot !== null}
          status={status}
          error={error}
          onRename={(title) =>
            setProject((previous) => ({
              ...previous,
              network: { ...previous.network, title },
            }))
          }
          onSave={persistNetwork}
          onNew={newNetwork}
          onOpen={openNetwork}
          clearError={() => setError("")}
          canClear={nodes.length > 0}
          onClear={() =>
            open({
              kind: "confirm",
              title: "Clear the canvas?",
              description:
                "All nodes and connections will be removed. Your stored project files will remain available.",
              confirm: () => save(clearCanvas(project)),
            })
          }
        />
        <aside className="sidebar">
          <div
            className="sidebar-tabs"
            role="tablist"
            aria-label="Project sidebar"
          >
            <button
              id="files-tab"
              role="tab"
              aria-selected={tab === "files"}
              aria-controls="files-panel"
              onClick={() => setTab("files")}
            >
              Project files <span>{files.length}</span>
            </button>
            <button
              id="graph-tab"
              role="tab"
              aria-selected={tab === "graph"}
              aria-controls="graph-panel"
              onClick={() => setTab("graph")}
            >
              Graph <span>{nodes.length}</span>
            </button>
          </div>
          {tab === "files" ? (
            <section
              className="sidebar-panel"
              id="files-panel"
              role="tabpanel"
              aria-labelledby="files-tab"
            >
              <div className="section-heading">
                <h2>Data sources</h2>
                <button
                  className="primary compact"
                  disabled={disabled}
                  onClick={() => uploadInput.current?.click()}
                >
                  + Upload files
                </button>
              </div>
              <input
                ref={uploadInput}
                type="file"
                multiple
                className="sr-only"
                aria-label="Upload project files"
                onChange={async (event) => {
                  if (event.target.files) await upload(event.target.files);
                  event.target.value = "";
                }}
              />
              <p className="sidebar-help">
                Store files here, then reuse them in your pipeline.
              </p>
              {!files.length && (
                <div className="sidebar-empty">
                  <span className="empty-icon">▤</span>
                  <strong>No project files yet</strong>
                  <p>Upload CSV, JSON, NumPy, or IDX data to get started.</p>
                </div>
              )}
              <ul className="item-list">
                {files.map((file) => {
                  const uses = project.datasets.filter(
                    (item) => item.source.path === file.path,
                  ).length;
                  return (
                    <li key={file.id} className="file-item">
                      <div className="file-summary">
                        <span className="file-icon">▤</span>
                        <div>
                          <strong title={file.name}>{file.name}</strong>
                          <small>
                            {formatBytes(file.size)}
                            {uses
                              ? ` · ${uses} source${uses === 1 ? "" : "s"}`
                              : ""}
                          </small>
                        </div>
                      </div>
                      <div className="file-actions">
                        <button
                          disabled={disabled}
                          onClick={() => open({ kind: "source", file })}
                        >
                          + Add to canvas
                        </button>
                        <button
                          className="danger"
                          aria-label={`Delete file ${file.name}`}
                          title={
                            uses
                              ? "Remove data source nodes using this file first"
                              : "Delete stored file"
                          }
                          disabled={disabled || uses > 0}
                          onClick={() => deleteFile(file)}
                        >
                          Delete
                        </button>
                      </div>
                    </li>
                  );
                })}
              </ul>
              <p className="storage-note">
                Files are shared across networks. Use File → Save to keep this
                network for later.
              </p>
            </section>
          ) : (
            <section
              className="sidebar-panel"
              id="graph-panel"
              role="tabpanel"
              aria-labelledby="graph-tab"
            >
              <div className="section-heading">
                <h2>Pipeline</h2>
                <span className="muted">{nodes.length} nodes</span>
              </div>
              <p className="sidebar-help">
                Select a node to locate it. All wiring appears as connections.
              </p>
              <h3>
                Data sources <span>{project.datasets.length}</span>
              </h3>
              {!project.datasets.length && (
                <p className="muted list-empty">No data sources</p>
              )}
              <ul className="item-list">
                {project.datasets.map((item) => (
                  <li className="graph-item" key={item.id}>
                    <button
                      className="item-name"
                      onClick={() => focusNode(item.id)}
                    >
                      {item.id}
                      <small>
                        {item.source.type.toUpperCase()} ·{" "}
                        {item.transforms.length} transforms
                      </small>
                    </button>
                    <button
                      aria-label={`Edit ${item.id}`}
                      disabled={disabled}
                      onClick={() => act("edit", item.id)}
                    >
                      Edit
                    </button>
                    <button
                      className="danger"
                      aria-label={`Delete ${item.id}`}
                      disabled={disabled}
                      onClick={() => act("delete", item.id)}
                    >
                      ×
                    </button>
                  </li>
                ))}
              </ul>
              <h3>
                Layers <span>{project.network.layers.length}</span>
              </h3>
              {!project.network.layers.length && (
                <p className="muted list-empty">No layers</p>
              )}
              <ul className="item-list">
                {project.network.layers.map((item) => (
                  <li className="graph-item" key={item.id}>
                    <button
                      className="item-name"
                      onClick={() => focusNode(item.id)}
                    >
                      {item.id}
                      <small>
                        {item.size} units · {item.activation}
                      </small>
                    </button>
                    <button
                      aria-label={`Edit ${item.id}`}
                      disabled={disabled}
                      onClick={() => act("edit", item.id)}
                    >
                      Edit
                    </button>
                    <button
                      className="danger"
                      aria-label={`Delete ${item.id}`}
                      disabled={disabled}
                      onClick={() => act("delete", item.id)}
                    >
                      ×
                    </button>
                  </li>
                ))}
              </ul>
              <h3>
                Connections <span>{allConnections.length}</span>
              </h3>
              {!allConnections.length && (
                <p className="muted list-empty">No connections</p>
              )}
              <ul className="item-list">
                {allConnections.map((item) => (
                  <li className="graph-item" key={item.id}>
                    <button
                      className="item-name"
                      disabled={disabled}
                      onClick={() => editConnection(item.id)}
                    >
                      {item.source} → {item.target}
                      <small>Edit connection</small>
                    </button>
                    <button
                      className="danger"
                      aria-label={`Delete connection ${item.source} to ${item.target}`}
                      disabled={disabled}
                      onClick={() =>
                        void save(removeItems(project, [], [item.id]))
                      }
                    >
                      ×
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </aside>
        <main className="flow" aria-label="Pipeline canvas">
          <div className="canvas-toolbar">
            <button
              disabled={disabled}
              onClick={() => open({ kind: "source" })}
            >
              + Data source
            </button>
            <span>Hover on a node’s right edge to build your pipeline</span>
          </div>
          {error && !dialog && (
            <div className="error-banner" role="alert">
              <span>{error}</span>
              {!ready ? (
                <button
                  disabled={busy}
                  onClick={() => {
                    setBusy(true);
                    setError("");
                    void load();
                  }}
                >
                  Retry
                </button>
              ) : (
                <button aria-label="Dismiss error" onClick={() => setError("")}>
                  ×
                </button>
              )}
            </div>
          )}
          <ReactFlow
            colorMode="light"
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            onInit={(instance) => {
              flow.current = instance;
            }}
            fitView
            fitViewOptions={{ padding: 0.3 }}
            minZoom={0.25}
            maxZoom={1.6}
            nodesDraggable={!disabled}
            nodesConnectable={!disabled}
            elementsSelectable={!disabled}
            deleteKeyCode={disabled || dialog ? null : ["Backspace", "Delete"]}
            onNodeDragStop={savePositions}
            onSelectionDragStop={savePositions}
            onNodeDoubleClick={(_, node) => act("edit", node.id)}
            onEdgeClick={(_, edge) => {
              if (!disabled) editConnection(edge.id);
            }}
            onConnect={(connection) => {
              if (!disabled)
                open({
                  kind: "connection",
                  source: connection.source,
                  target: connection.target,
                });
            }}
            onBeforeDelete={async ({
              nodes: removedNodes,
              edges: removedEdges,
            }) => {
              if (!disabled && !document.querySelector("dialog[open]"))
                await save(
                  removeItems(
                    project,
                    removedNodes.map((node) => node.id),
                    removedEdges.map((edge) => edge.id),
                  ),
                );
              return false;
            }}
            onPaneClick={(event) => {
              if (ready && !busy && !nodes.length) {
                const point = flow.current?.screenToFlowPosition({
                  x: event.clientX,
                  y: event.clientY,
                });
                open({
                  kind: "source",
                  position: point ? [point.x, point.y] : undefined,
                });
              }
            }}
            proOptions={{ hideAttribution: true }}
          >
            <Background
              variant={BackgroundVariant.Dots}
              gap={16}
              size={1}
              color="#c1c8d1"
            />
            <Controls showInteractive={false} />
          </ReactFlow>
          {ready && !nodes.length && (
            <div className="canvas-empty">
              <div className="empty-icon">+</div>
              <h1>Start with your data</h1>
              <p>
                Click anywhere on the canvas to add a data source.
                <br />
                Then build your pipeline, one layer at a time.
              </p>
              <button
                className="primary"
                disabled={disabled}
                onClick={() => open({ kind: "source" })}
              >
                Add data source
              </button>
            </div>
          )}
          <div className="canvas-footer">
            <span>
              {project.datasets.length} data sources ·{" "}
              {project.network.layers.length} layers · {allConnections.length}{" "}
              connections
            </span>
            <span>Select + Delete to remove · Click a connection to edit</span>
          </div>
        </main>
        {dialog && (
          <EditorDialog
            dialog={dialog}
            project={project}
            files={files}
            busy={busy}
            error={error}
            save={save}
            upload={upload}
            close={() => {
              setDialog(null);
              setError("");
            }}
          />
        )}
      </div>
    </EditorContext.Provider>
  );
}
function formatBytes(bytes: number) {
  return bytes < 1024
    ? `${bytes} B`
    : bytes < 1024 * 1024
      ? `${(bytes / 1024).toFixed(1)} KB`
      : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
