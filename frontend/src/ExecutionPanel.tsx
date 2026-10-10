import { useState } from "react";
import { computationKey, request, type ForwardResult, type ProjectAnalysis, type ProjectSpec } from "./network";

export default function ExecutionPanel({ project, analysis, disabled }: {
  project: ProjectSpec; analysis: ProjectAnalysis | null; disabled: boolean;
}) {
  const [index, setIndex] = useState(0);
  const [seed, setSeed] = useState(0);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<{ value: ForwardResult; key: string } | null>(null);
  const key = computationKey(project);
  const inputSources = project.feeds.filter((feed) => feed.role === "input").map((feed) => feed.dataset);
  const counts = inputSources.map((id) => analysis?.datasets[id]?.sample_count);
  const count = counts.length && counts.every((value) => value !== undefined)
    ? Math.min(...counts as number[]) : undefined;

  async function run() {
    setRunning(true);
    setError("");
    try {
      const value = await request<ForwardResult>("/executions/forward", "POST", { project, sample_index: index, seed });
      setResult({ value, key });
    } catch (error) {
      setError(error instanceof Error ? error.message : "Forward pass failed");
    } finally {
      setRunning(false);
    }
  }
  return <section className="execution-panel" aria-label="Forward execution">
    <h2>Forward pass</h2>
    <p className="muted">Run one example with untrained weights. The seed keeps weights repeatable.</p>
    <div className="form-row">
      <label>Example index (from 0)
        <input type="number" min="0" max={count === undefined ? undefined : count - 1} step="1" value={index}
          onChange={(event) => setIndex(Number(event.target.value))} />
      </label>
      <label>Weight seed
        <input type="number" min="0" max={4294967295} step="1" value={seed}
          onChange={(event) => setSeed(Number(event.target.value))} />
      </label>
    </div>
    {count !== undefined && <p>{count.toLocaleString()} examples available. Input sources must have aligned rows.</p>}
    <div className="sample-controls">
      <button disabled={running || index <= 0} onClick={() => setIndex(Math.max(0, index - 1))}>Previous</button>
      <button disabled={running || count === undefined || index >= count - 1} onClick={() => setIndex(index + 1)}>Next</button>
      <button disabled={running || !count} onClick={() => { if (count) setIndex(Math.floor(Math.random() * count)); }}>Random example</button>
    </div>
    {analysis && analysis.issues.length > 0 && <ul className="execution-issues">
      {analysis.issues.map((issue, i) => <li key={i}><strong>{issue.node_id}</strong>: {issue.message}</li>)}
    </ul>}
    <button className="primary" disabled={disabled || running || !Number.isSafeInteger(index) || index < 0 ||
      !Number.isSafeInteger(seed) || seed < 0 || seed > 4294967295 || (count !== undefined && index >= count)}
      onClick={() => void run()}>{running ? "Running…" : "Run forward pass"}</button>
    {error && <p role="alert" className="execution-error">{error}</p>}
    {result && <div aria-live="polite">
      <h3>Result · example {result.value.sample_index}</h3>
      {result.key !== key && <p role="status">This result belongs to an earlier graph configuration.</p>}
      <p className="muted">Untrained · seed {result.value.seed}</p>
      {Object.entries(result.value.outputs).map(([id, output]) => <div key={id}>
        <h3>{id} · {output.shape.join(" × ")} · {output.activation}</h3>
        {output.activation === "softmax" && output.activation_applied && <p>Predicted class index: {output.values.reduce((best, value, i, values) => value > values[best] ? i : best, 0)}</p>}
        <pre>{output.values.slice(0, 50).map((value) => Number(value.toPrecision(6))).join(", ")}</pre>
        {output.values.length > 50 && <p>Showing first 50 of {output.values.length.toLocaleString()} values.</p>}
      </div>)}
      <details><summary>Execution revisions</summary>
        <dl>{(["execution_id", "project_revision", "model_revision", "data_revision"] as const).map((name) =>
          <div key={name}><dt>{name.replaceAll("_", " ")}</dt><dd><code>{result.value[name]}</code></dd></div>)}</dl>
      </details>
    </div>}
  </section>;
}
