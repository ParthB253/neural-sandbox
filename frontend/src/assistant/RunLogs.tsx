import { useState, useSyncExternalStore } from "react";
import { runLogs, type RunLog, type TraceEvent } from "./diagnostics";

function duration(ms?: number): string {
  return ms === undefined ? "" : ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}

function download(logs: RunLog[]) {
  const url = URL.createObjectURL(new Blob([JSON.stringify({ version: 1, exportedAt: new Date().toISOString(), runs: logs }, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `assistant-logs-${new Date().toISOString().replaceAll(":", "-")}.json`;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function EventDetails({ event }: { event: TraceEvent }) {
  return <details className="assistant-log-event">
    <summary>
      <span>{event.kind} · {event.label}</span>
      <small>{event.state} {duration(event.durationMs)}</small>
    </summary>
    <small>{new Date(event.startedAt).toLocaleTimeString()} · {event.id}</small>
    <h4>Input</h4><pre>{event.input}</pre>
    {event.output !== undefined && <><h4>Result</h4><pre>{event.output}</pre></>}
    {event.error !== undefined && <><h4>Error</h4><pre className="assistant-log-error">{event.error}</pre></>}
  </details>;
}

export default function RunLogs({ selectedId, select }: { selectedId: string | null; select: (id: string) => void }) {
  const logs = useSyncExternalStore(runLogs.subscribe, runLogs.getSnapshot);
  const [kind, setKind] = useState("all");
  const selected = logs.find((run) => run.id === selectedId) ?? logs[0];
  return <section className="assistant-logs" aria-label="Assistant run logs">
    <div className="assistant-actions">
      <strong>Run logs</strong>
      <button type="button" disabled={!logs.length} onClick={() => download(logs)}>Export all</button>
    </div>
    <small>Saved in this browser, including project inputs and results. Keys and private reasoning are omitted. Older logs and large details are trimmed to limit storage.</small>
    {runLogs.warning && <p role="status">{runLogs.warning}</p>}
    {selectedId && !logs.some((run) => run.id === selectedId) && !!logs.length && <small>The selected run is no longer stored. Showing the most recent run.</small>}
    {!logs.length ? <p className="assistant-hint">No recorded runs yet. New assistant requests will appear here.</p> : <>
      <label className="assistant-log-selector">Run
        <select value={selected.id} onChange={(event) => select(event.target.value)}>
          {logs.map((run) => <option key={run.id} value={run.id}>{new Date(run.startedAt).toLocaleString()} · {run.state} · {run.prompt.slice(0, 60)}</option>)}
        </select>
      </label>
      <p className="assistant-log-prompt">{selected.prompt}</p>
      <dl className="assistant-log-metadata">
        <dt>Network</dt><dd>{selected.networkTitle}</dd>
        <dt>Model</dt><dd>{selected.model}</dd>
        <dt>Status</dt><dd>{selected.state} {duration(selected.durationMs)}</dd>
        <dt>Run ID</dt><dd>{selected.id}</dd>
      </dl>
      {selected.outcome && <details><summary>Run outcome</summary><pre>{selected.outcome}</pre></details>}
      <div className="assistant-actions">
        <label className="assistant-log-selector">Show
          <select value={kind} onChange={(event) => setKind(event.target.value)} aria-label="Filter log events">
            <option value="all">All events</option><option value="errors">Errors and cancellations</option>
            {(["model", "tool", "backend", "commit"] as const).map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
        </label>
        <button type="button" onClick={() => download([selected])}>Export run</button>
      </div>
      {!!selected.omittedEvents && <small>{selected.omittedEvents} earlier events omitted to limit storage.</small>}
      <div>{selected.events.filter((event) => kind === "all" || event.kind === kind ||
        (kind === "errors" && ["error", "cancelled", "interrupted"].includes(event.state))).map((event) => <EventDetails key={event.id} event={event} />)}</div>
      <button type="button" onClick={() => runLogs.clearFinished()}>Clear finished logs</button>
    </>}
  </section>;
}
