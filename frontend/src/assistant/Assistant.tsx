import { lazy, Suspense, useEffect, useRef, useState, type FormEvent } from "react";
import type { ProjectSpec } from "../network";
import { useAssistant } from "./useAssistant";
import RunLogs from "./RunLogs";

import { messageError, type Connection } from "./types";
import "./assistant.css";

const AssistantMarkdown = lazy(() => import("./AssistantMarkdown"));

interface Props {
  project: ProjectSpec;
  disabled: boolean;
  getProject: () => ProjectSpec;
  commit: (next: ProjectSpec, expected: ProjectSpec) => Promise<ProjectSpec>;
}

function Brain() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 5c-1-3-5-2-5 1-3 0-4 4-2 6-2 3 0 6 3 6 0 3 4 4 4 1V5Zm0 0c1-3 5-2 5 1 3 0 4 4 2 6 2 3 0 6-3 6 0 3-4 4-4 1" />
      <path d="M7 6c0 2 1 3 2 3m-4 3 3-1m0 7v-4m9-8c0 2-1 3-2 3m4 3-3-1m0 7v-4" />
    </svg>
  );
}

export default function Assistant(props: Props) {
  // Switching networks resets conversation while retaining the connection settings.
  const [connection, setConnection] = useState<Connection>({
    apiKey: import.meta.env.OPENROUTER_API_KEY ?? "",
    model: import.meta.env.OPENROUTER_MODEL || "minimax/minimax-m3.1-flash:free",
  });
  return (
    <AssistantPanel key={props.project.network.id} {...props} connection={connection} setConnection={setConnection} />
  );
}

function AssistantPanel({
  connection,
  setConnection,
  ...host
}: Props & {
  connection: Connection;
  setConnection: (connection: Connection) => void;
}) {
  const agent = useAssistant(host);
  const [expanded, setExpanded] = useState(false);
  const [settings, setSettings] = useState(false);
  const [logs, setLogs] = useState(false);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState("");
  const [undoing, setUndoing] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const connected = !!connection.apiKey.trim() && !!connection.model.trim();
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "nearest" });
  }, [agent.turns, expanded]);

  function submit(event: FormEvent) {
    event.preventDefault();
    if (!draft.trim() || host.disabled || agent.running) return;
    setExpanded(true);
    setLogs(false);
    setError("");
    if (!connected) {
      setSettings(true);
      return;
    }
    const prompt = draft.trim();
    setDraft("");
    void agent.send(prompt, { apiKey: connection.apiKey.trim(), model: connection.model.trim() });
  }

  return (
    <section className={`assistant${expanded ? " assistant--expanded" : ""}`} aria-label="Network assistant">
      {expanded && (
        <header className="assistant-header">
          <Brain />
          <strong>Network assistant</strong>
          <button
            type="button"
            aria-pressed={logs}
            onClick={() => {
              setLogs(!logs);
              setSettings(false);
            }}
          >
            Run logs
          </button>
          <button
            type="button"
            aria-label="Assistant settings"
            aria-pressed={settings}
            onClick={() => {
              setSettings(!settings);
              setLogs(false);
            }}
          >
            Settings
          </button>
          <button
            type="button"
            aria-label="Collapse assistant"
            onClick={() => {
              setExpanded(false);
              setSettings(false);
              input.current?.focus();
            }}
          >
            ⌃
          </button>
        </header>
      )}
      {expanded && (
        <div id="assistant-content" className="assistant-content">
          {logs ? (
            <RunLogs selectedId={selectedRun} select={setSelectedRun} />
          ) : settings ? (
            <form
              className="assistant-settings"
              onSubmit={(event) => {
                event.preventDefault();
                setSettings(false);
                input.current?.focus();
              }}
            >
              <strong>Connect to OpenRouter</strong>
              <label>
                API key
                <input
                  type="password"
                  autoComplete="off"
                  required
                  value={connection.apiKey}
                  onChange={(event) => setConnection({ ...connection, apiKey: event.target.value })}
                  placeholder="OpenRouter API key"
                  disabled={agent.running}
                />
              </label>
              <label>
                Model
                <input
                  required
                  value={connection.model}
                  onChange={(event) => setConnection({ ...connection, model: event.target.value })}
                  placeholder="provider/model-id"
                  disabled={agent.running}
                />
              </label>
              <small>
                Use a model that supports tools. Requests go directly to OpenRouter. A key from .env is included in the
                browser build; manually entered keys stay in memory. Network configuration and tool results are shared
                with the selected model.
              </small>
              <div className="assistant-actions">
                <a href="https://openrouter.ai/models?supported_parameters=tools" target="_blank" rel="noreferrer">
                  Browse models
                </a>
                <button type="submit" disabled={agent.running}>
                  Done
                </button>
              </div>
            </form>
          ) : (
            <>
              {!agent.turns.length && (
                <p className="assistant-hint">Describe a change to your network, or ask how it works.</p>
              )}
              <div
                className="assistant-transcript"
                role="log"
                aria-label="Assistant conversation"
                aria-live="polite"
                aria-relevant="additions text"
              >
                {agent.turns.map((turn) => (
                  <article className="assistant-turn" key={turn.id}>
                    <p className="assistant-request">
                      <span className="sr-only">You: </span>
                      {turn.request}
                    </p>
                    {!!turn.activities.length && (
                      <details className="assistant-activity">
                        <summary>
                          {turn.activities.length} tool {turn.activities.length === 1 ? "step" : "steps"}
                        </summary>
                        <ul>
                          {turn.activities.map((activity) => (
                            <li key={activity.id}>
                              {activity.state === "running" ? "…" : activity.state === "done" ? "✓" : "×"}{" "}
                              {activity.label}
                              {activity.state === "error" ? " · unsuccessful" : ""}
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                    {turn.logId && (
                      <button
                        type="button"
                        className="assistant-view-log"
                        onClick={() => {
                          setSelectedRun(turn.logId!);
                          setLogs(true);
                        }}
                      >
                        View run log
                      </button>
                    )}
                    {turn.response && (
                      <Suspense fallback={<p className="assistant-response">{turn.response}</p>}>
                        <AssistantMarkdown>{turn.response}</AssistantMarkdown>
                      </Suspense>
                    )}
                    {turn.error && (
                      <p className="assistant-error" role="alert">
                        {turn.error}
                      </p>
                    )}
                  </article>
                ))}
              </div>
              {agent.running && (
                <p className="assistant-hint" role="status">
                  {agent.turns.at(-1)?.activities.find((item) => item.state === "running")?.label ?? "Thinking…"}
                </p>
              )}
              <div ref={bottom} />
            </>
          )}
        </div>
      )}
      {expanded && !settings && !logs && !!agent.turns.length && !agent.running && (
        <div className="assistant-actions assistant-history-actions">
          <button
            type="button"
            disabled={host.disabled || undoing}
            onClick={() => {
              agent.clear();
              setError("");
            }}
          >
            New chat
          </button>
          {agent.canUndo && (
            <button
              type="button"
              disabled={host.disabled || undoing}
              onClick={async () => {
                setUndoing(true);
                setError("");
                try {
                  await agent.undoChanges();
                } catch (error) {
                  setError(messageError(error));
                } finally {
                  setUndoing(false);
                }
              }}
            >
              {undoing ? "Restoring…" : "Undo edits"}
            </button>
          )}
        </div>
      )}
      {error && (
        <p className="assistant-error" role="alert">
          {error}
        </p>
      )}
      <form className="assistant-composer" onSubmit={submit}>
        {!expanded && <Brain />}
        <input
          ref={input}
          aria-label="Ask AI to edit your network"
          placeholder={agent.running ? "Working on your request…" : "Ask AI to edit your network…"}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          disabled={agent.running || undoing}
        />
        {agent.running ? (
          <button type="button" onClick={agent.stop} aria-label="Stop assistant">
            Stop
          </button>
        ) : (
          <button type="submit" disabled={!draft.trim() || host.disabled || undoing} aria-label="Send request">
            ↑
          </button>
        )}
        {!expanded && (
          <button
            type="button"
            aria-label="Expand assistant"
            aria-expanded={expanded}
            aria-controls="assistant-content"
            onClick={() => {
              setExpanded(true);
              if (!connected) setSettings(true);
            }}
          >
            ⌄
          </button>
        )}
      </form>
      {!expanded && agent.running && (
        <span className="sr-only" role="status">
          Assistant is working. Expand to view progress.
        </span>
      )}
    </section>
  );
}
