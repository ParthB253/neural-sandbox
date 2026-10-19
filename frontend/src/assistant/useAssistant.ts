import { useEffect, useRef, useState } from "react";
import { request, type ProjectSpec } from "../network";
import { runLogs } from "./diagnostics";
import { complete } from "./openrouter";
import { instructions, runAgent } from "./runner";
import { createTools } from "./tools";
import { messageError, type Activity, type Connection, type Message } from "./types";

export interface Turn {
  id: string;
  request: string;
  logId?: string;
  response?: string;
  error?: string;
  activities: Activity[];
}

export function useAssistant(host: {
  project: ProjectSpec;
  getProject: () => ProjectSpec;
  commit: (next: ProjectSpec, expected: ProjectSpec) => Promise<ProjectSpec>;
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [running, setRunning] = useState(false);
  const [undo, setUndo] = useState<{ before: ProjectSpec; after: ProjectSpec } | null>(null);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => () => controller.current?.abort(), []);

  async function send(prompt: string, connection: Connection) {
    if (controller.current) return;
    const current = new AbortController();
    controller.current = current;
    setRunning(true);
    setUndo(null);
    const id = crypto.randomUUID();
    const trace = runLogs.start(
      {
        id,
        networkId: host.project.network.id,
        networkTitle: host.project.network.title,
        model: connection.model,
        prompt,
      },
      connection.apiKey,
    );
    const turn: Turn = { id, logId: id, request: prompt, activities: [] };
    setTurns((previous) => [...previous, turn]);
    const patch = (values: Partial<Turn>) =>
      setTurns((previous) => previous.map((item) => (item.id === id ? { ...item, ...values } : item)));
    let before: ProjectSpec | undefined;
    let after: ProjectSpec | undefined;
    try {
      const tools = await createTools(
        {
          getProject: host.getProject,
          request: (path, method, body, signal) =>
            trace.step(
              "backend",
              `${method ?? "GET"} ${path}`,
              { path, method: method ?? "GET", body },
              () => request(path, method, body, signal),
              signal,
            ),
          commit: async (next, expected) => {
            current.signal.throwIfAborted();
            const saved = await trace.step("commit", "PUT /project", next, () => host.commit(next, expected));
            // Undo must never cross an intervening manual edit.
            if (!before || (after && JSON.stringify(expected) !== JSON.stringify(after))) {
              before = structuredClone(expected);
            }
            after = structuredClone(saved);
            return saved;
          },
        },
        current.signal,
      );
      // Keep bounded conversational context; refresh all tool observations each turn.
      const history: Message[] = turns.slice(-6).flatMap((item): Message[] => [
        { role: "user", content: item.request },
        { role: "assistant", content: item.response ?? item.error ?? "Request stopped." },
      ]);
      const messages: Message[] = [
        { role: "system", content: instructions },
        ...history,
        { role: "user", content: prompt },
      ];
      const response = await runAgent({
        messages,
        tools: tools.definitions,
        complete: (messages, definitions, signal) => complete(connection, messages, definitions, signal, trace),
        execute: tools.execute,
        trace,
        signal: current.signal,
        activity: (activity) =>
          setTurns((previous) =>
            previous.map((item) =>
              item.id !== id
                ? item
                : {
                    ...item,
                    activities: [...item.activities.filter((entry) => entry.id !== activity.id), activity],
                  },
            ),
          ),
      });
      trace.finish("done", response);
      patch({ response });
    } catch (error) {
      trace.finish(current.signal.aborted ? "cancelled" : "error", error);
      patch({ error: current.signal.aborted ? "Stopped. Any applied edits are kept." : messageError(error) });
    } finally {
      if (before && after) setUndo({ before, after });
      setTurns((previous) =>
        previous.map((item) =>
          item.id !== id
            ? item
            : {
                ...item,
                activities: item.activities.map((activity) =>
                  activity.state === "running" ? { ...activity, state: "error" } : activity,
                ),
              },
        ),
      );
      controller.current = null;
      setRunning(false);
    }
  }

  async function undoChanges() {
    if (!undo || controller.current) return;
    const id = crypto.randomUUID();
    const trace = runLogs.start({
      id,
      networkId: host.project.network.id,
      networkTitle: host.project.network.title,
      model: "Editor",
      prompt: "Undo assistant edits",
    });
    try {
      await trace.step("commit", "Undo · PUT /project", undo.before, () => host.commit(undo.before, undo.after));
      trace.finish("done", "Restored the network to before my last edits.");
      setUndo(null);
      setTurns((previous) => [
        ...previous,
        {
          id,
          logId: id,
          request: "Undo assistant edits",
          response: "Restored the network to before my last edits.",
          activities: [],
        },
      ]);
    } catch (error) {
      trace.finish("error", error);
      throw error;
    }
  }

  return {
    turns,
    running,
    send,
    undoChanges,
    canUndo: !!undo && JSON.stringify(host.project) === JSON.stringify(undo.after),
    stop: () => controller.current?.abort(),
    clear: () => {
      if (!controller.current) {
        setTurns([]);
        setUndo(null);
      }
    },
  };
}
