export type TraceState = "running" | "done" | "error" | "cancelled" | "interrupted";
export interface TraceEvent {
  id: string;
  kind: "model" | "tool" | "backend" | "commit";
  label: string;
  startedAt: string;
  durationMs?: number;
  state: TraceState;
  input: string;
  output?: string;
  error?: string;
}
export interface RunLog {
  id: string;
  networkId: string;
  networkTitle: string;
  model: string;
  prompt: string;
  startedAt: string;
  durationMs?: number;
  state: TraceState;
  outcome?: string;
  events: TraceEvent[];
  omittedEvents?: number;
}

const STORAGE_KEY = "neuralsandbox.assistant.logs.v1";
const MAX_RUNS = 20;
const MAX_RUN_CHARS = 240_000;
const MAX_STORED_CHARS = 1_500_000;
const MAX_DETAIL_CHARS = 16_000;
type LogStorage = Pick<Storage, "getItem" | "setItem">;

export function serializeDetail(value: unknown, secret = ""): string {
  const json = (JSON.stringify(value, (key, item: unknown) => {
    if (/^(authorization|api[-_]?key|access[-_]?token|password|secret)$/i.test(key)) return "[redacted]";
    if (/^reasoning(_details)?$/.test(key)) return "[omitted]";
    if (item instanceof Error) return { name: item.name, message: item.message, stack: item.stack, cause: item.cause };
    if (typeof item === "string" && secret) return item.replaceAll(secret, "[redacted]");
    return item;
  }, 2) ?? "null").replace(/\uFEFF/g, "\\uFEFF");
  return json.length > MAX_DETAIL_CHARS ? `${json.slice(0, MAX_DETAIL_CHARS)}\n[truncated after ${MAX_DETAIL_CHARS} characters]` : json;
}

export class RunLogStore {
  private logs: RunLog[] = [];
  private listeners = new Set<() => void>();
  private storage?: LogStorage;
  warning = "";

  constructor(storage?: LogStorage) {
    this.storage = storage;
    try {
      const stored: unknown = JSON.parse(storage?.getItem(STORAGE_KEY) ?? "[]");
      if (!Array.isArray(stored)) throw new Error("Invalid log history");
      this.logs = stored.filter((run): run is RunLog => !!run && typeof run.id === "string" &&
        typeof run.startedAt === "string" && typeof run.prompt === "string" && Array.isArray(run.events) &&
        run.events.every((event: TraceEvent) => !!event && typeof event.id === "string" && typeof event.label === "string" && typeof event.input === "string"))
        .slice(0, MAX_RUNS).map((run) => ({ ...run,
          state: run.state === "running" ? "interrupted" : run.state,
          events: run.events.map((event) => ({ ...event, state: event.state === "running" ? "interrupted" : event.state })),
        }));
    } catch {
      this.warning = "Previous run logs could not be read. New logs are available for this session.";
    }
  }

  getSnapshot = (): RunLog[] => this.logs;
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };

  private publish() {
    this.logs = this.logs.slice(0, MAX_RUNS);
    // Retain recent events when an unusually long run exceeds its size budget.
    this.logs = this.logs.map((run) => {
      const events = [...run.events];
      while (events.length > 1 && JSON.stringify({ ...run, events }).length > MAX_RUN_CHARS) events.shift();
      return { ...run, events, omittedEvents: (run.omittedEvents ?? 0) + run.events.length - events.length };
    });
    while (this.logs.length > 1 && JSON.stringify(this.logs).length > MAX_STORED_CHARS) this.logs.pop();
    if (this.storage) {
      try { this.storage.setItem(STORAGE_KEY, JSON.stringify(this.logs)); this.warning = ""; }
      catch { this.warning = "Browser storage is full or unavailable. Logs are available for this session; export them before reloading."; }
    }
    this.listeners.forEach((listener) => listener());
  }

  start(run: Pick<RunLog, "id" | "networkId" | "networkTitle" | "model" | "prompt">, secret = ""): RunTrace {
    const text = (value: string) => (secret ? value.replaceAll(secret, "[redacted]") : value).slice(0, 8000);
    const safe = { ...run, prompt: text(run.prompt), networkTitle: text(run.networkTitle), model: text(run.model) };
    this.logs = [{ ...safe, startedAt: new Date().toISOString(), state: "running", events: [] }, ...this.logs];
    this.publish();
    return new RunTrace(this, run.id, secret);
  }

  update(id: string, change: (run: RunLog) => RunLog) {
    this.logs = this.logs.map((run) => run.id === id ? change(run) : run);
    this.publish();
  }

  clearFinished() {
    this.logs = this.logs.filter((run) => run.state === "running");
    this.publish();
  }
}

export class RunTrace {
  private store: RunLogStore;
  readonly id: string;
  private secret: string;
  private started = performance.now();

  constructor(store: RunLogStore, id: string, secret: string) {
    this.store = store;
    this.id = id;
    this.secret = secret;
  }

  start(kind: TraceEvent["kind"], label: string, input: unknown) {
    const id = crypto.randomUUID();
    const started = performance.now();
    this.store.update(this.id, (run) => ({ ...run, events: [...run.events, {
      id, kind, label, startedAt: new Date().toISOString(), state: "running", input: serializeDetail(input, this.secret),
    }] }));
    const settle = (state: TraceState, output?: unknown, error?: unknown) => this.store.update(this.id, (run) => ({
      ...run, events: run.events.map((event) => event.id !== id ? event : {
        ...event, state, durationMs: Math.round(performance.now() - started),
        output: output === undefined ? undefined : serializeDetail(output, this.secret),
        error: error === undefined ? undefined : serializeDetail(error, this.secret),
      }),
    }));
    return {
      done: (output: unknown) => settle("done", output),
      fail: (error: unknown, cancelled = false, output?: unknown) => settle(cancelled ? "cancelled" : "error", output, error),
    };
  }

  async step<T>(kind: TraceEvent["kind"], label: string, input: unknown, action: () => Promise<T>, signal?: AbortSignal): Promise<T> {
    const event = this.start(kind, label, input);
    try {
      const result = await action();
      event.done(result);
      return result;
    } catch (error) {
      event.fail(error, signal?.aborted);
      throw error;
    }
  }

  finish(state: Exclude<TraceState, "running">, outcome: unknown) {
    this.store.update(this.id, (run) => ({ ...run, state,
      durationMs: Math.round(performance.now() - this.started), outcome: serializeDetail(outcome, this.secret),
    }));
  }
}

function browserStore(): RunLogStore {
  try { return new RunLogStore(typeof window === "undefined" ? undefined : window.localStorage); }
  catch { const store = new RunLogStore(); store.warning = "Browser storage is unavailable. Logs are kept for this session only."; return store; }
}
export const runLogs = browserStore();
