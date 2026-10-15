# Training workspace implementation plan

Status: first working implementation delivered, 12 October 2026.

Implementation notes: the existing Flow shell is retained with separate training
components and a run hook, rather than a wholesale ProjectProvider refactor.
SSE sends authoritative snapshots on connection and change, rather than replaying
individual events. Evaluation cadence is currently epoch-based. The sidebar edits
next-run settings while the selected run retains its recorded configuration.
Detailed traces, streaming datasets, automatic restart-resume, and configurable
optimisation remain outside this version. Live charts retain 2,000 recent sampled
observations; the run directory retains the sampled metric journal.

Validation: frontend build/lint/tests, backend tests including numerical softmax,
paired-data validation, cancellation while paused, persistence, SSE snapshot,
and checkpoint execution; a 512-example MNIST run completed 21 updates with
held-out evaluation and checkpoint prediction. The following sections retain the
original design and acceptance criteria as the implementation reference.

## Decisions and scope

- Preserve the existing full-width header and the position/width of the sidebar.
- Put a floating Build / Train segmented control at the top centre of the right-hand pane, outside React Flow's transformed viewport. It remains in the same place on both pages.
- Build retains the graph and existing sidebar tools. Train shows run configuration in the existing sidebar and progress/results in the right-hand pane.
- Omit optimiser selection and learning-rate controls for now. Use the existing backend's direct gradient update rule; do not expose a setting that silently does nothing.
- Start with one active run per local backend, with cooperative pause/resume/cancel, disk-backed run records, and SSE progress.
- Defer distributed workers, hyperparameter sweeps, cross-run chart comparison, detailed activation/gradient traces, streaming datasets, and automatic recovery of interrupted training.

Confirmed data model: the two transformed sources are **inputs** and **expected outputs (targets)**, such as MNIST images and labels, paired by row. Use these labels in the interface. Training and validation instead describe partitions of these paired examples.

## 1. Data bindings and persistence

The current `ProjectSpec` already contains datasets and feeds. `FeedSpec` owns the selected source, field, target layer, input/target role, and ordered transforms. `network_store.py` saves these in `workspaces/networks/<network-id>.json` on File → Save. Uploaded files are stored separately. There is no training configuration or run persistence today.

Reuse that model:

- The normal setup is two imported sources, one connected as input and one as target. Keep support for different fields of one source; do not impose exactly two physical files or break multiple-input/output graphs.
- Reuse the existing transform editor. Training's Data section shows a summary for each binding, resolved shape/count, and an Edit action using that editor. There is no second copy of the transform pipeline and no pairing-algorithm dropdown.
- Form example records using the same row index across all input and target feeds. Validate equal, nonzero **post-transform** row counts before training; never silently truncate through `zip`.
- Row correspondence is the user's responsibility. Provide a paired-example preview and a short explanation that corresponding rows must describe the same example. Equal counts cannot prove semantic alignment.
- Shuffle/sample the common index list once and apply it to all bindings. Never shuffle images and labels separately.
- Require one compatible target feed per trained output for v1. Validate target width, numeric values, category encoding, loss compatibility, and finite data before updates begin. Report problems against the relevant feed/output.

Add an optional, defaulted `TrainingSpec` to `ProjectSpec`, stored alongside graph configuration. It contains batch size, epoch budget, sampling mode, seeds, output-to-loss configuration, and evaluation settings. Input/target bindings remain in `feeds` rather than being duplicated here.

All project operations must preserve/reset this field deliberately: GET/PUT project, File Save/Open/Save as, New, Clear, frontend removal helpers, and backend granular edit routes. Older saves load with defaults. Configuration edits make the document dirty. Deleting a referenced output clears its loss configuration or creates an actionable incomplete setup; it must not leave an invisible stale reference. Opening an older file must not discard its existing feed transforms.

Recommended first evaluation mode: an optional fixed, seeded validation holdout from the aligned examples, plus evaluation cadence in epochs or steps. Persist the split policy in the project and the resolved membership in the run. With evaluation off, hide evaluation metrics. Separate external evaluation datasets and a final untouched test set can follow; never portray training-set accuracy as held-out performance. Fix categorical encoding consistently across partitions; any future learned preprocessing must be fitted on training rows only.

Save each run separately under `workspaces/runs/<run-id>/`:

- Immutable project/configuration snapshot, network ID, data fingerprints, resolved split, initialization/sampling seeds, and runtime/update-rule version.
- Run state and timestamps, metric history, and errors.
- Latest and best checkpoints with layer/link IDs and architecture revision.

Starting a run records the current draft without implicitly overwriting File → Save. Save as copies configuration into a new network identity, not another network's run history. Protect uploaded files referenced by retained runs as well as saved graphs; detect changed external file contents before reuse. Snapshot references/hashes do not themselves preserve modified source bytes.

## 2. Page layout and frontend responsibilities

Extract the shared layout and document state from the current large `Flow.tsx` before adding training UI:

| Component | Responsibility |
| --- | --- |
| `Workspace` / `ProjectProvider` | Shared header, document state, save/open lifecycle, page selection |
| `WorkspaceModeSwitch` | Build / Train control positioned relative to the right pane |
| `BuildSidebar` / `BuildCanvas` | Existing files, graph, single-example execution, and React Flow |
| `TrainingSidebar` | Setup and Runs tabs, data bindings, sampling, loss, evaluation |
| `TrainingPage` | Run actions, current stage, loss chart, evaluation results, activity |
| `TrainingProvider` / `useTrainingRun` | REST requests, selected run, SSE lifecycle, reconnect and event reduction |

Build → Train changes sidebar content and right-pane content together. Preserve the graph viewport, selection, and each mode's selected sidebar tab. The switch does not pan/zoom with the graph. Disable graph keyboard handlers while Train is visible. Training continues while switching modes or inspecting another network; run subscriptions and controls are explicitly scoped by run/network ID.

Train sidebar, Setup tab:

- Data: input/target summaries and paired preview.
- Sampling: sequential or shuffle each epoch for v1, batch size, epoch budget, seed. Show derived batches/updates per epoch. Balanced and replacement sampling are later additions with their own explicit semantics.
- Objective: a supported loss for each output; start with equally weighted mean losses rather than a new weighting UI.
- Evaluation: holdout size, split seed, cadence, appropriate metrics.

The sidebar's Runs tab lists previous runs and allows selecting one or copying its settings. During a run show its immutable configuration. An Edit as new run action creates editable settings for a future run and does not mutate the active run.

The right pane reserves space below the segmented control for:

1. Run name, Start/Pause/Resume/Cancel, and snapshot revision.
2. Overall epoch/step progress and a separate current-stage message.
3. Dominant loss chart with actual validation measurement points and a concise metric summary.
4. Evaluation details and expandable run activity.

Before starting, show readiness/errors and an empty chart, not fabricated values. During loading, display indeterminate progress unless a real total exists. Use stages such as Loading sources → Applying transforms → Preparing sample order → Training batch → Evaluating → Saving checkpoint. Sampling and evaluation can recur; a static one-way stepper must not imply otherwise. Report epoch, batch, global update count, elapsed time, and an ETA only when there is enough measured throughput. Downsample long chart histories without dropping final/error events.

## 3. Prepare an actual training runtime

The current forward endpoint constructs a new model on every request and supplies a `no_backward` function. `Network.train()` synchronously loops over examples, updates weights with a unit gradient step, and prints epoch cost. Neither currently supports training jobs, persistent weights, progress events, evaluation, or controls.

- Extract shared project preparation, binding, initialization, activation, and revision helpers from `execution.py` into a runtime module. Keep HTTP handling separate from computation.
- Construct one model per run; retain its weights through all batches and evaluations. Load each source and materialize each transformed feed once per run, reusing the existing source/feed cache. Add a bounded batch-index API to `Feed`; avoid repeated full-dataset copies.
- Add real activation backward operations and cost functions. The current core assumes elementwise activation derivatives; softmax needs a vector-Jacobian product or a correctly fused softmax/cross-entropy path. Do not insert an elementwise softmax derivative just to make MNIST start.
- Define mean loss/gradient reduction across a batch, including the final short batch. Preserve the current direct gradient-update rule with no optimiser configuration. Its fixed unit step may limit convergence; record that rule and fail clearly on non-finite values. Configurable optimisation remains separate future work.
- Add a step-oriented runner around forward/backprop/update with events and control checks. Keep the existing `Network.train()` convenience method working, ideally delegating to the same stepping logic.
- Evaluate with the current weights without applying gradients. Aggregate losses by example count. Provide classification accuracy only when the objective/targets make it meaningful; otherwise show the applicable loss/metric.
- Snapshot weights at safe boundaries. Save latest after each epoch and on pause/cancel/completion; save best when validation improves. A checkpoint should carry batch cursor, sample order/RNG state, and counters needed for consistent in-process resume, although restart-resume is deferred.
- Extend single-example execution to accept an optional checkpoint reference. Keep today's seed-based fresh-weight execution available, label the selected weight source, and reject incompatible topology/checkpoint combinations.

## 4. Run service, SSE, and REST

SSE is suitable: training sends progress one way and ordinary REST handles commands. The transport is a small part of this work; the run owner, lifecycle, and controllable computation are the substantive additions.

For this local version, use a lifespan-managed `TrainingRunManager` in one FastAPI server process and one dedicated training worker thread. NumPy computation stays out of the async event loop. Use thread-safe synchronization and event handoff; the worker must not touch asyncio queues directly. The worker owns its mutable model. If profiling shows unacceptable responsiveness, a child process can replace the worker behind the same service contract; multiple API workers require a shared job service and are outside v1.

| Endpoint | Contract |
| --- | --- |
| `POST /training/runs` | Validate request structure, snapshot draft/settings, create run, return 202 and run ID; expensive preparation belongs to the worker |
| `GET /training/runs?network_id=…` | List this network's runs |
| `GET /training/runs/{id}` | Authoritative current snapshot, metrics, checkpoint metadata, event cursor |
| `GET /training/runs/{id}/events` | SSE subscription to an existing run; does not start or own computation |
| `POST /training/runs/{id}/pause` | Request cooperative pause |
| `POST /training/runs/{id}/resume` | Resume a live paused run |
| `POST /training/runs/{id}/cancel` | Request terminal cancellation while retaining results/checkpoints |

Reject a second active run with a useful conflict response in v1 rather than adding a queue. Make creation retry-safe with a client-generated request ID and commands idempotent where applicable. Validate state transitions under a lock, including cancel while paused and completion racing with pause.

Keep **status** separate from **stage**. Statuses include preparing, running, pausing, paused, cancelling, completed, cancelled, failed, and interrupted. Stage describes current work. Command acceptance is not pause completion: show Pausing… until the worker acknowledges a safe boundary.

Check controls between batches, evaluation chunks, transforms, and source loads where possible. A single NumPy operation or existing full-file load cannot be interrupted mid-call; show the pending command honestly. Resume must preserve weights, sample order, counters, and RNG state. Browser navigation/disconnection only removes a subscriber and never cancels training.

Use Starlette `StreamingResponse` with `text/event-stream`, correctly framed JSON events, heartbeat comments, monotonically increasing event IDs, and no-cache headers. Event types: state, progress, metrics, checkpoint, and failure. Include run ID, sequence, timestamp, and explicit counters. Throttle ordinary progress to around 5 updates/second; emit state transitions immediately. Keep bounded subscriber buffers and durable metric history so slow clients cannot stall computation or exhaust memory.

On connection/reconnection, deliver a consistent snapshot plus events after its cursor. Honour `Last-Event-ID`; if replay history is unavailable, send a fresh snapshot. Deduplicate by sequence and close the client EventSource on terminal status. Do not confuse a disconnected stream with a failed run. Hydrate after refresh using REST and reconnect to the same run ID. Test event creation/subscription races and multiple observers.

Manage startup/shutdown through FastAPI lifespan. Persist final state atomically and stop accepting runs during shutdown. Previously active/paused runs found after restart are marked Interrupted; history/checkpoints remain available, but automatic resume is not promised. Bound shutdown waits and document how development reload affects active training.

Relevant platform references: [Starlette streaming responses](https://starlette.dev/responses/#streamingresponse), [FastAPI background task caveat for heavy computation](https://fastapi.tiangolo.com/tutorial/background-tasks/#caveat), and [browser SSE events/reconnection](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events). FastAPI's request `BackgroundTasks` helper alone does not provide the required run ownership and controls.

## 5. Delivery order and acceptance checks

1. **Document model and shell:** optional training settings with backwards-compatible saves; extract workspace components; put the segmented control in the right pane and swap sidebar content. Check Save/Open/Save as/New/Clear and viewport preservation.
2. **Data preparation and training correctness:** paired preview/validation, common-index sampling, fixed holdout, real derivatives/losses, batched updates, evaluation, checkpoints. Verify numerical gradients (especially softmax/cross-entropy and DAG fan-in), loss reduction on a deterministic small dataset, mismatched rows/targets, reproducible pairing, and no train/validation overlap.
3. **Run lifecycle and transport:** manager, worker, persistence, REST, SSE. Test command races, cancellation while paused, equivalence of paused/resumed and uninterrupted deterministic runs, refresh/reconnect without restarting work, subscriber cleanup, and restart-to-Interrupted behavior.
4. **Training workspace integration:** wire real metrics, stages, setup errors, run history, and controls. Test stream deduplication, chart recovery, frozen run configuration, page/network switching, and responsive layout with the retained sidebar/header.
5. **End-to-end usable result:** run a small MNIST subset, pause/resume, evaluate, reopen its history, and execute an example using its saved checkpoint. Confirm existing forward execution, numerical-gradient tests, frontend build/lint/tests, and backend tests still pass; update README with limits and restart behavior.

Completion means a user can prepare aligned inputs/targets in the existing editor, save their setup, start a real training run, understand its current stage and loss, pause/resume/cancel, reopen its results, and inspect predictions using trained weights without losing the surrounding workspace layout.
