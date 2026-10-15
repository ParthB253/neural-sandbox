# Training page improvements

Planning and implementation date: 14 October 2026.

## Follow-up: browser memory and range navigation

The initial full-history approach was revised after the user reported roughly 5 GB Chrome memory usage during training. The frontend had retained the complete journal (including per-class tables and confusion matrices for every evaluation), fetched and merged it every ten seconds, and received repeated full recent histories and split-membership arrays through SSE. These are concrete sources of growth and allocation churn; the exact 5 GB Chrome session was not reproduced.

- Browser run snapshots now retain at most 200 scalar observations and one detailed evaluation. Split membership stays on the backend; counts remain visible in configuration. Compact SSE frames carry immutable project configuration once and incremental measurements thereafter, with detailed evaluation tables sent when the evaluation changes. Reconnect sends an authoritative initial snapshot.
- Chart history uses `compact=true&max_points=1200` plus metric/output selection and optional step bounds. The client enforces the same budget, replaces caches rather than appending, aborts stale requests and keeps only an overview and one selected-range view. Full raw history remains available on disk and through the original API behavior. Server sampling preserves endpoints and bucket extrema; narrower ranges retrieve finer recorded detail.
- The Recharts Brush depended on changing sampled-data indices. It was replaced with an absolute-step navigator supporting handles, selected-range dragging and keyboard controls. History responses cannot modify the selected range. Range requests are debounced while dragging; explicit From/To controls remain available.
- Latest evaluation objects and immutable project snapshots are reused between live updates, and per-class table lookup is linear rather than repeated scanning.

Validation: frontend build/lint/tests pass and all 29 focused backend training/metrics/view tests pass. Regression fixtures cover 100,000 observations, dense-curve peaks, raw history preservation, incremental SSE, retained class details and stable range panning. A separate Node data-cache stress run processed 2,000,000 observations across 20 refreshes and 100,000 incremental updates: post-GC heap stayed approximately 7.72–7.78 MiB, retaining 200 live records and a bounded sampled history. This measures the data-cache path, not total Chrome/Recharts memory. Browser interaction and a Chrome heap capture remain unverified because browser-control troubleshooting was previously deferred at the user's request.

## Original implementation notes

Implementation status: delivered with three parallel implementation subagents and primary-agent integration. Recharts provides brush/drag/range zoom and metric/output selection; full recorded history is available at `GET /training/runs/{id}/metrics` with optional inclusive `start_step` / `end_step` filters. As requested, API payloads remain straightforward and full history is fetched initially; compact list responses and bounded history payloads are deferred until network usage warrants them. Chart rendering alone is downsampled, preserving extrema and endpoints.

Fixed-model training/validation evaluation now includes per-output classification/regression metrics, final off-cadence evaluation, best-run scores and checkpoint metadata. The sidebar, setup, activity, immutable configuration and result displays have been rebuilt as separate components. Run cards show a latest-run badge and each run's own best validation loss; a cross-run winner is intentionally omitted without proven comparability. Existing run records remain readable and cannot acquire metrics that were never recorded.

Validation: frontend build, lint and tests pass; all 24 focused training/metrics backend tests pass. The full backend run also exposed six existing execution-test failures: those tests request untrained forward passes while the existing execution endpoint requires a training checkpoint. Browser visual verification was left incomplete after the user reported browser-control issues and requested that troubleshooting be skipped. Temporary preview servers were stopped.

## Outcome

Make long runs inspectable, show trustworthy training and validation metrics, and give the run history, activity and configuration the same visual care as the main workspace. Keep the existing Build / Train shell and sidebar width.

## Findings that shape the work

- `TrainingPage.tsx` currently draws a custom SVG with a dot for every validation observation and no zoom.
- Live run snapshots contain only the most recent 2,000 sampled metric entries. Older recorded observations are journalled in `metrics.jsonl`; training loss is sampled, rather than recorded at every optimiser step. Full-run navigation needs history retrieval as well as a new chart.
- Only validation accuracy is exposed today, under the ambiguous field `accuracy`. Loss and accuracy must identify their split and evaluation step.
- Run creation timestamps already exist. `updated_at` is available on the backend but missing from the frontend type.
- `best` and `latest` currently refer to checkpoints within a run. Numerical checkpoints are retained only for the newest run. A run-ranking badge must not imply that older weights remain available.
- Existing uncommitted workspace work must be preserved. Keep the original `TRAINING_PLAN.md` as the first-version reference.

## Parallel implementation ownership

The planning review was divided among three subagents: `chart_plan`, `metrics_plan` and `layout_plan`. Use the following ownership for implementation so agents can work in parallel without editing the same files.

| Owner | Responsibility | Owned files |
| --- | --- | --- |
| Metrics/backend agent | Evaluation calculations, recorded metrics, history retrieval, summary metadata and backend tests | `backend/src/neuralsandbox/training.py`, `training_runtime.py`, new `training_metrics.py` and metric models, `backend/tests/test_training.py`, new metric tests |
| Chart agent | Recharts, zoom controls, series preparation and chart performance | New `frontend/src/training/TrainingChart.tsx`, `chartData.ts`, chart styles and helper tests; `frontend/package.json` and lockfile |
| Layout agent | Run list, setup sections, activity timeline, recorded configuration and metrics presentation | New `RunList.tsx`, `TrainingSetup.tsx`, `RunActivity.tsx`, `RunConfiguration.tsx`, `RunMetrics.tsx`, `training.css`; `TrainingSidebar.tsx` |
| Primary agent / integrator | Shared contract, fetching/subscriptions, page composition, shared stylesheet cleanup and end-to-end verification | `frontend/src/training/useTraining.ts`, new `types.ts`, `TrainingPage.tsx`, targeted changes in `frontend/src/index.css` |

Agents must agree on the shared contract before wiring components. The integrator owns shared types; backend and frontend agents consume that contract rather than modifying it independently. Chart styles belong in a separate chart stylesheet; the layout agent owns the common training stylesheet.

## 1. Contract and semantics first

- Keep sampled batch loss for live feedback. Add evaluation records with step, epoch, time, split, example count and metrics per output. Do not mix an evolving batch estimate with a fixed-model whole-split evaluation under the same label.
- Keep new fields optional when reading old runs. Interpret legacy `accuracy` as validation accuracy. Missing metrics remain unavailable, never zero.
- Return a compact run summary for the sidebar and bounded live snapshots. Expose recorded history through a step-range endpoint with a display point budget. Preserve measurement counts/ranges so the chart can distinguish a sampled overview from detailed recorded observations.
- Add evaluation and checkpoint summaries recording their step, time and metric value. Keep checkpoint availability separate from historical performance.
- Define `Latest run` by creation time. Define any cross-run `Best validation` badge only within a comparable cohort: same data revision, split membership, target/output definition, loss and metric semantics. Runs outside that cohort should not compete for the badge. Ties are deterministic. If comparability cannot be established, omit the best badge.

## 2. Zoomable learning curves — chart agent

- Replace the SVG with a responsive Recharts line chart. Hide persistent dots for dense series; show the selected observation on hover. Use blue for training and purple for validation, with a real legend and precise tooltip.
- Add an overview brush, drag-to-select zoom, start/end step controls and Reset zoom. Keyboard users can operate the step controls without dragging.
- Recalculate the vertical domain for the visible range with sensible padding. Keep loss and percentage metrics on separate views with a metric selector.
- Preserve zoom while new snapshots arrive. Follow new data only when the user has selected the live/latest view. Reset the range when changing runs and cancel stale history requests.
- Merge sparse records by step without filling absent values with zero. Validation lines connect recorded evaluations; tooltip values must identify actual measurement steps.
- Fetch detailed recorded samples for the selected range. Downsample for rendering only, preserving extrema and endpoints; keep underlying observations intact. Show empty, single-observation, loading and history-unavailable states.
- Remove or scope the broad `.training-page svg` and SVG text rules, which would affect Recharts internals.

Reference: https://recharts.github.io/en-US/examples/HighlightAndZoomLineChart/ demonstrates selection-based zoom using axis domains and ReferenceArea.

## 3. Expanded, correct metrics — metrics/backend agent

- At the configured evaluation cadence, evaluate the training partition and validation partition with fixed weights, in batches and without gradient updates. Always evaluate the final completed model even when the final epoch is off cadence. With validation disabled, still evaluate training and omit validation results.
- Classification outputs: accuracy, macro precision, macro recall, macro F1, weighted F1, and per-class precision/recall/F1/support. Show balanced accuracy when all required classes have support; document unavailable cases.
- Cross-entropy also accepts soft probability targets. Emit discrete class metrics only for eligible hard targets; otherwise explain why those metrics are unavailable rather than silently treating soft targets as class labels.
- Aggregate confusion counts across the whole split before computing precision, recall and F1. Never average batch F1 values. Define zero-denominator behavior explicitly and serialize finite values or null.
- Explain per-class recall as the proportion of actual examples of that class predicted correctly; avoid a redundant, ambiguous “per-class accuracy” score. If class labels cannot be recovered reliably from transforms, use `Class 0`, `Class 1`, etc.
- For large class counts, return compact per-class counts and use a pageable/searchable table. A full confusion matrix is optional and bounded to small class sets.
- Regression outputs: MAE and RMSE alongside objective loss, with R² only when defined. Support mixed and multiple outputs through an output selector rather than silently averaging incompatible scores.
- Summary cards show paired training/validation loss, accuracy and F1 where applicable, plus best validation loss and its epoch. Secondary metrics and per-class results live in expandable tables with clear definitions and sample counts.
- Keep pause/cancel boundaries inside evaluation loops. Preserve SSE responsiveness, disk recovery and historical run loading. Limit expensive per-class payloads in live snapshots and fetch details on demand if needed.

## 4. Polished training workspace — layout agent

- Main pane: compact run heading and controls, status/progress summary, metric cards, chart, evaluation details, then activity and recorded configuration. Maintain clear spacing and readable widths at narrow window sizes.
- Run sidebar: full-width selectable cards showing short run ID, status indicator and text, local creation date/time, epoch/step progress and available validation score. Use a clear selected state and `Latest run` / scoped `Best validation` badges. Include an empty state and maintain chronological order.
- Setup sidebar: group fields into Data bindings, Training, Objectives and Evaluation sections. Give feeds compact cards with input/target labels, dataset/field, transform count and Edit action. Retain the paired preview and explain that settings apply to the next run.
- Activity: replace paragraphs with a chronological timeline showing timestamp, readable event title, status icon and detail. Coalesce repetitive evaluation progress into a stage entry rather than flooding the timeline. Keep failures visible and preserve the user's scroll position during updates.
- Recorded configuration: readable label/value sections for training settings, objective per output, split/evaluation policy and data bindings from the immutable run snapshot. Put raw JSON behind an optional technical-details disclosure.
- Checkpoints: cards for Best and Latest with step/time/score and actual availability. Preserve the existing checkpoint retention policy and download behavior unless explicitly expanded in a later task.
- Use existing colors, typography and controls; status must remain understandable without color. Add proper tab semantics, focus states and accessible names.

## Sequence and completion checks

1. Integrator establishes optional shared types, legacy adapters and endpoint/component contracts. Backend starts the metric evaluator; chart and layout agents build against fixtures in parallel.
2. Backend exposes summaries/history; integrator connects fetching and live updates. Chart and layout components are composed into the existing page.
3. Run focused correctness tests and existing frontend build/lint/tests and backend tests. Fix integration issues, then visually review the training page at desktop and narrow widths.

Acceptance checks:

- A 10,000-observation fixture supports overview, range selection, zoom/reset and readable hover values. Earlier recorded history remains reachable after the live 2,000-entry cap.
- Live updates do not reset a user-selected range or display another run's stale response. Empty, single-point and legacy runs render correctly.
- Known confusion matrices verify class metrics, imbalanced classes, absent classes, soft targets and zero predicted positives. Uneven batches produce the same results as whole-split evaluation.
- Training evaluation does not mutate weights or contaminate validation. Regression and mixed-output runs show appropriate metrics; disabled validation and off-cadence completion behave correctly.
- Latest/best run badges remain distinct from checkpoint availability, including after a newer run prunes older weights.
- Run selection, copy settings, feed editing, paired preview, start/pause/resume/cancel, reconnect and weight download retain their behavior.
- Activity and configuration are readable without opening raw JSON, all controls work by keyboard, and no chart/sidebar overflow occurs at narrow widths.

Deferred: cross-run curve comparison, hyperparameter search, ROC/PR curves, editable metric thresholds, new checkpoint retention policies and distributed training.
