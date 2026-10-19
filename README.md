# NeuralSandbox — a neural network you can see

A learning project for making neural networks less opaque. It starts with raw
NumPy and notebooks, then carries the same ideas into a visual builder in the
browser.

## Why

You can build a neural network in PyTorch in about six lines:

```python
model = nn.Sequential(nn.Linear(2, 16), ...)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for x, y in data:
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
```

If you already know what's happening here, it's great — you don't want to
re-derive the chain rule every morning. As a beginner I don't understand a single
line. It's boilerplate wrapped around mysterious optimisation and math;
`loss.backward()` does... what, exactly?

So this is me trying to actually understand it:

- No PyTorch, no TensorFlow for this first pass. Just vectors and NumPy.
- Clarity over speed. The optimisation is left bad on purpose, so long as every line
  is readable and you can see why it's there.

Eventually, PyTorch may have a place here too, where its abstractions genuinely
help. The aim is to move up that abstraction ladder deliberately — for me while
building it, and for anyone using the visual builder — without losing sight of
what each layer is doing.

Start with the simplest thing that works — a feed-forward net — and write it out
by hand.

## Where it's at

**`notebooks/mnist_nn.ipynb`** — where I started: with the basics of gradient
descent and execution written out by hand. It is a fixed `[784, 16, 16, 10]`
feed-forward net with ReLU hidden layers, a softmax output, cross-entropy cost,
and backprop term by term. On MNIST it reaches **~95.4% test accuracy after 10
training iterations**. It is slow, but each step is there to inspect.

**`backend/src/neuralsandbox/core.py`** — the next step towards generalising that
exercise. Working through one fixed network made it clearer that the architecture,
rather than the surface shape of the code, is what matters. This is where
`Layer` / `Link` / `Network` live:

- a network is an arbitrary DAG of layers, not just a stack
- the forward pass runs in topological order
- `Network.backprop` walks it in reverse and returns the gradients — it computes
  them; applying them is a separate step
- activation, the fan-in combinator, and the cost are all passed in alongside
  their derivatives; nothing is hard-coded
- backprop is checked against numerical gradients

`Network.train` is deliberately bare: it applies the gradients directly, without
an optimiser hiding the mechanics. The visual editor supports topology, data
sources, connections, transforms, and named network documents. It can derive
input shapes and execute a single forward pass; training is available in the editor’s Train workspace.

## Where it's going

The point isn't the code. The goal is a **visual builder** in the browser: drag
layer blocks onto a canvas, wire them together, press train, and watch the data
move through the graph. The architecture is the thing you touch directly.
Optimiser details are there if you open them and folded away if you don't. It
abstracts away just the parts that were boilerplate to begin with — and nothing
else.

**Next:** capture inspectable execution traces and connect training to the editor.

## Pending

The next substantial work is training and tracing execution through a network:
connecting the editor's pipeline to the core, making each pass visible, and
showing how values and gradients move through the graph. The UI is still being
iterated on and should be treated as work in progress rather than a finished
builder.

## Layout

```
backend/          Python: NumPy core + FastAPI service
  src/neuralsandbox/
    core.py         Layer / Link / Network — the math
    schemas.py      pydantic topology models (the wire format)
    api.py          FastAPI app
  tests/
frontend/         React + TypeScript pipeline editor (Vite + React Flow)
notebooks/        exploratory, hand-written nets
mnist_dataset/    IDX files, git-ignored, local only
```

The frontend renders and edits a `ProjectSpec` (network topology, data sources,
and feeds — no weights);
Python stays the single source of truth and the only place the math lives.

## Running it

```
cd backend
uv sync
uv run uvicorn neuralsandbox.api:app --reload   # http://localhost:8000, docs at /docs
uv run pytest
```

Notebooks use the same environment: `uv run jupyter lab` from `backend/`.

## Pipeline editor

The frontend is a React + TypeScript editor using React Flow. Run it alongside
the backend (Node 22.18+ is needed for the frontend test runner):

```sh
cd frontend
pnpm install
pnpm dev                 # http://localhost:5173
pnpm build
pnpm lint
pnpm test
```

- **Project files:** upload files once and reuse them across data-source nodes.
  CSV, JSON, NumPy and IDX sources can be configured in the editor. Uploads are
  limited to 100 MB each and stored on disk under `workspaces/files/` (git-ignored).
  Set `NEURALSANDBOX_UPLOAD_DIR` to use a different storage directory.
- **Network documents:** startup and **File → New network** begin with an empty
  canvas. Edit the title beside the nn icon. **File → Save** (Ctrl/⌘ S) writes the
  current network to disk; **Save as** creates a new network with its own ID and
  title. **Open saved network** restores a saved graph, its source configuration,
  transforms, and connections. Switching networks prompts before discarding
  unsaved changes. Uploaded files remain shared, independent resources.
- **Start a pipeline:** click an empty canvas to configure a data source, select a
  stored file, or upload a new one. Use **+ Data source** for additional sources.
  **Clear canvas** removes the graph while retaining uploaded files.
- **Build from a node:** hover at its right edge (or focus/click its + button) to
  add a connected layer, connect to an existing layer, edit, or delete. Data
  source connections own their ordered transforms. Both kinds of backend wiring
  (feeds and links) appear as **connections**. Edit a data-source connection to
  choose its field, input/target role, and transform pipeline. Multiple feeds can
  share one source while selecting and transforming different CSV columns.
  Canvas feed labels show the field, role, and transform count. Older saved
  source pipelines migrate to each outgoing feed when opened.
- **Graph tab:** lists data sources, layers, and all connections. Select a node
  to locate it; edit or delete items here. Deleting a node removes its attached
  connections. Selected canvas items can also be removed with Delete/Backspace.
  Files used by the current draft or any saved network are protected from
  deletion until their source nodes are removed and the affected networks saved.

Canvas edits update the current in-memory draft through `GET/PUT /project`.
Use **File → Save** to persist a network across backend restarts; a restart opens
a fresh empty draft, and saved networks remain available through the File menu.
Each `NetworkSpec` has a UUID `id` and an independent `title`. Clearing the canvas
preserves both; creating a new network or using Save as allocates a new ID.

`GET /networks` lists saved networks; `PUT /networks/{id}` saves one atomically;
`POST /networks` saves a copy; `GET /networks/{id}` reads a saved snapshot; and
`POST /networks/{id}/open` loads it into the current draft. Snapshots are stored in
`workspaces/networks/`, configurable with `NEURALSANDBOX_NETWORK_DIR`. They contain
configuration and references to shared files, not copies of uploaded data.
Uploaded file bytes and metadata also persist across restarts.

The backend must be running to edit. Failed saves show an error and preserve the
previous disk snapshot. Set `VITE_API_URL` if the API is not at
`http://localhost:8000`.
Python executes transforms and forward passes; the browser presents their results.
Training is available through the Build / Train switch.

## Shapes and forward execution (v0)

Sources expose each transformed field's dtype, sample shape and example count.
Adding a layer from a source defaults to **Auto from input feed** sizing: flattened
MNIST images resolve to 784 neurons. Auto sizing follows changes to the source,
field and transforms; a weighted connection to a hidden layer does not determine
that layer's width. Existing saved layers default to Manual sizing. Scalar numeric
fields supply one feature; matrix/image fields need Flatten before a feed.

The canvas shows resolved widths and weight-matrix dimensions. The **Run** tab
lets you select an example by zero-based index, Previous/Next or Random, choose
a weight seed, and run a forward pass. Outputs use untrained weights, recreated
deterministically from the seed for each request. Input layers take their feed
values directly; their activation is not applied. Hidden/output activations support
identity, ReLU, sigmoid, tanh and numerically stable softmax. Multiple weighted
predecessors contribute by addition, as in the core.

`POST /project/analysis` accepts a `ProjectSpec` and returns source schemas,
resolved layer widths, link shapes and node/connection-specific issues.
`POST /executions/forward` accepts `{project, sample_index, seed}` and returns
output values, resolved shapes, an execution ID, and project/model/data revisions.
Forward execution does not mutate the draft or save weights. Node positions and
the document title do not affect computation revisions. Data revisions identify
the transformed input data; model revisions identify the configuration and actual
parameters. Results from an earlier graph configuration are labelled in the UI.

Every root needs one input feed. Multiple input feeds into one layer, or an input
feed combined with weighted predecessors, are rejected. Input sources must have
equal example counts and aligned rows; matching counts cannot establish semantic
alignment, so pairing rows correctly is the user's responsibility. Missing fields,
non-numeric feeds, incompatible manual widths, empty datasets, invalid indices,
unsupported activations, and non-finite inputs/results produce actionable errors.
The v0 runtime is limited to two million parameters and 100,000 returned output
values; the UI initially displays the first 50 values per output.

Analysis and execution currently load and transform data on each request, reusing
the loaded source within that request. No cross-request dataset cache or streaming
loader is implemented yet. Computation completes before returning a result;
traces, lazy trace retrieval, playback, training and checkpoints are future work.


## Training workspace

Build / Train switches the right-hand pane and sidebar while retaining the header
and graph viewport. Connect inputs and expected targets with the existing feed
editor. Transforms and bindings save with the network, as do sampling, batch size,
epochs, seed, loss, and validation settings. Corresponding rows must describe the
same example; training checks counts and shapes but cannot infer row semantics.
Use Preview paired examples to inspect the first three pairs.

Train supports sequential or shuffled batches, mean squared error, and
softmax cross-entropy, with an optional fixed validation holdout. Updates use the
core's existing fixed unit gradient step. No optimiser or learning-rate selector
is exposed; convergence is not guaranteed for arbitrary networks and data scales.

One local worker owns the run and weights. SSE sends authoritative snapshots;
REST controls pause, resume and cancellation at batch/evaluation boundaries.
Closing the page does not cancel training. Large source loads cannot be interrupted
mid-operation. Use one API server worker. Restarted runs are marked Interrupted;
automatic restart-resume is not supported.

Runs, configuration snapshots, split membership, sampled metric history (the live snapshot retains
2,000 recent observations; `metrics.jsonl` retains the recorded history), and latest/best NumPy checkpoints persist under
`workspaces/runs/` (`NEURALSANDBOX_RUN_DIR` overrides it). Run-referenced uploaded
files are protected from deletion. The Build sidebar's Run tab can execute one
example using a saved checkpoint or fresh seeded weights.

`POST /training/runs` starts a run; `GET /training/runs?network_id=…` lists runs;
`GET /training/runs/{id}` reads state; `/events` streams snapshots. POST to
`/pause`, `/resume`, or `/cancel` controls that run. Reconnecting receives a fresh
snapshot rather than replaying individual events. `POST /training/preview`
validates the setup and returns paired examples.

MNIST input feeds should include **Flatten → Scale (divisor 255)**; label feeds
should use One hot. Scaling is explicit and never applied behind the editor's
back. Large input ranges produce a training warning. Cross-entropy is computed
from logits with log-softmax and a fused output gradient, so confidently wrong
predictions retain a gradient even when softmax probabilities underflow to zero.
Older runs/checkpoints retain their original results; start a fresh run after
correcting input transforms.

## Network assistant

The floating assistant beside Build / Train runs in the browser and calls
OpenRouter directly. Set `OPENROUTER_API_KEY` in `frontend/.env` (a repository-root
`.env` also works); optionally set `OPENROUTER_MODEL` to a tool-capable model ID.
The default is `openrouter/auto`. Restart Vite after changing environment values.
Settings in the assistant let you override the key and model for the current
page session. Environment-provided keys are included in the client bundle: use
this setup locally, not for a public deployment with a shared secret.

The frontend separates the OpenRouter adapter, bounded agent loop, tool registry,
conversation state, and floating UI under `frontend/src/assistant/`. Each turn
repeats model → sequential tool execution → tool observations until a final reply,
cancellation, or the 12-round / 48-call limit. Tool errors are returned to the model
so it can correct its request. OpenRouter reasoning metadata is replayed without
showing private reasoning in the UI. The conversation shows tool execution status;
**Run logs** in the assistant header opens the detailed diagnostic history.

For small changes, `edit_project` applies JSON Patch operations to the last
inspected draft, retaining all unrelated values. `update_project` remains available
for complete replacements, but rejects missing network/configuration fields before
they can be filled with backend defaults. Configuration conflicts require a fresh
inspection; layout-only moves are merged while preserving user positions. Three
identical failed tool requests stop the loop early.

Tools use the existing `/files`, `/node-types`, `/project/analysis`, and `/project`
endpoints. The project tool schema comes from `/openapi.json`. Changes are validated
by project analysis and committed atomically with `PUT /project`, then reconciled
with the canvas. No new backend agent endpoint or agent framework is required.
The canvas remains interactive during inference. Writes require the latest
inspected snapshot; conflicts tell the agent to inspect again. Undo is available
while the canvas still matches the assistant's latest commit and never crosses
an intervening manual edit. Cancelling stops inference and pending tools; an
already-started write is reconciled before finishing. Switching networks cancels
the conversation. Saved networks and file deletion are not exposed as tools.

Verification: `npm test`, `npm run build`, and `npm run lint` from `frontend/`.
Tests cover tool observations, recoverable errors, cancellation, loop limits,
conflicting edits, validation, schema expansion, and the OpenRouter wire format.

### Assistant run logs

Open the assistant and select **Run logs**, or **View run log** beneath a new
request. Expand events to inspect tool arguments, results, errors, timestamps,
and durations. Model events also include the request body and provider response
(including usage, finish reasons, response IDs and HTTP errors when returned).
Backend validation errors preserve status codes and field locations. Project
commits and Undo are recorded separately. Filters show tool, model, backend,
commit, or unsuccessful events; **Export run** / **Export all** downloads JSON.

Logs are stored locally in this browser, independently of chat history and
network switching. Capture begins with requests made after this feature was
added; earlier activity cannot be reconstructed. Reloading marks unfinished
runs interrupted. **New chat** retains logs; **Clear finished logs** removes
completed history while preserving active runs. Storage is bounded to 20 runs,
roughly 1.5 million serialized characters overall and 240,000 per run. Individual
details truncate after 16,000 characters, and omitted events are indicated.
If browser storage is unavailable, logs remain in memory with an export reminder.
API keys and authorization fields are redacted before storage; private reasoning
is omitted. Logs do contain prompts, network configuration and tool results.

### Combining input columns

A feed delivers only its **Field**. **Select** retains named columns for later
transforms; it does not join them. To build one input vector from several CSV
columns, cast/scale numeric fields and one-hot encode categories first, then add
**Concatenate** with the fields in the desired order and an output field such as
`features`. Set the feed's Field to `features` and the input layer to Auto size.
Five scalar fields plus 9- and 15-column one-hot fields produce 29 input features.
Concatenate requires numeric scalars or vectors; flatten images/matrices first.
It preserves source fields and leaves other feeds unchanged.

`POST /project/analysis` now includes a `feeds` report with each valid feed's
selected field, delivered width, and transformed field schemas. The assistant can
inspect these directly instead of guessing widths or changing CSV parsing options.
Invisible BOM characters are escaped as `\uFEFF` in diagnostic logs.
