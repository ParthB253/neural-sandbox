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
input shapes and execute a single forward pass; training is not wired into the
editor yet.

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
  sources also offer an ordered transform editor. Both kinds of backend wiring
  (feeds and links) appear as **connections**. A data-source connection lets you
  choose a field and whether it supplies inputs or training targets.
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
Training is not yet available in the editor.

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
