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

- No PyTorch, no TensorFlow. Just vectors and NumPy.
- Clarity over speed. The optimisation is left bad on purpose, so long as every line
  is readable and you can see why it's there.

Start with the simplest thing that works — a feed-forward net — and write it out
by hand.

## Where it's at

**`notebooks/mnist_nn.ipynb`** — the hand-written version. A fixed
`[784, 16, 16, 10]` feed-forward net: ReLU hidden layers, softmax output,
cross-entropy cost, backprop written out term by term, plain gradient descent.
On MNIST it reaches **~95.4% test accuracy after 10 training iterations**. Slow,
but every line is legible.

**`backend/src/neuralsandbox/core.py`** — the generalisation. Once I'd struggled through
gradient descent for one fixed network, it was clear the shape of the code barely
matters — the _architecture_ does. So this is `Layer` / `Link` / `Network`:

- a network is an arbitrary DAG of layers, not just a stack
- the forward pass runs in topological order
- `Network.backprop` walks it in reverse and returns the gradients — it computes
  them; applying them is a separate step
- activation, the fan-in combinator, and the cost are all passed in alongside
  their derivatives; nothing is hard-coded
- backprop is checked against numerical gradients

`Network.train` is deliberately bare: it applies the gradients directly, without
an optimiser hiding the mechanics. The visual editor supports topology, data
sources, connections, transforms, and named network documents; it does not run
the pipeline or train a network yet.

## Where it's going

The point isn't the code. The goal is a **visual builder** in the browser: drag
layer blocks onto a canvas, wire them together, press train, and watch the data
move through the graph. The architecture is the thing you touch directly.
Optimiser details are there if you open them and folded away if you don't. It
abstracts away just the parts that were boilerplate to begin with — and nothing
else.

**Next:** connect the editable pipeline to network execution and training.

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
This remains a topology/configuration editor: transforms and training are not
executed by the browser.
