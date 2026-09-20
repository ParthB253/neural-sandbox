# nn-sandbox — a neural network you can see

A learning project. The end goal is a visual builder for neural networks in the
browser, with a React pipeline editor backed by raw NumPy and notebooks.

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

**`backend/src/nnsandbox/core.py`** — the generalisation. Once I'd struggled through
gradient descent for one fixed network, it was clear the shape of the code barely
matters — the _architecture_ does. So this is `Layer` / `Link` / `Network`:

- a network is an arbitrary DAG of layers, not just a stack
- the forward pass runs in topological order
- `Network.backprop` walks it in reverse and returns the gradients — it computes
  them; applying them is a separate step
- activation, the fan-in combinator, and the cost are all passed in alongside
  their derivatives; nothing is hard-coded
- backprop is checked against numerical gradients

There's no training loop attached to `Network` yet. The visual editor now supports
topology, data sources, connections, and transform configuration.

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
  src/nnsandbox/
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
uv run uvicorn nnsandbox.api:app --reload   # http://localhost:8000, docs at /docs
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
  Set `NNSANDBOX_UPLOAD_DIR` to use a different storage directory.
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
  Files in use are protected from deletion until their source nodes are removed.

Canvas edits are validated and saved atomically through `GET/PUT /project`.
They survive page reloads but, like the original API draft, reset when the
backend restarts. Uploaded file bytes and metadata persist across restarts.
The backend must be running to edit; failed saves keep the last saved graph and
show an error. Set `VITE_API_URL` if the API is not at `http://localhost:8000`.
This remains a topology/configuration editor: transforms and training are not
executed by the browser.
