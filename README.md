# ml — a neural network you can see

A learning project. The end goal is a visual builder for neural networks in the
browser; right now it's raw NumPy and notebooks.

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

**`visualiser/objects.py`** — the generalisation. Once I'd struggled through
gradient descent for one fixed network, it was clear the shape of the code barely
matters — the _architecture_ does. So this is `Layer` / `Link` / `Network`:

- a network is an arbitrary DAG of layers, not just a stack
- the forward pass runs in topological order
- `Network.backprop` walks it in reverse and returns the gradients — it computes
  them; applying them is a separate step
- activation, the fan-in combinator, and the cost are all passed in alongside
  their derivatives; nothing is hard-coded
- backprop is checked against numerical gradients

There's no training loop attached to `Network` yet. That, and anything visual, is
still ahead.

## Where it's going

The point isn't the code. The goal is a **visual builder** in the browser: drag
layer blocks onto a canvas, wire them together, press train, and watch the data
move through the graph. The architecture is the thing you touch directly.
Optimiser details are there if you open them and folded away if you don't. It
abstracts away just the parts that were boilerplate to begin with — and nothing
else.

**Next:** static rendering — draw a `Network` as a diagram (layers, links,
shapes) before making any of it interactive.

## Running it

```
uv sync
```

Then open `notebooks/mnist_nn.ipynb` in your editor. The MNIST IDX files should live in
`mnist_dataset/`.
