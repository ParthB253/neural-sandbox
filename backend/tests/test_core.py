"""Backprop vs. finite-difference gradients on a non-trivial graph."""
import numpy as np

from neuralsandbox.core import Layer, Network

sig = lambda z: 1 / (1 + np.exp(-z))
d_sig = lambda z: sig(z) * (1 - sig(z))
ident = lambda z: z
d_ident = lambda z: np.ones_like(z)


def _mse(preds, targets):
    return 0.5 * sum(np.sum((p - t) ** 2) for p, t in zip(preds, targets))


def _d_mse(preds, targets):
    return [p - t for p, t in zip(preds, targets)]


def test_backprop_matches_numerical_gradient():
    np.random.seed(0)

    # diamond: a -> b -> {c, d} -> e  (fan-out at b, fan-in at e)
    a = Layer(ident, d_ident, from_array=np.zeros(3))
    b = Layer(sig, d_sig, size=4)
    c = Layer(sig, d_sig, size=3)
    d = Layer(sig, d_sig, size=2)
    e = Layer(ident, d_ident, size=2)
    net = Network(_mse, _d_mse, [a, b, c, d, e])
    net.connect(a, b)
    net.connect(b, c)
    net.connect(b, d)
    net.connect(c, e)
    net.connect(d, e)

    x = {a: np.random.rand(1, 3)}
    y = {e: np.random.rand(1, 2)}
    _, grad_w, grad_b = net.backprop(x, y)

    def loss_only():
        for layer in net.inputs:
            layer.values = x[layer]
        for layer in net.topo_order():
            layer.compute()
        return net.cost([o.values for o in net.outputs], [y[o] for o in net.outputs])

    eps = 1e-6

    for link in net.links:
        W = link.weights
        num = np.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i, j] += eps; hi = loss_only()
                W[i, j] -= 2 * eps; lo = loss_only()
                W[i, j] += eps
                num[i, j] = (hi - lo) / (2 * eps)
        assert np.allclose(num, grad_w[link], atol=1e-6)

    for layer in (b, c, d, e):
        bias = layer.bias
        num = np.zeros_like(bias)
        for j in range(bias.shape[1]):
            bias[0, j] += eps; hi = loss_only()
            bias[0, j] -= 2 * eps; lo = loss_only()
            bias[0, j] += eps
            num[0, j] = (hi - lo) / (2 * eps)
        assert np.allclose(num, grad_b[layer], atol=1e-6)
