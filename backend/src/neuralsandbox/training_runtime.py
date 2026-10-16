"""Prepared, row-aligned training data and a run-owned numerical model."""
import hashlib
from collections.abc import Callable
from typing import Any
import numpy as np
from .core import Layer, Network
from .execution import ACTIVATIONS, _prepare, MAX_PARAMETERS
from .schemas import ProjectSpec
from .training_metrics import OutputAccumulator, hard_targets


def activation_vjp(name: str, z: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    a = ACTIVATIONS[name](z)
    if name == "softmax":
        return a * (gradient - (gradient * a).sum(axis=1, keepdims=True))
    derivative = {"identity": lambda: 1, "relu": lambda: z > 0,
                  "sigmoid": lambda: a * (1-a), "tanh": lambda: 1-a*a}
    return gradient * derivative[name]()


class TrainingRuntime:
    def __init__(self, project: ProjectSpec, initialize: bool = True) -> None:
        report, _, bindings, sizes = _prepare(project)
        assert not report['issues'], '; '.join(i['message'] for i in report['issues'])
        assert project.network.links, 'Add a trainable connection'
        assert sum(sizes[l.source]*sizes[l.target] for l in project.network.links)+sum(sizes.values()) <= MAX_PARAMETERS, 'Network is too large'

        self.project = project
        rng = np.random.default_rng(project.training.seed)
        self.layers = {s.id: Layer(ACTIVATIONS[s.activation], None, from_array=np.zeros(sizes[s.id]),
            bias=np.zeros((1, sizes[s.id])),
            initialize=initialize, activation_vjp=lambda z, g, name=s.activation: activation_vjp(name, z, g)) for s in project.network.layers}

        self.network = Network(self.cost, self.derivative, list(self.layers.values()))
        for link in project.network.links:
            shape = (sizes[link.source], sizes[link.target])
            fan_in = sum(sizes[item.source] for item in project.network.links if item.target == link.target)
            method = project.training.initialization
            weights = np.array(0)
            if initialize:
                if method == "uniform":
                    weights = rng.uniform(-0.5, 0.5, shape)
                else:
                    variance = 2 / fan_in if method == "he_normal" else 2 / (fan_in + sizes[link.target])
                    weights = rng.normal(0, np.sqrt(variance), shape)
            self.network.connect(self.layers[link.source], self.layers[link.target], weights, initialize=initialize)

        self.network.calculate_io()
        self.inputs, self.targets = {}, {}
        self.warnings: list[str] = []
        digest = hashlib.sha256()
        for spec in project.feeds:
            feed = bindings[spec.id][0]
            assert feed.feature_count == sizes[spec.layer], f"{spec.id}: transformed width does not match layer"
            values = np.asarray(feed.data.fields[spec.field], dtype=np.float64).reshape(-1, sizes[spec.layer])
            assert np.isfinite(values).all(), f'{spec.id}: non-finite data'
            if spec.role == 'input' and values.size and np.max(np.abs(values)) > 10:
                self.warnings.append(
                    f"{spec.id}: input range is {values.min():g} to {values.max():g}. "
                    "Large input magnitudes can destabilise training. "
                    "For MNIST pixels, add a Scale transform with divisor 255 to the input feed."
                )
            destination = self.inputs if spec.role == 'input' else self.targets
            layer = self.layers[spec.layer]
            assert layer not in destination, f'{spec.layer}: duplicate {spec.role} feed'
            destination[layer] = values
            digest.update(spec.id.encode()); digest.update(values.tobytes())

        assert set(self.inputs) == set(self.network.inputs), 'Every input needs one input feed'
        assert set(self.targets) == set(self.network.outputs), 'Every output needs one target feed'
        counts = {len(v) for v in [*self.inputs.values(), *self.targets.values()]}
        assert len(counts)==1 and next(iter(counts))>0, 'Inputs and targets must have equal nonzero row counts'
        self.count = next(iter(counts)); self.data_revision = digest.hexdigest()

        names = {v:k for k,v in self.layers.items()}
        self.losses = [project.training.losses.get(names[l], 'cross_entropy' if next(s.activation for s in project.network.layers if s.id==names[l])=='softmax' else 'mse') for l in self.network.outputs]
        for layer, loss in zip(self.network.outputs, self.losses):
            if loss=='cross_entropy':
                assert next(s.activation for s in project.network.layers if s.id==names[layer])=='softmax', 'Cross entropy requires softmax output'
                y=self.targets[layer]
                assert (y>=0).all() and np.allclose(y.sum(axis=1),1), 'Cross entropy targets must be one-hot or probability vectors'

        indices = np.random.default_rng(project.training.seed).permutation(self.count)
        n = int(self.count * project.training.validation_fraction)
        assert project.training.validation_fraction==0 or n>0, 'Not enough examples for the selected validation fraction'
        self.validation, self.train = indices[:n], indices[n:]
        assert len(self.train)>0, 'Training partition is empty'

    def cost(self, predictions: list[np.ndarray], targets: list[np.ndarray]) -> float:
        losses = []
        for layer, prediction, target, loss in zip(self.network.outputs, predictions, targets, self.losses):
            if loss == 'cross_entropy':
                # Log-softmax stays finite even when a probability underflows to zero.
                shifted = layer.pre_activation - layer.pre_activation.max(axis=1, keepdims=True)
                log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
                losses.append(-(target * log_probabilities).sum(axis=1).mean())
            else:
                losses.append(np.square(prediction - target).mean())
        return float(np.mean(losses))

    def derivative(self, predictions: list[np.ndarray], targets: list[np.ndarray]) -> list[np.ndarray]:
        # Cross-entropy is differentiated directly with respect to output logits.
        return [(np.zeros_like(p) if loss == 'cross_entropy' else 2*(p-y)/p.size)/len(predictions)
                for p, y, loss in zip(predictions, targets, self.losses)]

    def output_delta(self, layer: Layer, targets: np.ndarray) -> np.ndarray | None:
        index = self.network.outputs.index(layer)
        if self.losses[index] == 'cross_entropy':
            return (layer.values - targets) / (len(targets) * len(self.network.outputs))
        return None

    def step(self, indices: np.ndarray, update: bool) -> tuple[float, float | None]:
        inputs={l:v[indices] for l,v in self.inputs.items()}; targets={l:v[indices] for l,v in self.targets.items()}
        if update:
            loss, weights, biases = self.network.backprop(inputs, targets, self.output_delta)
            for link, gradient in weights.items(): link.weights -= self.project.training.learning_rate * gradient
            for layer, gradient in biases.items(): layer.bias -= self.project.training.learning_rate * gradient
        else:
            for layer, values in inputs.items(): layer.values=values
            for layer in self.network.topo_order(): layer.compute()
            loss=self.cost([l.values for l in self.network.outputs],[targets[l] for l in self.network.outputs])

        assert np.isfinite(loss) and all(np.isfinite(l.weights).all() for l in self.network.links), 'Training produced non-finite values with the configured learning rate'
        accuracy = float(np.mean([np.mean(l.values.argmax(axis=1)==targets[l].argmax(axis=1)) for l in self.network.outputs])) if all(loss == 'cross_entropy' and hard_targets(targets[layer]) for layer, loss in zip(self.network.outputs, self.losses)) else None
        return float(loss), accuracy

    def evaluate(self, indices: np.ndarray, boundary: Callable[[], bool]) -> dict[str, Any] | None:
        """Evaluate a partition without changing parameters; stop at cooperative boundaries."""
        assert len(indices) > 0, 'Evaluation partition must not be empty'
        names = {layer: name for name, layer in self.layers.items()}
        accumulators = [OutputAccumulator('classification' if loss == 'cross_entropy' else 'regression', self.targets[layer].shape[1])
                        for layer, loss in zip(self.network.outputs, self.losses)]
        total_loss = 0.

        for offset in range(0, len(indices), self.project.training.batch_size):
            if not boundary(): return None

            batch = indices[offset:offset + self.project.training.batch_size]
            loss, _ = self.step(batch, False)
            total_loss += loss * len(batch)
            for layer, objective, accumulator in zip(self.network.outputs, self.losses, accumulators):
                target = self.targets[layer][batch]
                if objective == 'cross_entropy':
                    shifted = layer.pre_activation - layer.pre_activation.max(axis=1, keepdims=True)
                    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
                    output_loss = float(-(target * log_probabilities).sum(axis=1).mean())
                else:
                    output_loss = float(np.square(layer.values - target).mean())
                accumulator.add(layer.values, target, output_loss)
        return {'loss': total_loss / len(indices), 'count': len(indices),
                'outputs': {names[layer]: accumulator.result() for layer, accumulator in zip(self.network.outputs, accumulators)}}

    def weights(self) -> dict[str, np.ndarray]:
        return {**{f'b:{id}':l.bias for id,l in self.layers.items()}, **{f'w:{s.id}':link.weights for s in self.project.network.links for link in self.layers[s.source].to_links if link.to_layer is self.layers[s.target]}}
