"""Whole-partition metrics accumulated from fixed-weight evaluation batches."""
from typing import Any
import numpy as np


def hard_targets(targets: np.ndarray) -> bool:
    """Discrete metrics require one-hot targets, rather than probability labels."""
    return bool(np.all((targets == 0) | (targets == 1)) and np.all(targets.sum(axis=1) == 1))


class OutputAccumulator:
    def __init__(self, kind: str, width: int) -> None:
        self.kind = kind
        self.width = width
        self.count = 0
        self.loss_sum = 0.
        self.discrete = True
        self.support = np.zeros(width, dtype=np.int64)
        self.predicted = np.zeros(width, dtype=np.int64)
        self.correct = np.zeros(width, dtype=np.int64)
        self.confusion = np.zeros((width, width), dtype=np.int64) if kind == 'classification' and width <= 30 else None
        self.absolute_error = 0.
        self.squared_error = 0.
        self.target_mean = np.zeros(width)
        self.target_m2 = np.zeros(width)

    def add(self, predictions: np.ndarray, targets: np.ndarray, loss: float) -> None:
        assert predictions.shape == targets.shape, 'Predictions and targets must have matching shapes'
        assert len(targets) > 0, 'Metric batches must contain examples'

        count = len(targets)
        self.loss_sum += loss * count
        if self.kind == 'classification':
            self.discrete &= hard_targets(targets)
            if self.discrete:
                actual, predicted = targets.argmax(axis=1), predictions.argmax(axis=1)
                self.support += np.bincount(actual, minlength=self.width)
                self.predicted += np.bincount(predicted, minlength=self.width)
                self.correct += np.bincount(actual[actual == predicted], minlength=self.width)
                if self.confusion is not None:
                    np.add.at(self.confusion, (actual, predicted), 1)
        else:
            errors = predictions - targets
            self.absolute_error += float(np.abs(errors).sum())
            self.squared_error += float(np.square(errors).sum())
            # Parallel variance combination avoids cancellation for nearly constant targets.
            mean = targets.mean(axis=0)
            delta = mean - self.target_mean
            total = self.count + count
            self.target_m2 += np.square(targets - mean).sum(axis=0) + delta**2 * self.count * count / total
            self.target_mean += delta * count / total

        self.count += count

    def result(self) -> dict[str, Any]:
        assert self.count > 0, 'Metrics require at least one example'
        result: dict[str, Any] = {'kind': self.kind, 'loss': self.loss_sum / self.count}
        if self.kind == 'classification':
            if not self.discrete:
                return {**result, 'accuracy': None, 'unavailable_reason': 'Discrete class metrics require one-hot targets; this partition contains soft probability targets.'}
            # Undefined class denominators remain null; macro scores use zero for them.
            precision = np.divide(self.correct, self.predicted, out=np.zeros(self.width), where=self.predicted > 0)
            recall = np.divide(self.correct, self.support, out=np.zeros(self.width), where=self.support > 0)
            f1 = np.divide(2 * self.correct, self.support + self.predicted, out=np.zeros(self.width), where=(self.support + self.predicted) > 0)
            result.update(accuracy=float(self.correct.sum() / self.count), precision_macro=float(precision.mean()),
                          recall_macro=float(recall.mean()), f1_macro=float(f1.mean()),
                          f1_weighted=float(np.dot(f1, self.support) / self.count),
                          balanced_accuracy=float(recall.mean()) if (self.support > 0).all() else None,
                          per_class=[{'index': i, 'label': f'Class {i}', 'support': int(self.support[i]),
                                      'precision': float(precision[i]) if self.predicted[i] else None,
                                      'recall': float(recall[i]) if self.support[i] else None,
                                      'f1': float(f1[i]) if self.support[i] + self.predicted[i] else None} for i in range(self.width)])
            if self.confusion is not None:
                result['confusion_matrix'] = self.confusion.tolist()
        else:
            denominator = float(self.target_m2.sum())
            result.update(mae=self.absolute_error / (self.count * self.width),
                          rmse=float(np.sqrt(self.squared_error / (self.count * self.width))),
                          r2=1 - self.squared_error / denominator if denominator > 0 else None)
        return result
