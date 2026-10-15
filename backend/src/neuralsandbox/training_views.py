"""Bounded browser projections; full evaluations remain in the run journal."""
from typing import Any
import math


def scalar_metric(record: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in record.items() if key not in {'training', 'validation'}}
    for split in ('training', 'validation'):
        if split in record:
            evaluation = record[split]
            result[split] = {**evaluation, 'outputs': {
                name: {key: value for key, value in output.items() if key not in {'per_class', 'confusion_matrix'}}
                for name, output in evaluation['outputs'].items()}}
    return result


def sample_metrics(records: list[dict[str, Any]], budget: int, metric: str = 'loss', output: str = '') -> list[dict[str, Any]]:
    """Bound points while retaining each displayed series' extrema per time bucket."""
    assert budget >= 8, 'Curve budget must allow endpoints and series extrema'
    merged: dict[int, dict[str, Any]] = {}
    for record in records:
        step = record['step']
        merged[step] = {**merged.get(step, {}), **record}
    points = sorted(merged.values(), key=lambda record: record['step'])
    if len(points) <= budget:
        return points

    def values(record: dict[str, Any]) -> list[float | None]:
        scores = [record.get('loss') if metric == 'loss' else None]
        for split in ('training', 'validation'):
            evaluation = record.get(split, {})
            score = (evaluation.get('outputs', {}).get(output, {}) if output else evaluation).get(metric)
            if split == 'validation' and not evaluation:
                score = record.get('validation_loss' if metric == 'loss' else 'accuracy' if metric == 'accuracy' else '')
            scores.append(score if isinstance(score, (int, float)) and math.isfinite(score) else None)
        return scores

    selected = {0, len(points) - 1}
    buckets = max(1, (budget - 2) // 6)
    width = math.ceil(len(points) / buckets)
    for start in range(0, len(points), width):
        candidates = [(index, values(points[index])) for index in range(start, min(start + width, len(points)))]
        for series in range(3):
            available = [(index, scores[series]) for index, scores in candidates if scores[series] is not None]
            if available:
                selected.add(min(available, key=lambda pair: pair[1])[0])
                selected.add(max(available, key=lambda pair: pair[1])[0])
    return [points[index] for index in sorted(selected)]


def browser_run(record: dict[str, Any]) -> dict[str, Any]:
    """Send small live curves and only one detailed evaluation, never split membership."""
    result = {key: value for key, value in record.items() if key not in {'metrics', 'train_indices', 'validation_indices'}}
    for split in ('train', 'validation'):
        if f'{split}_indices' in record:
            result[f'{split}_count'] = len(record[f'{split}_indices'])
    result['metrics'] = [scalar_metric(metric) for metric in record.get('metrics', [])[-200:]]
    return result
