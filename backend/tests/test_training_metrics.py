"""Split metrics use whole-partition sufficient statistics and explicit missing values."""
import numpy as np
import pytest
from neuralsandbox.training_metrics import OutputAccumulator
from neuralsandbox.training_runtime import TrainingRuntime
from neuralsandbox.training import RunManager, StartRequest
from test_training import project


def test_class_metrics_match_confusion_counts_and_uneven_batches():
    targets = np.eye(3)[[0, 0, 0, 1, 1, 2]]
    predictions = np.eye(3)[[0, 0, 1, 1, 2, 2]]
    whole = OutputAccumulator('classification', 3)
    whole.add(predictions, targets, .4)
    batched = OutputAccumulator('classification', 3)
    for start, end in [(0, 1), (1, 4), (4, 6)]:
        batched.add(predictions[start:end], targets[start:end], .4)
    actual = batched.result()
    assert actual == whole.result()
    assert actual['confusion_matrix'] == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]
    assert actual['accuracy'] == pytest.approx(4/6)
    assert actual['precision_macro'] == pytest.approx(2/3)
    assert actual['recall_macro'] == pytest.approx((2/3 + 1/2 + 1)/3)
    assert actual['f1_macro'] == pytest.approx((.8 + .5 + 2/3)/3)
    assert actual['f1_weighted'] == pytest.approx((.8*3 + .5*2 + 2/3)/6)


def test_missing_classes_and_soft_labels_remain_explicit():
    metric = OutputAccumulator('classification', 3)
    metric.add(np.eye(3)[[0, 0]], np.eye(3)[[0, 1]], .4)
    result = metric.result()
    assert result['balanced_accuracy'] is None
    assert result['per_class'][1]['precision'] is None
    assert result['per_class'][2]['recall'] is None
    assert result['per_class'][2]['f1'] is None
    metric.add(np.array([[.3, .3, .4]]), np.array([[.5, .5, 0]]), .9)
    assert metric.result()['accuracy'] is None
    assert 'per_class' not in metric.result()
    assert 'soft probability' in metric.result()['unavailable_reason']


def test_regression_aggregation_and_constant_targets():
    targets = np.array([[1., 2.], [3., 6.], [4., 8.]])
    predictions = targets + np.array([[1., -1.], [2., 0.], [0., 2.]])
    whole = OutputAccumulator('regression', 2)
    whole.add(predictions, targets, 10/6)
    batched = OutputAccumulator('regression', 2)
    batched.add(predictions[:1], targets[:1], 1)
    batched.add(predictions[1:], targets[1:], 2)
    assert batched.result() == pytest.approx(whole.result())
    assert whole.result()['mae'] == 1
    assert whole.result()['rmse'] == pytest.approx(np.sqrt(10/6))
    constant = OutputAccumulator('regression', 1)
    constant.add(np.zeros((2, 1)), np.ones((2, 1)), 1)
    assert constant.result()['r2'] is None


def test_evaluation_preserves_weights_and_is_batch_independent(tmp_path):
    runtime = TrainingRuntime(project(tmp_path))
    before = {key: value.copy() for key, value in runtime.weights().items()}
    actual = runtime.evaluate(runtime.train, lambda: True)
    runtime.project.training.batch_size = 100
    whole = runtime.evaluate(runtime.train, lambda: True)
    assert actual['loss'] == pytest.approx(whole['loss'])
    assert actual['outputs']['o']['accuracy'] == whole['outputs']['o']['accuracy']
    assert actual['outputs']['o']['confusion_matrix'] == whole['outputs']['o']['confusion_matrix']
    for key, value in before.items():
        np.testing.assert_array_equal(value, runtime.weights()[key])
    assert runtime.evaluate(runtime.train, lambda: False) is None


def test_final_evaluation_without_validation_and_off_cadence(tmp_path, monkeypatch):
    import neuralsandbox.training as service
    monkeypatch.setattr(service, 'RUN_DIR', tmp_path/'runs')
    local = RunManager()
    p = project(tmp_path, epochs=3)
    p.training.validation_fraction = 0
    p.training.evaluation_every = 2
    run = local.start(StartRequest(project=p))
    local.worker.join(10)
    run = local.get(run['id'])
    assert run['status'] == 'completed', run.get('error')
    evaluation = run['evaluation_summary']
    assert evaluation['epoch'] == 3
    assert evaluation['step'] == 12
    assert evaluation['training']['count'] == 8
    assert 'validation' not in evaluation
    assert [m['epoch'] for m in run['metrics'] if 'training' in m] == [2, 3]
    assert run['checkpoint_details']['latest']['step'] == 12
    assert run['completed_at'] >= run['created_at']


def test_history_ranges_recovery_and_legacy_snapshots(tmp_path, monkeypatch):
    import neuralsandbox.training as service
    monkeypatch.setattr(service, 'RUN_DIR', tmp_path/'runs')
    local = RunManager()
    run = local.start(StartRequest(project=project(tmp_path)))
    local.worker.join(10)
    id = run['id']
    full = local.history(id)
    assert full['total'] == len(full['metrics'])
    window = local.history(id, 4, 6)
    assert window['total'] == full['total']
    assert all(4 <= metric['step'] <= 6 for metric in window['metrics'])
    path = service.RUN_DIR/id/'metrics.jsonl'
    with path.open('a') as journal:
        journal.write('{"step":')
    assert local.history(id) == full
    path.unlink()
    assert local.history(id)['metrics'] == full['metrics']
    with pytest.raises(service.HTTPException) as error:
        local.history(id, 8, 2)
    assert error.value.status_code == 422


def test_history_endpoint_validation(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from neuralsandbox.api import app
    import neuralsandbox.training as service
    monkeypatch.setattr(service, 'RUN_DIR', tmp_path/'runs')
    local = RunManager()
    monkeypatch.setattr(service, 'manager', local)
    run = local.start(StartRequest(project=project(tmp_path)))
    local.worker.join(10)
    client = TestClient(app)
    assert client.get(f"/training/runs/{run['id']}/metrics").json()['total'] > 0
    assert client.get(f"/training/runs/{run['id']}/metrics?start_step=-1").status_code == 422
    assert client.get('/training/runs/absent/metrics').status_code == 404


def test_mixed_outputs_and_soft_targets(tmp_path):
    data = project(tmp_path).model_dump(mode='json')
    data['network']['layers'].append({'id': 'regression', 'size': 2, 'activation': 'identity'})
    data['network']['links'].append({'id': 'second', 'source': 'i', 'target': 'regression'})
    data['feeds'].append({'id': 'second-target', 'dataset': 'y', 'field': 'value', 'layer': 'regression', 'role': 'target'})
    from neuralsandbox.schemas import ProjectSpec
    runtime = TrainingRuntime(ProjectSpec.model_validate(data))
    actual = runtime.evaluate(runtime.train, lambda: True)
    assert actual['outputs']['o']['kind'] == 'classification'
    assert actual['outputs']['regression']['kind'] == 'regression'
    assert actual['outputs']['regression']['rmse'] >= 0
    assert runtime.step(runtime.train, False)[1] is None
    output = runtime.layers['o']
    runtime.targets[output][:] = [.25, .75]
    actual = runtime.evaluate(runtime.train, lambda: True)
    assert actual['outputs']['o']['accuracy'] is None
    assert runtime.step(runtime.train, False)[1] is None


def test_final_validation_best_metadata_and_evaluation_cancel(tmp_path, monkeypatch):
    import neuralsandbox.training as service
    monkeypatch.setattr(service, 'RUN_DIR', tmp_path/'runs')
    local = RunManager()
    p = project(tmp_path, epochs=3)
    p.training.evaluation_every = 2
    run = local.start(StartRequest(project=p))
    local.worker.join(10)
    run = local.get(run['id'])
    assert run['evaluation_summary']['epoch'] == 3
    assert run['evaluation_summary']['validation']['count'] == 2
    assert run['best_epoch'] in (2, 3)
    assert run['checkpoint_details']['best']['validation_loss'] == run['best_validation_loss']
    assert run['checkpoint_details']['best']['step'] == run['best_step']
    original = TrainingRuntime.evaluate

    def cancel_during_evaluation(runtime, indices, boundary):
        local.command(local.newest(str(p.network.id))['id'], 'cancel')
        return original(runtime, indices, boundary)

    monkeypatch.setattr(TrainingRuntime, 'evaluate', cancel_during_evaluation)
    cancelled = local.start(StartRequest(project=p))
    local.worker.join(10)
    cancelled = local.get(cancelled['id'])
    assert cancelled['status'] == 'cancelled'
    assert 'evaluation_summary' not in cancelled
    assert 'best' not in cancelled['checkpoints']


def test_history_reaches_observations_beyond_live_snapshot_cap(tmp_path, monkeypatch):
    import json
    import neuralsandbox.training as service
    monkeypatch.setattr(service, 'RUN_DIR', tmp_path/'runs')
    local = RunManager()
    observations = [{'step': index, 'loss': 1 / index} for index in range(1, 2501)]
    local.runs['recorded'] = {'id': 'recorded', 'metrics': observations[-2000:], 'sequence': 0}
    folder = service.RUN_DIR/'recorded'
    folder.mkdir(parents=True)
    path = folder/'metrics.jsonl'
    path.write_text(''.join(json.dumps(metric) + '\n' for metric in observations) + '{"step":')
    window = local.history('recorded', 1, 5)
    assert window['total'] == 2500
    assert window['metrics'] == observations[:5]
    local.publish('recorded', metrics=[{'step': 2501, 'loss': .001}])
    assert local.history('recorded')['total'] == 2501
    assert local.history('recorded', 2501, 2501)['metrics'][0]['loss'] == .001
