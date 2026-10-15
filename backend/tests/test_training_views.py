"""Browser memory budgets and sampling preserve full records on the server."""
import json
from neuralsandbox.training_views import browser_run, sample_metrics, scalar_metric


def test_browser_snapshot_keeps_only_latest_class_details():
    evaluation = {'loss': .4, 'count': 2048, 'outputs': {'o': {
        'kind': 'classification', 'loss': .4, 'accuracy': .8,
        'per_class': [{'index': i, 'precision': .8, 'recall': .8, 'f1': .8, 'support': 10} for i in range(100)],
        'confusion_matrix': [[1] * 30 for _ in range(30)]}}}
    records = [{'step': step, 'training': evaluation, 'validation': evaluation} for step in range(2000)]
    run = {'metrics': records, 'evaluation_summary': records[-1], 'train_indices': list(range(100000)), 'validation_indices': [100000]}
    compact = browser_run(run)
    assert len(compact['metrics']) == 200
    assert compact['train_count'] == 100000
    assert compact['validation_count'] == 1
    assert 'train_indices' not in compact
    assert 'per_class' not in compact['metrics'][0]['training']['outputs']['o']
    assert compact['evaluation_summary']['training']['outputs']['o']['per_class'] == evaluation['outputs']['o']['per_class']
    assert len(json.dumps(compact)) < len(json.dumps(run)) / 100
    assert 'per_class' in run['metrics'][0]['training']['outputs']['o']


def test_sample_budget_and_zoom_preserve_extrema():
    records = [{'step': step, 'loss': 20 if step == 4567 else .4,
                'validation_loss': .1 if step == 7001 else .5} for step in range(100000)]
    records.append({'step': 4567, 'accuracy': .8})
    sampled = sample_metrics(records, 1200)
    assert len(sampled) <= 1200
    assert sampled[0]['step'] == 0
    assert sampled[-1]['step'] == 99999
    assert next(point for point in sampled if point['step'] == 4567)['loss'] == 20
    assert next(point for point in sampled if point['step'] == 4567)['accuracy'] == .8
    assert any(point['step'] == 7001 for point in sampled)
    window = sample_metrics([record for record in records if 4560 <= record['step'] <= 4570], 1200)
    assert len(window) == 11


def test_scalar_projection_and_sampling_follow_selected_output_metric():
    records = [{'step': step, 'validation': {'loss': .5, 'count': 10, 'outputs': {
        'o': {'accuracy': .99 if step == 1234 else .5, 'per_class': [{'index': 1}]}}}} for step in range(5000)]
    compact = [scalar_metric(record) for record in records]
    sampled = sample_metrics(compact, 100, 'accuracy', 'o')
    assert len(sampled) <= 100
    assert any(point['step'] == 1234 for point in sampled)
    assert 'per_class' not in compact[0]['validation']['outputs']['o']


def test_compact_history_api_keeps_raw_history_available(tmp_path, monkeypatch):
    import neuralsandbox.training as service
    from fastapi.testclient import TestClient
    from neuralsandbox.api import app
    monkeypatch.setattr(service, 'RUN_DIR', tmp_path)
    local = service.RunManager()
    monkeypatch.setattr(service, 'manager', local)
    evaluation = {'loss': .4, 'count': 20, 'outputs': {'o': {'loss': .4, 'accuracy': .8, 'per_class': [{'index': 0}]}}}
    records = [{'step': step, 'loss': 20 if step == 1234 else .4, 'training': evaluation} for step in range(5000)]
    local.runs['recorded'] = {'id': 'recorded', 'metrics': records[-2000:], 'sequence': 0, 'created_at': 1, 'network_id': 'network'}
    folder = tmp_path/'recorded'
    folder.mkdir()
    (folder/'metrics.jsonl').write_text(''.join(json.dumps(metric) + '\n' for metric in records))
    client = TestClient(app)
    bounded = client.get('/training/runs/recorded/metrics?compact=true&max_points=1200').json()
    assert bounded['total'] == 5000
    assert len(bounded['metrics']) <= 1200
    assert any(metric['step'] == 1234 for metric in bounded['metrics'])
    assert 'per_class' not in bounded['metrics'][0]['training']['outputs']['o']
    window = client.get('/training/runs/recorded/metrics?compact=true&max_points=1200&start_step=1230&end_step=1240').json()
    assert len(window['metrics']) == 11
    raw = client.get('/training/runs/recorded/metrics').json()
    assert len(raw['metrics']) == 5000
    assert raw['metrics'][0]['training']['outputs']['o']['per_class']
    compact_list = client.get('/training/runs?network_id=network&compact=true').json()
    assert len(compact_list[0]['metrics']) == 200
    assert client.get('/training/runs/recorded/metrics?max_points=100000').status_code == 422


def test_compact_sse_sends_incremental_metrics_and_static_config_once(monkeypatch):
    import asyncio
    import neuralsandbox.training as service
    local = service.RunManager()
    monkeypatch.setattr(service, 'manager', local)
    evaluation = {'step': 2, 'time': 10, 'training': {'loss': .2, 'count': 10, 'outputs': {}}}
    local.runs['live'] = {'id': 'live', 'sequence': 1, 'status': 'running', 'project': {'large': 'immutable'}, 'metrics': [{'step': 1, 'loss': .4}]}

    class ConnectedRequest:
        async def is_disconnected(self) -> bool:
            return False

    async def frames():
        response = await service.events('live', ConnectedRequest(), compact=True)
        iterator = response.body_iterator
        first = json.loads((await anext(iterator)).split('data: ')[1])
        local.runs['live'].update(sequence=2, metrics=[{'step': 1, 'loss': .4}, {'step': 2, 'loss': .3}], evaluation_summary=evaluation)
        second = json.loads((await anext(iterator)).split('data: ')[1])
        local.runs['live'].update(sequence=3, metrics=[{'step': 1, 'loss': .4}, {'step': 2, 'loss': .3}, {'step': 3, 'loss': .2}], status='completed')
        third = json.loads((await anext(iterator)).split('data: ')[1])
        await iterator.aclose()
        return first, second, third

    first, second, third = asyncio.run(frames())
    assert first['project'] == {'large': 'immutable'}
    assert 'project' not in second
    assert second['evaluation_summary'] == evaluation
    assert 'evaluation_summary' not in third
    assert [point['step'] for point in third['metrics']] == [2, 3]
