"""Numerical correctness, paired data, and the local training lifecycle."""
import time
import numpy as np
import pytest
from neuralsandbox.schemas import ProjectSpec
from neuralsandbox.training_runtime import TrainingRuntime, activation_vjp
from neuralsandbox.training import RunManager, StartRequest, TERMINAL


def project(tmp_path, epochs=3):
    x=np.array([[0.],[1.],[0.],[1.],[0.],[1.],[0.],[1.]])
    y=np.eye(2)[x[:,0].astype(int)]
    np.save(tmp_path/'x.npy',x); np.save(tmp_path/'y.npy',y)
    return ProjectSpec.model_validate({'network':{'layers':[{'id':'i','size':1,'activation':'identity'},{'id':'o','size':2,'activation':'softmax'}], 'links':[{'id':'w','source':'i','target':'o'}]},'datasets':[{'id':'x','source':{'type':'npy','path':str(tmp_path/'x.npy')}},{'id':'y','source':{'type':'npy','path':str(tmp_path/'y.npy')}}], 'feeds':[{'id':'fx','dataset':'x','field':'value','layer':'i','role':'input'},{'id':'fy','dataset':'y','field':'value','layer':'o','role':'target'}], 'training':{'epochs':epochs,'batch_size':2,'validation_fraction':.25}})


def test_softmax_vjp():
    z=np.array([[.2,-.4,1.1],[.5,.1,-.2]]); g=np.array([[.4,.7,-.2],[-.1,.8,.2]])
    from neuralsandbox.execution import _softmax
    actual=activation_vjp('softmax',z,g)
    for index in np.ndindex(z.shape):
        plus=z.copy();minus=z.copy();plus[index]+=1e-6;minus[index]-=1e-6
        numeric=np.sum((_softmax(plus)-_softmax(minus))*g)/2e-6
        assert actual[index]==pytest.approx(numeric,abs=1e-8)


def test_loss_decreases_and_split_is_disjoint(tmp_path):
    runtime=TrainingRuntime(project(tmp_path))
    before=runtime.step(runtime.train,False)[0]
    for _ in range(40):runtime.step(runtime.train,True)
    assert runtime.step(runtime.train,False)[0]<before
    assert not set(runtime.train)&set(runtime.validation)
    assert len(runtime.train)+len(runtime.validation)==runtime.count


def test_rejects_mismatched_rows(tmp_path):
    p=project(tmp_path);np.save(tmp_path/'y.npy',np.eye(2))
    with pytest.raises(AssertionError,match='row counts'):TrainingRuntime(p)


def test_run_persists_and_reuses_request(tmp_path,monkeypatch):
    import neuralsandbox.training as service
    monkeypatch.setattr(service,'RUN_DIR',tmp_path/'runs')
    manager=RunManager();request=StartRequest(project=project(tmp_path))
    record=manager.start(request);manager.worker.join(10)
    record=manager.get(record['id'])
    assert record['status']=='completed',record.get('error')
    assert record['step']==9
    assert set(record['checkpoints'])=={'latest','best'}
    assert manager.start(request)['id']==record['id']
    restored=RunManager();restored.recover()
    assert restored.get(record['id'])['metrics']==record['metrics']


def test_cancel_while_paused(tmp_path,monkeypatch):
    import neuralsandbox.training as service
    monkeypatch.setattr(service,'RUN_DIR',tmp_path/'runs')
    manager=RunManager();record=manager.start(StartRequest(project=project(tmp_path,10000)))
    manager.command(record['id'],'pause')
    deadline=time.monotonic()+5
    while manager.get(record['id'])['status'] not in {'paused',*TERMINAL} and time.monotonic()<deadline:time.sleep(.01)
    assert manager.get(record['id'])['status']=='paused'
    manager.command(record['id'],'cancel');manager.worker.join(5)
    assert manager.get(record['id'])['status']=='cancelled'


def test_trained_checkpoint_forward(tmp_path,monkeypatch):
    import neuralsandbox.training as service
    from neuralsandbox.execution import forward, ForwardRequest
    monkeypatch.setattr(service,'RUN_DIR',tmp_path/'runs')
    local=RunManager();monkeypatch.setattr(service,'manager',local)
    p=project(tmp_path);record=local.start(StartRequest(project=p));local.worker.join(10)
    result=forward(ForwardRequest(project=p,checkpoint_run_id=record['id']))
    assert result['trained']
    with np.load(tmp_path/'runs'/record['id']/'latest.npz') as weights:
        from neuralsandbox.execution import _softmax
        expected=_softmax(weights['b:o'])
    np.testing.assert_allclose(result['outputs']['o']['values'],expected[0])


def test_sse_snapshot_and_training_settings_roundtrip(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from neuralsandbox.api import app
    import neuralsandbox.training as service
    monkeypatch.setattr(service,'RUN_DIR',tmp_path/'runs')
    local=RunManager();monkeypatch.setattr(service,'manager',local)
    client=TestClient(app)
    p=project(tmp_path)
    saved=client.put('/project',json=p.model_dump(mode='json'))
    assert saved.status_code==200
    assert client.get('/project').json()['training']['batch_size']==2
    response=client.post('/training/runs',json={'project':p.model_dump(mode='json')})
    assert response.status_code==202
    local.worker.join(10)
    id=response.json()['id']
    events=client.get(f'/training/runs/{id}/events',headers={'Last-Event-ID':'0'})
    assert events.status_code==200
    assert 'event: snapshot' in events.text and 'completed' in events.text


def test_saturated_softmax_cross_entropy_keeps_gradient(tmp_path):
    runtime = TrainingRuntime(project(tmp_path))
    output = runtime.network.outputs[0]
    link = runtime.network.links[0]
    link.weights[:] = 0
    output.bias[:] = [1000, -1000]
    indices = np.array([1, 3])  # Both targets are class 1, whose probability underflows.
    inputs = {layer: values[indices] for layer, values in runtime.inputs.items()}
    targets = {layer: values[indices] for layer, values in runtime.targets.items()}
    loss, weights, biases = runtime.network.backprop(inputs, targets, runtime.output_delta)
    assert loss == pytest.approx(2000)
    np.testing.assert_allclose(biases[output], [[1, -1]])
    np.testing.assert_allclose(weights[link], [[1, -1]])
    runtime.step(indices, True)
    assert runtime.step(indices, False)[0] < loss


def test_fused_cross_entropy_gradients_match_finite_differences(tmp_path):
    runtime = TrainingRuntime(project(tmp_path))
    indices = np.array([0, 1, 3])
    inputs = {layer: values[indices] for layer, values in runtime.inputs.items()}
    targets = {layer: values[indices] for layer, values in runtime.targets.items()}
    _, weights, biases = runtime.network.backprop(inputs, targets, runtime.output_delta)
    for parameter, gradient in [*[(link.weights, grad) for link, grad in weights.items()],
                                *[(layer.bias, grad) for layer, grad in biases.items()]]:
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + 1e-5
            plus = runtime.step(indices, False)[0]
            parameter[index] = original - 1e-5
            minus = runtime.step(indices, False)[0]
            parameter[index] = original
            assert gradient[index] == pytest.approx((plus-minus)/2e-5, abs=1e-7)


@pytest.mark.parametrize("method", ["uniform", "he_normal", "xavier_normal"])
def test_initialization_and_learning_rate(tmp_path, method):
    p = project(tmp_path)
    p.training.initialization = method
    runtime = TrainingRuntime(p)
    repeated = TrainingRuntime(p)
    np.testing.assert_array_equal(runtime.network.links[0].weights, repeated.network.links[0].weights)
    assert all(np.count_nonzero(layer.bias) == 0 for layer in runtime.layers.values())
    before = runtime.network.links[0].weights.copy()
    inputs = {layer: values[runtime.train] for layer, values in runtime.inputs.items()}
    targets = {layer: values[runtime.train] for layer, values in runtime.targets.items()}
    _, gradients, _ = runtime.network.backprop(inputs, targets, runtime.output_delta)
    expected = before - p.training.learning_rate * gradients[runtime.network.links[0]]
    runtime.step(runtime.train, True)
    np.testing.assert_allclose(runtime.network.links[0].weights, expected)
    preview = TrainingRuntime(p, initialize=False)
    assert preview.network.links[0].weights is None
    assert preview.network.links[0].weight_shape == (1, 2)
    assert preview.layers["o"].bias is None
    assert preview.layers["o"].bias_shape == (1, 2)


def test_new_run_removes_old_weights_and_recovery_preserves_order(tmp_path, monkeypatch):
    import neuralsandbox.training as service
    monkeypatch.setattr(service, "RUN_DIR", tmp_path / "runs")
    local = RunManager()
    p = project(tmp_path)
    first = local.start(StartRequest(project=p))
    local.worker.join(10)
    assert (service.RUN_DIR / first["id"] / "latest.npz").exists()
    second = local.start(StartRequest(project=p))
    local.worker.join(10)
    assert not (service.RUN_DIR / first["id"] / "latest.npz").exists()
    assert not (service.RUN_DIR / first["id"] / "best.npz").exists()
    assert local.get(first["id"])["checkpoints"] == []
    restored = RunManager()
    restored.recover()
    assert restored.newest(str(p.network.id))["id"] == second["id"]
    monkeypatch.setattr(service, "manager", restored)
    assert service.runs(str(p.network.id))[0]["id"] == second["id"]
    with pytest.raises(service.HTTPException, match="409"):
        service.download_weights(first["id"])


def test_forward_requires_training(tmp_path):
    from fastapi import HTTPException
    from neuralsandbox.execution import forward, ForwardRequest
    with pytest.raises(HTTPException) as error:
        forward(ForwardRequest(project=project(tmp_path)))
    assert error.value.status_code == 422
