"""Local run ownership, persistence, cooperative controls and snapshot SSE."""
import asyncio
import json
import os
import threading
import time
from pathlib import Path
from uuid import UUID, uuid4
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from .schemas import ProjectSpec
from .training_runtime import TrainingRuntime
from .training_views import browser_run, sample_metrics, scalar_metric

router = APIRouter()
TERMINAL = {'completed','cancelled','failed','interrupted'}
RUN_DIR = Path(os.environ.get('NEURALSANDBOX_RUN_DIR', Path(__file__).resolve().parents[3]/'workspaces'/'runs'))


class StartRequest(BaseModel):
    project: ProjectSpec
    request_id: UUID = Field(default_factory=uuid4)

class RunManager:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.runs: dict[str, dict] = {}
        self.worker: threading.Thread | None = None
        self.stopping = False

    def recover(self) -> None:
        with self.lock:
            self.stopping=False

            if RUN_DIR.exists():
                for path in RUN_DIR.glob('*/run.json'):
                    record=json.loads(path.read_text())
                    if record['status'] not in TERMINAL: record['status']='interrupted'
                    self.runs[record['id']]=record
                    self.persist(record)
            self.prune_weights()

    def newest(self, network_id: str) -> dict | None:
        records = [r for r in self.runs.values() if r['network_id'] == network_id]
        return max(records, key=lambda r: r['created_at'], default=None)

    def prune_weights(self) -> None:
        """Retain numerical checkpoints only for each network's newest run."""
        for record in self.runs.values():
            if record is self.newest(record['network_id']):
                continue
            for name in ('latest', 'best'):
                (RUN_DIR / record['id'] / f'{name}.npz').unlink(missing_ok=True)
            record['checkpoints'] = []
            self.persist(record)

    def persist(self, record: dict) -> None:
        folder=RUN_DIR/record['id']; folder.mkdir(parents=True, exist_ok=True)
        temporary=folder/'run.tmp'; temporary.write_text(json.dumps(record, allow_nan=False)); temporary.replace(folder/'run.json')

    def publish(self, id: str, **values) -> None:
        with self.lock:
            record=self.runs[id]
            if 'status' in values or ('stage' in values and values['stage'] != record.get('stage')):
                record.setdefault('activity', []).append({'time':time.time(), 'message':values.get('status', values.get('stage'))})
                record['activity']=record['activity'][-100:]

            if values.get('metrics'):
                path = RUN_DIR / id / 'metrics.jsonl'
                # Separate a torn final record from the next valid append after recovery.
                needs_separator = False
                if path.exists() and path.stat().st_size:
                    with path.open('rb') as journal:
                        journal.seek(-1, 2)
                        needs_separator = journal.read(1) != b'\n'
                with path.open('a') as history:
                    if needs_separator:
                        history.write('\n')
                    history.write(json.dumps(values['metrics'][-1], allow_nan=False)+'\n')

            record.update(values); record['sequence']+=1
            if values.get('status') in TERMINAL:
                record['completed_at'] = time.time()
            record['updated_at']=time.time(); self.persist(record)

    def get(self, id: str, compact: bool = False) -> dict:
        with self.lock:
            if id not in self.runs: raise HTTPException(404, 'Unknown training run')
            record = browser_run(self.runs[id]) if compact else self.runs[id]
            return json.loads(json.dumps(record))

    def history(self, id: str, start_step: int | None = None, end_step: int | None = None) -> dict:
        """Read full recorded observations, with legacy snapshot and partial journal recovery."""
        assert start_step is None or start_step >= 0, 'Start step must be nonnegative'
        assert end_step is None or end_step >= 0, 'End step must be nonnegative'
        if start_step is not None and end_step is not None and start_step > end_step:
            raise HTTPException(422, 'Start step must not exceed end step')
        with self.lock:
            record = self.get(id)
            path = RUN_DIR / id / 'metrics.jsonl'
            observations = []
            if path.exists():
                for line in path.read_text().splitlines():
                    try:
                        metric = json.loads(line)
                        if isinstance(metric, dict) and isinstance(metric.get('step'), int):
                            observations.append(metric)
                    except json.JSONDecodeError:
                        # A process interruption may leave an incomplete final journal line.
                        continue
            # Older versions may have persisted a snapshot without a complete journal.
            seen = {json.dumps(metric, sort_keys=True) for metric in observations}
            for metric in record.get('metrics', []):
                key = json.dumps(metric, sort_keys=True)
                if key not in seen:
                    observations.append(metric)
                    seen.add(key)

            observations.sort(key=lambda metric: metric['step'])
            lower = start_step if start_step is not None else min((m['step'] for m in observations), default=0)
            upper = end_step if end_step is not None else max((m['step'] for m in observations), default=0)
            return {'metrics': [m for m in observations if lower <= m['step'] <= upper],
                    'total': len(observations), 'start_step': lower, 'end_step': upper}

    def start(self, request: StartRequest) -> dict:
        with self.lock:
            for record in self.runs.values():
                if record['request_id']==str(request.request_id):
                    if record['project']!=request.project.model_dump(mode='json'): raise HTTPException(409,'Request ID already used with different settings')
                    return self.get(record['id'])
            if self.stopping or (self.worker and self.worker.is_alive()): raise HTTPException(409,'One training run is already active')

            id=str(uuid4())
            self.runs[id]={'id':id,'request_id':str(request.request_id),'network_id':str(request.project.network.id),
                'project':request.project.model_dump(mode='json'),'status':'preparing','stage':'Loading sources and transforms',
                'sequence':0,'step':0,'epoch':0,'total_steps':0,'metrics':[],'created_at':time.time(), 'checkpoints':[], 'update_rule':'learning-rate-gradient-logsoftmax-v3', 'activity':[]}
            self.persist(self.runs[id])
            self.prune_weights()
            self.worker=threading.Thread(target=self.execute,args=(id,request.project.model_copy(deep=True)),daemon=True)
            self.worker.start()
            return self.get(id)

    def command(self, id: str, command: str) -> dict:
        with self.lock:
            record=self.get(id); status=record['status']
            if command=='cancel':
                if status not in TERMINAL: self.publish(id,status='cancelling')
            elif command=='pause':
                if status in {'running','preparing'}: self.publish(id,status='pausing')
                elif status not in {'pausing','paused'}: raise HTTPException(409,'Run cannot be paused')
            elif command=='resume':
                if status=='paused': self.publish(id,status='running')
                elif status!='running': raise HTTPException(409,'Run is not paused')
            else: raise HTTPException(404,'Unknown command')
            return self.get(id)

    def checkpoint(self, id: str, runtime: TrainingRuntime, name: str) -> None:
        folder=RUN_DIR/id
        with (folder/f'{name}.tmp').open('wb') as file: np.savez(file,**runtime.weights())
        (folder/f'{name}.tmp').replace(folder/f'{name}.npz')
        with self.lock:
            names=self.runs[id]['checkpoints']
            if name not in names: names.append(name)
            record = self.runs[id]
            details = {'step': record['step'], 'epoch': record['epoch'], 'time': time.time()}
            summary = record.get('evaluation_summary', {})
            if summary.get('step') == record['step'] and 'validation_loss' in summary:
                details['validation_loss'] = summary['validation_loss']
            record.setdefault('checkpoint_details', {})[name] = details
            self.publish(id)

    def boundary(self, id: str, runtime: TrainingRuntime) -> bool:
        while True:
            with self.lock: status=self.runs[id]['status']; stopping=self.stopping

            if stopping or status=='cancelling':
                self.checkpoint(id,runtime,'latest'); self.publish(id,status='interrupted' if stopping else 'cancelled'); return False
            if status=='pausing':
                with self.lock:
                    if self.runs[id]['status']=='pausing':
                        self.checkpoint(id,runtime,'latest'); self.publish(id,status='paused')
            elif status!='paused': return True

            time.sleep(.05)

    def execute(self, id: str, project: ProjectSpec) -> None:
        try:
            runtime=TrainingRuntime(project); settings=project.training
            rng=np.random.default_rng(settings.seed)
            total=int(np.ceil(len(runtime.train)/settings.batch_size))*settings.epochs
            self.publish(id,total_steps=total,warnings=runtime.warnings,data_revision=runtime.data_revision,train_indices=runtime.train.tolist(),validation_indices=runtime.validation.tolist())

            if not self.boundary(id,runtime): return
            with self.lock:
                if self.runs[id]['status']=='preparing': self.publish(id,status='running')
            step=0; best=float('inf'); metrics=[]; last=0.
            for epoch in range(1,settings.epochs+1):
                self.publish(id,stage='Preparing sample order',epoch=epoch)
                order=rng.permutation(runtime.train) if settings.sampling=='shuffle' else np.sort(runtime.train)

                for offset in range(0,len(order),settings.batch_size):
                    if not self.boundary(id,runtime): return
                    loss,_=runtime.step(order[offset:offset+settings.batch_size],True); step+=1
                    now=time.monotonic()
                    if now-last>=.2 or offset+settings.batch_size>=len(order):
                        metrics.append({'step':step,'epoch':epoch,'time':time.time(),'loss':loss}); last=now
                        self.publish(id,stage='Training',step=step,metrics=metrics[-2000:])

                if epoch % settings.evaluation_every == 0 or epoch == settings.epochs:
                    self.publish(id, stage='Evaluating training partition', step=step)
                    training = runtime.evaluate(runtime.train, lambda: self.boundary(id, runtime))
                    if training is None: return
                    point = {'step': step, 'epoch': epoch, 'time': time.time(), 'training': training}
                    if len(runtime.validation):
                        self.publish(id, stage='Evaluating validation partition')
                        validation = runtime.evaluate(runtime.validation, lambda: self.boundary(id, runtime))
                        if validation is None: return
                        point.update(validation=validation, validation_loss=validation['loss'])
                        outputs = list(validation['outputs'].values())
                        # Legacy accuracy is unambiguous only for one eligible classification output.
                        point['accuracy'] = outputs[0].get('accuracy') if len(outputs) == 1 else None

                    if not self.boundary(id, runtime): return
                    metrics.append(point)
                    self.publish(id, metrics=metrics[-2000:], evaluation_summary=point)
                    if 'validation_loss' in point and point['validation_loss'] < best:
                        best = point['validation_loss']
                        self.publish(id, best_step=step, best_epoch=epoch, best_validation_loss=best)
                        self.checkpoint(id, runtime, 'best')
                self.checkpoint(id,runtime,'latest')

            if not self.boundary(id,runtime): return
            with self.lock:
                if self.runs[id]['status']=='cancelling':
                    self.publish(id,status='cancelled')
                else: self.publish(id,status='completed',stage='Complete',step=step)
        except Exception as error:
            self.publish(id,status='failed',stage='Failed',error=str(error))

    def shutdown(self) -> None:
        with self.lock: self.stopping=True
        if self.worker: self.worker.join(timeout=5)

manager=RunManager()


@router.post('/training/runs',status_code=202)
def start(request: StartRequest): return manager.start(request)

@router.get('/training/runs')
def runs(network_id: str, compact: bool = False):
    with manager.lock: return sorted([manager.get(id, compact=compact) for id,r in manager.runs.items() if r['network_id']==network_id], key=lambda r: r['created_at'], reverse=True)


@router.get('/training/runs/{id}')
def get(id: str): return manager.get(id)

@router.get('/training/runs/{id}/metrics')
def metric_history(id: str, start_step: int | None = Query(default=None, ge=0), end_step: int | None = Query(default=None, ge=0),
                   max_points: int | None = Query(default=None, ge=8, le=2000), compact: bool = False,
                   metric: str = 'loss', output: str = ''):
    history = manager.history(id, start_step, end_step)
    if compact:
        history['metrics'] = [scalar_metric(record) for record in history['metrics']]
    if max_points is not None:
        history['metrics'] = sample_metrics(history['metrics'], max_points, metric, output)
    return history


@router.get('/training/runs/{id}/weights')
def download_weights(id: str) -> FileResponse:
    record = manager.get(id)
    if manager.newest(record['network_id'])['id'] != id:
        raise HTTPException(409, 'Only the most recent run retains weights')
    if record['status'] != 'completed':
        raise HTTPException(409, 'Weights can only be saved after a run completes')
    path = RUN_DIR / id / 'latest.npz'
    if not path.is_file():
        raise HTTPException(404, 'The final weights file is unavailable')
    return FileResponse(path, media_type='application/octet-stream', filename=f'{id}-weights.npz')

@router.post('/training/runs/{id}/{command}')
def command(id: str, command: str): return manager.command(id,command)

@router.get('/training/runs/{id}/events')
async def events(id: str, request: Request, compact: bool = False):
    manager.get(id)
    async def stream():
        sequence=-1
        last_metric_step=-1
        last_evaluation=None
        while not await request.is_disconnected():
            record=manager.get(id, compact=compact)
            if record['sequence']!=sequence:
                if compact:
                    if sequence >= 0:
                        record.pop('project', None)
                        record['metrics'] = [metric for metric in record['metrics'] if metric['step'] >= last_metric_step]
                    last_metric_step = max((metric['step'] for metric in record['metrics']), default=last_metric_step)
                    summary = record.get('evaluation_summary')
                    evaluation_key = (summary.get('step'), summary.get('time')) if summary else None
                    if evaluation_key == last_evaluation:
                        record.pop('evaluation_summary', None)
                    last_evaluation = evaluation_key
                sequence=record['sequence']
                yield f'id: {sequence}\nevent: snapshot\ndata: {json.dumps(record)}\n\n'
            else: yield ': heartbeat\n\n'
            if record['status'] in TERMINAL: return
            await asyncio.sleep(.2)
    return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

@router.post('/training/preview')
def preview(project: ProjectSpec):
    try:
        runtime=TrainingRuntime(project, initialize=False)
        return {'count':runtime.count,'examples':[{'row':i,'inputs':{k:v[i].tolist()[:50] for k,l in runtime.layers.items() if (v:=runtime.inputs.get(l)) is not None},'targets':{k:v[i].tolist()[:50] for k,l in runtime.layers.items() if (v:=runtime.targets.get(l)) is not None}} for i in range(min(3,runtime.count))]}
    except (AssertionError,ValueError,KeyError) as error: raise HTTPException(422,str(error)) from error
