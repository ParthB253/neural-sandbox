import { terminal, type TrainingState } from './useTraining';
import { lazy, Suspense } from 'react';
const TrainingChart=lazy(()=>import('./TrainingChart'));
import RunMetrics from './RunMetrics';
import RunActivity from './RunActivity';
import RunConfiguration from './RunConfiguration';
import RunCheckpoints from './RunCheckpoints';
import './training.css';

export default function TrainingPage({state,ready}:{state:TrainingState;ready:boolean}) {
 const {run,error,pending,act}=state;
 const activeRun=state.runs.find(candidate=>!terminal(candidate));
 const progress=run?.total_steps ? Math.min(100,100*run.step/run.total_steps) : null;
 return <section className="training-page">
  <div className="training-heading">
   <div><p className="training-eyebrow">Training workspace</p><h1>{run?`Run ${run.id.slice(0,8)}`:'Train your network'}</h1>
    <p>{run?`${run.status} · Epoch ${run.epoch} of ${run.project.training?.epochs??'—'}`:'Configure inputs, targets and sampling in the sidebar.'}</p>
   </div>
   <div className="sample-controls">
    <button className="primary" disabled={!ready||pending||!!activeRun} onClick={()=>void act('start')}>Start new run</button>
    {run&&!terminal(run)&&<>
     <button disabled={pending||['pausing','cancelling'].includes(run.status)} onClick={()=>void act(run.status==='paused'?'resume':'pause')}>{run.status==='paused'?'Resume':'Pause'}</button>
     <button disabled={pending||run.status==='cancelling'} onClick={()=>void act('cancel')}>Cancel</button>
    </>}
   </div>
  </div>
  {!ready&&!run&&<p role="status">Connect valid input and target feeds to a trainable network in Build to start.</p>}
  {(error||run?.error)&&<p role="alert" className="execution-error">{error||run?.error}</p>}
  {run?.warnings?.map(message=><p key={message} role="status" className="training-warning">{message}</p>)}
  {run ? <>
   <div className="training-progress" role="status">
    <div className="training-progress-heading"><strong>{run.stage}</strong><span>{progress===null?'Preparing':`${progress.toFixed(0)}%`}</span></div>
    <progress aria-label="Training progress" value={run.total_steps?run.step:undefined} max={run.total_steps||1}/>
    <div className="training-progress-heading"><span>Step {run.step.toLocaleString()} / {run.total_steps.toLocaleString()}</span><time dateTime={new Date(run.created_at*1000).toISOString()}>Started {new Date(run.created_at*1000).toLocaleString()}</time></div>
   </div>
   <RunMetrics key={`metrics-${run.id}`} run={run}/>
   <Suspense fallback={<p role="status">Loading learning curves…</p>}><TrainingChart key={run.id} run={run}/></Suspense>
   <RunCheckpoints run={run}/>
   <RunActivity run={run}/>
   <RunConfiguration run={run}/>
  </> : <div className="training-empty"><h2>Your next experiment starts here</h2><p>Choose your data and training settings, then start a run to see learning curves and evaluation results.</p></div>}
 </section>;
}
