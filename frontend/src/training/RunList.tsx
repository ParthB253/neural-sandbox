import type { TrainingRun } from './types';
import './training.css';
export default function RunList({runs,selected,onSelect}:{runs:TrainingRun[];selected:string|null;onSelect:(id:string)=>void}) {
 const ordered=[...runs].sort((a,b)=>b.created_at-a.created_at||a.id.localeCompare(b.id));
 if(!ordered.length)return <div className="training-empty"><strong>No runs yet</strong><p>Your training history will appear here after you start a run.</p></div>;
 return <div className="training-run-list" aria-label="Training runs">{ordered.map((run,index)=>{
  const evaluation=run.evaluation_summary??run.metrics.filter(m=>m.validation_loss!==undefined||m.validation).at(-1);
  const loss=evaluation?.validation?.loss??evaluation?.validation_loss;
  const progress=run.total_steps?Math.min(100,run.step/run.total_steps*100):0;
  return <button key={run.id} className={`training-run-card ${selected===run.id?'is-selected':''}`} aria-pressed={selected===run.id} onClick={()=>onSelect(run.id)}>
   <span className="training-run-top"><strong>Run {run.id.slice(0,8)}</strong>{index===0&&<span className="training-badge">Latest run</span>}</span>
   <span className={`training-status status-${run.status}`}><i aria-hidden="true"/>{run.status}</span>
   <time dateTime={new Date(run.created_at*1000).toISOString()}>{new Date(run.created_at*1000).toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'})}</time>
   <span className="training-run-meta"><span>Epoch {run.epoch.toLocaleString()}</span><span>{run.step.toLocaleString()} / {run.total_steps.toLocaleString()} steps</span></span>
   <span className="training-run-progress" aria-hidden="true"><span style={{width:`${progress}%`}}/></span>
   {loss!==undefined&&<span className="training-run-score">Validation loss <strong>{loss.toPrecision(4)}</strong></span>}
   {run.best_validation_loss!==undefined&&<span className="training-run-score">Best in run <strong>{run.best_validation_loss.toPrecision(4)}</strong></span>}
  </button>;
 })}</div>;
}
