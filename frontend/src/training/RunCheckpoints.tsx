import type { TrainingRun } from './types';
import './training.css';
export default function RunCheckpoints({run}:{run:TrainingRun}) {
 const apiUrl=import.meta.env.VITE_API_URL??'http://localhost:8000';
 const downloadable=run.status==='completed'&&run.checkpoints.length>0;
 return <section className="training-checkpoints" aria-label="Run checkpoints"><div className="training-panel-heading"><div><h2>Checkpoints</h2><p className="muted">Weight availability is separate from historical performance.</p></div>{downloadable&&<button onClick={()=>window.location.assign(`${apiUrl}/training/runs/${run.id}/weights`)}>Save weights</button>}</div><div className="training-config-grid">{['best','latest'].map(name=>{
 const detail=run.checkpoint_details?.[name],available=run.checkpoints.includes(name);
 return <div className="training-config-card" key={name}><div className="training-run-top"><h3>{name==='best'?'Best checkpoint':'Latest checkpoint'}</h3><span className={`training-badge ${available?'is-available':''}`}>{available?'Available':'Unavailable'}</span></div>{detail?<dl><dt>Step</dt><dd>{detail.step.toLocaleString()}</dd><dt>Epoch</dt><dd>{detail.epoch}</dd><dt>Saved</dt><dd>{new Date(detail.time*1000).toLocaleString()}</dd>{detail.validation_loss!=null&&<><dt>Validation loss</dt><dd>{detail.validation_loss.toPrecision(4)}</dd></>}</dl>:<p className="muted">{available?'Weights recorded; detailed metadata unavailable.':'No retained weights for this checkpoint.'}</p>}</div>;
 })}</div></section>;
}
