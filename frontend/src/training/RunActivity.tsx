import type { TrainingRun } from './types';
import './training.css';
function activityTitle(message:string):string {
 if(/fail|error/i.test(message))return 'Run error';
 if(/cancel/i.test(message))return 'Run cancelled';
 if(/paus/i.test(message))return 'Training paused';
 if(/resum/i.test(message))return 'Training resumed';
 if(/complet|finished/i.test(message))return 'Stage complete';
 if(/evaluat|validat/i.test(message))return 'Evaluating model';
 if(/checkpoint/i.test(message))return 'Checkpoint saved';
 if(/start|created/i.test(message))return 'Run started';
 return 'Run update';
}
export default function RunActivity({run}:{run:TrainingRun}) {
 const events:(NonNullable<TrainingRun['activity']>[number]&{count:number})[]=[];
 for(const event of run.activity??[]) {
  const previous=events.at(-1),repetitive=/evaluat|validat/i.test(event.message)&&!/fail|error|complet|finish/i.test(event.message);
  if(previous&&repetitive&&/evaluat|validat/i.test(previous.message)&&!/fail|error|complet|finish/i.test(previous.message)){previous.count+=1;previous.message=event.message;previous.time=event.time}
  else events.push({...event,count:1});
 }
 return <details className="training-disclosure training-activity" open><summary>Run activity <span className="training-count">{events.length}</span></summary>{events.length?<ol className="training-timeline">{events.map((event,index)=><li key={index} className={/fail|error/i.test(event.message)?'is-error':''}><span className="training-timeline-dot" aria-hidden="true"/><div><div className="training-event-heading"><strong>{activityTitle(event.message)}</strong><time dateTime={new Date(event.time*1000).toISOString()}>{new Date(event.time*1000).toLocaleTimeString()}</time></div><p>{event.message}</p>{event.count>1&&<small className="muted">{event.count} evaluation updates grouped</small>}</div></li>)}</ol>:<p className="muted">No activity has been recorded yet.</p>}</details>;
}
