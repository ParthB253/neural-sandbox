import { useEffect, useState } from 'react';
import { request, type ProjectSpec } from '../network';
import { applyRunUpdate, compactRun } from './runSnapshot';
import type { TrainingRun } from './types';
export type { TrainingRun } from './types';
export interface TrainingSettings { learning_rate:number; initialization:"uniform"|"he_normal"|"xavier_normal"; epochs:number; batch_size:number; sampling:'shuffle'|'sequential'; seed:number; validation_fraction:number; evaluation_every:number; losses:Record<string,'mse'|'cross_entropy'> }
export const defaults:TrainingSettings={learning_rate:0.01,initialization:"xavier_normal",epochs:10,batch_size:128,sampling:'shuffle',seed:42,validation_fraction:.1,evaluation_every:1,losses:{}};
export const terminal=(run:TrainingRun)=>['completed','cancelled','failed','interrupted'].includes(run.status);
export function useTraining(project:ProjectSpec) {
 const [runs,setRuns]=useState<TrainingRun[]>([]),[selected,setSelected]=useState<string|null>(null),[error,setError]=useState(''),[pending,setPending]=useState(false);
 const run=runs.find(r=>r.id===selected);
 useEffect(()=>{let active=true; request<TrainingRun[]>(`/training/runs?network_id=${project.network.id}&compact=true`).then(r=>{if(active){setRuns(r.map(run=>compactRun(run)));setSelected(r[0]?.id??null);setError('')}}).catch(e=>{if(active)setError(String(e))});return()=>{active=false}},[project.network.id]);
 // Keep active runs subscribed even while inspecting a historical run.
 const activeRunIds=runs.filter(r=>!terminal(r)).map(r=>r.id).sort().join(',');
 useEffect(()=>{
  if(!activeRunIds)return;
  const sources=activeRunIds.split(',').map(id=>{
   const source=new EventSource(`${import.meta.env.VITE_API_URL??'http://localhost:8000'}/training/runs/${id}/events?compact=true`);
   source.addEventListener('snapshot',event=>{
    const next=JSON.parse((event as MessageEvent).data) as TrainingRun;
    setRuns(old=>old.map(r=>r.id===next.id&&next.sequence>=r.sequence?applyRunUpdate(r,next):r));
    if(terminal(next))source.close();
   });
   return source;
  });
  return()=>sources.forEach(source=>source.close());
 },[activeRunIds]);
 async function act(command:string){setPending(true);setError('');try{const next=command==='start'?await request<TrainingRun>('/training/runs','POST',{project,request_id:crypto.randomUUID()}):await request<TrainingRun>(`/training/runs/${selected}/${command}`,'POST');setRuns(old=>[compactRun(next,old.find(r=>r.id===next.id)),...old.filter(r=>r.id!==next.id).map(r=>command==='start'?{...r,checkpoints:[]}:r)].sort((a,b)=>b.created_at-a.created_at));setSelected(next.id)}catch(e){setError(e instanceof Error?e.message:String(e))}finally{setPending(false)}}
 return {runs,run,selected,setSelected,error,pending,act};
}
export type TrainingState=ReturnType<typeof useTraining>;
