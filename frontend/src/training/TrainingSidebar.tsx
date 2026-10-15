import { useState } from 'react';
import type { ProjectSpec } from '../network';
import type { TrainingState } from './useTraining';
import TrainingSetup from './TrainingSetup';
import RunList from './RunList';
import './training.css';
export default function TrainingSidebar({project,change,state,edit}:{project:ProjectSpec;change:(p:ProjectSpec)=>void;state:TrainingState;edit:(id:string)=>void}) {
 const [tab,setTab]=useState('setup');
 return <><div className="sidebar-tabs" role="tablist" aria-label="Training sidebar" onKeyDown={event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const next=event.key==='Home'?'setup':event.key==='End'?'runs':tab==='setup'?'runs':'setup';setTab(next);document.getElementById(`training-${next}-tab`)?.focus()}}><button id="training-setup-tab" role="tab" tabIndex={tab==='setup'?0:-1} aria-selected={tab==='setup'} aria-controls="training-sidebar-panel" onClick={()=>setTab('setup')}>Setup</button><button id="training-runs-tab" role="tab" tabIndex={tab==='runs'?0:-1} aria-selected={tab==='runs'} aria-controls="training-sidebar-panel" onClick={()=>setTab('runs')}>Runs ({state.runs.length})</button></div><section id="training-sidebar-panel" role="tabpanel" aria-labelledby={`training-${tab}-tab`} className="sidebar-panel training-setup">{tab==='runs'?<RunList runs={state.runs} selected={state.selected} onSelect={state.setSelected}/>:<TrainingSetup project={project} change={change} run={state.run} edit={edit}/>}</section></>;
}
