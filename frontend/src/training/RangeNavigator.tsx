import { useRef } from 'react';
import { panRange, type StepRange } from './chartData';

/** Absolute optimiser steps never depend on indices in a sampled/refreshed curve. */
export default function RangeNavigator({min,max,value,onChange}:{min:number;max:number;value:StepRange;onChange:(start:number,end:number)=>void}) {
 const drag=useRef<{x:number;width:number;range:StepRange;min:number;max:number}|null>(null);
 const span=Math.max(1,max-min),left=(value[0]-min)/span*100,right=(value[1]-min)/span*100;
 return <div className="training-range-navigator" role="group" aria-label="Full run range selector">
  <div className="training-range-track" aria-hidden="true"><div onPointerDown={event=>{
   const width=event.currentTarget.parentElement!.getBoundingClientRect().width;
   drag.current={x:event.clientX,width,range:value,min,max};
   event.currentTarget.setPointerCapture(event.pointerId);event.preventDefault();
  }} onPointerMove={event=>{
   const start=drag.current;if(!start||!start.width)return;
   const next=panRange(start.range,(event.clientX-start.x)/start.width*(start.max-start.min),start.min,start.max);
   onChange(next[0],next[1]);
  }} onPointerUp={()=>{drag.current=null}} onPointerCancel={()=>{drag.current=null}} style={{left:`${left}%`,width:`${Math.max(0,right-left)}%`}}/></div>
  <input aria-label="Range start step" type="range" min={min} max={max} step="any" value={value[0]} onChange={event=>onChange(Math.min(Number(event.target.value),value[1]-1),value[1])}/>
  <input aria-label="Range end step" type="range" min={min} max={max} step="any" value={value[1]} onChange={event=>onChange(value[0],Math.max(Number(event.target.value),value[0]+1))}/>
  <div className="training-range-labels"><span>{min.toLocaleString()}</span><span>{max.toLocaleString()}</span></div>
 </div>;
}
