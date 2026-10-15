import { useEffect, useState } from 'react';
import type { ChartMetric, StepRange } from './chartData';
import type { MetricHistory } from './types';
import { boundedHistory, CURVE_BUDGET } from './curveCache';


/** Replace bounded views instead of retaining every historical evaluation in the page. */
export function useCurveHistory(runId:string,finished:boolean,metric:ChartMetric,output:string,range:StepRange|null) {
 const [overview,setOverview]=useState<{key:string;data:MetricHistory}|null>(null);
 const [detail,setDetail]=useState<{key:string;data:MetricHistory}|null>(null);
 const [failure,setFailure]=useState('');
 const baseKey=`${runId}:${metric}:${output}`,start=range?.[0],end=range?.[1];
 const detailKey=`${baseKey}:${start}:${end}`;
 useEffect(()=>{
  const controller=new AbortController();let fetching=false;
  async function load(){
   if(fetching)return;fetching=true;
   try {
    const query=new URLSearchParams({compact:'true',max_points:String(CURVE_BUDGET),metric,output});
    const response=await fetch(`${import.meta.env.VITE_API_URL??'http://localhost:8000'}/training/runs/${encodeURIComponent(runId)}/metrics?${query}`,{signal:controller.signal});
    if(!response.ok)throw new Error('Earlier history is unavailable.');
    const data=boundedHistory(await response.json() as MetricHistory,metric,output);
    if(!controller.signal.aborted){setOverview({key:baseKey,data});setFailure('')}
   }catch {if(!controller.signal.aborted)setFailure(baseKey)}
   finally{fetching=false}
  }
  void load();
  const interval=finished?undefined:window.setInterval(()=>void load(),10000);
  return()=>{controller.abort();window.clearInterval(interval)};
 },[runId,finished,metric,output,baseKey]);
 useEffect(()=>{
  if(start===undefined||end===undefined)return;
  const controller=new AbortController();let fetching=false;
  async function load(){
   if(fetching)return;fetching=true;
   try {
    const query=new URLSearchParams({compact:'true',max_points:String(CURVE_BUDGET),metric,output,start_step:String(start),end_step:String(end)});
    const response=await fetch(`${import.meta.env.VITE_API_URL??'http://localhost:8000'}/training/runs/${encodeURIComponent(runId)}/metrics?${query}`,{signal:controller.signal});
    if(!response.ok)throw new Error('Range history is unavailable.');
    const data=boundedHistory(await response.json() as MetricHistory,metric,output);
    if(!controller.signal.aborted){setDetail({key:detailKey,data});setFailure('')}
   }catch {if(!controller.signal.aborted)setFailure(detailKey)}
   finally{fetching=false}
  }
  // Let the navigator move freely; fetch finer observations after dragging settles.
  const timer=window.setTimeout(()=>void load(),250);
  const interval=finished?undefined:window.setInterval(()=>void load(),10000);
  return()=>{controller.abort();window.clearTimeout(timer);window.clearInterval(interval)};
 },[runId,finished,metric,output,start,end,detailKey]);
 return {overview:overview?.key===baseKey?overview.data:null,detail:detail?.key===detailKey?detail.data:null,
  unavailable:failure===baseKey||failure===detailKey};
}
