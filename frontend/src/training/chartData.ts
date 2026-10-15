import type { TrainingMetric } from './types.ts';

export type ChartMetric = 'loss' | 'accuracy' | 'f1_macro' | 'precision_macro' | 'recall_macro' | 'mae' | 'rmse';
export interface ChartPoint { step:number; batch?:number; training?:number; validation?:number }
export type StepRange = readonly [number, number];

/** New snapshots can contain a batch observation and an evaluation at the same step. */
export function mergeMetrics(...collections:TrainingMetric[][]):TrainingMetric[] {
 const byStep=new Map<number,TrainingMetric>();
 for(const collection of collections)for(const metric of collection){
  const previous=byStep.get(metric.step);
  byStep.set(metric.step,{...previous,...metric});
 }
 return [...byStep.values()].sort((a,b)=>a.step-b.step);
}
const finite=(value:number|null|undefined):number|undefined=>value!=null&&Number.isFinite(value)?value:undefined;
export function chartPoints(metrics:TrainingMetric[],metric:ChartMetric,output:string):ChartPoint[]{
 return metrics.map(record=>({
  step:record.step,
  batch:metric==='loss'?finite(record.loss):undefined,
  training:finite(metric==='loss'?(output?record.training?.outputs[output]?.loss:record.training?.loss):record.training?.outputs[output]?.[metric]),
  validation:finite(metric==='loss'?(output?record.validation?.outputs[output]?.loss:record.validation?.loss)??(!record.validation?.outputs?record.validation_loss:undefined):record.validation?.outputs[output]?.[metric]??(metric==='accuracy'&&!record.validation?.outputs?record.accuracy:undefined)),
 })).filter(point=>point.batch!==undefined||point.training!==undefined||point.validation!==undefined);
}
/** Rendering reduction preserves each series' bucket extrema and boundary observations. */
export function downsample(points:ChartPoint[],budget=900):ChartPoint[]{
 if(points.length<=budget)return points;
 const indices=new Set([0,points.length-1]);
 const bucketSize=Math.ceil(points.length/Math.max(1,Math.floor((budget-2)/6)));
 for(let start=0;start<points.length;start+=bucketSize){
  const end=Math.min(points.length,start+bucketSize);
  for(const key of ['batch','training','validation'] as const){
   let min=-1,max=-1;
   for(let index=start;index<end;index++)if(points[index][key]!==undefined){
    if(min<0||points[index][key]!<points[min][key]!)min=index;
    if(max<0||points[index][key]!>points[max][key]!)max=index;
   }
   if(min>=0)indices.add(min);if(max>=0)indices.add(max);
  }
 }
 return [...indices].sort((a,b)=>a-b).map(index=>points[index]);
}
export function valueDomain(points:ChartPoint[],percentage:boolean):[number,number]{
 const values=points.flatMap(point=>[point.batch,point.training,point.validation].filter((value):value is number=>value!==undefined));
 if(!values.length)return [0,1];
 let min=Infinity,max=-Infinity;for(const value of values){min=Math.min(min,value);max=Math.max(max,value)}
 const padding=Math.max((max-min)*.12,Math.abs(max)*.025,.001);
 return [Math.max(0,min-padding),percentage?Math.min(1,max+padding):max+padding];
}

/** Pan a selection in absolute steps, preserving its width at either boundary. */
export function panRange(value:StepRange,delta:number,min:number,max:number):StepRange {
 const width=Math.min(max-min,value[1]-value[0]);
 const start=Math.max(min,Math.min(max-width,Math.round(value[0]+delta)));
 return [start,start+width];
}
