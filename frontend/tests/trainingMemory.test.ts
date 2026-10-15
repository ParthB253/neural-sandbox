import { test } from 'node:test';
import assert from 'node:assert/strict';
import { compactRun, LIVE_METRIC_LIMIT } from '../src/training/runSnapshot.ts';
import { boundedHistory, CURVE_BUDGET } from '../src/training/curveCache.ts';
import type { TrainingRun, TrainingMetric } from '../src/training/types.ts';
import { mergeMetrics } from '../src/training/chartData.ts';

const evaluation={loss:.4,count:2048,outputs:{o:{kind:'classification' as const,loss:.4,accuracy:.8,
 per_class:Array.from({length:100},(_,index)=>({index,label:`Class ${index}`,precision:.8,recall:.8,f1:.8,support:20})),
 confusion_matrix:Array.from({length:30},()=>Array(30).fill(1) as number[])}}};

test('live run storage is bounded and retains only one detailed evaluation',()=>{
 const metrics=Array.from({length:2000},(_,step)=>({step,training:evaluation,validation:evaluation}));
 const run={id:'run',project:{network:{id:'network'}},metrics,evaluation_summary:metrics.at(-1),train_indices:Array.from({length:100000},(_,i)=>i)} as TrainingRun;
 const next=compactRun(run);
 assert.equal(next.metrics.length,LIVE_METRIC_LIMIT);
 assert.equal(next.train_count,100000);
 assert.equal(next.train_indices,undefined);
 assert.equal(next.metrics[0].training?.outputs.o.per_class,undefined);
 assert.equal(next.evaluation_summary?.training?.outputs.o.per_class?.length,100);
 assert.ok(JSON.stringify(next).length<JSON.stringify(run).length/100);
 assert.equal(compactRun({...run,project:{...run.project}},next).project,next.project);
});

test('large server histories are reduced before caching, retaining peaks and zoom detail',()=>{
 const metrics:TrainingMetric[]=Array.from({length:100000},(_,step)=>({step,loss:step===4567?20:.4,
 ...(step%100===0?{training:evaluation,validation:evaluation}:{})}));
 const data={metrics,total:metrics.length,start_step:0,end_step:99999};
 const cache=boundedHistory(data,'loss','o');
 assert.ok(cache.metrics.length<=CURVE_BUDGET);
 assert.ok(cache.metrics.some(metric=>metric.step===4567));
 assert.equal(cache.metrics[0].step,0);assert.equal(cache.metrics.at(-1)?.step,99999);
 assert.ok(cache.metrics.every(metric=>!metric.training?.outputs.o.per_class));
 const window=boundedHistory({...data,metrics:metrics.slice(4560,4571)},'loss','o');
 assert.equal(window.metrics.length,11);
 assert.equal(window.metrics.find(metric=>metric.step===4567)?.loss,20);
});

test('batch snapshots cannot erase evaluation at the same step',()=>{
 const run={id:'run',metrics:[{step:1,loss:.3}]} as TrainingRun;
 const merged=mergeMetrics([{step:1,training:evaluation}],compactRun(run).metrics);
 assert.equal(merged[0].training?.loss,.4);
 assert.equal(merged[0].loss,.3);
});

test('incremental snapshots preserve static config and class tables without growing the cache',async()=>{
 const {applyRunUpdate}=await import('../src/training/runSnapshot.ts');
 let run={id:'run',project:{network:{id:'network'}},metrics:[],evaluation_summary:{step:1,time:1,training:evaluation},train_count:2048} as unknown as TrainingRun;
 const config=run.project,summary=run.evaluation_summary;
 for(let step=1;step<=10000;step++)run=applyRunUpdate(run,{step,metrics:[{step,loss:1/step}]});
 assert.equal(run.metrics.length,LIVE_METRIC_LIMIT);
 assert.equal(run.metrics[0].step,10000-LIVE_METRIC_LIMIT+1);
 assert.equal(run.project,config);
 assert.equal(run.evaluation_summary,summary);
 assert.equal(run.train_count,2048);
});
