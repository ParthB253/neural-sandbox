import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chartPoints, downsample, mergeMetrics, valueDomain } from '../src/training/chartData.ts';

test('merge preserves batch loss when evaluation arrives at the same step',()=>{
 const merged=mergeMetrics([{step:20,loss:.8},{step:10,loss:.9}],[{step:20,validation_loss:.5}]);
 assert.deepEqual(merged,[{step:10,loss:.9},{step:20,loss:.8,validation_loss:.5}]);
});
test('legacy values remain validation measurements and sparse values are never zero-filled',()=>{
 const points=chartPoints([{step:1,loss:.9},{step:3,validation_loss:.5,accuracy:.8}],'accuracy','');
 assert.deepEqual(points,[{step:3,batch:undefined,training:undefined,validation:.8}]);
 assert.deepEqual(chartPoints([{step:1,loss:.9}],'loss','')[0],{step:1,batch:.9,training:undefined,validation:undefined});
});
test('dense rendering retains peaks, sparse evaluations and endpoints',()=>{
 const points=Array.from({length:10000},(_,step)=>({step,batch:step===4567?20:1,validation:step===5001?3:undefined}));
 const rendered=downsample(points,900);
 assert.ok(rendered.length<=900);
 assert.equal(rendered[0].step,0);assert.equal(rendered.at(-1)?.step,9999);
 assert.ok(rendered.some(point=>point.step===4567));assert.ok(rendered.some(point=>point.step===5001));
 assert.equal(points.length,10000);
});
test('vertical domain fits actual observations, including single point and percentages',()=>{
 const [min,max]=valueDomain([{step:1,validation:.8}],true);
 assert.ok(min<.8&&max>.8&&max<=1);
 assert.deepEqual(valueDomain([],false),[0,1]);
 const domain=valueDomain([{step:1,batch:.001}],false);assert.ok(domain[0]>=0&&domain[1]>.001);
});
test('missing output metrics do not fall back to aggregate validation scores',()=>{
 const records=[{step:5,validation_loss:.4,accuracy:.9,validation:{loss:.4,count:10,outputs:{other:{kind:'classification' as const,loss:.4,accuracy:.9}}}}];
 assert.deepEqual(chartPoints(records,'loss','missing'),[]);
 assert.deepEqual(chartPoints(records,'accuracy','missing'),[]);
});

test('navigator pans absolute steps without snapping when sampled history changes',async()=>{
 const {panRange}=await import('../src/training/chartData.ts');
 const selected=[8000,9000] as const;
 const refreshed=Array.from({length:10000},(_,step)=>({step,batch:1}));
 downsample(refreshed,450);
 assert.deepEqual(panRange(selected,800,0,10000),[8800,9800]);
 assert.deepEqual(panRange(selected,5000,0,10000),[9000,10000]);
 assert.deepEqual(panRange(selected,-9000,0,10000),[0,1000]);
 // Live-run growth changes only the upper limit, not the selected step values.
 assert.deepEqual(panRange(selected,0,0,11000),selected);
});
