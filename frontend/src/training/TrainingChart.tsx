import { useMemo, useState } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { chartPoints, downsample, mergeMetrics, valueDomain, type ChartMetric, type StepRange } from './chartData';
import type { TrainingRun } from './types';
import { useCurveHistory } from './useCurveHistory';
import RangeNavigator from './RangeNavigator';
import './chart.css';

const metricLabels:Record<ChartMetric,string>={loss:'Loss',accuracy:'Accuracy',f1_macro:'Macro F1',precision_macro:'Macro precision',recall_macro:'Macro recall',mae:'MAE',rmse:'RMSE'};
const terminal=(status:string)=>['completed','cancelled','failed','interrupted'].includes(status);

export default function TrainingChart({run}:{run:TrainingRun}){
 return <RunChart key={run.id} run={run}/>;
}
function RunChart({run}:{run:TrainingRun}){
 const [metric,setMetric]=useState<ChartMetric>('loss'),[output,setOutput]=useState('');
 const [range,setRange]=useState<StepRange|null>(null),[selection,setSelection]=useState<StepRange|null>(null);
 const [rangeError,setRangeError]=useState('');
 const [draftRange,setDraftRange]=useState<[string,string]|null>(null);
 const finished=terminal(run.status);
 const outputs=useMemo(()=>[...new Set([...Object.keys(run.evaluation_summary?.training?.outputs??{}),...Object.keys(run.evaluation_summary?.validation?.outputs??{}),...run.metrics.flatMap(record=>[...Object.keys(record.training?.outputs??{}),...Object.keys(record.validation?.outputs??{})])])],[run.evaluation_summary,run.metrics]);
 const selectedOutput=output||outputs[0]||'';
 const history=useCurveHistory(run.id,finished,metric,selectedOutput,range);
 const metrics=useMemo(()=>mergeMetrics(history.overview?.metrics??[],history.detail?.metrics??[],run.metrics),[history.overview,history.detail,run.metrics]);
 const points=useMemo(()=>chartPoints(metrics,metric,selectedOutput),[metrics,metric,selectedOutput]);
 // Navigator bounds follow recorded steps, never sampled array positions or requested bounds.
 const first=0,last=Math.max(1,run.step);
 const visibleRange=useMemo<StepRange>(()=>range??[first,last],[range,first,last]);
 const visible=useMemo(()=>points.filter(point=>point.step>=visibleRange[0]&&point.step<=visibleRange[1]),[points,visibleRange]);
 const plotted=useMemo(()=>downsample(visible),[visible]);
 const overview=useMemo(()=>downsample(chartPoints(mergeMetrics(history.overview?.metrics??[],run.metrics),metric,selectedOutput),450),[history.overview,run.metrics,metric,selectedOutput]);
 const percentage=['accuracy','f1_macro','precision_macro','recall_macro'].includes(metric);
 const domain=valueDomain(visible,percentage);
 function zoom(from:number,to:number){
  const start=Math.max(first,Math.round(Math.min(from,to))),end=Math.min(last,Math.round(Math.max(from,to)));
  if(!Number.isFinite(start)||!Number.isFinite(end)||start>=end){setRangeError('Enter an end step greater than the start step.');return}
  setRange([start,end]);setDraftRange(null);setRangeError('');
 }
 const formatValue=(value:number)=>percentage?`${(value*100).toFixed(1)}%`:value.toLocaleString(undefined,{maximumSignificantDigits:4});
 return <section className="training-chart" aria-label="Learning curves">
  <div className="training-chart-heading"><div><h2>Learning curves</h2><p>Compare fixed-model evaluations; batch loss shows live training feedback.</p></div><div className="training-chart-selectors">
   <label>Metric<select value={metric} onChange={event=>setMetric(event.target.value as ChartMetric)}>{Object.entries(metricLabels).map(([value,label])=><option key={value} value={value}>{label}</option>)}</select></label>
   {outputs.length>1&&<label>Output<select value={selectedOutput} onChange={event=>setOutput(event.target.value)}>{outputs.map(id=><option key={id} value={id}>{run.project.network.layers.find(layer=>layer.id===id)?.id??id}</option>)}</select></label>}
  </div></div>
  <form className="training-chart-range" onSubmit={event=>{event.preventDefault();const data=new FormData(event.currentTarget);zoom(Number(data.get('start')),Number(data.get('end')))}}>
   <label>From step<input name="start" type="number" min={first} max={last} value={draftRange?.[0]??visibleRange[0]} onChange={event=>setDraftRange([event.target.value,draftRange?.[1]??String(visibleRange[1])])} required/></label>
   <label>To step<input name="end" type="number" min={first} max={last} value={draftRange?.[1]??visibleRange[1]} onChange={event=>setDraftRange([draftRange?.[0]??String(visibleRange[0]),event.target.value])} required/></label>
   <button type="submit">Apply range</button><button type="button" onClick={()=>{setRange(null);setDraftRange(null);setRangeError('')}} disabled={!range}>Reset zoom</button>
   <span className="training-chart-follow">{range?'Viewing selected range':finished?'Full recorded run':'Following latest steps'}</span>
  </form>
  {rangeError&&<p role="alert">{rangeError}</p>}
  <p className="training-chart-hint">Drag across the chart to zoom, or use the range controls. Hover for recorded values.</p>
  {points.length>0?<>
   <div className="training-chart-plot" role="img" aria-label={`${metricLabels[metric]} from step ${visibleRange[0]} to ${visibleRange[1]}. ${visible.length} recorded observations.`}>
    <ResponsiveContainer width="100%" height="100%" minWidth={0}>
     <LineChart data={plotted} margin={{top:12,right:20,bottom:16,left:8}} onMouseDown={event=>{if(event?.activeLabel!==undefined)setSelection([Number(event.activeLabel),Number(event.activeLabel)])}} onMouseMove={event=>{if(selection&&event?.activeLabel!==undefined)setSelection([selection[0],Number(event.activeLabel)])}} onMouseUp={()=>{if(selection&&selection[0]!==selection[1])zoom(selection[0],selection[1]);setSelection(null)}} onMouseLeave={()=>setSelection(null)}>
      <CartesianGrid strokeDasharray="3 3" vertical={false}/>
      <XAxis type="number" dataKey="step" domain={[...visibleRange]} allowDataOverflow tickFormatter={value=>Number(value).toLocaleString()} label={{value:'Optimiser step',position:'insideBottom',offset:-10}}/>
      <YAxis type="number" domain={domain} allowDataOverflow width={70} tickFormatter={formatValue}/>
      <Tooltip filterNull labelFormatter={value=>`Step ${Number(value).toLocaleString()}`} formatter={value=>typeof value==='number'?formatValue(value):'—'}/>
      <Legend verticalAlign="top"/>
      {metric==='loss'&&<Line type="linear" dataKey="batch" name={outputs.length>1?"Training · batch loss (all outputs)":"Training · batch loss"} stroke="#8eaac9" strokeWidth={1} dot={visible.filter(point=>point.batch!==undefined).length===1?{r:4}:false} activeDot={{r:4}} connectNulls isAnimationActive={false}/>}
      <Line type="linear" dataKey="training" name="Training · evaluation" stroke="#3276d2" strokeWidth={2} dot={visible.filter(point=>point.training!==undefined).length===1?{r:4}:false} activeDot={{r:4}} connectNulls isAnimationActive={false}/>
      <Line type="linear" dataKey="validation" name="Validation · evaluation" stroke="#9464ad" strokeWidth={2} dot={visible.filter(point=>point.validation!==undefined).length===1?{r:4}:false} activeDot={{r:4}} connectNulls isAnimationActive={false}/>
      {selection&&<ReferenceArea x1={Math.min(...selection)} x2={Math.max(...selection)} fill="#3276d2" fillOpacity={.14}/>}
     </LineChart>
    </ResponsiveContainer>
   </div>
   {!visible.length&&<p role="status">No {metricLabels[metric].toLowerCase()} observations in this range. Expand the range to see recorded evaluations.</p>}
   <div className="training-chart-overview" aria-label="Full run overview"><ResponsiveContainer width="100%" height="100%" minWidth={0}><LineChart data={overview} margin={{top:4,right:20,bottom:0,left:78}}><XAxis type="number" domain={[first,last]} dataKey="step" hide/><Line dataKey="training" stroke="#3276d2" dot={false} connectNulls isAnimationActive={false}/><Line dataKey="validation" stroke="#9464ad" dot={false} connectNulls isAnimationActive={false}/>{metric==='loss'&&<Line dataKey="batch" stroke="#94a6bd" dot={false} connectNulls isAnimationActive={false}/>}</LineChart></ResponsiveContainer></div>
   <RangeNavigator min={first} max={last} value={visibleRange} onChange={zoom}/>

   <p className="training-chart-hint">Bounded preview of recorded observations{plotted.length<visible.length?` · ${plotted.length.toLocaleString()} shown, preserving peaks and endpoints`:''}. Missing measurements are left unavailable.</p>
  </>:<p className="training-chart-empty" role="status">{metric==='loss'?'Learning curves will appear once training starts.':`No ${metricLabels[metric].toLowerCase()} evaluations are recorded for this output. Select a different metric or output.`}</p>}
  {!history.overview&&!history.unavailable&&<p className="training-chart-hint" role="status">Loading earlier recorded history…</p>}
  {history.unavailable&&<p className="training-chart-hint" role="status">Earlier history is unavailable. Showing available live observations; history will retry while training.</p>}
 </section>;
}
