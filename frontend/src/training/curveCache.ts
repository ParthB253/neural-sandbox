import { chartPoints, downsample, mergeMetrics, type ChartMetric } from './chartData.ts';
import { scalarMetric } from './runSnapshot.ts';
import type { MetricHistory } from './types.ts';
export const CURVE_BUDGET=1200;

/** Enforce the client budget too, including when connected to an older API. */
export function boundedHistory(data:MetricHistory,metric:ChartMetric,output:string):MetricHistory {
 const records=mergeMetrics(data.metrics.map(scalarMetric));
 const selected=new Set(downsample(chartPoints(records,metric,output),CURVE_BUDGET).map(point=>point.step));
 return {...data,metrics:records.filter(record=>selected.has(record.step))};
}
