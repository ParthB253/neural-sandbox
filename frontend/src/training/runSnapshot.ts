import { mergeMetrics } from './chartData.ts';
import type { EvaluationMetrics, TrainingMetric, TrainingRun } from './types.ts';
export const LIVE_METRIC_LIMIT=200;
function scalarEvaluation(evaluation:EvaluationMetrics):EvaluationMetrics {
 return {...evaluation,outputs:Object.fromEntries(Object.entries(evaluation.outputs).map(([name,output])=>{
  const {per_class,confusion_matrix,...scores}=output;
  void per_class;void confusion_matrix;
  return [name,scores];
 }))};
}
export function scalarMetric(metric:TrainingMetric):TrainingMetric {
 return {...metric,...(metric.training?{training:scalarEvaluation(metric.training)}:{}),...(metric.validation?{validation:scalarEvaluation(metric.validation)}:{})};
}
/** Also bounds snapshots from older servers; retain class tables only for the latest evaluation. */
export function compactRun(next:TrainingRun,previous?:TrainingRun):TrainingRun {
 const {train_indices,validation_indices,...record}=next;
 const summary=next.evaluation_summary??next.metrics.findLast(metric=>metric.training||metric.validation);
 const unchanged=previous?.evaluation_summary&&summary?.step===previous.evaluation_summary.step&&summary?.time===previous.evaluation_summary.time;
 return {...record,project:previous?.id===next.id?previous.project:next.project,
  train_count:next.train_count??train_indices?.length,validation_count:next.validation_count??validation_indices?.length,
  evaluation_summary:unchanged?previous.evaluation_summary:summary,
  metrics:next.metrics.slice(-LIVE_METRIC_LIMIT).map(scalarMetric)};
}

/** Compact SSE frames carry static configuration once and incremental curve measurements. */
export function applyRunUpdate(previous:TrainingRun,update:Partial<TrainingRun>):TrainingRun {
 return compactRun({...previous,...update,metrics:update.metrics?mergeMetrics(previous.metrics,update.metrics):previous.metrics},previous);
}
