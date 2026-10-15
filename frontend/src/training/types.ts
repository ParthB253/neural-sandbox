import type { ProjectSpec } from '../network';

export interface ClassMetrics { index:number; label:string; support:number; precision:number|null; recall:number|null; f1:number|null }
export interface OutputMetrics {
 kind:'classification'|'regression'; loss:number; accuracy?:number|null;
 precision_macro?:number|null; recall_macro?:number|null; f1_macro?:number|null;
 f1_weighted?:number|null; balanced_accuracy?:number|null;
 mae?:number|null; rmse?:number|null; r2?:number|null;
 per_class?:ClassMetrics[]; confusion_matrix?:number[][]; unavailable_reason?:string;
}
export interface EvaluationMetrics { loss:number; count:number; outputs:Record<string,OutputMetrics> }
export interface TrainingMetric {
 step:number; epoch?:number; time?:number; loss?:number; validation_loss?:number; accuracy?:number|null;
 training?:EvaluationMetrics; validation?:EvaluationMetrics;
}
export interface TrainingRun {
 created_at:number; updated_at?:number; completed_at?:number; id:string; sequence:number;
 status:string; stage:string; step:number; total_steps:number; epoch:number; project:ProjectSpec;
 error?:string; warnings?:string[]; activity?:{time:number;message:string}[];
 checkpoints:string[]; checkpoint_details?:Record<string,{step:number;epoch:number;time:number;validation_loss?:number|null}>;
 metrics:TrainingMetric[]; evaluation_summary?:TrainingMetric;
 best_step?:number; best_epoch?:number; best_validation_loss?:number;
 train_count?:number; validation_count?:number;
 data_revision?:string; train_indices?:number[]; validation_indices?:number[];
}
export interface MetricHistory { metrics:TrainingMetric[]; total:number; start_step:number; end_step:number }
