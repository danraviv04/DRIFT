"""Strictly optional W&B training/evaluation monitoring for local DRIFT sweeps.

Scientific results remain authoritative in local CSV tables and figures. This
module gives every candidate run the same compact training, evaluation, and
progress metric schema; model choices belong in run config rather than metric
paths.
"""

import time


def start_wandb_progress(project, entity, group, name, config, tags=None):
    """Start one candidate run with progress, training, and evaluation axes."""
    import wandb  # local import: W&B remains untouched unless explicitly enabled

    config = dict(config)
    config.setdefault("wandb_role", "candidate training and evaluation monitor")
    config.setdefault("scientific_results", "local CSV tables and figures")
    run = wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=name,
        config=config,
        tags=list(tags) if tags else None,
        reinit="finish_previous",
    )
    # Do not define a wildcard: doing so auto-created dozens of one-point AUC
    # panels for single-size sweeps.
    wandb.define_metric("progress/fraction")
    wandb.define_metric(
        "progress/elapsed_minutes", step_metric="progress/fraction"
    )
    wandb.define_metric("progress/eta_minutes", step_metric="progress/fraction")
    wandb.define_metric("d1_size")
    for metric in (
        "evaluation/fusion_macro_auc",
        "evaluation/fusion_macro_ap",
        "evaluation/fusion_macro_f1_at_0_5",
        "evaluation/fusion_macro_precision_at_0_5",
        "evaluation/fusion_macro_recall_at_0_5",
        "evaluation/fusion_macro_mcc_at_0_5",
        "evaluation/logreg_macro_auc",
        "evaluation/logreg_macro_ap",
        "evaluation/afm_macro_auc",
        "evaluation/afm_macro_ap",
        "evaluation/d1_macro_auc",
        "evaluation/best_epoch_mean",
    ):
        wandb.define_metric(metric, step_metric="d1_size")
    wandb.define_metric("training/epoch")
    for metric in (
        "training/train_loss_mean",
        "training/inner_val_loss_mean",
        "training/inner_val_ap_mean",
        "training/inner_val_auc_mean",
        "training/inner_val_f1_at_0_5_mean",
        "training/inner_val_precision_at_0_5_mean",
        "training/inner_val_recall_at_0_5_mean",
        "training/inner_val_mcc_at_0_5_mean",
        "training/active_outer_folds",
    ):
        wandb.define_metric(metric, step_metric="training/epoch")
    return run


class WandbProgress:
    """Map nested fold work onto one monotonic sweep-completion stream."""

    def __init__(self, run, total_evaluations):
        if int(total_evaluations) <= 0:
            raise ValueError("total_evaluations must be positive")
        self.run = run
        self.total = int(total_evaluations)
        self.completed = 0
        self.started = time.monotonic()
        self.last_equivalent = -1.0
        self._emit(0.0, "starting")

    def _set_summary(self, equivalent_completed, status):
        elapsed = max(0.0, (time.monotonic() - self.started) / 60.0)
        fraction = min(1.0, max(0.0, equivalent_completed / self.total))
        self.run.summary["progress/status"] = str(status)
        self.run.summary["progress/completed_evaluations"] = int(self.completed)
        self.run.summary["progress/total_evaluations"] = int(self.total)
        self.run.summary["progress/fraction"] = float(fraction)
        self.run.summary["progress/elapsed_minutes"] = float(elapsed)
        return fraction, elapsed

    def _emit(self, equivalent_completed, status):
        fraction, elapsed = self._set_summary(equivalent_completed, status)
        payload = {
            "progress/fraction": float(fraction),
            "progress/elapsed_minutes": float(elapsed),
        }
        if fraction > 0:
            payload["progress/eta_minutes"] = float(
                elapsed * (1.0 - fraction) / fraction
            )
        self.run.log(payload)
        self.last_equivalent = float(equivalent_completed)

    def set_status(self, status):
        self._set_summary(self.completed, status)

    def callback(self, label):
        """Return a fold callback that maps one evaluation onto global progress."""
        base_completed = int(self.completed)

        def report(completed_folds, total_folds, detail=None):
            total_folds = max(1, int(total_folds))
            within = min(1.0, max(0.0, float(completed_folds) / total_folds))
            detail = detail or {}
            item = (
                detail.get("held_out_gene")
                or detail.get("outer_gene")
                or detail.get("candidate_id")
            )
            status = f"{label}; {item}" if item else str(label)
            self._emit(base_completed + within, status)

        return report

    def complete(self, label):
        self.completed += 1
        if self.last_equivalent < self.completed - 1e-12:
            self._emit(float(self.completed), label)
        else:
            self._set_summary(float(self.completed), label)

    def finish(self):
        self.completed = self.total
        if self.last_equivalent < self.total - 1e-12:
            self._emit(float(self.total), "completed")
        else:
            self._set_summary(float(self.total), "completed")
