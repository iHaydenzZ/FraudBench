import csv
import os
from datetime import datetime
from typing import Dict, Any

class ExperimentRegistry:
    def __init__(self, registry_path: str = "results/registry.csv"):
        self.registry_path = registry_path
        self._ensure_registry_exists()

    def _ensure_registry_exists(self):
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Define schema
                writer.writerow([
                    "timestamp",
                    "experiment_name",
                    "seed",
                    "dataset",
                    "model_type",
                    "defence_type",
                    "attack_type",
                    "attack_epsilon",
                    "validity_rate",
                    "adv_validity_rate",
                    "clean_pr_auc",
                    "clean_precision",
                    "clean_recall",
                    "clean_f1",
                    "robust_pr_auc",
                    "robust_precision",
                    "robust_recall",
                    "robust_f1",
                    "clean_accuracy",
                    "robust_accuracy",
                    "train_time_sec",
                    "attack_time_sec"
                ])

    def log_experiment(
        self,
        config: Dict[str, Any],
        metrics_clean: Dict[str, float],
        metrics_robust: Dict[str, float] = None,
        validity_rate: float = 1.0,
        adv_validity_rate: float = None,
        train_time_sec: float = None,
        attack_time_sec: float = None
    ):
        if metrics_robust is None:
            metrics_robust = {}

        row = [
            datetime.now().isoformat(),
            config.get('experiment_name', 'n/a'),
            config.get('seed', 42),
            config['dataset']['name'],
            config['model']['type'],
            config.get('defence', {}).get('type', 'none'),
            config.get('attack', {}).get('type', 'none'),
            config.get('attack', {}).get('epsilon', 0.0),
            f"{validity_rate:.4f}",
            f"{adv_validity_rate:.4f}" if adv_validity_rate is not None else "n/a",
            f"{metrics_clean.get('pr_auc', 0):.4f}",
            f"{metrics_clean.get('precision', 0):.4f}",
            f"{metrics_clean.get('recall', 0):.4f}",
            f"{metrics_clean.get('f1', 0):.4f}",
            f"{metrics_robust.get('pr_auc', 0):.4f}",
            f"{metrics_robust.get('precision', 0):.4f}",
            f"{metrics_robust.get('recall', 0):.4f}",
            f"{metrics_robust.get('f1', 0):.4f}",
            f"{metrics_clean.get('accuracy', 0):.4f}",
            f"{metrics_robust.get('accuracy', 0):.4f}",
            f"{train_time_sec:.2f}" if train_time_sec is not None else "n/a",
            f"{attack_time_sec:.2f}" if attack_time_sec is not None else "n/a"
        ]
        
        with open(self.registry_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
        print(f"  Experiment logged to {self.registry_path}")
