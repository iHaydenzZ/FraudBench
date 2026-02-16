"""Tests for experiment registry."""

import csv
import pytest
from evaluation.registry import ExperimentRegistry


@pytest.fixture
def registry(tmp_path):
    """Create a fresh registry in a temp directory."""
    path = str(tmp_path / "registry.csv")
    return ExperimentRegistry(registry_path=path)


class TestRegistrySchema:
    """Tests for registry CSV schema."""

    def test_registry_header_contains_seed(self, registry):
        """Seed column must exist in the registry header."""
        with open(registry.registry_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "seed" in header

    def test_log_experiment_writes_seed(self, registry):
        """log_experiment must write the seed value from config."""
        config = {
            "experiment_name": "test_exp",
            "seed": 123,
            "dataset": {"name": "ccfd"},
            "model": {"type": "neural"},
            "defence": {"type": "none"},
            "attack": {"type": "capgd", "epsilon": 0.1},
        }
        metrics_clean = {"pr_auc": 0.8, "precision": 0.7, "recall": 0.6, "f1": 0.65, "accuracy": 0.9}
        metrics_robust = {"pr_auc": 0.5, "precision": 0.4, "recall": 0.3, "f1": 0.35, "accuracy": 0.7}

        registry.log_experiment(
            config,
            metrics_clean,
            metrics_robust,
            validity_rate=1.0,
            adv_validity_rate=0.99,
            train_time_sec=10.0,
            attack_time_sec=1.0,
        )

        with open(registry.registry_path, "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["seed"] == "123"

    def test_log_experiment_default_seed_42(self, registry):
        """If config has no seed key, registry should default to 42."""
        config = {
            "experiment_name": "test_exp",
            "dataset": {"name": "ccfd"},
            "model": {"type": "neural"},
            "defence": {"type": "none"},
            "attack": {"type": "capgd", "epsilon": 0.1},
        }
        metrics_clean = {"pr_auc": 0.8, "precision": 0.7, "recall": 0.6, "f1": 0.65, "accuracy": 0.9}

        registry.log_experiment(config, metrics_clean, None, validity_rate=1.0)

        with open(registry.registry_path, "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert row["seed"] == "42"
