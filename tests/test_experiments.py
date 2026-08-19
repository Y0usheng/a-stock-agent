import tempfile
import unittest
from pathlib import Path

from evals.experiments import (canonical_hash, comparable_experiments,
                               decision_log_variant, list_experiments,
                               record_experiment)


class ExperimentRegistryTests(unittest.TestCase):
    def test_canonical_hash_ignores_mapping_order(self):
        self.assertEqual(canonical_hash({"a": 1, "b": 2}),
                         canonical_hash({"b": 2, "a": 1}))

    def test_run_preserves_dataset_and_variant_fingerprints(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "experiments.db"
            row = record_experiment("ab", "us", {"items": [1, 2]},
                                    {"prompt_version": "manager-v2"},
                                    {"accuracy": 0.5}, db_path=db,
                                    created_at="2026-08-20T00:00:00+00:00")
            loaded = list_experiments(db)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["dataset_fingerprint"], row["dataset_fingerprint"])
            self.assertEqual(loaded[0]["variant"]["prompt_version"], "manager-v2")

    def test_decision_variant_contains_prompt_and_role_models(self):
        variant = decision_log_variant({"strategy_version": "x", "agent_sdk": {
            "prompt": {"version": "manager-v2", "sha256": "abc"},
            "role_models": {"manager": {"provider": "openai", "model": "gpt"}},
        }})
        self.assertEqual(variant["prompt_version"], "manager-v2")
        self.assertIn("manager", variant["role_models"])

    def test_only_same_dataset_different_variants_form_an_ab_group(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "experiments.db"
            dataset = {"fingerprint": "same-data"}
            record_experiment("model-ab", "us", dataset, {"prompt_version": "v1"},
                              {"accuracy": 0.5}, db_path=db,
                              created_at="2026-01-01T00:00:00+00:00")
            record_experiment("model-ab", "us", dataset, {"prompt_version": "v2"},
                              {"accuracy": 0.7}, db_path=db,
                              created_at="2026-01-02T00:00:00+00:00")
            groups = comparable_experiments(db)
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0]["variants"][1]["accuracy_delta_vs_baseline"], 0.2)


if __name__ == "__main__":
    unittest.main()
