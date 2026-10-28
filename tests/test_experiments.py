import tempfile
import unittest
from pathlib import Path
from copy import deepcopy

from evals.experiments import (canonical_hash, comparable_experiments,
                               decision_log_variant, list_experiments,
                               record_experiment)
from agent.sdk_runner import build_prompt_manifest
from evals.score import group_scored_rows, record_scored_experiments


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

    def test_same_template_groups_across_dates_and_separates_models_and_sources(self):
        def variant(day, symbols, model="model-a", offline=True):
            return decision_log_variant({"mode": "openai_agents_sdk", "offline": offline,
                "agent_sdk": {"prompt": build_prompt_manifest("us", day, symbols, "manager-v1"),
                              "role_models": {"manager": {"model": model}}}})
        first = variant("2026-09-01", ["AAPL"])
        self.assertEqual(first, variant("2026-09-02", ["META"]))
        self.assertNotEqual(first, variant("2026-09-01", ["AAPL"], model="model-b"))
        self.assertNotEqual(first, variant("2026-09-01", ["AAPL"], offline=False))

    def test_cross_date_scoring_calibrates_each_variant_and_forms_strict_ab(self):
        rows = []
        for version in ("manager-v1", "manager-v2"):
            for day in ("2026-08-10", "2026-08-11"):
                variant = decision_log_variant({"mode": "openai_agents_sdk", "execution_mode": "shadow",
                    "agent_sdk": {"prompt": build_prompt_manifest("us", day, ["AAPL"], version)}})
                for symbol in ("AAPL", "MSFT", "META", "AMD"):
                    rows.append({"date": day, "market": "us", "symbol": symbol, "actual": "up",
                                 "horizon_days": 5, "confidence": 0.7, "correct": True,
                                 "variant": variant, "input_fingerprint": day,
                                 "outcome_fingerprint": day + symbol})
        self.assertEqual(len(group_scored_rows(rows)), 2)
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "experiments.db"
            runs = record_scored_experiments(rows, db)
            self.assertEqual([run["metrics"]["sample_count"] for run in runs], [8, 8])
            self.assertTrue(all(run["metrics"]["calibration"]["status"] == "complete" for run in runs))
            self.assertEqual(len(comparable_experiments(db)), 1)
        changed = deepcopy(rows)
        changed[-1]["input_fingerprint"] = "different-account"
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "experiments.db"
            record_scored_experiments(changed, db)
            self.assertEqual(comparable_experiments(db), [])

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
