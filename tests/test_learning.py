from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from agent.learning import score, propose, review, past_logs, calibrate_predictions, public_summary
from agent.strategy_evolution import deploy, healthcheck, load, checked_function
from evals.experiments import canonical_hash
from tests.test_investment import example


def sample_rows(confidence=.9, correct=False):
    # Explicit test fixtures, not historical investment results.
    dates = [f'2020-01-{d:02d}' for d in range(1, 9)] + ['2022-01-01', '2022-01-02']
    return [{'id': f'{day}:{i}', 'symbol': str(i), 'date': day, 'decision_at': day + 'T17:30:00+08:00',
             'outcome_date': '2021-06-01' if day.startswith('2020') else '2023-06-01',
             'confidence': confidence, 'correct': correct} for day in dates for i in range(5)]


class LearningTests(unittest.TestCase):
    def test_exact_future_session_and_full_252_session_maturity(self):
        sessions = pd.bdate_range('2020-01-01', periods=255).strftime('%Y-%m-%d').tolist()
        log = {'timing': {'decision_at': sessions[0] + 'T17:30:00+08:00', 'execution_session': sessions[1]},
               'predictions': [{'symbol': '300750', 'direction': 'up', 'confidence': .6, 'horizon_days': 252}]}
        target = sessions[253]
        histories = {'300750': [{'date': day, 'open': 100., 'close': 110. if day == target else 100., 'volume': 1000} for day in sessions]}
        rows, pending = score([log], histories, sessions, target + 'T14:59:00+08:00')
        self.assertEqual((rows, pending), ([], 1))
        rows, pending = score([log, log], histories, sessions, target + 'T17:30:00+08:00')
        self.assertEqual((len(rows), pending), (1, 0))
        self.assertEqual(rows[0]['outcome_date'], target)
        self.assertTrue(rows[0]['correct'])
        self.assertEqual(score([log], {}, sessions, target + 'T17:30:00+08:00'), ([], 1))
        log['timing']['execution_session'] = sessions[0]  # same known close cannot be a forward anchor
        self.assertEqual(score([log], histories, sessions, target + 'T17:30:00+08:00'), ([], 0))

    def test_purges_training_labels_unavailable_at_validation_decision(self):
        rows = sample_rows()
        for row in rows:
            if row['date'].startswith('2020'):
                row['outcome_date'] = '2022-02-01'
        report, parameters = propose(rows)
        self.assertEqual(report['train_count'], 0)
        self.assertIsNone(parameters)

    def test_improving_candidate_and_regression_control(self):
        evaluation, parameters = propose(sample_rows())
        self.assertEqual(evaluation['status'], 'adopted')
        self.assertEqual((evaluation['train_count'], evaluation['test_count']), (40, 10))
        self.assertEqual(parameters['version'], canonical_hash({k: v for k, v in parameters.items() if k != 'version'}))
        report, unchanged = propose(sample_rows(1., True))
        self.assertEqual(report['status'], 'rejected')
        self.assertIsNone(unchanged)
        self.assertEqual(propose(sample_rows(), parameters)[0]['test_count'], 0)

    def test_code_activation_has_tests_control_rollback_and_preserves_raw_probability(self):
        _, parameters = propose(sample_rows())
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            code = deploy(parameters, home, '2024-01-01T17:30:00+08:00')
            self.assertEqual((code['tests'], code['control'], code['rollback']), ('passed',)*3)
            result = {'predictions': [{'confidence': .9}]}
            calibrate_predictions(result, {'active_parameters': parameters, 'strategy_code': code})
            self.assertEqual(result['predictions'][0]['confidence'], .9)
            self.assertEqual(result['predictions'][0]['calibrated_confidence'], 0.)
            entry = load(home, code['version'], '2024-01-02T17:30:00+08:00')
            with self.assertRaises(ValueError):
                checked_function(entry['source'] + '\nimport os\n', parameters['calibrator'])
            with self.assertRaises(RuntimeError):
                healthcheck(home, '2023-01-01T17:30:00+08:00')
            (home / 'versions' / (code['version'] + '.py')).write_text('raise Exception()', encoding='utf-8')
            self.assertEqual(healthcheck(home, '2024-01-02T17:30:00+08:00')['status'], 'rolled_back_to_raw')

    def test_corrupt_candidate_rolls_back_to_previous_checked_code(self):
        _, first = propose(sample_rows())
        _, second = propose(sample_rows(.8))
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            original = deploy(first, home, '2024-01-01T17:30:00+08:00')
            new = deploy(second, home, '2024-01-02T17:30:00+08:00')
            (home / 'versions' / (new['version'] + '.py')).write_text('corrupt', encoding='utf-8')
            result = healthcheck(home, '2024-01-03T17:30:00+08:00')
            self.assertEqual(result['status'], 'rolled_back')
            self.assertEqual(result['version'], original['version'])

    def test_failed_control_cannot_activate_code(self):
        _, parameters = propose(sample_rows())
        parameters['validation']['status'] = 'rejected'
        parameters['version'] = canonical_hash({k: v for k, v in parameters.items() if k != 'version'})
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            deploy(parameters, Path(temp), '2024-01-01T17:30:00+08:00')

    def test_cold_start_monotonic_cutoff_and_public_projection(self):
        with tempfile.TemporaryDirectory() as temp:
            snapshot = review([], {}, [], '2026-10-30T17:30:00+08:00', home=Path(temp))
            self.assertEqual(snapshot['mature_samples'], 0)
            self.assertIsNone(snapshot['active_parameters'])
            self.assertEqual(snapshot['status'], 'insufficient_data')
            self.assertNotIn('active_parameters', public_summary(snapshot))
            self.assertNotIn('source', public_summary(snapshot))
            with self.assertRaises(ValueError):
                review([], {}, [], '2026-10-29T17:30:00+08:00', home=Path(temp))

    def test_review_activates_and_uses_recovered_previous_parameters(self):
        with tempfile.TemporaryDirectory() as temp, patch('agent.learning.score', return_value=(sample_rows(), 0)):
            home = Path(temp)
            snapshot = review([], {}, [], '2024-01-01T17:30:00+08:00', home=home)
            self.assertEqual(snapshot['strategy_code']['status'], 'activated')
            first = snapshot['active_parameter_version']
            with patch('agent.learning.score', return_value=(sample_rows(.8), 0)):
                # Old holdout cannot be reused, even if the proposed raw confidence changes.
                snapshot = review([], {}, [], '2024-01-02T17:30:00+08:00', home=home)
            self.assertEqual(snapshot['active_parameter_version'], first)
            self.assertEqual(snapshot['evaluation']['test_count'], 0)
            source_path = home / 'strategy' / 'versions' / (snapshot['strategy_code']['version'] + '.py')
            source_path.write_text('broken', encoding='utf-8')
            snapshot = review([], {}, [], '2024-01-03T17:30:00+08:00', home=home)
            # Rollback does not grant permission to choose again on the spent holdout.
            self.assertIsNone(snapshot['active_parameter_version'])
            self.assertEqual(snapshot['strategy_code']['status'], 'rolled_back_to_raw')
            self.assertEqual(snapshot['evaluation']['test_count'], 0)

    def test_only_formal_validated_past_logs_can_train(self):
        _, log = example()
        log.update(execution_mode='paper', timing={'decision_at': '2026-10-28T17:30:00+08:00'},
                   agent_sdk={'role_models': {'investment-manager': {'model': 'gpt-6.1-sol'}}})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'after_close').mkdir()
            for name, changes in [('valid', {}), ('shadow', {'execution_mode': 'shadow'}),
                ('mock', {'data_sources': ['mock']}), ('future', {'timing': {'decision_at': '2026-11-01T17:30:00+08:00'}}),
                ('changed', {'dataset_fingerprint': 'bad'}), ('failed', {'risk_review': {}})]:
                (root / 'after_close' / (name + '.cn.json')).write_text(json.dumps({**deepcopy(log), **changes}), encoding='utf-8')
            self.assertEqual(len(past_logs('2026-10-30T17:30:00+08:00', root)), 1)
