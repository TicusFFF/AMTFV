"""Offline checks for the native-output adapter and the public scoring entry."""
import json
import socket
import unittest
from unittest.mock import patch

import evaluate
import prepare_inputs


class OfflineEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.network = patch.object(socket.socket, 'connect', side_effect=AssertionError('Unexpected network call'))
        self.connect = self.network.start()
        self.addCleanup(self.network.stop)

    def test_score_native_final_responses_and_keep_empty_outputs(self):
        data = [{'q': 'First?', 'A_original': '0.5'}, {'q': 'Second?', 'A_original': 0}]
        records = {'args': {'dataset_key': 'example'}, 'logs': [
            {'id': 'example_1', 'query': 'Second?', 'final_response': '', 'final_result': '0'},
            {'id': 'example_0', 'query': 'First?', 'final_response': r'\boxed{\frac{1}{2}}',
             'turns': [{'response': 'Incorrect earlier response.'}]},
        ]}
        with patch.object(prepare_inputs, 'read_json', side_effect=[data, records]):
            doc = prepare_inputs.prepare_inputs([('example', 'dataset.json', 'generated.json')])
        self.assertEqual(doc['expected_ids'], ['example_0', 'example_1'])
        scores = evaluate.evaluate_batch(doc)
        self.assertEqual((scores['status'], scores['n_expected'], scores['n_no_answer'], scores['acc']),
                         ('completed', 2, 1, 50.0))
        self.assertEqual([r['correct'] for r in scores['rows']], [True, False])
        self.assertEqual(scores['rows'][0]['extraction']['answer'], r'\frac{1}{2}')
        self.connect.assert_not_called()

    def test_reject_mismatched_question_before_scoring(self):
        data = [{'problem': 'First?', 'answer': 2}]
        records = {'logs': [{'id': 'example_0', 'query': 'Unrelated?', 'final_response': 'Final Answer: 2'}]}
        with patch.object(prepare_inputs, 'read_json', side_effect=[data, records]):
            with self.assertRaisesRegex(ValueError, 'does not match'):
                prepare_inputs.prepare_inputs([('example', 'dataset.json', 'generated.json')])

    def test_extractor_cannot_see_reference_answer(self):
        calls = []

        def extract(**kwargs):
            calls.append(kwargs['messages'])
            return 'seventeen'

        with patch.object(evaluate.extractor, 'call_llm', side_effect=extract), \
                patch.object(evaluate.judge, '_call_llm', return_value='False'):
            row = evaluate.evaluate_answer('A sample question?', 'I settled on seventeen.', 'reference-only text')
        self.assertEqual((row['status'], row['correct']), ('completed', False))
        self.assertEqual(len(calls), 1)
        user_data = json.loads(calls[0][1]['content'])
        self.assertEqual(user_data, {'problem': 'A sample question?', 'solution_response': 'I settled on seventeen.'})


if __name__ == '__main__':
    unittest.main()
