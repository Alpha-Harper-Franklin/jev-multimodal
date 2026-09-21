import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from jev_multimodal.adapters import transcript
from jev_multimodal.cuda_qwen import stable_prefix, suffix_layout
from jev_multimodal.evidence import Evidence, Snapshot, ExtractionCache, RevisionGate, visual_evidence
from jev_multimodal.metrics import binary_metrics, fit_temperature
from jev_multimodal.schema import Question, normalized
from jev_multimodal.typesafe import JevClient, JevError


def evidence(revision='1', timestamp=100, locator=None, modality='image'):
    return Evidence('camera', revision, modality, 'test:1', '0'*64, {'text': 'Stop'}, locator or {'frame': 1}, timestamp)


class EvidenceTests(unittest.TestCase):
    def test_conflicting_revisions_rejected(self):
        with self.assertRaises(ValueError):
            Snapshot((evidence(), evidence('2', locator={'frame': 2})))

    def test_duplicate_location_rejected(self):
        with self.assertRaises(ValueError):
            Snapshot((evidence(), evidence()))

    def test_pages_same_source_valid(self):
        s = Snapshot((evidence(locator={'page': 1}), evidence(locator={'page': 2})))
        self.assertEqual(Snapshot.from_dict(s.state()).digest, s.digest)

    def test_freshness_missing_stale_and_future(self):
        Snapshot((evidence(),)).validate_freshness(5, now=105)
        for stamp in (None, 99, 106):
            with self.assertRaises(ValueError):
                Snapshot((evidence(timestamp=stamp),)).validate_freshness(5, now=105)

    def test_nonfinite_timestamp_rejected(self):
        for x in (math.nan, math.inf, True):
            with self.assertRaises(ValueError):
                evidence(timestamp=x)

    def test_snapshot_isolated_from_caller_mutation(self):
        e = evidence(); s = Snapshot((e,)); before = s.digest
        e.data['text'] = 'Go'
        exposed = s.state(); exposed['evidence'][0]['data']['text'] = 'Go'
        self.assertEqual(before, s.digest)

    def test_late_result_dropped_even_same_bytes(self):
        gate = RevisionGate(); s = Snapshot((evidence(),))
        old = gate.submit(s); new = gate.submit(s)
        self.assertFalse(gate.accepts(old, s)); self.assertTrue(gate.accepts(new, s))

    def test_compact_bridge_retains_probabilities_and_provenance(self):
        prediction = {'model': 'vision', 'backend': 'local', 'image_sha256': '1'*64,
                      'answers': [{'id': 'person', 'type': 'noul', 'choice': 'yes',
                                   'probabilities': {'yes': .87654321, 'no': .12345679},
                                   'candidate_logits': [8, 6], 'prompt_sha256': '2'*64}]}
        q = Question.yes_no('person', 'Is a person visible?')
        item = visual_evidence(prediction, 'camera', 'r2', questions=[q])[0]
        self.assertEqual(item.content_sha256, '1'*64)
        got = item.data['visual_judgments'][0]
        self.assertEqual(got['probabilities']['yes'], .87654321)
        self.assertEqual(got['question'], q.instructions)
        self.assertNotIn('candidate_logits', got)
        prediction['answers'][0]['probabilities']['yes'] = 0
        self.assertEqual(got['probabilities']['yes'], .87654321)

    def test_cache_content_and_engine_invalidation(self):
        cache = ExtractionCache(2); calls = []
        def compute():
            calls.append(1); return {'words': ['stop']}
        got, hit = cache.get_or_compute('a', {'engine': 1}, compute)
        got['words'].clear()
        again, hit = cache.get_or_compute('a', {'engine': 1}, compute)
        self.assertTrue(hit); self.assertEqual(again['words'], ['stop'])
        cache.get_or_compute('a', {'engine': 2}, compute)
        cache.get_or_compute('b', {'engine': 1}, compute)
        cache.get_or_compute('a', {'engine': 1}, compute)
        self.assertEqual(len(calls), 4)

    def test_transcript_preserves_times_and_rejects_bad_span(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'words.json'
            path.write_text(json.dumps([{'start_s': 1, 'end_s': 2, 'text': 'hello'}]))
            e = transcript(path, 'microphone', 'r1')[0]
            self.assertEqual(e.locator, {'segment': 0, 'start_s': 1, 'end_s': 2})
            path.write_text(json.dumps([{'start_s': 2, 'end_s': 1, 'text': 'hello'}]))
            with self.assertRaises(ValueError):
                transcript(path, 'microphone', 'r1')


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.questions = [Question.yes_no('visible', 'Is a sign visible?'), Question('action', 'What next?', {'stop': 'Stop', 'go': 'Go'})]
        self.response = {'model': 'test-model', 'answers': {'visible': {'type': 'noul', 'noul': .8},
                            'action': {'type': 'choice', 'choice': 'stop', 'probabilities': {'stop': .9, 'go': .1}, 'confidence': .8}},
                         'usage': {'input_tokens': 10}}

    def client(self, response=None):
        return JevClient('unit-test-placeholder', transport=lambda payload: copy.deepcopy(response or self.response))

    def test_batch_one_request_and_provenance(self):
        requests = []
        def transport(payload):
            requests.append(payload); return self.response
        result = JevClient('unit-test-placeholder', transport=transport).judge(Snapshot((evidence(),)), self.questions)
        self.assertEqual(len(requests), 1)
        self.assertEqual(set(requests[0]['questions']), {'visible', 'action'})
        self.assertEqual(result['answers'][0]['noul'], .8)
        self.assertNotIn('confidence', result['answers'][0])
        self.assertEqual(requests[0]['state']['evidence'][0]['revision'], '1')

    def test_missing_extra_wrong_type_rejected(self):
        variants = []
        r = copy.deepcopy(self.response); del r['answers']['visible']; variants.append(r)
        r = copy.deepcopy(self.response); r['answers']['extra'] = {}; variants.append(r)
        r = copy.deepcopy(self.response); r['answers']['visible']['type'] = 'choice'; variants.append(r)
        for r in variants:
            with self.assertRaises(JevError):
                self.client(r).judge(Snapshot((evidence(),)), self.questions)

    def test_boolean_nan_and_invalid_distributions_rejected(self):
        for invalid in (True, math.nan, -.1, 1.1):
            r = copy.deepcopy(self.response); r['answers']['visible']['noul'] = invalid
            with self.assertRaises(JevError):
                self.client(r).judge(Snapshot((evidence(),)), self.questions)
        for probs in ({'stop': .1, 'go': .9}, {'stop': .9, 'other': .1}, {'stop': .9, 'go': .5}):
            r = copy.deepcopy(self.response); r['answers']['action']['probabilities'] = probs
            with self.assertRaises(JevError):
                self.client(r).judge(Snapshot((evidence(),)), self.questions)

    def test_expired_evidence_does_not_call_transport(self):
        requests = []
        client = JevClient('unit-test-placeholder', transport=lambda x: requests.append(x))
        with self.assertRaises(ValueError):
            client.judge(Snapshot((evidence(),)), self.questions, max_age_s=1)
        self.assertFalse(requests)

    def test_large_evidence_never_silently_truncated(self):
        s = Snapshot((evidence(),), {'large': 'x'*100000})
        with self.assertRaises(ValueError):
            self.client().judge(s, self.questions)

    def test_duplicate_question_ids_rejected(self):
        with self.assertRaises(ValueError):
            self.client().judge(Snapshot((evidence(),)), [self.questions[0]]*2)


class NumericalTests(unittest.TestCase):
    def test_prefix_catches_tokenizer_boundary_merge(self):
        self.assertEqual(stable_prefix([1, 2, 3], [[1, 2, 4, 5], [1, 2, 3, 6]]), 2)
        with self.assertRaises(ValueError):
            stable_prefix([1], [[2]])

    def test_padding_masks_and_answer_positions(self):
        ids, masks, pos, end = suffix_layout([[8], [9, 10, 11]], 4, 7, 0)
        self.assertEqual(masks[0], [1]*5+[0, 0])
        self.assertEqual(pos, [[7, 0, 0], [7, 8, 9]])
        self.assertEqual(end, [0, 2])

    def test_probabilities_finite_under_extreme_logits(self):
        self.assertEqual(normalized([1000, -1000]), [1., 0.])
        with self.assertRaises(ValueError):
            normalized([math.nan, 0])

    def test_known_metrics(self):
        result = binary_metrics([.8, .2], [1, 0])
        self.assertAlmostEqual(result['brier'], .04)
        self.assertAlmostEqual(result['ece_10_top_label'], .2)
        self.assertEqual(result['accuracy'], 1)

    def test_temperature_does_not_change_choice(self):
        logits = [[4, 0], [0, 4], [4, 0], [0, 4]]
        t = fit_temperature(logits, [1, 0, 0, 1])
        self.assertGreater(t, 1)
        self.assertTrue(all((normalized(z)[0] >= .5) == (normalized(z, t)[0] >= .5) for z in logits))


if __name__ == '__main__':
    unittest.main()
