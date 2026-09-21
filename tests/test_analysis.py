"""Analysis must reject incorrect joins and expose error/coverage tradeoffs."""
import unittest

from experiments.analyze_pope import paired_accuracy, analyze
from experiments.analyze_routing import threshold_for, routing


class AnalysisTests(unittest.TestCase):
    def test_paired_accuracy_preserves_image_clusters(self):
        truth = {str(i): {'label':'yes','split':'test'} for i in range(4)}
        base = [{'image_id':'a','answers':[{'id':'0','choice':'yes'},{'id':'1','choice':'no'}]},
                {'image_id':'b','answers':[{'id':'2','choice':'yes'},{'id':'3','choice':'yes'}]}]
        candidate = [{'image_id':'a','answers':[{'id':'0','choice':'yes'},{'id':'1','choice':'yes'}]},
                     {'image_id':'b','answers':[{'id':'2','choice':'no'},{'id':'3','choice':'yes'}]}]
        result = paired_accuracy(base, candidate, truth, 'test')
        self.assertEqual(result['image_clusters'], 2)
        self.assertEqual(result['accuracy_delta'], 0)
        self.assertEqual((result['choice_flips'],result['improved'],result['regressed']), (2,1,1))
        self.assertEqual(result['image_bootstrap_95_ci'], [-.5,.5])

    def test_wrong_image_join_is_rejected(self):
        labels = [{'question_id':'q','image_id':'correct-image','label':'yes','split':'test'}]
        rows = [{'mode':m,'image_id':'wrong-image','answers':[{'id':'q'}]}
                for m in ('independent','shared')]
        with self.assertRaisesRegex(ValueError, 'wrong image'):
            analyze(rows, labels)

    def test_calibration_risk_target_can_fail_on_test(self):
        calibration = [(.99,1)]*50+[(.55,0)]*50
        threshold = threshold_for(calibration,.05)
        self.assertEqual(routing(calibration,threshold)['errors'], 0)
        test = [(.99,0)]*50+[(.55,1)]*50
        shifted = routing(test,threshold)
        self.assertEqual(shifted['coverage'], .5)
        self.assertEqual(shifted['accepted_error_rate'], 1)
        self.assertIsNone(threshold_for([(.99,0)]*100,.05))
        self.assertEqual(routing(test,None)['accepted'], 0)


if __name__ == '__main__':
    unittest.main()
