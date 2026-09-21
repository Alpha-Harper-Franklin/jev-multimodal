"""Exercise the source CLI through separate processes, without GPU or API access."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CliTests(unittest.TestCase):
    def test_visual_transcript_merge_preserves_sources(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            env = dict(os.environ, PYTHONPATH=str(root))
            def run(*args):
                result = subprocess.run([sys.executable, '-m', 'jev_multimodal', *args],
                                        cwd=base, env=env, text=True, capture_output=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)
            # Use a recorded real-model prediction; the test does not rerun inference.
            record = json.loads((root/'benchmarks/capabilities/batched-parity.json').read_text())['shared']
            (base/'visual.json').write_text(json.dumps(record))
            questions = [{'id':'person','type':'noul','instructions':'Is there a person in image 1?'},
                         {'id':'setting','type':'choice','instructions':'What setting?',
                          'criteria':{'indoor':'Indoors','outdoor':'Outdoors','uncertain':'Uncertain'}},
                         {'id':'text','type':'score','instructions':'How much text?',
                          'criteria':['None','A few words','Many words']}]
            (base/'questions.json').write_text(json.dumps(questions))
            (base/'speech.json').write_text(json.dumps([{'start_s':1.2,'end_s':2.4,'text':'Inspect the first image'}]))
            run('visual-evidence','--predictions','visual.json','--questions','questions.json',
                '--source-id','camera','--revision','1','--output','vision-evidence.json')
            run('extract','transcript','speech.json','--source-id','microphone','--revision','2',
                '--output','speech-evidence.json')
            run('merge','--evidence','vision-evidence.json','speech-evidence.json','--output','combined.json')
            result = json.loads((base/'combined.json').read_text())
            self.assertEqual([e['source_id'] for e in result['evidence']], ['camera','microphone'])
            self.assertEqual(result['evidence'][0]['data']['ordered_image_sha256s'], record['image_sha256s'])
            self.assertEqual(result['evidence'][1]['locator']['start_s'], 1.2)


if __name__ == '__main__':
    unittest.main()
