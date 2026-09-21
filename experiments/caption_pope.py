"""Generic image captions for a measured caption -> hosted Jev baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_multimodal.cuda_qwen import QwenVision

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--manifest', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
from PIL import Image, ImageOps
engine = QwenVision(args.model)
rows = [json.loads(s) for s in Path(args.manifest).read_text().splitlines()]
with Path(args.output).open('x') as stream:
    for i, row in enumerate(rows):
        engine.sync(); began = time.perf_counter()
        if hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest() != row['image_sha256']:
            raise ValueError('Image hash mismatch')
        with Image.open(row['image_path']) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        prompt = 'Describe the visible objects and scene in this image accurately and concisely. Do not infer objects that are not visible.'
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = engine.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = engine.processor(text=[text], images=[image], return_tensors='pt').to(engine.device)
        engine.model.model.rope_deltas = None
        with engine.torch.inference_mode():
            output = engine.model.generate(**inputs, max_new_tokens=160, do_sample=False)
        generated = output[:, inputs['input_ids'].shape[1]:]
        caption = engine.processor.batch_decode(generated, skip_special_tokens=True)[0]
        engine.sync()
        record = {'image_id': row['image_id'], 'image_sha256': row['image_sha256'], 'caption': caption,
                  'elapsed_ms': (time.perf_counter()-began)*1000, 'generated_tokens': generated.shape[1],
                  'questions': row['questions'], 'model': Path(args.model).name, 'max_new_tokens': 160,
                  'prompt': prompt, 'cold_first_image': i == 0}
        stream.write(json.dumps(record)+'\n'); stream.flush()
        print(json.dumps({'captions': i+1, 'total': len(rows)}), flush=True)
