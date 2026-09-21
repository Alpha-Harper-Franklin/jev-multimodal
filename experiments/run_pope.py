"""Paired visual inference; labels deliberately not read by this process."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_multimodal import Question
from jev_multimodal.cuda_qwen import QwenVision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
    engine = QwenVision(args.model, batch_size=args.batch_size)
    import torch
    import transformers
    fingerprint = {str(p.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in Path(__file__).resolve().parents[1].glob('jev_multimodal/*.py')}
    metadata = {'python': platform.python_version(), 'torch': torch.__version__,
                'transformers': transformers.__version__, 'gpu': torch.cuda.get_device_name(),
                'model': Path(args.model).name, 'model_config_sha256': hashlib.sha256((Path(args.model)/'config.json').read_bytes()).hexdigest(),
                'source_sha256': fingerprint, 'batch_size': args.batch_size,
                'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                'warmup': 'first image, both modes, excluded from latency',
                'order': 'independent/shared on even index; shared/independent on odd index'}
    (target/'environment.json').write_text(json.dumps(metadata, indent=2))
    first = rows[0]
    for mode in ('independent', 'shared'):
        engine.judge(first['image_path'], [Question.from_dict(q) for q in first['questions']], mode=mode)
    torch.cuda.reset_peak_memory_stats()
    with (target/'predictions.jsonl').open('x') as stream:
        for i, row in enumerate(rows):
            if hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest() != row['image_sha256']:
                raise ValueError('Image hash mismatch')
            questions = [Question.from_dict(q) for q in row['questions']]
            for mode in (('independent', 'shared') if i % 2 == 0 else ('shared', 'independent')):
                result = engine.judge(row['image_path'], questions, mode=mode)
                result.update(image_id=row['image_id'])
                stream.write(json.dumps(result)+'\n'); stream.flush()
            print(json.dumps({'completed_images': i+1, 'total': len(rows)}), flush=True)
    # Duplicated question content is a throughput diagnostic, never additional accuracy data.
    scaling = []
    for count in (1, 4, 16, 64):
        questions = [Question.yes_no(f'scale_{j}', first['questions'][j % 6]['instructions']) for j in range(count)]
        for mode in ('independent', 'shared'):
            result = engine.judge(first['image_path'], questions, mode=mode)
            scaling.append({'count': count, 'mode': mode, 'metrics': result['metrics']})
    (target/'scaling.json').write_text(json.dumps(scaling, indent=2))
    (target/'complete.json').write_text(json.dumps({'images': len(rows), 'max_memory_allocated_bytes': torch.cuda.max_memory_allocated(), 'completed_unix': time.time()}, indent=2))


if __name__ == '__main__':
    main()
