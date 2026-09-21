import argparse
import json
from pathlib import Path

from .schema import Question
from .evidence import Snapshot


def main():
    parser = argparse.ArgumentParser(description='Image decisions and multimodal evidence for Jev')
    commands = parser.add_subparsers(dest='command', required=True)
    vision = commands.add_parser('vision', help='Local Qwen2.5-VL finite-candidate scoring')
    vision.add_argument('--image', required=True)
    vision.add_argument('--model', required=True, help='Local Qwen2.5-VL checkpoint directory')
    vision.add_argument('--questions', required=True)
    vision.add_argument('--state')
    vision.add_argument('--mode', choices=['shared', 'independent'], default='shared')
    vision.add_argument('--batch-size', type=int, default=8)
    vision.add_argument('--device', default='cuda:0')
    judge = commands.add_parser('judge', help='Send extracted evidence to hosted Jev')
    judge.add_argument('--evidence', required=True, help='Snapshot JSON, never raw media')
    judge.add_argument('--questions', required=True)
    judge.add_argument('--max-age', type=float)
    extract = commands.add_parser('extract', help='Produce provenance-preserving evidence JSON')
    extract.add_argument('kind', choices=['transcript', 'pdf', 'ocr', 'audio'])
    extract.add_argument('path')
    extract.add_argument('--source-id', required=True)
    extract.add_argument('--revision', required=True)
    extract.add_argument('--observed-at', type=float)
    extract.add_argument('--asr-model')
    extract.add_argument('--language', default='eng')
    for command in (vision, judge, extract):
        command.add_argument('--output', required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        parser.error('Output already exists; choose a new path')
    if args.command in ('vision', 'judge'):
        questions = [Question.from_dict(q) for q in json.loads(Path(args.questions).read_text(encoding='utf-8'))]
    if args.command == 'vision':
        from .cuda_qwen import QwenVision
        state = json.loads(Path(args.state).read_text(encoding='utf-8')) if args.state else {}
        result = QwenVision(args.model, device=args.device, batch_size=args.batch_size).judge(args.image, questions, state, args.mode)
    elif args.command == 'judge':
        from .typesafe import JevClient
        snapshot = Snapshot.from_dict(json.loads(Path(args.evidence).read_text(encoding='utf-8')))
        result = JevClient().judge(snapshot, questions, max_age_s=args.max_age)
    else:
        from . import adapters
        options = {'source_id': args.source_id, 'revision': args.revision, 'observed_at': args.observed_at}
        if args.kind == 'audio':
            if not args.asr_model:
                parser.error('--asr-model is required for audio')
            evidence = adapters.audio_asr(args.path, model_path=args.asr_model, **options)
        elif args.kind == 'ocr':
            evidence = adapters.ocr_image(args.path, language=args.language, **options)
        else:
            fn = adapters.transcript if args.kind == 'transcript' else adapters.pdf_text
            evidence = fn(args.path, **options)
        result = Snapshot(evidence).state()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(str(target.resolve()))


if __name__ == '__main__':
    main()
