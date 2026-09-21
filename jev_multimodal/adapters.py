"""Optional extractors. None sends raw image/audio bytes to hosted Jev."""
import hashlib
import json
from pathlib import Path

from .evidence import Evidence


def _file(path, limit=50_000_000):
    p = Path(path)
    if p.stat().st_size > limit:
        raise ValueError('Media exceeds the configured input size limit')
    data = p.read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def transcript(path, source_id, revision, observed_at=None):
    """Import [{start_s, end_s, text}] from an ASR provider, retaining times."""
    data, digest = _file(path)
    rows = json.loads(data)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty JSON list of transcript segments')
    evidence = []
    for index, row in enumerate(rows):
        import math
        start, end = row['start_s'], row['end_s']
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in (start, end)) or not 0 <= start <= end:
            raise ValueError('Invalid transcript timestamp')
        if not isinstance(row['text'], str) or not row['text'].strip():
            raise ValueError('Transcript text is empty')
        evidence.append(Evidence(source_id, revision, 'transcript', 'transcript-json:v1', digest,
                                 {'text': row['text']}, {'segment': index, 'start_s': start, 'end_s': end}, observed_at))
    return tuple(evidence)


def pdf_text(path, source_id, revision, observed_at=None, max_pages=100):
    from pypdf import PdfReader
    import io
    data, digest = _file(path)
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise ValueError('Encrypted PDFs require explicit decryption before extraction')
    if not 1 <= len(reader.pages) <= max_pages:
        raise ValueError('PDF page count outside configured limit')
    result = []
    for index, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ''
        result.append(Evidence(source_id, revision, 'pdf', 'pypdf:'+__import__('pypdf').__version__, digest,
                               {'text': text, 'needs_ocr': not bool(text.strip())}, {'page': index}, observed_at))
    return tuple(result)


def ocr_image(path, source_id, revision, observed_at=None, language='eng', cache=None):
    from PIL import Image, ImageOps
    import pytesseract
    import io
    data, digest = _file(path, 25_000_000)
    engine = 'tesseract:'+str(pytesseract.get_tesseract_version())
    def extract():
        with Image.open(io.BytesIO(data)) as opened:
            if opened.width*opened.height > 24_000_000:
                raise ValueError('Image exceeds 24 megapixels')
            image = ImageOps.exif_transpose(opened).convert('RGB')
        rows = pytesseract.image_to_data(image, lang=language, timeout=30, output_type=pytesseract.Output.DICT)
        words = []
        for i, word in enumerate(rows['text']):
            if not word.strip():
                continue
            words.append({'text': word, 'bbox_xywh': [int(rows[key][i]) for key in ('left', 'top', 'width', 'height')],
                          'ocr_score': float(rows['conf'][i]), 'score_semantics': 'OCR engine score, not calibrated correctness'})
        return {'words': words, 'width': image.width, 'height': image.height, 'text': ' '.join(x['text'] for x in words)}
    if cache:
        result, _ = cache.get_or_compute(digest, {'engine': engine, 'language': language, 'adapter': 1}, extract)
    else:
        result = extract()
    return (Evidence(source_id, revision, 'image', engine, digest, result, {'frame': 0}, observed_at),)


def audio_asr(path, source_id, revision, model_path, observed_at=None):
    """Local faster-whisper checkpoint only; no automatic model download."""
    from faster_whisper import WhisperModel
    import faster_whisper
    import io
    data, digest = _file(path)
    if not Path(model_path).is_dir():
        raise ValueError('Provide an existing local faster-whisper model directory')
    model = WhisperModel(str(model_path), device='cpu', compute_type='int8', local_files_only=True)
    segments, info = model.transcribe(io.BytesIO(data), beam_size=1, vad_filter=True)
    return tuple(Evidence(source_id, revision, 'audio', 'faster-whisper:'+faster_whisper.__version__+':'+Path(model_path).name,
                          digest, {'text': s.text, 'language': info.language, 'avg_logprob': s.avg_logprob},
                          {'segment': i, 'start_s': s.start, 'end_s': s.end}, observed_at)
                 for i, s in enumerate(segments))
