"""Real PDF/ASR extraction integration, not an accuracy benchmark.

Supply the public whisper tests/jfk.flac sample and a local faster-whisper
checkpoint. This script creates an authored two-page PDF fixture (one blank).
It does not download weights or contact Jev.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_multimodal.adapters import audio_asr, pdf_text
from jev_multimodal.evidence import Snapshot


def word_errors(reference, hypothesis):
    a = re.findall(r"[a-z]+", reference.lower())
    b = re.findall(r"[a-z]+", hypothesis.lower())
    previous = list(range(len(b)+1))
    for i, word in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(word != other)))
        previous = current
    return {'reference_words': len(a), 'edit_distance': previous[-1], 'wer': previous[-1]/len(a)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audio', required=True)
    p.add_argument('--asr-model', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                             NameObject('/Subtype'): NameObject('/Type1'),
                             NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
    stream = DecodedStreamObject()
    stream.set_data(b'BT /F1 16 Tf 60 700 Td (Service window: 09:00 to 17:00 UTC.) Tj 0 -30 Td (Refund deadline: 30 days after purchase.) Tj ET')
    page[NameObject('/Contents')] = writer._add_object(stream)
    writer.add_blank_page(width=612, height=792)
    writer.write(out/'authored-fixture.pdf')
    started = time.perf_counter()
    document = pdf_text(out/'authored-fixture.pdf', 'document', '1')
    pdf_ms = (time.perf_counter()-started)*1000
    assert len(document) == 2 and document[0].locator == {'page': 1}
    assert '30 days' in document[0].data['text'] and not document[0].data['needs_ocr']
    assert document[1].data['needs_ocr'] and document[1].locator == {'page': 2}
    started = time.perf_counter()
    audio = audio_asr(a.audio, 'speech', '1', a.asr_model)
    asr_ms = (time.perf_counter()-started)*1000
    assert audio, 'No ASR segments'
    assert all(0 <= e.locator['start_s'] <= e.locator['end_s'] for e in audio)
    spoken = ' '.join(e.data['text'].strip() for e in audio)
    reference = 'And so my fellow Americans ask not what your country can do for you ask what you can do for your country'
    snapshots = {'document': Snapshot(document), 'speech': Snapshot(audio)}
    snapshots['combined'] = Snapshot.combine(snapshots.values())
    for name, snapshot in snapshots.items():
        (out/(name+'.json')).write_text(json.dumps(snapshot.state(), indent=2)+'\n')
    result = {'scope': 'Integration fixture: one public speech clip and one authored PDF, not a dataset evaluation',
              'audio_sha256': hashlib.sha256(Path(a.audio).read_bytes()).hexdigest(),
              'asr_model': Path(a.asr_model).name,
              'model_sha256': hashlib.sha256((Path(a.asr_model)/'model.bin').read_bytes()).hexdigest(),
              'runtime': {'python': platform.python_version(), **{x: importlib.metadata.version(x) for x in ('pypdf','faster-whisper','ctranslate2')}},
              'pdf': {'pages': 2, 'text_pages': 1, 'needs_ocr_pages': 1, 'elapsed_ms': pdf_ms},
              'audio': {'segments': len(audio), 'text': spoken, 'elapsed_ms_including_cpu_model_load': asr_ms,
                        'reference': reference, **word_errors(reference, spoken)},
              'combined_evidence_count': len(snapshots['combined'].evidence)}
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
