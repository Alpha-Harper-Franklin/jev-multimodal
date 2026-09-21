"""Run Tesseract on authored images and check content-based cache invalidation."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jev_multimodal.adapters import ocr_image
from jev_multimodal.evidence import ExtractionCache, Snapshot


class MeasuredCache(ExtractionCache):
    def __init__(self):
        super().__init__()
        self.hits = []

    def get_or_compute(self, *args):
        result, hit = super().get_or_compute(*args)
        self.hits.append(hit)
        return result, hit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--tesseract', help='Optional explicit Tesseract executable')
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    if a.tesseract:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = a.tesseract
    from PIL import Image, ImageDraw, ImageFont
    for deadline in (30,90):
        image = Image.new('RGB',(900,210),'white')
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=32)
        draw.text((30,35),'SERVICE HOURS 09:00 - 17:00 UTC',fill='black',font=font)
        draw.text((30,100),f'REFUND DEADLINE {deadline} DAYS',fill='black',font=font)
        image.save(out/f'authored-{deadline}.png')
    cache = MeasuredCache()
    records, timings = [], []
    for index, deadline in enumerate((30,30,90),1):
        started = time.perf_counter()
        evidence = ocr_image(out/f'authored-{deadline}.png', 'screen', str(index), cache=cache)
        timings.append((time.perf_counter()-started)*1000)
        data = evidence[0].data
        assert f'{deadline}' in data['text'] and 'REFUND' in data['text']
        for word in data['words']:
            x,y,width,height = word['bbox_xywh']
            assert 0 <= x < x+width <= data['width'] and 0 <= y < y+height <= data['height']
        records.append(Snapshot(evidence).state())
    assert cache.hits == [False,True,False]
    assert records[0]['evidence'][0]['content_sha256'] == records[1]['evidence'][0]['content_sha256']
    assert records[1]['evidence'][0]['content_sha256'] != records[2]['evidence'][0]['content_sha256']
    for index, record in enumerate(records,1):
        (out/f'evidence-{index}.json').write_text(json.dumps(record,indent=2)+'\n')
    summary = {'scope':'Two authored high-contrast images; extraction/cache integration, not an OCR dataset benchmark',
               'engine':records[0]['evidence'][0]['extractor'],
               'cache_hits':cache.hits,'elapsed_ms':timings,
               'transcripts':[r['evidence'][0]['data']['text'] for r in records],
               'word_counts':[len(r['evidence'][0]['data']['words']) for r in records],
               'boxes_within_image':True,'changed_image_recomputed':True}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
