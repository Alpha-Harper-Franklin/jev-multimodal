"""Evidence preserves acquisition time, source revision, and modality provenance."""
from collections import OrderedDict
import copy
from dataclasses import asdict, dataclass, field
import math
import threading
import time

from .schema import canonical_digest


@dataclass(frozen=True)
class Evidence:
    source_id: str
    revision: str
    modality: str
    extractor: str
    content_sha256: str
    data: dict
    locator: dict = field(default_factory=dict)
    observed_at: float | None = None

    def __post_init__(self):
        if any(not isinstance(x, str) or not x for x in (self.source_id, self.revision, self.extractor)):
            raise ValueError('Evidence needs source ID, revision, and extractor version')
        if self.modality not in ('image', 'pdf', 'audio', 'transcript', 'structured', 'text'):
            raise ValueError('Unknown evidence modality')
        if len(self.content_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.content_sha256):
            raise ValueError('A lowercase SHA-256 content digest is required')
        if not isinstance(self.data, dict) or not isinstance(self.locator, dict):
            raise ValueError('Evidence data and locator must be JSON objects')
        if self.observed_at is not None and (type(self.observed_at) not in (float, int) or not math.isfinite(self.observed_at)):
            raise ValueError('observed_at must be a finite Unix timestamp')
        canonical_digest(asdict(self))
        object.__setattr__(self, 'data', copy.deepcopy(self.data))
        object.__setattr__(self, 'locator', copy.deepcopy(self.locator))


@dataclass(frozen=True)
class Snapshot:
    evidence: tuple[Evidence, ...]
    context: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.evidence or not isinstance(self.context, dict):
            raise ValueError('Snapshot needs evidence and an object context')
        revisions = {}
        identities = set()
        for item in self.evidence:
            if item.source_id in revisions and revisions[item.source_id] != item.revision:
                raise ValueError('One snapshot cannot mix revisions of the same source')
            revisions[item.source_id] = item.revision
            identity = (item.source_id, canonical_digest(item.locator), item.extractor)
            if identity in identities:
                raise ValueError('Duplicate evidence locator; use distinct page/segment/question IDs')
            identities.add(identity)
        object.__setattr__(self, 'evidence', copy.deepcopy(tuple(self.evidence)))
        object.__setattr__(self, 'context', copy.deepcopy(self.context))
        canonical_digest(self.state())

    def state(self):
        return {'context': copy.deepcopy(self.context), 'evidence': [asdict(e) for e in self.evidence]}

    @property
    def digest(self):
        return canonical_digest(self.state())

    def validate_freshness(self, max_age_s, now=None):
        if type(max_age_s) not in (int, float) or not math.isfinite(max_age_s) or max_age_s < 0:
            raise ValueError('Freshness limit must be finite and nonnegative')
        now = time.time() if now is None else now
        for item in self.evidence:
            if item.observed_at is None:
                raise ValueError('Timestamp missing from live evidence')
            age = now-item.observed_at
            if age < 0 or age > max_age_s:
                raise ValueError('Evidence is stale or timestamp is in the future')

    @classmethod
    def from_dict(cls, value):
        return cls(tuple(Evidence(**item) for item in value['evidence']), value.get('context', {}))

    @classmethod
    def combine(cls, snapshots):
        """Join modalities without silently replacing context, revisions or locations."""
        evidence, context = [], {}
        for snapshot in snapshots:
            evidence.extend(snapshot.evidence)
            for key, value in snapshot.context.items():
                if key in context and canonical_digest(context[key]) != canonical_digest(value):
                    raise ValueError('Conflicting snapshot context: '+key)
                context[key] = value
        return cls(tuple(evidence), context)


class RevisionGate:
    """Reject results superseded by a later acquisition, even if the bytes repeat."""
    def __init__(self):
        self._lock = threading.Lock()
        self._generation = 0
        self._digest = None

    def submit(self, snapshot):
        with self._lock:
            self._generation += 1
            self._digest = snapshot.digest
            return (self._generation, self._digest)

    def accepts(self, ticket, snapshot):
        with self._lock:
            return ticket == (self._generation, self._digest) and snapshot.digest == self._digest


class ExtractionCache:
    """Bounded content cache. Caller attaches current timestamps after extraction.

    Keys must include engine/version/options; no timestamps or response decisions
    are reused. Returning a deep copy prevents one caller corrupting future hits.
    """
    def __init__(self, capacity=32):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError('Positive cache capacity required')
        self.capacity = capacity
        self._values = OrderedDict()
        self._lock = threading.Lock()

    def get_or_compute(self, content_sha256, extractor_config, compute):
        key = canonical_digest([content_sha256, extractor_config])
        with self._lock:
            if key in self._values:
                self._values.move_to_end(key)
                return copy.deepcopy(self._values[key]), True
        result = compute()
        canonical_digest(result)
        with self._lock:
            self._values[key] = copy.deepcopy(result)
            self._values.move_to_end(key)
            while len(self._values) > self.capacity:
                self._values.popitem(last=False)
        return copy.deepcopy(result), False


def visual_evidence(result, source_id, revision, observed_at=None, questions=None):
    """Group predictions once per image; raw logits stay in the local result.

    Repeating hashes, prompts, and vocabulary diagnostics per question costs
    API tokens without adding visual evidence. Do not round probabilities.
    """
    specs = {q.id: q for q in questions or ()}
    judgments = []
    for answer in result['answers']:
        item = {k: copy.deepcopy(answer[k]) for k in ('id', 'type', 'choice', 'probabilities')}
        if answer['id'] in specs:
            spec = specs[answer['id']]
            item['question'] = spec.instructions
            if spec.kind != 'noul':
                item['criteria'] = dict(spec.criteria)
        for key in ('score', 'legend'):
            if key in answer:
                item[key] = copy.deepcopy(answer[key])
        judgments.append(item)
    image_hashes = result.get('image_sha256s', [result['image_sha256']])
    return (Evidence(source_id, revision, 'image', result['model']+':'+result['backend'],
                     result['image_sha256'],
                     {'visual_judgments': judgments, 'uncertainty': 'uncalibrated model predictions, not observation labels',
                      **({'ordered_image_sha256s': image_hashes} if len(image_hashes)>1 else {})},
                     {'frame': 0} if len(image_hashes)==1 else {'ordered_frames': list(range(len(image_hashes)))}, observed_at),)
