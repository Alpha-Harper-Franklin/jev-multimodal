"""Finite question contracts shared by local vision and evidence backends."""
from dataclasses import dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class Question:
    id: str
    instructions: str
    criteria: dict[str, str]
    kind: str = 'choice'

    def __post_init__(self):
        if not isinstance(self.id,str) or not self.id or not isinstance(self.instructions,str) or not self.instructions.strip():
            raise ValueError('Question ID and instructions are required')
        if self.kind not in ('choice','noul'):
            raise ValueError('Only choice and noul are implemented')
        if not isinstance(self.criteria,dict) or not 2 <= len(self.criteria) <= 26:
            raise ValueError('Provide 2 to 26 named candidates')
        if any(not isinstance(k,str) or not k or not isinstance(v,str) or not v for k,v in self.criteria.items()):
            raise ValueError('Candidate IDs and descriptions must be nonempty strings')
        if self.kind == 'noul' and list(self.criteria) != ['yes','no']:
            raise ValueError('Noul uses ordered yes/no candidates')
        for text in [self.instructions,*self.criteria.keys(),*self.criteria.values()]:
            if '<|' in text or '|>' in text:
                raise ValueError('Reserved model control tokens are not evidence')

    @classmethod
    def yes_no(cls, id, instructions):
        return cls(id,instructions,{'yes':'Yes','no':'No'},'noul')

    @classmethod
    def from_dict(cls, value):
        if value.get('type') == 'noul':
            return cls.yes_no(value['id'],value['instructions'])
        return cls(value['id'],value['instructions'],value['criteria'],value.get('type','choice'))


def normalized(logits, temperature=1.0):
    if not logits or type(temperature) not in (int,float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Nonempty logits and positive temperature required')
    if any(not math.isfinite(x) for x in logits):
        raise ValueError('Non-finite candidate score')
    peak = max(logits)
    values = [math.exp((x-peak)/temperature) for x in logits]
    return [x/sum(values) for x in values]


def make_answer(question, logits, candidate_mass, temperature=1.0):
    if len(logits) != len(question.criteria):
        raise ValueError('Candidate score count mismatch')
    probs = dict(zip(question.criteria,normalized(logits,temperature)))
    result = {'id':question.id,'type':question.kind,'choice':max(probs,key=probs.get),
              'probabilities':probs,'candidate_logits':logits,'candidate_probability_mass':candidate_mass,
              'probability_semantics':'conditional candidate scores; not calibrated correctness',
              'temperature':temperature}
    if question.kind == 'noul':
        result['noul'] = probs['yes']
    return result


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
