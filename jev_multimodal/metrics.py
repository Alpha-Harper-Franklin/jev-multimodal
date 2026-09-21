"""Binary reliability diagnostics; fitting and reporting take separate samples."""
import math
from .schema import normalized


def binary_metrics(probabilities, labels, bins=10):
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError('Matched nonempty probabilities and labels required')
    if any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError('Binary integer labels required')
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError('Invalid probabilities')
    n = len(labels)
    tp = sum(p >= .5 and y == 1 for p,y in zip(probabilities,labels))
    fp = sum(p >= .5 and y == 0 for p,y in zip(probabilities,labels))
    fn = sum(p < .5 and y == 1 for p,y in zip(probabilities,labels))
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    correct = [int((p >= .5) == y) for p, y in zip(probabilities, labels)]
    confidence = [max(p, 1-p) for p in probabilities]
    ece = 0.
    for b in range(bins):
        indices = [i for i, c in enumerate(confidence) if min(int(c*bins), bins-1) == b]
        if indices:
            ece += abs(sum(confidence[i]-correct[i] for i in indices))/n
    return {'n': n, 'correct': sum(correct), 'accuracy': sum(correct)/n,
            'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.,
            'yes_rate':sum(p>=.5 for p in probabilities)/n,
            'brier': sum((p-y)**2 for p, y in zip(probabilities, labels))/n,
            'nll': -sum(math.log(max(1e-12, p if y else 1-p)) for p, y in zip(probabilities, labels))/n,
            'ece_10_top_label': ece}


def fit_temperature(logits, labels):
    if not logits or len(logits) != len(labels):
        raise ValueError('Calibration data required')
    candidates = [math.exp(-3+i*6/240) for i in range(241)]
    return min(candidates, key=lambda t: binary_metrics([normalized(z, t)[0] for z in logits], labels)['nll'])
