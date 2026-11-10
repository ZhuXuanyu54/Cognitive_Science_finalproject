"""Score a mixture and match unordered separation outputs to references."""
from itertools import permutations
import numpy as np

SISDR_LIMIT_DB = 120.0

def _unit(signal):
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - signal.mean()
    norm = np.linalg.norm(signal)
    return None if norm == 0 else signal / norm

def sisdr(reference, estimate):
    reference, estimate = _unit(reference), _unit(estimate)
    if reference is None or estimate is None:
        return -SISDR_LIMIT_DB
    scale = np.dot(reference, estimate)
    residual = estimate - scale * reference
    return float(np.clip(10 * np.log10(max(scale * scale, 1e-12) / max(np.dot(residual, residual), 1e-12)), -SISDR_LIMIT_DB, SISDR_LIMIT_DB))

def evaluate(mixture, references, estimates):
    """Evaluate aligned mono arrays with shapes (T,), (K,T), (K,T)."""
    references, estimates = np.asarray(references), np.asarray(estimates)
    scores = np.array([[sisdr(ref, est) for est in estimates] for ref in references])
    order = max(permutations(range(len(references))), key=lambda p: sum(scores[i, j] for i, j in enumerate(p)))
    per_source = []
    for i, j in enumerate(order):
        baseline = sisdr(references[i], mixture)
        per_source.append({"reference_index": i, "estimate_index": j, "sisdr": float(scores[i,j]), "baseline_sisdr": baseline, "sisdri": float(scores[i,j] - baseline)})
    return {"reference_to_estimate": list(order), "per_source": per_source, "mean_sisdr": float(np.mean([x["sisdr"] for x in per_source])), "mean_baseline_sisdr": float(np.mean([x["baseline_sisdr"] for x in per_source])), "mean_sisdri": float(np.mean([x["sisdri"] for x in per_source])), "min_sisdri": min(x["sisdri"] for x in per_source)}
