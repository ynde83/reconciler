from dataclasses import dataclass, field
import numpy as np

EPS = 1e-6

@dataclass
class DriftReport():
    psi_by_feature: dict[str, float]
    drifted_features: list[str]
    drift_score: float
    is_drift: bool

def bin_shares(values: np.ndarray, inner_edges: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("no any values to split")

    idx = np.searchsorted(inner_edges, values, side = "right")
    counts = np.bincount(idx, minlength=len(inner_edges) + 1)

    return counts / counts.sum()

def psi(expected_probs, actual_values, inner_edges):
    actual_probs = bin_shares(actual_values, inner_edges)

    a = np.maximum(actual_probs, EPS)
    e = np.maximum(np.asarray(expected_probs, dtype=float), EPS)

    if len(a) != len(e):
        raise ValueError(f"baskets in the window {len(a)}, in baseline {len(e)}: please check edges")

    value = np.sum((a -e) * np.log(a / e))

    return float(value)

def check_drift(
    window: dict[str, np.ndarray],   
    baseline: dict,                  
    psi_threshold: float = 0.2,
    min_drifted: int = 3,
) -> DriftReport:
    psi_by_feature = {}
    for name, stats in baseline["features"].items():
        psi_by_feature[name] = psi(stats["props"], window[name], stats["edges"])

    drifted_features = [name for name, value in psi_by_feature.items() if value > psi_threshold]

    drift_score = float(np.mean([psi_by_feature[n] for n in drifted_features])) if drifted_features else 0

    return DriftReport (
        psi_by_feature=psi_by_feature,
        drifted_features=drifted_features,
        drift_score=drift_score,
        is_drift=len(drifted_features) >= min_drifted,
    )