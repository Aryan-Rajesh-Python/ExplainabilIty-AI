"""
Measured evolution metrics: entropy, routing drift, diffusion timestep phases.
"""

import math
from typing import Sequence

import numpy as np


def distribution_entropy(scores: Sequence[float]) -> float:
    """Shannon entropy of a normalized score vector (BART-MNLI routing)."""
    arr = np.array(scores, dtype=np.float64)
    if arr.size == 0 or arr.sum() <= 0:
        return 0.0
    p = arr / arr.sum()
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)))


def diffusion_timestep_phases(step_metrics):
    """
    Bucket by measured scheduler timestep (high t = noisy / coarse).
    Returns list of dicts with timestep range and narrative.
    """
    if not step_metrics:
        return []
    ordered = sorted(step_metrics, key=lambda m: m.get("timestep", 0), reverse=True)
    n = len(ordered)
    if n < 2:
        return []

    chunk_size = max(1, n // 4)
    phases = []
    phase_desc = [
        ("global scene topology and layout priors established", "high-noise"),
        ("subject identity and geometry stabilized", "mid-high"),
        ("structure, pose, and relations refined", "mid-low"),
        ("texture, edges, and illumination continuity emerged", "low-noise"),
    ]
    for i in range(0, n, chunk_size):
        chunk = ordered[i : i + chunk_size]
        if not chunk:
            continue
        t_hi = max(m["timestep"] for m in chunk)
        t_lo = min(m["timestep"] for m in chunk)
        idx = min(len(phases), len(phase_desc) - 1)
        desc, label = phase_desc[idx]
        phases.append({
            "timestep_hi": t_hi,
            "timestep_lo": t_lo,
            "label": label,
            "description": desc,
            "avg_latent_std": float(np.mean([m["latent_std"] for m in chunk])),
        })
    return phases


def latent_suppression_narrative(dominant: str, alternates: list) -> str:
    if not alternates:
        return ""
    alts = ", ".join(f'"{a}"' for a in alternates[:3])
    return (
        f"Latent routing converged toward \"{dominant}\" while suppressing "
        f"alternates ({alts}); source: measured routing scores."
    )


def global_attention_drift(early_label: str, late_label: str) -> str:
    if early_label == late_label:
        return (
            f"Global discourse attention remained anchored on \"{early_label}\" "
            "while detail elaboration increased downstream (endpoint routing)."
        )
    return (
        f"Global discourse attention shifted from \"{early_label}\" toward "
        f"\"{late_label}\" during downstream expansion (endpoint BART-MNLI routing)."
    )
