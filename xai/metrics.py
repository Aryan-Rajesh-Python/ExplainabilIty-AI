"""Shared measurable metrics with explicit computation sources."""

import numpy as np


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def section_embedding_transitions(sections, embedding_model, max_sections=8):
    """Measured cosine similarity between consecutive section embeddings."""
    secs = [s[:1500] for s in sections if s.strip()][:max_sections]
    if len(secs) < 2:
        return []
    embs = embedding_model.encode(secs)
    transitions = []
    for i in range(1, len(embs)):
        sim = cosine_similarity(embs[i - 1], embs[i])
        transitions.append({"from_idx": i - 1, "to_idx": i, "similarity": sim})
    return transitions


def diffusion_timestep_buckets(step_metrics, num_steps=None):
    """Group measured latent-std steps into early/mid/late denoising phases."""
    if not step_metrics:
        return []
    n = len(step_metrics)
    buckets = [
        ("early", step_metrics[: max(1, n // 3)]),
        ("mid", step_metrics[max(1, n // 3) : max(2, 2 * n // 3)]),
        ("late", step_metrics[max(2, 2 * n // 3) :]),
    ]
    narratives = []
    phase_desc = {
        "early": "global scene topology and coarse layout emerged",
        "mid": "subject geometry and spatial relations stabilized",
        "late": "photorealistic texture, edges, and lighting continuity emerged",
    }
    for phase, steps in buckets:
        if not steps:
            continue
        t_start = steps[0]["step"]
        t_end = steps[-1]["step"]
        avg_std = float(np.mean([s["latent_std"] for s in steps]))
        narratives.append({
            "phase": phase,
            "step_range": (t_start, t_end),
            "avg_latent_std": avg_std,
            "description": phase_desc[phase],
        })
    return narratives


TRACE_SOURCES = {
    "gradcam": "Grad-CAM activations (ResNet50)",
    "clip": "CLIP ViT-B/32 patch cosine similarity",
    "multiscale": "Gaussian pyramid edge-energy decomposition",
    "confidence_traj": "Multi-blur ResNet50 softmax trajectory",
    "zero_shot": "BART-MNLI hypothesis routing scores",
    "lime": "Masked-token local attribution",
    "shap": "Kernel SHAP token attribution",
    "mel": "Librosa mel-spectrogram band means",
    "mfcc": "Librosa MFCC temporal variance",
    "rms": "Librosa RMS energy envelope",
    "motion": "OpenCV frame-difference energy",
    "whisper": "Whisper transcription segments",
    "embedding": "SentenceTransformer cosine transitions",
    "diffusion": "Diffusers latent std per denoising step",
}
