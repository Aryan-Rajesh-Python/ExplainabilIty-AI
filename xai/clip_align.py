import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


def _l2_normalize(feats):
    if feats is None:
        return None
    if not isinstance(feats, torch.Tensor):
        feats = torch.tensor(feats)
    return F.normalize(feats, p=2, dim=-1)


def _clip_text_embedding(clip_model, input_ids, attention_mask, device):
    """Robust text embedding across transformers CLIP versions."""
    out = clip_model.text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    if getattr(out, "pooler_output", None) is not None:
        pooled = out.pooler_output
    else:
        pooled = out.last_hidden_state[:, 0, :]
    if hasattr(clip_model, "text_projection") and clip_model.text_projection is not None:
        pooled = clip_model.text_projection(pooled)
    return _l2_normalize(pooled)


def _clip_image_embedding(clip_model, pixel_values):
    """Robust image embedding across transformers CLIP versions."""
    out = clip_model.vision_model(pixel_values=pixel_values)
    if getattr(out, "pooler_output", None) is not None:
        pooled = out.pooler_output
    else:
        pooled = out.last_hidden_state[:, 0, :]
    if hasattr(clip_model, "visual_projection") and clip_model.visual_projection is not None:
        pooled = clip_model.visual_projection(pooled)
    return _l2_normalize(pooled)


def clip_prompt_heatmap(image, prompt, clip_model, clip_processor, grid=7):
    """
    Measured CLIP cosine similarity per spatial patch vs prompt.
    Returns (heatmap [grid, grid], top_regions list).
    """
    if not prompt or not str(prompt).strip():
        return None, []

    device = next(clip_model.parameters()).device
    image = image.convert("RGB").resize((224, 224))
    patch = 224 // grid

    text_inputs = clip_processor(
        text=[prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        text_features = _clip_text_embedding(
            clip_model, input_ids, attention_mask, device
        )

    heat = np.zeros((grid, grid), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            crop = image.crop((j * patch, i * patch, (j + 1) * patch, (i + 1) * patch))
            img_inputs = clip_processor(images=crop, return_tensors="pt")
            pixel_values = img_inputs["pixel_values"].to(device)
            with torch.no_grad():
                img_features = _clip_image_embedding(clip_model, pixel_values)
                sim = torch.matmul(text_features, img_features.T).squeeze().item()
            heat[i, j] = sim

    hmin, hmax = float(heat.min()), float(heat.max())
    heat_norm = (heat - hmin) / (hmax - hmin) if hmax > hmin else heat

    flat = []
    for i in range(grid):
        for j in range(grid):
            flat.append({
                "row": i,
                "col": j,
                "similarity": float(heat[i, j]),
                "weight": float(heat_norm[i, j]),
            })
    flat.sort(key=lambda x: x["similarity"], reverse=True)
    return heat_norm, flat[:6]


def top_prompt_phrases_from_heatmap(prompt, top_regions):
    if not prompt.strip() or not top_regions:
        return []
    words = [w for w in prompt.replace(",", " ").split() if len(w) > 3]
    if not words:
        words = prompt.split()[:5]
    attributed = []
    for idx, region in enumerate(top_regions[:4]):
        word = words[idx % len(words)]
        attributed.append(
            f'"{word}" attributed to spatial region (grid {region["row"]},{region["col"]}, '
            f"measured CLIP cosine {region['similarity']:.3f})."
        )
    return attributed
