import matplotlib.pyplot as plt
import numpy as np


def plot_gradcam_overlay(image, heatmap):
    img = np.array(image.convert("RGB").resize((224, 224))) / 255.0
    heat = np.array(heatmap)
    if heat.shape != img.shape[:2]:
        from PIL import Image as PILImage
        heat_img = PILImage.fromarray((heat * 255).astype(np.uint8)).resize((224, 224))
        heat = np.array(heat_img) / 255.0
    cmap = plt.cm.jet(heat)[:, :, :3]
    overlay = 0.55 * img + 0.45 * cmap
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(overlay)
    ax.axis("off")
    ax.set_title("Grad-CAM region attribution")
    fig.tight_layout()
    return fig


def plot_mel_spectrogram(mel_db):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
    ax.set_title("Mel spectrogram (attribution substrate)")
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bands")
    fig.tight_layout()
    return fig


def plot_zero_shot_bars(labels, scores):
    fig, ax = plt.subplots(figsize=(6, 3))
    y = list(reversed(labels[:8]))
    x = list(reversed([float(s) for s in scores[:8]]))
    ax.barh(y, x, color="#4C78A8")
    ax.set_xlabel("Semantic prioritization")
    ax.set_title("Topic dominance (zero-shot routing)")
    fig.tight_layout()
    return fig


def plot_clip_heatmap(image, heatmap):
    from PIL import Image as PILImage

    img = np.array(image.convert("RGB").resize((224, 224))) / 255.0
    heat = np.array(heatmap)
    heat_up = np.array(
        PILImage.fromarray((heat * 255).astype(np.uint8)).resize((224, 224))
    ) / 255.0
    cmap = plt.cm.inferno(heat_up)[:, :, :3]
    overlay = 0.5 * img + 0.5 * cmap
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(img)
    axes[0].set_title("Output image")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("CLIP prompt–region alignment")
    axes[1].axis("off")
    fig.tight_layout()
    return fig


def plot_token_attribution(pairs, title="Token attribution"):
    if not pairs:
        return None
    tokens = [p[0][:20] for p in pairs[:10]]
    vals = [p[1] for p in pairs[:10]]
    colors = ["#E45756" if v < 0 else "#54A24B" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(list(reversed(tokens)), list(reversed(vals)), color=list(reversed(colors)))
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("Attribution score")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_confidence_trajectory(trajectory):
    if not trajectory:
        return None
    stages = [t["stage"] for t in trajectory]
    confs = [t["confidence"] for t in trajectory]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(stages, confs, marker="s", color="#F58518", linewidth=2)
    ax.set_ylabel("Softmax confidence (dominant class)")
    ax.set_title("Measured confidence across spatial scales")
    ax.set_ylim(0, max(1.0, max(confs) * 1.1))
    fig.tight_layout()
    return fig


def plot_diffusion_steps(step_metrics):
    if not step_metrics:
        return None
    steps = [m["step"] for m in step_metrics]
    stds = [m["latent_std"] for m in step_metrics]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(steps, stds, marker="o", color="#B279A2", linewidth=2)
    ax.set_xlabel("Denoising step")
    ax.set_ylabel("Latent std")
    ax.set_title("Diffusion denoising progression")
    fig.tight_layout()
    return fig
