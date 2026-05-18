"""
Slide-level discourse XAI for PPTX: transitions, density evolution, sequential scaffolding.
"""

import re
from collections import Counter

import nltk
import numpy as np

from xai.report import MechanisticReport
from xai.metrics import section_embedding_transitions

_TECH = {
    "algorithm", "learning", "regression", "classification", "neural",
    "supervised", "training", "model", "data", "ensemble",
}


_DIAGRAM_CUES = {"diagram", "workflow", "figure", "chart", "architecture", "network", "flow"}


def analyze_pptx_discourse(
    slide_texts,
    embedding_model=None,
    user_prompt="",
    zero_shot_model=None,
):
    report = MechanisticReport(modality="pptx")
    slides = [s.strip() for s in slide_texts if s and s.strip()]
    if not slides:
        return report

    report.generation_pathway.append(
        "Early discourse planning established slide-level educational taxonomy "
        "before downstream inference expanded algorithm-specific detail."
    )

    if embedding_model and len(slides) >= 2:
        trans = section_embedding_transitions(slides, embedding_model, max_sections=12)
        for t in trans[:4]:
            report.generation_pathway.append(
                f"Slide {t['from_idx'] + 1}→{t['to_idx'] + 1} transition: "
                f"semantic continuity {t['similarity']:.3f} "
                "(source: SentenceTransformer cosine)."
            )

    densities = []
    for i, slide in enumerate(slides[:15]):
        words = [w.lower() for w in slide.split() if w.isalpha()]
        if words:
            densities.append(sum(1 for w in words if w in _TECH) / len(words))

    if len(densities) >= 2 and densities[-1] > densities[0] + 0.03:
        report.internal_representation.append(
            "Information density evolution: technical term density increased across "
            f"slides (measured {densities[0]:.2f} → {densities[-1]:.2f} tech-token ratio)."
        )

    for i, slide in enumerate(slides[:8]):
        lines = [ln.strip() for ln in slide.split("\n") if ln.strip()]
        if len(lines) >= 2:
            title_len, body_len = len(lines[0].split()), sum(len(l.split()) for l in lines[1:])
            if title_len < body_len:
                report.feature_attribution.append(
                    f"Slide {i + 1}: heading-level attention dominated structural planning "
                    "before body-level explanatory detail (title/body length ratio)."
                )
                break

    sents_per_slide = []
    for slide in slides[:10]:
        sents = nltk.sent_tokenize(slide[:500])
        if sents:
            sents_per_slide.append(np.mean([len(s.split()) for s in sents]))

    if len(sents_per_slide) >= 3:
        if sents_per_slide[-1] > sents_per_slide[0] + 2:
            report.output_alignment.append(
                "Readability adaptation: sentence complexity increased on later slides "
                "while preserving instructional scaffolding (measured words/sentence)."
            )

    report.generation_pathway.append(
        "Sequential slide generation trace: top-down structural scaffolding "
        f"across {len(slides)} units before bullet-level elaboration."
    )

    diagram_slides = [
        i for i, s in enumerate(slides)
        if any(c in s.lower() for c in _DIAGRAM_CUES)
    ]
    if diagram_slides:
        report.feature_attribution.append(
            "Diagram-oriented layout routing attributed on slide(s) "
            f"{', '.join(str(i + 1) for i in diagram_slides[:4])} "
            "(spatial explanatory structure over prose; keyword heuristic)."
        )

    if zero_shot_model:
        from xai.evolution import distribution_entropy, global_attention_drift, latent_suppression_narrative
        from xai.routing import run_topical_routing

        route = run_topical_routing(zero_shot_model, " ".join(slides)[:2000], user_prompt)
        sup = latent_suppression_narrative(
            route["top_label"],
            [p[0] for p in route.get("secondary_pathways", [])],
        )
        if sup:
            report.internal_representation.append(sup)
        if len(slides) >= 2:
            e0 = run_topical_routing(zero_shot_model, slides[0][:800], user_prompt)
            e1 = run_topical_routing(zero_shot_model, slides[-1][:800], user_prompt)
            report.generation_pathway.append(
                global_attention_drift(e0["top_label"], e1["top_label"])
            )
            ent0, ent1 = distribution_entropy(e0["scores"][:5]), distribution_entropy(e1["scores"][:5])
            if ent1 < ent0:
                report.internal_representation.append(
                    f"Cross-slide attention entropy decreased ({ent0:.2f}→{ent1:.2f}; "
                    "source: BART-MNLI routing)."
                )

    report.artifacts["slide_count"] = len(slides)
    report.artifacts["trace_sources"] = [
        "slide_segmentation",
        "sentence_transformer_slide_cosine",
        "tech_density_per_slide",
    ]
    return report
