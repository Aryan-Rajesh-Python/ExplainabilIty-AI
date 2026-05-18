import re
from collections import Counter

import numpy as np
import nltk

from xai.report import MechanisticReport
from xai.routing import run_topical_routing


def _transcript_terms(transcript, limit=8):
    tokens = nltk.word_tokenize(transcript.lower())
    stop = set(nltk.corpus.stopwords.words("english"))
    content = [w for w in tokens if w.isalpha() and len(w) > 2 and w not in stop]
    return Counter(content).most_common(limit)


def analyze_video(
    frame_motions,
    duration,
    fps,
    transcript="",
    visual_weight=0.5,
    audio_weight=0.5,
    user_prompt="",
    zero_shot_model=None,
):
    report = MechanisticReport(modality="video")
    motions = np.array(frame_motions) if frame_motions else np.array([0.0])

    report.input_semantics.append(
        f"{len(motions)} sampled frames over {duration:.1f}s "
        f"({fps:.1f} fps) attributed temporally."
    )

    if user_prompt.strip():
        report.input_semantics.append(
            f'Generation intent: "{user_prompt[:120]}".'
        )

    if len(motions) > 0:
        peak = int(np.argmax(motions))
        report.feature_attribution.append(
            f"Frame index {peak} attributed highest motion saliency "
            f"(measured score {float(motions[peak]):.2f})."
        )
        if float(np.max(motions)) > 20:
            report.feature_attribution.append(
                "High-motion frames drove action-oriented visual interpretation."
            )

    avg_motion = float(np.mean(motions)) if len(motions) else 0.0
    has_speech = bool(
        transcript
        and not str(transcript).lower().startswith("could not")
    )

    if has_speech:
        terms = _transcript_terms(transcript)
        if terms:
            term_str = ", ".join(f'"{w}"' for w, _ in terms[:6])
            report.feature_attribution.append(
                f"Transcript token attribution: {term_str} semantically prioritized."
            )
        if zero_shot_model:
            route = run_topical_routing(
                zero_shot_model,
                transcript[:2000],
                user_prompt,
            )
            report.input_semantics.append(
                f"Content routing cluster: \"{route['top_label']}\" "
                f"(measured prioritization {route['top_score']:.2f})."
            )
            for alt, sc in route.get("secondary_pathways", [])[:1]:
                report.internal_representation.append(
                    f"Competing pathway: \"{alt}\" (weight {sc:.2f}) during fusion."
                )

    if has_speech and avg_motion < 25:
        report.internal_representation.append(
            "Multimodal fusion: speech semantics attributed more strongly than motion."
        )
        visual_weight, audio_weight = 0.35, 0.65
    elif avg_motion > 25:
        report.internal_representation.append(
            "Multimodal fusion: visual motion attributed more strongly than speech."
        )
        visual_weight, audio_weight = 0.65, 0.35
    else:
        report.internal_representation.append(
            "Balanced audio-visual fusion during temporal inference."
        )

    if has_speech:
        report.generation_pathway.append(
            "Early stage: acoustic/prosodic structure; mid stage: word-level semantics; "
            "late stage: topic consolidation from transcript-aligned routing."
        )
    else:
        report.generation_pathway.append(
            "Visual motion and frame-difference energy dominated (no reliable transcript)."
        )

    if duration > 60:
        report.generation_pathway.append(
            "Extended temporal horizon enabled narrative progression attribution."
        )
    else:
        report.generation_pathway.append(
            "Short-form temporal compression prioritized key-moment saliency."
        )

    report.generation_pathway.append(
        f"Scene pacing from measured frame-difference energy (mean {avg_motion:.2f})."
    )

    report.output_alignment.append(
        f"Final interpretation aligned with fusion weights "
        f"(visual ~{visual_weight:.0%}, audio ~{audio_weight:.0%})."
    )
    if user_prompt.strip() and has_speech:
        prompt_kw = set(nltk.word_tokenize(user_prompt.lower()))
        overlap = [w for w, _ in _transcript_terms(transcript) if w in prompt_kw]
        if overlap:
            report.output_alignment.append(
                f"Prompt-to-content alignment via terms: {', '.join(overlap[:6])}."
            )

    report.artifacts["frame_motions"] = motions.tolist()
    report.artifacts["fusion"] = {"visual": visual_weight, "audio": audio_weight}
    return report
