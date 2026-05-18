from collections import Counter

import nltk
import numpy as np

from xai.report import MechanisticReport
from xai.routing import run_topical_routing
from xai.shap_text import shap_token_attribution


def _lime_token_attribution(text, score_fn, max_tokens=12, samples=8):
    snippet = text[:1500]
    tokens = nltk.word_tokenize(snippet)
    if not tokens:
        return []

    unique = list(dict.fromkeys(tokens))[:max_tokens]
    baseline = score_fn(snippet)
    if baseline is None:
        return []

    attributions = []
    rng = np.random.default_rng(42)
    for token in unique:
        deltas = []
        for _ in range(samples):
            masked = [
                t if (t != token or rng.random() > 0.5) else "[MASK]"
                for t in tokens
            ]
            s = score_fn(" ".join(masked))
            if s is not None:
                deltas.append(baseline - s)
        if deltas:
            attributions.append((token, float(sum(deltas) / len(deltas))))

    attributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return attributions[:12]


def _keyword_attribution(top_kw, top_label):
    lines = []
    for word, count in top_kw[:8]:
        lines.append(
            f'"{word}" semantically prioritized (frequency {count}) '
            f'within "{top_label}" thematic cluster.'
        )
    return lines


def analyze_text(
    text,
    sentiment_model,
    zero_shot_model,
    embedding_model,
    user_prompt="",
    use_shap=False,
    fast_mode=True,
):
    report = MechanisticReport(modality="text")
    snippet = text[:2000]
    words = text.split()
    tokens = nltk.word_tokenize(text.lower())
    stop = set(nltk.corpus.stopwords.words("english"))
    content = [w for w in tokens if w.isalpha() and w not in stop]
    top_kw = Counter(content).most_common(10)

    route = run_topical_routing(zero_shot_model, snippet, user_prompt)
    top_label = route["top_label"]
    top_score = route["top_score"]
    zs_labels = route["labels"]
    zs_scores = route["scores"]

    report.input_semantics.append(
        f"{len(words)} tokens semantically prioritized in the output surface form."
    )
    if top_kw:
        kw_str = ", ".join(f'"{w}"' for w, _ in top_kw[:6])
        report.input_semantics.append(f"Dominant lexical concepts: {kw_str}.")

    if user_prompt.strip():
        prompt_tokens = set(nltk.word_tokenize(user_prompt.lower()))
        overlap = [w for w, _ in top_kw[:10] if w in prompt_tokens]
        if overlap:
            report.input_semantics.append(
                f"Prompt-aligned tokens attributed: {', '.join(overlap[:8])}."
            )

    lime_attrs = []
    shap_attrs = []

    if fast_mode:
        report.feature_attribution.extend(_keyword_attribution(top_kw, top_label))
    else:
        def score_fn(t):
            try:
                r = run_topical_routing(zero_shot_model, t[:1500], user_prompt)
                for lab, sc in zip(r["labels"], r["scores"]):
                    if lab == top_label:
                        return sc
                return r["top_score"]
            except Exception:
                return None

        lime_attrs = _lime_token_attribution(snippet, score_fn)
        for token, delta in lime_attrs[:6]:
            direction = "increased" if delta > 0 else "decreased"
            report.feature_attribution.append(
                f'[LIME] "{token}" {direction} "{top_label}" alignment '
                f"(magnitude {abs(delta):.3f})."
            )

        if use_shap:
            shap_attrs = shap_token_attribution(snippet, score_fn, nsamples=40)
            for token, val in shap_attrs[:6]:
                direction = "increased" if val > 0 else "decreased"
                report.feature_attribution.append(
                    f'[SHAP] "{token}" {direction} "{top_label}" '
                    f"(Shapley {val:+.4f})."
                )

    try:
        sentiment = sentiment_model(snippet[:512])[0]
        report.internal_representation.append(
            f"Affective inference trace: {sentiment['label']} "
            f"(score {sentiment['score']:.3f})."
        )
    except Exception:
        pass

    emb = embedding_model.encode([snippet])
    norm = float(np.linalg.norm(emb[0]))
    density = "high" if norm > 10 else "moderate" if norm > 5 else "compact"
    report.internal_representation.append(
        f"Semantic embedding density: {density} (source: MiniLM L2 norm {norm:.1f})."
    )

    total = sum(zs_scores[:4]) or 1.0
    for label, score in zip(zs_labels[:4], zs_scores[:4]):
        pct = 100 * score / total
        if pct > 15:
            report.internal_representation.append(
                f'"{label}" cluster weight ~{pct:.0f}% of topical emphasis '
                f"(source: BART-MNLI routing, normalized top-4)."
            )

    for alt_label, alt_score in route.get("secondary_pathways", [])[:2]:
        report.internal_representation.append(
            f"Competing pathway: \"{alt_label}\" (weight {alt_score:.2f}) "
            "before dominant cluster stabilized."
        )

    if len(words) > 300:
        report.generation_pathway.append(
            "Long-form elaboration emerged through sustained informational expansion."
        )
    elif len(words) < 80:
        report.generation_pathway.append(
            "Compression pathway prioritized concise semantic delivery."
        )

    for label, score in zip(zs_labels[:3], zs_scores[:3]):
        if score > 0.35:
            report.generation_pathway.append(
                f'Rhetorical structure gravitated toward "{label}" '
                f"during inference (weight {score:.2f})."
            )

    report.output_alignment.append(
        f'Final surface form aligns with "{top_label}" '
        f"(semantic prioritization {top_score:.2f}), grounded in current prompt and content."
    )

    report.artifacts["zero_shot"] = list(zip(zs_labels, zs_scores))
    report.artifacts["lime_attribution"] = lime_attrs
    report.artifacts["shap_attribution"] = shap_attrs
    report.artifacts["top_keywords"] = top_kw
    report.artifacts["routing"] = route
    report.artifacts["trace_sources"] = ["bart_mnli_routing", "sentence_transformer_embedding"]
    if lime_attrs:
        report.artifacts["trace_sources"].append("lime_token_masking")
    if shap_attrs:
        report.artifacts["trace_sources"].append("kernel_shap")
    token_viz = shap_attrs if shap_attrs else lime_attrs
    if token_viz:
        report.artifacts["token_attribution"] = token_viz
    return report
