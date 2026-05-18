"""
Prompt- and content-grounded topical routing (avoids stale / mismatched clusters).
"""

import re
from collections import Counter

import nltk

# (substring triggers in prompt+content, label) — order matters
_THEME_RULES = [
    (
        ["machine learning", " ml ", "algorithm", "neural network", "supervised",
         "unsupervised", "deep learning", "classification", "regression"],
        "machine learning taxonomy explanation",
    ),
    (
        ["recommend", "personalization", "collaborative filtering"],
        "recommendation systems explanation",
    ),
    (
        ["ppt", "pptx", "slide", "presentation", "deck"],
        "presentation slide generation",
    ),
    (
        ["pdf", "report", "whitepaper"],
        "formal report documentation",
    ),
    (
        ["docx", "word document", "manuscript"],
        "structured document generation",
    ),
    (
        ["tutorial", "how to", "step-by-step", "guide"],
        "tutorial documentation",
    ),
    (
        ["research", "paper", "citation", "hypothesis"],
        "academic research exposition",
    ),
]

_BASE_LABELS = [
    "educational explanation",
    "technical instruction",
    "analytical writing",
    "scientific exposition",
    "summarization",
    "instructional documentation",
    "creative writing",
    "storytelling",
]


def _content_keywords(text, limit=15):
    tokens = nltk.word_tokenize(text.lower())
    stop = set(nltk.corpus.stopwords.words("english"))
    content = [w for w in tokens if w.isalpha() and len(w) > 2 and w not in stop]
    return Counter(content).most_common(limit)


def build_candidate_labels(user_prompt: str, text: str) -> list[str]:
    """Labels derived from THIS prompt + document only."""
    keywords = _content_keywords(text[:4000])
    combined = f"{user_prompt} {' '.join(w for w, _ in keywords)}".lower()

    labels = []
    for triggers, label in _THEME_RULES:
        if any(t in combined for t in triggers):
            labels.append(label)

    for label in _BASE_LABELS:
        if label not in labels:
            labels.append(label)

    return list(dict.fromkeys(labels))[:14]


def _label_allowed(label: str, user_prompt: str, keywords) -> bool:
    """Block 'recommendation' unless prompt/content support it."""
    combined = f"{user_prompt} {' '.join(w for w, _ in keywords)}".lower()
    if "recommendation" in label.lower():
        return any(
            t in combined
            for t in ["recommend", "personalization", "collaborative"]
        )
    return True


def run_topical_routing(zero_shot_model, text: str, user_prompt: str):
    """
    Classify using prompt + content jointly. Returns dict with
    labels, scores, top_label, top_score, secondary_pathways.
    """
    snippet = text[:2000]
    keywords = _content_keywords(snippet)
    labels = build_candidate_labels(user_prompt, snippet)

    hypothesis = (
        f"User prompt that caused this generation:\n{user_prompt[:400]}\n\n"
        f"Generated output excerpt:\n{snippet}"
    )

    zs = zero_shot_model(hypothesis[:2000], labels)

    paired = list(zip(zs["labels"], [float(s) for s in zs["scores"]]))
    paired = [
        (lab, sc)
        for lab, sc in paired
        if _label_allowed(lab, user_prompt, keywords)
    ]
    if not paired:
        paired = list(zip(zs["labels"], [float(s) for s in zs["scores"]]))

    paired.sort(key=lambda x: x[1], reverse=True)
    top_label, top_score = paired[0]

    secondary = []
    for lab, sc in paired[1:4]:
        if sc >= 0.22 and (top_score - sc) < 0.25:
            secondary.append((lab, sc))

    return {
        "labels": [p[0] for p in paired],
        "scores": [p[1] for p in paired],
        "top_label": top_label,
        "top_score": top_score,
        "secondary_pathways": secondary,
        "keywords": keywords,
        "candidate_labels": labels,
    }
