import numpy as np
import nltk


def shap_token_attribution(text, score_fn, max_tokens=18, nsamples=80):
    """
    Kernel SHAP over token presence features for a scalar score_fn(text)->float.
    Returns list of (token, shap_value).
    """
    try:
        import shap
    except ImportError:
        return []

    tokens = nltk.word_tokenize(text[:1500])
    if len(tokens) < 3:
        return []

    tokens = tokens[:max_tokens]
    n = len(tokens)
    baseline_text = " ".join(tokens)
    baseline = score_fn(baseline_text)
    if baseline is None:
        return []

    rng = np.random.default_rng(0)

    def predict(mask_batch):
        scores = []
        for mask in mask_batch:
            words = [
                tokens[i] if mask[i] > 0.5 else "[MASK]"
                for i in range(n)
            ]
            s = score_fn(" ".join(words))
            scores.append(s if s is not None else baseline)
        return np.array(scores, dtype=np.float64)

    background = rng.integers(0, 2, size=(8, n)).astype(np.float64)
    explainer = shap.KernelExplainer(predict, background)
    try:
        shap_values = explainer.shap_values(
            np.ones((1, n), dtype=np.float64),
            nsamples=nsamples,
            silent=True,
        )
    except Exception:
        return []

    if isinstance(shap_values, list):
        vals = np.array(shap_values[0]).flatten()
    else:
        vals = np.array(shap_values).flatten()

    pairs = [(tokens[i], float(vals[i])) for i in range(n)]
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:12]
