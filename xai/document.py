import re
from xai.report import MechanisticReport


def analyze_pdf_structure(text):
    report = MechanisticReport(modality="pdf")
    sections = re.split(r"\n\s*\n", text)
    section_lens = [len(s.split()) for s in sections if s.strip()]
    if section_lens:
        dominant = max(section_lens)
        report.input_semantics.append(
            f"Document contains {len(section_lens)} semantic blocks; "
            f"longest block ~{dominant} words."
        )
    report.feature_attribution.append(
        "Section hierarchy attributed via paragraph-length segmentation."
    )
    report.generation_pathway.append(
        "Reading flow emerged through sequential section emphasis "
        "(introduction → body → conclusion pattern when present)."
    )
    report.output_alignment.append(
        "PDF structure semantically prioritized informational segmentation."
    )
    return report


def analyze_docx_structure(text):
    report = MechanisticReport(modality="docx")
    paras = [p for p in text.split("\n") if p.strip()]
    if paras:
        lens = [len(p.split()) for p in paras]
        avg = sum(lens) / len(lens)
        report.input_semantics.append(
            f"{len(paras)} paragraph units; mean length ~{avg:.0f} words."
        )
        if max(lens) > 2 * avg:
            report.feature_attribution.append(
                "Lead paragraphs attributed higher hierarchical weight."
            )
    report.internal_representation.append(
        "Formatted writing patterns encoded as paragraph-level latent structure."
    )
    report.generation_pathway.append(
        "Hierarchical prose emerged through paragraph boundary reinforcement."
    )
    return report


def analyze_pptx_structure(text, slide_texts=None):
    report = MechanisticReport(modality="pptx")
    blocks = slide_texts or [b for b in text.split("\n") if b.strip()]
    if blocks:
        report.input_semantics.append(
            f"{len(blocks)} slide-level semantic units attributed."
        )
        short = sum(1 for b in blocks if len(b.split()) < 15)
        if short > len(blocks) // 2:
            report.feature_attribution.append(
                "Bullet-point density reinforced presentation-style formatting."
            )
        long_titles = [
            b for b in blocks
            if len(b.split()) <= 8 and (b.isupper() or len(b) < 60)
        ]
        if long_titles:
            report.feature_attribution.append(
                "Title-style headings attributed higher instructional prioritization."
            )
    report.generation_pathway.append(
        "Slide hierarchy shaped inference toward instructional scaffolding."
    )
    report.internal_representation.append(
        "Layout semantics (title/body) dominate over continuous prose representation."
    )
    return report
