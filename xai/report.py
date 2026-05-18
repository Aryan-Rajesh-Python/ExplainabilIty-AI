from dataclasses import dataclass, field
from typing import Any


@dataclass
class MechanisticReport:
    """Five-section model-process-level explainability structure."""

    input_semantics: list[str] = field(default_factory=list)
    feature_attribution: list[str] = field(default_factory=list)
    internal_representation: list[str] = field(default_factory=list)
    generation_pathway: list[str] = field(default_factory=list)
    output_alignment: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    modality: str = "text"

    def extend(self, other: "MechanisticReport") -> None:
        for attr in (
            "input_semantics",
            "feature_attribution",
            "internal_representation",
            "generation_pathway",
            "output_alignment",
        ):
            getattr(self, attr).extend(getattr(other, attr))
        self.artifacts.update(other.artifacts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_semantics": self.input_semantics,
            "feature_attribution": self.feature_attribution,
            "internal_representation": self.internal_representation,
            "generation_pathway": self.generation_pathway,
            "output_alignment": self.output_alignment,
            "artifacts": self.artifacts,
            "modality": self.modality,
        }


def report_to_context(report: MechanisticReport) -> str:
    sections = [
        ("1. Input Semantics", report.input_semantics),
        ("2. Feature Attribution", report.feature_attribution),
        ("3. Internal Representation", report.internal_representation),
        ("4. Generation Pathway", report.generation_pathway),
        ("5. Final Output Alignment", report.output_alignment),
    ]
    lines = [f"Modality: {report.modality}", ""]
    for title, items in sections:
        lines.append(title)
        if items:
            for item in items:
                lines.append(f"- {item}")
        else:
            lines.append("- (no attributions computed)")
        lines.append("")
    return "\n".join(lines).strip()
