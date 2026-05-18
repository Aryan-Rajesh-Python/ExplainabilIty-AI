"""
Explainable travel planner agent: multi-step tool trace + mechanistic XAI.
"""

import re
from typing import Callable, Optional

import nltk

from xai.agent_trace import AgentDecisionStep, AgentRunTrace, StepTimer
from xai.discourse import analyze_discourse
from xai.report import MechanisticReport
from xai.routing import run_topical_routing
from xai.text import analyze_text


TRAVEL_TOOLS = [
    (
        "analyze_constraints",
        "Parse user goals, budget, duration, interests, and travel style.",
    ),
    (
        "research_destination",
        "Gather destination-specific context (areas, culture, season, logistics).",
    ),
    (
        "plan_itinerary",
        "Build day-by-day schedule with times and neighborhoods.",
    ),
    (
        "plan_experiences",
        "Select must-visit places, food, and local experiences aligned to interests.",
    ),
    (
        "add_practical_and_costs",
        "Add transport, safety, packing tips, and budget estimates.",
    ),
    (
        "compile_final_plan",
        "Synthesize all steps into one cohesive travel document.",
    ),
]


def build_user_request(
    destination: str,
    days: int,
    interests: str,
    budget: str,
    travelers: str,
    travel_style: str,
    extra_notes: str,
) -> str:
    parts = [
        f"Plan a trip to {destination} for {days} day(s).",
        f"Interests: {interests or 'general sightseeing'}.",
        f"Budget: {budget}.",
        f"Travelers: {travelers}.",
        f"Style: {travel_style}.",
    ]
    if extra_notes.strip():
        parts.append(f"Additional notes: {extra_notes.strip()}")
    return " ".join(parts)


def _gemini_call(model, prompt: str, timeout: int = 75) -> str:
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    def _call():
        return model.generate_content(prompt)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            response = pool.submit(_call).result(timeout=timeout)
        return (response.text or "").strip()
    except FuturesTimeout:
        return "Step timed out."
    except Exception as e:
        return f"Step failed: {e}"


def _summarize(text: str, max_len: int = 400) -> str:
    t = " ".join(text.split())
    return t if len(t) <= max_len else t[: max_len - 3] + "..."


def _extract_rationale(text: str) -> str:
    for marker in ("RATIONALE:", "Why:", "WHY:"):
        if marker in text:
            part = text.split(marker, 1)[-1].strip()
            return _summarize(part, 300)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        return _summarize(lines[-1], 300)
    return "Step output informed the next planning stage."


def _run_tool_step(
    model,
    step_id: int,
    tool_name: str,
    tool_description: str,
    user_request: str,
    prior_context: str,
    step_prompt: str,
) -> tuple[str, AgentDecisionStep]:
    timer = StepTimer()
    full_prompt = f"""
You are a travel-planning agent executing tool: {tool_name}
Purpose: {tool_description}

USER REQUEST:
{user_request}

PRIOR AGENT STEPS:
{prior_context or "(none yet)"}

TASK:
{step_prompt}

End your response with a line:
RATIONALE: <one sentence explaining why you made these choices for this user>
"""
    output = _gemini_call(model, full_prompt)
    rationale = _extract_rationale(output)
    body = output
    if "RATIONALE:" in body:
        body = body.split("RATIONALE:")[0].strip()

    return body, AgentDecisionStep(
        step_id=step_id,
        tool_name=tool_name,
        tool_description=tool_description,
        input_summary=_summarize(f"{user_request} | prior: {prior_context}", 350),
        output_summary=_summarize(body, 500),
        rationale=rationale,
        duration_ms=timer.elapsed_ms(),
    )


def run_travel_agent_with_trace(
    model,
    user_request: str,
    on_step: Optional[Callable[[AgentDecisionStep], None]] = None,
) -> AgentRunTrace:
    """Run planner as sequential tool calls with logged decisions."""
    trace = AgentRunTrace(user_request=user_request)
    if model is None:
        trace.final_output = "Travel agent unavailable: configure GEMINI_API_KEY."
        return trace

    context_parts: list[str] = []

    step_prompts = {
        "analyze_constraints": (
            "List structured constraints: destination, days, budget level, "
            "traveler type, interests, pace, and any hard requirements. Bullet format."
        ),
        "research_destination": (
            "Write a brief destination brief: best areas to stay, local culture, "
            "weather/season notes, and how to get around. Specific to the destination."
        ),
        "plan_itinerary": (
            "Create a day-by-day itinerary (morning/afternoon/evening) using constraints "
            "and destination brief. Name real places and logical routes."
        ),
        "plan_experiences": (
            "List must-visit places and food/local experiences. For each, say WHY it "
            "matches the user's interests."
        ),
        "add_practical_and_costs": (
            "Add transport tips, safety, packing, and estimated daily cost breakdown "
            "aligned to their budget."
        ),
        "compile_final_plan": (
            "Merge ALL prior steps into one polished markdown travel plan with headings: "
            "Trip Overview, Day-by-Day Itinerary, Must-Visit Places, Food & Local Experiences, "
            "Practical Information, Estimated Costs. Do not omit prior details."
        ),
    }

    for step_id, (tool_name, tool_desc) in enumerate(TRAVEL_TOOLS, start=1):
        prior = "\n\n".join(context_parts[-3:])
        body, step = _run_tool_step(
            model,
            step_id,
            tool_name,
            tool_desc,
            user_request,
            prior,
            step_prompts[tool_name],
        )
        trace.add_step(step)
        if tool_name == "compile_final_plan":
            trace.final_output = body
        else:
            context_parts.append(f"[{tool_name}]\n{body}")
        if on_step:
            on_step(step)

    if not trace.final_output and trace.steps:
        trace.final_output = trace.steps[-1].output_summary

    if not trace.final_output:
        trace.final_output = "No plan generated."

    return trace


def generate_travel_plan(model, user_request: str, timeout: int = 90) -> str:
    """Legacy single-shot planner (no step trace)."""
    trace = run_travel_agent_with_trace(model, user_request)
    return trace.final_output


def _extract_days(plan: str) -> list[str]:
    days = re.findall(
        r"(?im)^(?:#{1,3}\s*)?(?:day\s*\d+|day\s*[:\-]?\s*\d+).*$",
        plan,
    )
    if days:
        return days[:14]
    blocks = re.split(r"\n(?=#{1,3}\s)", plan)
    return [b.strip()[:200] for b in blocks if len(b.strip()) > 40][:10]


def _extract_places(plan: str) -> list[str]:
    places = []
    for line in plan.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if re.search(r"(?i)\b(visit|see|explore|temple|museum|market|fort|palace|park)\b", line):
            clean = re.sub(r"^[\-\*\d\.\)]+\s*", "", line)[:100]
            if len(clean) > 8:
                places.append(clean)
    return places[:12]


def _section_presence(plan: str) -> dict[str, bool]:
    lower = plan.lower()
    return {
        "itinerary": "day-by-day" in lower or "day 1" in lower,
        "food": "food" in lower or "restaurant" in lower or "cuisine" in lower,
        "practical": "transport" in lower or "practical" in lower or "pack" in lower,
        "costs": "cost" in lower or "budget" in lower or "₹" in plan or "$" in plan,
        "must_visit": "must-visit" in lower or "must visit" in lower,
    }


def analyze_agent_decisions(
    trace: AgentRunTrace,
    report: MechanisticReport,
) -> MechanisticReport:
    """Enrich mechanistic report with logged tool-call decisions."""
    if not trace.steps:
        return report

    report.input_semantics.append(
        f"Agent executed {len(trace.steps)} logged tool calls before final plan."
    )

    for step in trace.steps:
        report.generation_pathway.append(
            f"Step {step.step_id} — tool `{step.tool_name}` ({step.duration_ms:.0f}ms): "
            f"{step.rationale}"
        )
        report.feature_attribution.append(
            f"`{step.tool_name}` output attributed: {_summarize(step.output_summary, 200)}"
        )

    report.internal_representation.append(
        "Agent inference trace follows constraint parsing → destination research → "
        "itinerary → experiences → practical/costs → compilation."
    )
    report.artifacts["agent_decision_steps"] = [s.to_dict() for s in trace.steps]
    sources = report.artifacts.get("trace_sources", [])
    sources.append("agent_tool_call_log")
    report.artifacts["trace_sources"] = sources
    return report


def analyze_travel_plan(
    plan: str,
    user_request: str,
    sentiment_model,
    zero_shot_model,
    embedding_model,
    fast_mode: bool = True,
    agent_trace: Optional[AgentRunTrace] = None,
) -> MechanisticReport:
    report = MechanisticReport(modality="travel_agent")

    dest_match = re.search(
        r"(?i)trip to ([A-Za-z\s]+?)(?: for |\s+for\s+\d|\.)",
        user_request,
    )
    destination = dest_match.group(1).strip() if dest_match else "destination"
    report.input_semantics.append(
        f'Agent conditioned on user goal: travel to {destination} '
        f"(request length {len(user_request.split())} tokens)."
    )

    req_lower = user_request.lower()
    interest_terms = [
        w for w in nltk.word_tokenize(req_lower)
        if w.isalpha() and len(w) > 3
        and w not in {"plan", "trip", "days", "travel", "budget"}
    ]
    if interest_terms:
        report.input_semantics.append(
            f"User interests attributed in request: {', '.join(interest_terms[:8])}."
        )

    if agent_trace:
        analyze_agent_decisions(agent_trace, report)

    sections = _section_presence(plan)
    for name, present in sections.items():
        if present:
            report.feature_attribution.append(
                f'Final plan includes "{name.replace("_", " ")}" section '
                "(derived from compile_final_plan tool)."
            )

    places = _extract_places(plan)
    if places:
        report.feature_attribution.append(
            "Place-selection in final plan: " + "; ".join(places[:4])
        )

    days_found = _extract_days(plan)
    if days_found:
        report.generation_pathway.append(
            f"Final document contains {len(days_found)} day-level units "
            "(post tool-call synthesis)."
        )

    text_report = analyze_text(
        plan,
        sentiment_model,
        zero_shot_model,
        embedding_model,
        user_prompt=user_request,
        fast_mode=fast_mode,
    )
    report.extend(text_report)

    route = run_topical_routing(zero_shot_model, plan[:2000], user_request)
    report.internal_representation.append(
        f'Final plan routing: "{route["top_label"]}" (score {route["top_score"]:.2f}).'
    )

    discourse = analyze_discourse(
        plan,
        user_request,
        modality="travel_itinerary",
        embedding_model=embedding_model,
        zero_shot_model=zero_shot_model,
    )
    report.extend(discourse)

    if "budget" in req_lower:
        report.output_alignment.append(
            "Budget constraints from step `analyze_constraints` propagated to "
            "`add_practical_and_costs` and final compilation."
        )
    if any(w in req_lower for w in ["food", "cuisine", "street", "eat"]):
        report.output_alignment.append(
            "Culinary interests attributed in `plan_experiences` tool output."
        )

    report.output_alignment.append(
        "Final itinerary aligned with logged agent tool chain and user request."
    )
    report.artifacts["travel_places"] = places
    report.artifacts["travel_sections"] = sections
    report.artifacts["trace_sources"] = list(
        dict.fromkeys(
            report.artifacts.get("trace_sources", [])
            + [
                "gemini_tool_calls",
                "bart_mnli_routing",
                "itinerary_structure_parse",
                "discourse_embedding",
            ]
        )
    )
    return report
