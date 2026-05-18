"""
Streamlit UI for the Explainable Travel Planner Agent.
"""

import streamlit as st

from xai.travel_agent import analyze_travel_plan, build_user_request, run_travel_agent_with_trace


def _has_travel_plan() -> bool:
    plan = st.session_state.get("travel_plan", "")
    return bool(plan and str(plan).strip())


def _display_decision_trace(steps: list):
    st.subheader("🛠️ Agent decision trace (tool calls)")
    st.caption(
        "Each step is a logged tool invocation — inputs, outputs, and why the agent chose it."
    )
    for step in steps:
        label = f"Step {step['step_id']}: `{step['tool_name']}` ({step['duration_ms']:.0f} ms)"
        with st.expander(label, expanded=step["step_id"] <= 2):
            st.markdown(f"**Tool purpose:** {step['tool_description']}")
            st.markdown(f"**Input context:** {step['input_summary']}")
            st.markdown(f"**Output:** {step['output_summary']}")
            st.markdown(f"**Why this step:** {step['rationale']}")


def _run_travel_explanation(
    sentiment_model,
    zero_shot_model,
    embedding_model,
    fast_mode,
    run_mechanistic_narrative,
    reset_xai_session,
):
    reset_xai_session()
    agent_trace = st.session_state.get("travel_agent_trace")
    with st.status("Building mechanistic XAI trace for agent decisions…", expanded=True):
        report = analyze_travel_plan(
            st.session_state.travel_plan,
            st.session_state.travel_request,
            sentiment_model,
            zero_shot_model,
            embedding_model,
            fast_mode=fast_mode,
            agent_trace=agent_trace,
        )
        st.session_state.travel_report = report
    extra = ""
    agent_trace = st.session_state.get("travel_agent_trace")
    if agent_trace:
        extra = agent_trace.to_context()
    run_mechanistic_narrative(
        report,
        st.session_state.travel_request,
        title="🧠 Why the travel agent planned this way",
        extra_context=extra,
    )


def render_travel_agent(
    model,
    sentiment_model,
    zero_shot_model,
    embedding_model,
    fast_mode: bool,
    run_mechanistic_narrative,
    reset_xai_session,
):
    st.header("✈️ Explainable Travel Planner Agent")
    st.caption(
        "Multi-step travel agent with logged tool calls, then mechanistic XAI explains "
        "each decision and the final plan."
    )

    col1, col2 = st.columns(2)
    with col1:
        destination = st.text_input("Destination", placeholder="e.g. Delhi, India")
        days = st.number_input("Number of days", min_value=1, max_value=21, value=3)
        interests = st.text_input(
            "Interests",
            placeholder="history, street food, markets, photography…",
        )
    with col2:
        budget = st.selectbox("Budget", ["budget", "mid-range", "luxury"])
        travelers = st.selectbox(
            "Travelers",
            ["solo", "couple", "family with kids", "group of friends"],
        )
        travel_style = st.selectbox(
            "Travel style",
            ["relaxed", "packed schedule", "balanced"],
        )

    extra_notes = st.text_area(
        "Extra preferences (optional)",
        placeholder="e.g. avoid crowded places, vegetarian food only",
        height=80,
    )

    auto_explain = st.checkbox(
        "Automatically run XAI explanation after plan is generated",
        value=False,
    )

    user_request = build_user_request(
        destination,
        int(days),
        interests,
        budget,
        travelers,
        travel_style,
        extra_notes,
    )

    with st.expander("Full agent request", expanded=False):
        st.write(user_request)

    plan_btn = st.button("🗺️ Generate travel plan (multi-step agent)", type="primary")
    explain_btn = st.button("🔬 Explain why the agent planned this", type="secondary")

    if plan_btn:
        if not destination.strip():
            st.warning("Enter a destination.")
        elif model is None:
            st.error("Gemini is not configured. Set GEMINI_API_KEY.")
        else:
            reset_xai_session()
            progress = st.progress(0, text="Starting agent…")
            step_box = st.empty()
            steps_done: list = []

            def on_step(step):
                steps_done.append(step.to_dict())
                n = len(steps_done)
                progress.progress(
                    n / 6,
                    text=f"Tool call {n}/6: {step.tool_name}…",
                )
                step_box.info(f"Completed `{step.tool_name}` — {step.rationale}")

            with st.spinner("Agent running tool calls (6 steps)…"):
                trace = run_travel_agent_with_trace(
                    model,
                    user_request,
                    on_step=on_step,
                )

            progress.progress(1.0, text="Agent finished.")
            st.session_state.travel_plan = trace.final_output
            st.session_state.travel_request = user_request
            st.session_state.travel_agent_trace = trace
            st.session_state.travel_steps = [s.to_dict() for s in trace.steps]
            st.session_state.pop("travel_report", None)
            if auto_explain:
                st.session_state.pending_travel_explain = True
            st.rerun()

    if st.session_state.pop("pending_travel_explain", False) and _has_travel_plan():
        _run_travel_explanation(
            sentiment_model,
            zero_shot_model,
            embedding_model,
            fast_mode,
            run_mechanistic_narrative,
            reset_xai_session,
        )

    if _has_travel_plan():
        steps = st.session_state.get("travel_steps", [])
        if steps:
            _display_decision_trace(steps)

        st.subheader("📋 Final travel plan")
        st.markdown(st.session_state.travel_plan)
        st.success("Plan ready — run **Explain why the agent planned this** for full XAI.")
    else:
        st.info("Generate a plan first (agent runs 6 logged tool calls; may take 1–3 minutes).")

    if explain_btn:
        if not _has_travel_plan():
            st.warning("Generate a travel plan first.")
        else:
            _run_travel_explanation(
                sentiment_model,
                zero_shot_model,
                embedding_model,
                fast_mode,
                run_mechanistic_narrative,
                reset_xai_session,
            )
