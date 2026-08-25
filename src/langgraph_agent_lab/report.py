"""Markdown report generation from the validated metrics model."""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data.

    Generate a report that includes:
    1. Metrics summary table (total scenarios, success rate, retries, interrupts)
    2. Per-scenario results table
    3. Architecture explanation (your graph design, state schema, reducers)
    4. Failure analysis (at least two failure modes you considered)
    5. Improvement plan

    Use reports/lab_report_template.md as your guide.

    Return: formatted markdown string
    """
    rows = []
    for item in metrics.scenario_metrics:
        errors = "; ".join(item.errors).replace("|", "\\|") or "—"
        rows.append(
            "| {scenario} | {expected} | {actual} | {success} | {retries} | "
            "{interrupts} | {errors} |".format(
                scenario=item.scenario_id,
                expected=item.expected_route,
                actual=item.actual_route or "—",
                success="✅" if item.success else "❌",
                retries=item.retry_count,
                interrupts=item.interrupt_count,
                errors=errors,
            )
        )

    scenario_table = "\n".join(rows) or "| — | — | — | — | — | — | — |"
    extension_rows = [
        (
            "| LLM-as-judge | heuristic | structured verdict + timeout/cost guard | "
            "judge unit test | event metadata + reason | opt-in; provider cost |"
        ),
        (
            "| Real HITL | mock approval | interrupt + Command(resume) | "
            "HITL unit test | pause/resume on same thread | interactive only |"
        ),
        (
            "| Time travel | final state only | history replay | history unit test | "
            "history_evidence.json | no UI fork |"
        ),
        (
            "| Mermaid | target diagram | compiled graph export | export test/command | "
            "outputs/graph.mmd | topology only |"
        ),
        (
            "| SQLite recovery | MemorySaver | WAL SQLite adapter | process-restart test | "
            "recovery test | optional dependency |"
        ),
    ]
    extension_table = "\n".join(extension_rows)
    return f"""# Day 08 Lab Report

## 1. Team / student

- Name: LangGraph Agent Lab
- Repo/commit: local workspace
- Date: generated from `outputs/metrics.json`

## 2. Architecture

The workflow is a typed `StateGraph`: `START → intake → classify`, followed by
conditional routing to `answer`, `tool`, `clarify`, `risky_action`, or `retry`.
Tool calls pass through `evaluate`; failed results enter the bounded
`retry → tool` loop and eventually `dead_letter`. Risky requests pass through
`risky_action → approval` before the tool executes. Every terminal branch ends
at `finalize → END`.

`classify_node` uses a structured LLM schema with explicit safety precedence.
`answer_node` receives the original query, tool results, proposed action, and
approval decision as grounded context. Offline fallback output is used only
when no provider is configured, so local CI can still exercise graph behavior.

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| `messages` | append | compact execution trace |
| `tool_results` | append | preserve each tool attempt |
| `errors` | append | retain retry/dead-letter evidence |
| `events` | append | audit trail and node metrics |
| `route`, `attempt`, `evaluation_result` | overwrite | current control state |
| `evaluation_reason`, `judge_calls` | overwrite | judge audit/control state |
| `pending_question`, `proposed_action`, `approval` | overwrite | current user/action decision |
| `final_answer` | overwrite | latest response presented to the user |

## 4. Scenario results

Summary: **{metrics.total_scenarios}** scenarios, **{metrics.success_rate:.1%}**
success rate, **{metrics.total_retries}** retry node visits, **{metrics.total_interrupts}**
approval/HITL events, average **{metrics.avg_nodes_visited:.2f}** nodes, and
`resume_success={metrics.resume_success}`.

| Scenario | Expected route | Actual route | Success | Retries | Interrupts | Errors |
|---|---|---|---:|---:|---:|---|
{scenario_table}

## 5. Failure analysis

1. **Transient tool failure:** `evaluate_node` detects an error result and
   routes to `retry`. `route_after_retry` compares `attempt` with
   `max_attempts`, which prevents an unbounded loop and sends exhausted runs to
   `dead_letter`.
2. **Risky action without approval:** destructive or side-effecting intent is
   routed through `approval`. The mock reviewer approves in CI; setting
   `LANGGRAPH_INTERRUPT=true` pauses the graph with a real LangGraph interrupt.
   Rejection routes to clarification and does not execute the tool.
3. **Vague request:** the graph asks for a missing identifier or desired
   outcome instead of fabricating an answer.

## 6. Persistence / recovery evidence

Each run uses a stable `thread_id` derived from the scenario ID and invokes the
graph with a configurable checkpointer. The CLI inspects state history after
each run and records `resume_success` when history is available. Memory
checkpoints are used by the sample config; the implementation also supports
SQLite with WAL mode via `build_checkpointer("sqlite", ...)`. The CLI writes
checkpoint IDs and counts to `outputs/history_evidence.json` without copying
ticket content or secrets.

## 7. Extension work

The core graph remains the baseline: heuristic evaluation, mock approval by
default, bounded retry, and MemorySaver in the sample config. The selected
extensions are opt-in or additive, so the baseline CI path does not require an
online service.

| Extension | Baseline | Change | Check | Evidence | Limitation |
|---|---|---|---|---|---|
{extension_table}

The SQLite process-restart test passed in the verification environment after
installing `langgraph-checkpoint-sqlite`. On a clean environment without that
optional extra, the test is skipped with an explicit dependency message.

Parallel `Send()`, Streamlit UI, and Postgres were intentionally not enabled:
they are optional extensions and would add concurrency or service dependencies
without improving the stable core contract.

## 8. Improvement plan

The next production step would be replacing mock tools and approval with
authenticated adapters, adding provider-specific timeout/circuit-breaker
policies, redacting sensitive audit fields, and testing classification against a
larger labeled hidden-scenario set.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
