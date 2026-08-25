"""Node functions for the LangGraph support-ticket workflow.

The graph keeps node outputs as small, serializable state updates. The two
required LLM nodes use real LangChain model calls; an explicit local fallback
keeps the lab runnable for CI and demos without credentials.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, make_event


class ClassificationResult(BaseModel):
    """Structured contract returned by the classifier LLM."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    risk_level: Literal["low", "medium", "high"] = "low"
    rationale: str = Field(default="", max_length=500)


class EvaluationVerdict(BaseModel):
    """Bounded structured verdict returned by the optional LLM judge."""

    verdict: Literal["success", "needs_retry"]
    reason: str = Field(default="", max_length=500)


def _response_text(response: Any) -> str:  # noqa: ANN401
    """Extract text from common LangChain response shapes."""
    if isinstance(response, str):
        return response.strip()
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                value = block.get("text") or block.get("content")
                if value:
                    parts.append(str(value))
            else:
                parts.append(str(block))
        return "".join(parts).strip()
    return str(content).strip()


def _offline_classification(query: str) -> tuple[str, str]:
    """Conservative local fallback used only when an LLM is unavailable."""
    text = query.casefold()
    risky = re.compile(
        r"\b(refund|delete|delet|remove|erase|cancel|cancell|send\s+(?:a\s+)?"
        r"(?:email|message)|close\s+(?:the\s+)?account|disable|charge|transfer)\b"
    )
    tool = re.compile(
        r"\b(look\s*up|lookup|status|track(?:ing)?|search|find|check|order|"
        r"shipment|ticket\s+number)\b"
    )
    missing = re.compile(
        r"\b(it|this|that|something|anything)\b|\b(can\s+you\s+fix|help\s+me)\b"
    )
    error = re.compile(
        r"\b(timeout|timed\s*out|failure|failed|crash(?:ed)?|error|unavailable|down|"
        r"cannot\s+recover|system\s+failure|service\s+unavailable)\b"
    )
    # Match the documented safety precedence for ambiguous requests.
    if risky.search(text):
        return "risky", "high"
    if tool.search(text):
        return "tool", "low"
    if missing.search(text) and len(text.split()) <= 8:
        return "missing_info", "low"
    if error.search(text):
        return "error", "low"
    return "simple", "low"


def _fallback_answer(state: AgentState) -> str:
    """Return a safe, context-grounded answer when no LLM is configured."""
    query = state.get("query", "").strip()
    route = state.get("route", "simple")
    results = [item for item in state.get("tool_results", []) if "ERROR" not in item.upper()]
    if results:
        return f"I reviewed the available tool result for your request: {results[-1]}"
    if route == "risky":
        return "The requested action was approved and has been submitted for processing."
    return f"Here is general guidance for your request: {query}"


def _timed_event(
    node: str,
    event_type: str,
    message: str,
    started: float,
    **metadata: Any,  # noqa: ANN401
) -> dict[str, Any]:
    return make_event(
        node,
        event_type,
        message,
        latency_ms=int((time.perf_counter() - started) * 1000),
        **metadata,
    )


def _invoke_with_timeout(
    call: Callable[[], Any], timeout_seconds: float
) -> tuple[Any | None, str | None]:  # noqa: ANN401
    """Run an optional judge call with a hard wall-clock budget."""
    result: dict[str, Any] = {}

    def worker() -> None:
        try:
            result["value"] = call()
        except Exception as exc:  # pragma: no cover - provider-specific failures
            result["error"] = type(exc).__name__

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(max(0.0, timeout_seconds))
    if thread.is_alive():
        return None, "timeout"
    if "error" in result:
        return None, str(result["error"])
    return result.get("value"), None


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.05, float(os.getenv(name, str(default))))
    except ValueError:
        return default


def intake_node(state: AgentState) -> dict:
    """Normalize raw query and add the first audit event."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query with structured LLM output."""
    started = time.perf_counter()
    query = state.get("query", "")
    system_prompt = """
You classify support tickets into exactly one route.

Routes:
- risky: a side effect is requested, including refunds, deletion, cancellation,
  account changes, charges, or sending a message/email.
- tool: the user asks for an information lookup such as order status, tracking,
  search, or a record check.
- missing_info: the request is too vague to act on and lacks the needed object,
  identifier, or desired outcome.
- error: a system failure, timeout, crash, outage, or unrecoverable processing
  problem is being reported.
- simple: a general informational question answerable without an external tool.

When multiple signals are present, use this safety precedence: risky > tool >
missing_info > error > simple. Return only the structured schema.
""".strip()
    source = "llm"
    rationale = ""
    route: str
    risk_level: str
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(ClassificationResult)
        result = structured_llm.invoke(
            [("system", system_prompt), ("human", f"Support ticket: {query}")]
        )
        parsed = (
            result
            if isinstance(result, ClassificationResult)
            else ClassificationResult.model_validate(result)
        )
        route = parsed.route
        risk_level = parsed.risk_level
        rationale = parsed.rationale
    except Exception as exc:  # pragma: no cover - used when credentials are absent
        route, risk_level = _offline_classification(query)
        source = "offline_fallback"
        rationale = f"LLM unavailable: {type(exc).__name__}"

    return {
        "route": route,
        "risk_level": risk_level,
        "events": [
            _timed_event(
                "classify",
                "completed",
                f"classified as {route}",
                started,
                source=source,
                rationale=rationale,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call, including deterministic transient failures."""
    started = time.perf_counter()
    attempt = int(state.get("attempt", 0))
    route = state.get("route", "")
    # Error routes fail twice before succeeding. A custom tool scenario can
    # opt into one transient failure with should_retry=True.
    transient_failure = (route == "error" and attempt < 2) or (
        route == "tool" and bool(state.get("should_retry")) and attempt == 0
    )
    if transient_failure:
        result = f"ERROR: transient tool failure on attempt {attempt + 1}"
        event_type = "error"
        message = "tool returned a transient failure"
    elif route == "risky":
        result = "SUCCESS: approved action completed by mock tool"
        event_type = "completed"
        message = "approved action executed by mock tool"
    else:
        result = "SUCCESS: lookup completed by mock tool"
        event_type = "completed"
        message = "mock tool returned a result"
    return {
        "tool_results": [result],
        "messages": [f"tool:{result}"],
        "events": [_timed_event("tool", event_type, message, started, attempt=attempt)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate the latest tool result and drive the retry gate.

    The heuristic remains the safe default for CI. Setting
    ``LLM_JUDGE_ENABLED=true`` enables a structured judge with a timeout and a
    per-run call budget; any judge failure falls back to the same heuristic.
    """
    started = time.perf_counter()
    latest = (state.get("tool_results") or [""])[-1]
    heuristic = "needs_retry" if latest.lstrip().upper().startswith("ERROR:") else "success"
    evaluation = heuristic
    reason = "heuristic error-prefix check"
    source = "heuristic"
    judge_calls = int(state.get("judge_calls", 0))
    judge_enabled = os.getenv("LLM_JUDGE_ENABLED", "").casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    max_calls = _env_int("LLM_JUDGE_MAX_CALLS", 2)
    if judge_enabled and judge_calls < max_calls:
        timeout_seconds = _env_float("LLM_JUDGE_TIMEOUT_SECONDS", 5.0)
        prompt = (
            "Evaluate this tool result. Return verdict=needs_retry only if the "
            "result clearly reports a transient failure; otherwise return success. "
            "Give a short reason. Tool result: "
            f"{latest}"
        )
        raw, failure = _invoke_with_timeout(
            lambda: get_llm()
            .with_structured_output(EvaluationVerdict)
            .invoke(prompt),
            timeout_seconds,
        )
        judge_calls += 1
        if failure:
            reason = f"judge fallback: {failure}"
        else:
            try:
                verdict = (
                    raw
                    if isinstance(raw, EvaluationVerdict)
                    else EvaluationVerdict.model_validate(raw)
                )
                evaluation = verdict.verdict
                reason = verdict.reason or "structured judge verdict"
                source = "llm_judge"
            except Exception as exc:  # pragma: no cover - malformed provider output
                reason = f"judge fallback: {type(exc).__name__}"
    elif judge_enabled:
        reason = f"judge cost guard reached ({max_calls} calls)"
    else:
        reason = "LLM judge disabled; heuristic used"
    return {
        "evaluation_result": evaluation,
        "evaluation_reason": reason,
        "judge_calls": judge_calls,
        "events": [
            _timed_event(
                "evaluate",
                "completed",
                f"tool result evaluated as {evaluation}",
                started,
                source=source,
                reason=reason,
                judge_calls=judge_calls,
                judge_max_calls=max_calls,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a grounded final response with an LLM."""
    started = time.perf_counter()
    context = {
        "query": state.get("query", ""),
        "route": state.get("route", ""),
        "tool_results": state.get("tool_results", []),
        "approval": state.get("approval"),
        "proposed_action": state.get("proposed_action"),
    }
    prompt = """
You are a support agent. Answer the user's request using only the supplied
context. Do not invent order details, account facts, or actions. If a tool
result is present, summarize it clearly. If an action was approved, state what
was completed. Be concise and helpful.

Context:
{context}
""".strip().format(context=json.dumps(context, ensure_ascii=False))
    source = "llm"
    try:
        response = get_llm().invoke(prompt)
        answer = _response_text(response)
        if not answer:
            raise ValueError("LLM returned an empty answer")
    except Exception as exc:  # pragma: no cover - used when credentials are absent
        answer = _fallback_answer(state)
        source = "offline_fallback"
        context["fallback_reason"] = type(exc).__name__
    return {
        "final_answer": answer,
        "events": [
            _timed_event(
                "answer", "completed", "grounded answer generated", started, source=source
            )
        ],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    started = time.perf_counter()
    query = state.get("query", "").strip()
    prompt = f"""
Write one concise clarification question for this vague support request:
{query}
Ask for the missing identifier, object, or desired outcome. Do not guess the
user's intent and do not answer the request yet.
""".strip()
    source = "llm"
    failure = ""
    try:
        question = _response_text(get_llm().invoke(prompt))
        if not question:
            raise ValueError("LLM returned an empty clarification")
    except Exception as exc:  # pragma: no cover - used when credentials are absent
        question = (
            "Could you provide the relevant identifier or describe exactly what you want "
            f"us to fix? (Original request: {query})"
        )
        source = "offline_fallback"
        failure = type(exc).__name__
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [
            _timed_event(
                "clarify",
                "completed",
                "clarification requested",
                started,
                source=source,
                fallback_reason=failure,
            )
        ],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a side-effecting action for human approval."""
    started = time.perf_counter()
    query = state.get("query", "").strip()
    proposed = (
        f"Proposed side-effecting action: {query}. Review the target and scope before approval."
    )
    return {
        "proposed_action": proposed,
        "events": [
            _timed_event(
                "risky_action", "completed", "action prepared for approval", started
            )
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Record mock approval, or pause with LangGraph interrupt when enabled."""
    started = time.perf_counter()
    mode = "mock"
    decision: Any = {
        "approved": True,
        "reviewer": "mock-reviewer",
        "comment": "Approved for lab execution.",
    }
    if os.getenv("LANGGRAPH_INTERRUPT", "").casefold() in {"1", "true", "yes", "on"}:
        from langgraph.types import interrupt

        mode = "interrupt"
        decision = interrupt(
            {
                "type": "approval_request",
                "question": "Approve this side-effecting support action?",
                "proposed_action": state.get("proposed_action", ""),
            }
        )
    if isinstance(decision, bool):
        decision = {"approved": decision}
    if not isinstance(decision, dict):
        decision = {"approved": False, "comment": "Invalid approval response."}
    approval = {
        "approved": bool(decision.get("approved", False)),
        "reviewer": str(decision.get("reviewer", "mock-reviewer")),
        "comment": str(decision.get("comment", "")),
    }
    return {
        "approval": approval,
        "events": [
            _timed_event(
                "approval",
                "completed",
                "approval recorded",
                started,
                approved=approval["approved"],
                mode=mode,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Increment the bounded retry counter and record the failure."""
    started = time.perf_counter()
    attempt = int(state.get("attempt", 0)) + 1
    max_attempts = int(state.get("max_attempts", 3))
    error = f"Transient failure; retry attempt {attempt}/{max_attempts}"
    return {
        "attempt": attempt,
        "errors": [error],
        "messages": [f"retry:{attempt}"],
        "events": [
            _timed_event(
                "retry",
                "completed",
                "retry counter incremented",
                started,
                attempt=attempt,
                max_attempts=max_attempts,
            )
        ],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle an unresolvable failure after max retries."""
    started = time.perf_counter()
    attempt = int(state.get("attempt", 0))
    final_answer = (
        "I couldn't complete this request after the configured retry limit "
        f"({attempt} attempt(s)). The issue has been recorded for follow-up."
    )
    return {
        "final_answer": final_answer,
        "events": [
            _timed_event(
                "dead_letter", "completed", "request moved to dead letter handling", started
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit the final audit event; every route reaches this node."""
    started = time.perf_counter()
    return {
        "events": [_timed_event("finalize", "completed", "workflow finished", started)]
    }
