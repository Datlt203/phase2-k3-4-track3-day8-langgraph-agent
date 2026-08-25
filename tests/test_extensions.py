"""Offline proofs for the optional extension tracks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from langgraph.types import Command

from langgraph_agent_lab import nodes
from langgraph_agent_lab.graph import build_graph, export_mermaid
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state


class _StructuredFake:
    def __init__(self, payload: dict[str, str]) -> None:
        self.payload = payload

    def invoke(self, _prompt: object) -> dict[str, str]:
        return self.payload


class _TextResponse:
    content = "Approved response"


class _FakeLLM:
    def __init__(self, classification: dict[str, str] | None = None) -> None:
        self.classification = classification or {
            "route": "simple",
            "risk_level": "low",
            "rationale": "test",
        }

    def with_structured_output(self, schema: object) -> _StructuredFake:
        if schema is nodes.EvaluationVerdict:
            return _StructuredFake({"verdict": "success", "reason": "result is usable"})
        return _StructuredFake(self.classification)

    def invoke(self, _prompt: object) -> _TextResponse:
        return _TextResponse()


def test_llm_judge_is_structured_bounded_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nodes, "get_llm", lambda: _FakeLLM())
    monkeypatch.setenv("LLM_JUDGE_ENABLED", "true")
    monkeypatch.setenv("LLM_JUDGE_MAX_CALLS", "1")
    monkeypatch.setenv("LLM_JUDGE_TIMEOUT_SECONDS", "1")

    judged = nodes.evaluate_node(
        {"tool_results": ["SUCCESS: lookup completed"], "judge_calls": 0}
    )
    assert judged["evaluation_result"] == "success"
    assert judged["judge_calls"] == 1
    assert judged["events"][0]["metadata"]["source"] == "llm_judge"

    guarded = nodes.evaluate_node(
        {"tool_results": ["ERROR: transient"], "judge_calls": 1}
    )
    assert guarded["evaluation_result"] == "needs_retry"
    assert guarded["judge_calls"] == 1
    assert "cost guard" in guarded["evaluation_reason"]


def test_real_hitl_interrupt_and_resume_same_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        nodes,
        "get_llm",
        lambda: _FakeLLM(
            {"route": "risky", "risk_level": "high", "rationale": "side effect"}
        ),
    )
    monkeypatch.setenv("LANGGRAPH_INTERRUPT", "true")
    scenario = Scenario(
        id="extension-hitl",
        query="Refund this customer",
        expected_route=Route.RISKY,
        requires_approval=True,
    )
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": state["thread_id"]}}
    graph = build_graph(checkpointer=build_checkpointer("memory"))

    paused = graph.invoke(state, config=config)
    assert paused.get("__interrupt__")
    assert paused.get("approval") is None

    resumed = graph.invoke(
        Command(resume={"approved": True, "reviewer": "test-reviewer"}),
        config=config,
    )
    assert resumed["approval"]["approved"] is True
    assert resumed["final_answer"]
    assert resumed["events"][-1]["node"] == "finalize"


def test_time_travel_replay_and_mermaid_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(nodes, "get_llm", lambda: _FakeLLM())
    scenario = Scenario(id="extension-time", query="hello", expected_route=Route.SIMPLE)
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": state["thread_id"]}}
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    graph.invoke(state, config=config)

    history = list(graph.get_state_history(config))
    assert len(history) > 2
    replayed = graph.invoke(None, config=history[2].config)
    assert replayed["final_answer"]

    mermaid_path = export_mermaid(graph, tmp_path / "graph.mmd")
    diagram = mermaid_path.read_text(encoding="utf-8")
    assert "classify" in diagram
    assert "finalize" in diagram


def test_sqlite_checkpoint_survives_process_restart(tmp_path: Path) -> None:
    pytest.importorskip("langgraph.checkpoint.sqlite")
    database = repr(str(tmp_path / "recovery.sqlite"))
    source_root = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root) + os.pathsep + env.get("PYTHONPATH", "")
    for key in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        env[key] = ""

    seed_script = f"""
from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state
scenario = Scenario(id='sqlite-recovery', query='hello', expected_route=Route.SIMPLE)
state = initial_state(scenario)
config = {{'configurable': {{'thread_id': state['thread_id']}}}}
graph = build_graph(build_checkpointer('sqlite', {database}))
graph.invoke(state, config=config)
"""
    check_script = f"""
from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state
scenario = Scenario(id='sqlite-recovery', query='hello', expected_route=Route.SIMPLE)
state = initial_state(scenario)
config = {{'configurable': {{'thread_id': state['thread_id']}}}}
graph = build_graph(build_checkpointer('sqlite', {database}))
snapshot = graph.get_state(config)
assert snapshot.values.get('final_answer')
print('recovered')
"""
    seeded = subprocess.run(
        [sys.executable, "-c", seed_script], env=env, capture_output=True, text=True
    )
    assert seeded.returncode == 0, seeded.stderr
    recovered = subprocess.run(
        [sys.executable, "-c", check_script], env=env, capture_output=True, text=True
    )
    assert recovered.returncode == 0, recovered.stderr
    assert "recovered" in recovered.stdout
