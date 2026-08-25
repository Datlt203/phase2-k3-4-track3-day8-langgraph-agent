# Day 08 Lab Report

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
| `pending_question`, `proposed_action`, `approval` | overwrite | current user/action decision |
| `final_answer` | overwrite | latest response presented to the user |

## 4. Scenario results

Summary: **7** scenarios, **100.0%**
success rate, **3** retry node visits, **2**
approval/HITL events, average **6.43** nodes, and
`resume_success=True`.

| Scenario | Expected route | Actual route | Success | Retries | Interrupts | Errors |
|---|---|---|---:|---:|---:|---|
| S01_simple | simple | simple | ✅ | 0 | 0 | — |
| S02_tool | tool | tool | ✅ | 0 | 0 | — |
| S03_missing | missing_info | missing_info | ✅ | 0 | 0 | — |
| S04_risky | risky | risky | ✅ | 0 | 1 | — |
| S05_error | error | error | ✅ | 2 | 0 | Transient failure; retry attempt 1/3; Transient failure; retry attempt 2/3 |
| S06_delete | risky | risky | ✅ | 0 | 1 | — |
| S07_dead_letter | error | error | ✅ | 1 | 0 | Transient failure; retry attempt 1/1 |

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
SQLite with WAL mode via `build_checkpointer("sqlite", ...)`.

## 7. Extension work

Implemented SQLite checkpointer support, LangGraph interrupt-based approval,
per-node latency metadata, state-history evidence, and automatic Markdown
report generation.

## 8. Improvement plan

The next production step would be replacing mock tools and approval with
authenticated adapters, adding provider-specific timeout/circuit-breaker
policies, redacting sensitive audit fields, and testing classification against a
larger labeled hidden-scenario set.
