"""CLI for the lab."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer
import yaml  # type: ignore[import-untyped]

from .graph import build_graph, export_mermaid
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    resume_success = False
    history_evidence: list[dict[str, object]] = []
    for scenario in scenarios:
        state = initial_state(scenario)
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
        started = time.perf_counter()
        final_state = graph.invoke(state, config=run_config)  # type: ignore[call-overload]
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        metrics.append(
            metric_from_state(
                final_state,
                scenario.expected_route.value,
                scenario.requires_approval,
                latency_ms=elapsed_ms,
            )
        )
        if checkpointer is not None:
            try:
                history = list(graph.get_state_history(run_config))  # type: ignore[arg-type]
                resume_success = resume_success or len(history) > 1
                checkpoint_ids = []
                for snapshot in history:
                    snapshot_config = getattr(snapshot, "config", {})
                    configurable = (
                        snapshot_config.get("configurable", {})
                        if isinstance(snapshot_config, dict)
                        else {}
                    )
                    if isinstance(configurable, dict) and configurable.get("checkpoint_id"):
                        checkpoint_ids.append(str(configurable["checkpoint_id"]))
                history_evidence.append(
                    {
                        "thread_id": state["thread_id"],
                        "checkpoint_count": len(history),
                        "checkpoint_ids": checkpoint_ids,
                    }
                )
            except (AttributeError, TypeError, ValueError):
                # Some third-party checkpointers do not expose history.
                pass
    report = summarize_metrics(metrics, resume_success=resume_success)
    write_metrics(report, output)
    history_path = output.parent / "history_evidence.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(history_evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    typer.echo(f"Wrote metrics to {output}")


@app.command("export-graph")
def export_graph_command(output: Annotated[Path, typer.Option("--output")]) -> None:
    """Export Mermaid from the compiled graph topology."""
    graph = build_graph(checkpointer=None)
    path = export_mermaid(graph, output)
    typer.echo(f"Wrote Mermaid graph to {path}")


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


if __name__ == "__main__":
    app()
