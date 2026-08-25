"""Checkpointer adapter."""

from __future__ import annotations


def build_checkpointer(
    kind: str = "memory", database_url: str | None = None
) -> object | None:
    """Return a LangGraph checkpointer.

    MemorySaver is the zero-configuration option; SQLite is available through
    the optional checkpoint-sqlite dependency.

    For SQLite:
    - pip install langgraph-checkpoint-sqlite
    - Use SqliteSaver with sqlite3.connect() and WAL mode
    - See: https://langchain-ai.github.io/langgraph/how-tos/persistence/
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "SQLite persistence requires the optional dependency: "
                "pip install -e '.[sqlite]'"
            ) from exc

        path = database_url or "outputs/langgraph_checkpoints.sqlite"
        if path.startswith("sqlite:///"):
            path = path.removeprefix("sqlite:///")
        connection = sqlite3.connect(path, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.commit()
        return SqliteSaver(conn=connection)
    if kind == "postgres":
        raise RuntimeError(
            "Postgres persistence is optional and requires the checkpoint-postgres dependency."
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")
