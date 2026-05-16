"""AgentLoom CLI."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="agent-loom",
    help="A memory-augmented multi-agent harness.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the installed version."""
    from agent_loom import __version__

    typer.echo(__version__)


@app.command()
def run(goal: str) -> None:
    """Submit a goal to the harness (Phase 0+)."""
    typer.echo(f"[stub] Would run goal: {goal!r}")
    typer.echo("Implemented in Phase 0. See docs/PHASES.md.")


@app.command(name="memory-inspect")
def memory_inspect() -> None:
    """Inspect the episodic store and KG (Phase 1+)."""
    typer.echo("[stub] Memory inspector — implemented in Phase 1.")


if __name__ == "__main__":
    app()
