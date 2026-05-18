"""AgentLoom CLI."""

from __future__ import annotations

import asyncio

import typer

app = typer.Typer(
    name="agent-loom",
    help="A memory-augmented multi-agent harness.",
    no_args_is_help=True,
)

# Sub-app for `agent-loom memory <subcommand>`. Why nest: future commands
# (`memory promote`, `memory prune`) all live under the memory namespace.
memory_app = typer.Typer(help="Inspect and manage episodic memory + KG.")
app.add_typer(memory_app, name="memory")


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


@memory_app.command("inspect")
def memory_inspect(
    use_pg: bool = typer.Option(
        False,
        "--pg/--no-pg",
        help="Read from pgvector instead of an in-memory store.",
    ),
    top_n: int = typer.Option(
        5, "--top", help="How many highest-referenced episodes to list."
    ),
) -> None:
    """Print a KG snapshot — totals, failure mix, top referenced episodes."""

    async def _run() -> None:
        from agent_loom.memory.graph import InMemoryKnowledgeGraph
        from agent_loom.memory.inspect import build_memory_snapshot, format_snapshot

        store_obj = await _resolve_store(use_pg=use_pg)
        # The CLI inspects whatever store the user points at, but the KG
        # currently lives only in process — pgvector persistence of edges
        # arrives in a follow-up migration. For now we show an empty graph
        # snapshot when reading from pg so the output still parses.
        graph = InMemoryKnowledgeGraph()
        snap = await build_memory_snapshot(store=store_obj, graph=graph, top_n=top_n)
        typer.echo(format_snapshot(snap))

    asyncio.run(_run())


async def _resolve_store(*, use_pg: bool):
    """Pick the EpisodicStore for the CLI command.

    Keep the import lazy so the CLI starts fast even when pgvector deps are
    unavailable in the user's environment.
    """
    if use_pg:
        from agent_loom.memory.store_pg import PgvectorEpisodicStore

        return PgvectorEpisodicStore()
    from agent_loom.memory.store import InMemoryEpisodicStore

    return InMemoryEpisodicStore()


if __name__ == "__main__":
    app()
