"""Filesystem-backed prompt templates.

Why .md files instead of strings: prompts are content, not code, and reviewers
should be able to diff them like documents. The loader caches reads so we don't
pay disk cost on every LLM call.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPT_DIR = Path(__file__).parent


@lru_cache(maxsize=8)
def load_prompt(name: str) -> str:
    """Read prompts/<name>.md from this package. Cached.

    Raises FileNotFoundError if the prompt is missing — fail loud rather than
    silently substitute an empty string.
    """
    path = _PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")
