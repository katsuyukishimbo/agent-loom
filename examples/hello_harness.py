"""Top-level convenience shim. The canonical entrypoint is the package version.

Run either:
    python examples/hello_harness.py
    python -m agent_loom.examples.hello_harness   # preferred
"""

from __future__ import annotations

import asyncio

from agent_loom.examples.hello_harness import main

if __name__ == "__main__":
    asyncio.run(main())
