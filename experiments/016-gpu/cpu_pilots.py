"""pilot.py's test (net, shield or planner pilots; numpy, a pool of CPU processes) for networks
trained here (016-gpu/runs/<name>/), writing its results there too.

uv run experiments/016-gpu/cpu_pilots.py test --name g1 --mode shield --k 5
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "016-flawless"))
import pilot  # noqa: E402

if __name__ == "__main__":
    pilot.RUNS = HERE / "runs"
    pilot.main()
