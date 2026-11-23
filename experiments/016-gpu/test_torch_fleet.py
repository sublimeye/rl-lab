"""The torch step matches the numpy Fleet's (float64, to round-off; outcomes exactly).

CI installs no torch: this skips there.
"""

import pytest

torch = pytest.importorskip("torch")


def test_torch_step_matches_numpy():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    import parity

    torch.set_num_threads(2)
    ok, worst, events, checks = parity.run(drones=2, n_copies=4, horizon=40, seed=1)
    assert ok, events
    assert worst < 1e-9
    assert checks >= 5
