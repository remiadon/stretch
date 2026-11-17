"""Regression tests against upstream stretch-tscd 0.1.1: outputs, peak memory, speed.

Outputs: goldens live in tests/golden/. A missing golden is written from the current
code and the test is skipped; delete a golden only when an output change is intended.
Memory: each case fails if its peak exceeds upstream's, rounded up to the KiB
(pytest-memray, measured on macOS arm64 / Python 3.14). Lower the limit when a change shrinks it.
Speed: test_speed is timed by pytest-benchmark; see [tool.pytest.ini_options].
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stretch import STRETCH

GOLDEN = Path(__file__).parent / "golden"


def readme_data():
    # Verbatim from the README's five-minute example.
    rng = np.random.default_rng(7)
    n = 120
    temperature = rng.normal(size=n)
    demand = np.zeros(n)
    for t in range(2, n):
        delay = 1 if t < n // 2 else 2
        demand[t] = 0.8 * temperature[t - delay] + 0.2 * demand[t - 1]
    demand += 0.1 * rng.normal(size=n)
    return pd.DataFrame({"outdoor temperature": temperature, "power demand": demand})


def chain_data(n_vars=4, T=200, seed=0):
    # Chain 0->1->...; the delay switches between 1 and 3 every 50 steps.
    rng = np.random.default_rng(seed)
    X = np.zeros((T, n_vars))
    X[:, 0] = rng.normal(size=T)
    for j in range(1, n_vars):
        for t in range(3, T):
            d = 1 if (t // 50) % 2 == 0 else 3
            X[t, j] = np.tanh(X[t - d, j - 1]) + 0.3 * rng.normal()
    return pd.DataFrame(X, columns=[f"x{j}" for j in range(n_vars)])


CASES = {  # name: (data, STRETCH params, upstream peak memory)
    "readme_greedy": (readme_data, dict(search="greedy", max_lag=3), "1400 KB"),
    "readme_exhaustive": (readme_data, dict(search="exhaustive", max_lag=3), "587 KB"),
    "readme_topic": (readme_data, dict(search="topic", max_lag=3), "643 KB"),
    "chain4_greedy": (chain_data, dict(search="greedy", max_lag=3), "4198 KB"),
}


def to_json(result):
    return {
        "adjacency": result.adjacency.to_numpy().tolist(),
        "delay_functions": {
            f"{s} -> {t}": d.tolist() for (s, t), d in result.delay_functions.items()
        },
    }


def check_golden(case, result):
    got = to_json(result)
    path = GOLDEN / f"{case}.json"
    if not path.exists():
        path.write_text(json.dumps(got, indent=1) + "\n")
        pytest.skip(f"wrote {path.name}")
    assert got == json.loads(path.read_text())


# Separate from test_speed: the limit_memory marker turns memray on, which would skew timings.
@pytest.mark.parametrize(
    "case", [pytest.param(c, marks=pytest.mark.limit_memory(mem)) for c, (*_, mem) in CASES.items()]
)
def test_matches_upstream(case):
    make_data, params, _ = CASES[case]
    check_golden(case, STRETCH(**params).fit_result(make_data()))


@pytest.mark.parametrize("case", CASES)
def test_speed(case, benchmark):
    make_data, params, _ = CASES[case]
    data = make_data()
    # Fits take seconds: 3 timed rounds instead of auto-calibration; 1 warmup keeps JIT compiles out.
    result = benchmark.pedantic(
        lambda: STRETCH(**params).fit_result(data), rounds=3, iterations=1, warmup_rounds=1
    )
    check_golden(case, result)
