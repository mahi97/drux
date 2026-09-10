# -*- coding: utf-8 -*-
"""
Benchmark the array-native numerical core against the pinned np.vectorize baseline.

The pinned package is the pre-refactor implementation installed as
``drux_pinned`` (see install.sh). Large 1-D time arrays are used so the
comparison measures evaluation cost rather than Python call overhead.
"""

import argparse
import timeit

import numpy as np

from drux import (
    FirstOrderModel,
    HiguchiModel,
    HopfenbergModel,
    WeibullModel,
    ZeroOrderModel,
)

try:
    from drux_pinned.first_order import FirstOrderModel as PinnedFirstOrder
    from drux_pinned.higuchi import HiguchiModel as PinnedHiguchi
    from drux_pinned.hopfenberg import HopfenbergModel as PinnedHopfenberg
    from drux_pinned.weibull import WeibullModel as PinnedWeibull
    from drux_pinned.zero_order import ZeroOrderModel as PinnedZeroOrder
except ImportError:  # pragma: no cover - environment without the pin
    PinnedFirstOrder = None
    PinnedHiguchi = None
    PinnedHopfenberg = None
    PinnedWeibull = None
    PinnedZeroOrder = None


def build_model_pairs():
    """
    Return comparable live/pinned model pairs.

    :return: list of (name, live_model, pinned_model)
    """
    live = [
        ("zero_order", ZeroOrderModel(k0=0.1, M0=0.01)),
        ("first_order", FirstOrderModel(k=0.003, M0=0.1)),
        ("higuchi", HiguchiModel(D=1e-6, c0=1.0, cs=0.5)),
        ("weibull", WeibullModel(M=1.0, a=0.095, b=0.7)),
        ("hopfenberg", HopfenbergModel(M=1.0, k0=0.00067, c0=0.0374, a0=3.51, n=2)),
    ]
    if PinnedZeroOrder is None:
        return [(name, model, None) for name, model in live]
    pinned = {
        "zero_order": PinnedZeroOrder(k0=0.1, M0=0.01),
        "first_order": PinnedFirstOrder(k=0.003, M0=0.1),
        "higuchi": PinnedHiguchi(D=1e-6, c0=1.0, cs=0.5),
        "weibull": PinnedWeibull(M=1.0, a=0.095, b=0.7),
        "hopfenberg": PinnedHopfenberg(M=1.0, k0=0.00067, c0=0.0374, a0=3.51, n=2),
    }
    return [(name, model, pinned[name]) for name, model in live]


def _time_call(func, repeat, number):
    """
    Return the best-of-repeat wall time for ``number`` calls.

    :param func: zero-argument callable
    :param repeat: number of timeit repeats
    :param number: calls per repeat
    :return: minimum elapsed seconds for one batch of ``number`` calls
    """
    timer = timeit.Timer(func)
    return min(timer.repeat(repeat=repeat, number=number))


def benchmark_evaluate(n_points, repeat, number):
    """
    Time array evaluation for each model.

    :param n_points: length of the synthetic time array
    :param repeat: timeit repeats
    :param number: calls per repeat
    :return: list of result dictionaries
    """
    times = np.linspace(0.0, 1000.0, int(n_points))
    rows = []
    for name, live, pinned in build_model_pairs():
        live.evaluate(times)
        live_seconds = _time_call(lambda: live.evaluate(times), repeat, number)
        pinned_seconds = None
        if pinned is not None:
            vectorized = np.vectorize(pinned._model_function, otypes=[float])
            vectorized(times)
            pinned_seconds = _time_call(lambda: vectorized(times), repeat, number)
        rows.append({
            "name": name,
            "mode": "evaluate",
            "n_points": int(n_points),
            "live_s": live_seconds / number,
            "pinned_s": None if pinned_seconds is None else pinned_seconds / number,
        })
    return rows


def benchmark_simulate(duration, time_step, repeat, number):
    """
    Time simulate() for each model on a large divisible grid.

    :param duration: simulation duration (s)
    :param time_step: time step (s)
    :param repeat: timeit repeats
    :param number: calls per repeat
    :return: list of result dictionaries
    """
    rows = []
    n_points = int(round(duration / time_step)) + 1
    for name, live, pinned in build_model_pairs():
        live.simulate(duration=duration, time_step=time_step)
        live_seconds = _time_call(
            lambda: live.simulate(duration=duration, time_step=time_step),
            repeat,
            number)
        pinned_seconds = None
        if pinned is not None:
            pinned.simulate(duration=duration, time_step=time_step)
            pinned_seconds = _time_call(
                lambda: pinned.simulate(duration=duration, time_step=time_step),
                repeat,
                number)
        rows.append({
            "name": name,
            "mode": "simulate",
            "n_points": n_points,
            "live_s": live_seconds / number,
            "pinned_s": None if pinned_seconds is None else pinned_seconds / number,
        })
    return rows


def format_table(rows):
    """
    Format benchmark rows as a GitHub-flavored markdown table.

    :param rows: result dictionaries from the benchmark helpers
    :return: markdown table
    """
    lines = [
        "| Model | Mode | Points | Live (s) | Pinned vectorize (s) | Speedup |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["pinned_s"] is None:
            speedup = "n/a"
            pinned = "n/a"
        else:
            speedup = "{0:.1f}x".format(row["pinned_s"] / row["live_s"])
            pinned = "{0:.6f}".format(row["pinned_s"])
        lines.append(
            "| {0} | {1} | {2} | {3:.6f} | {4} | {5} |".format(
                row["name"],
                row["mode"],
                row["n_points"],
                row["live_s"],
                pinned,
                speedup))
    return "\n".join(lines)


def parse_args(argv=None):
    """
    Parse command-line arguments.

    :param argv: optional argument list
    :return: parsed namespace
    """
    parser = argparse.ArgumentParser(
        description="Benchmark the Drux numerical core against np.vectorize.")
    parser.add_argument("--points", type=int, default=1000000,
                        help="Length of the evaluate() time array.")
    parser.add_argument("--duration", type=float, default=100000.0,
                        help="simulate() duration in seconds.")
    parser.add_argument("--time-step", type=float, default=0.1,
                        help="simulate() time step in seconds.")
    parser.add_argument("--repeat", type=int, default=5,
                        help="timeit repeat count.")
    parser.add_argument("--number", type=int, default=3,
                        help="Calls per timeit repeat.")
    return parser.parse_args(argv)


def main(argv=None):
    """
    Run evaluate and simulate benchmarks and print a markdown table.

    :param argv: optional argument list
    :return: process exit code
    """
    args = parse_args(argv)
    evaluate_rows = benchmark_evaluate(args.points, args.repeat, args.number)
    simulate_rows = benchmark_simulate(
        args.duration, args.time_step, args.repeat, args.number)
    print(format_table(evaluate_rows + simulate_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
