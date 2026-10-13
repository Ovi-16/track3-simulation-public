#!/usr/bin/env python3
"""Phase 1c profiling driver — SPEC-01 §3/§6, measured optimization budget.

Runs INSIDE the baseline image (where ABIDES lives) and profiles the exact hot
region the ranked throughput number measures: ``abides.run(config)`` in
baselines/abides_fork/simulate.py, between its t0 and wall_clock_sec. It also
separately times the serialization tax (extract_trace + extract_message_trace +
to_parquet) which sits inside the container window but outside abides.run.

Output per scenario, written to --out-dir:
  <sid>.pstats       cProfile dump (open with pstats / snakeviz / tuna)
  <sid>.summary.txt  human-readable top-N cumulative + per-component wall-clock

This turns the SPEC-01 §3 bottleneck HYPOTHESES (kernel event-queue dispatch,
order-book ops, agent callbacks, latency sampling, serialization) into a MEASURED
ranked budget — so Phase 2 optimizes what actually dominates, not a guess. The
component wall-clock split also seeds the NVTX/SimProfile mapping for the Best
Systems Diagnosis award (matching / event_queue / latency / agent_logic / io).

Run via docker (see profiling/run_profile.sh), e.g.:
    python -m profiling.profile_hotloop --config /input/scenario.json --out-dir /output
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pathlib
import pstats
import time

# Imported from the adapter the image installs under /opt (PYTHONPATH=/opt).
from abides_core import abides  # type: ignore
from abides_fork.config import build_config  # type: ignore
from abides_fork.scenario_io import read_scenario  # type: ignore
from abides_fork.trace import extract_message_trace, extract_trace  # type: ignore
from abides_fork.simulate import reset_abides_counters  # type: ignore


def profile_scenario(config_path: str, out_dir: pathlib.Path, top_n: int = 40) -> dict:
    scenario = json.loads(read_scenario(config_path))
    sid = str(scenario.get("scenario_id", pathlib.Path(config_path).stem))

    reset_abides_counters()
    config = build_config(scenario)

    # --- Profile the hot loop: abides.run (the timed, ranked region) ---
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    end_state = abides.run(config)
    profiler.disable()
    run_sec = time.perf_counter() - t0

    # --- Time the serialization tax (inside container window, outside abides.run) ---
    t1 = time.perf_counter()
    trace = extract_trace(end_state)
    extract_sec = time.perf_counter() - t1

    t2 = time.perf_counter()
    message_trace = extract_message_trace(end_state)
    msg_extract_sec = time.perf_counter() - t2

    out_dir.mkdir(parents=True, exist_ok=True)
    t3 = time.perf_counter()
    trace.to_parquet(out_dir / f"{sid}.trace.parquet", compression="snappy", index=False)
    message_trace.to_parquet(out_dir / f"{sid}.mtrace.parquet", compression="snappy", index=False)
    write_sec = time.perf_counter() - t3

    n_events = int(len(trace))

    # --- Dump pstats + readable summary ---
    pstats_path = out_dir / f"{sid}.pstats"
    profiler.dump_stats(str(pstats_path))

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf)
    stats.sort_stats("cumulative").print_stats(top_n)
    cumulative_block = buf.getvalue()

    buf2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=buf2)
    stats2.sort_stats("tottime").print_stats(top_n)
    tottime_block = buf2.getvalue()

    total_window = run_sec + extract_sec + msg_extract_sec + write_sec
    summary = (
        f"=== Phase 1c profile: {sid} ===\n"
        f"n_events                 : {n_events}\n"
        f"abides.run (hot loop)    : {run_sec:.4f}s  ({100*run_sec/total_window:.1f}% of window)\n"
        f"extract_trace            : {extract_sec:.4f}s  ({100*extract_sec/total_window:.1f}%)\n"
        f"extract_message_trace    : {msg_extract_sec:.4f}s ({100*msg_extract_sec/total_window:.1f}%)\n"
        f"parquet write (both)     : {write_sec:.4f}s  ({100*write_sec/total_window:.1f}%)\n"
        f"total container window   : {total_window:.4f}s\n"
        f"events/sec (hot loop)    : {n_events/run_sec:.0f}\n"
        f"events/sec (full window) : {n_events/total_window:.0f}\n"
        f"\n--- TOP {top_n} BY CUMULATIVE TIME (where the hot loop spends time) ---\n"
        f"{cumulative_block}\n"
        f"--- TOP {top_n} BY SELF (tottime) TIME (the functions to compile first) ---\n"
        f"{tottime_block}"
    )
    (out_dir / f"{sid}.summary.txt").write_text(summary)

    return {
        "scenario_id": sid, "n_events": n_events,
        "run_sec": run_sec, "extract_sec": extract_sec,
        "msg_extract_sec": msg_extract_sec, "write_sec": write_sec,
        "total_window": total_window,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="profiling.profile_hotloop")
    ap.add_argument("--config", required=True, help="path to scenario.json")
    ap.add_argument("--out-dir", required=True, help="where to write .pstats + .summary.txt")
    ap.add_argument("--top-n", type=int, default=40)
    args = ap.parse_args(argv)

    result = profile_scenario(args.config, pathlib.Path(args.out_dir), args.top_n)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
