#!/usr/bin/env python3
"""Equivalence harness — SPEC-01 Phase 1b, the correctness moat.

PURPOSE
-------
Prove, on every commit, that a candidate simulator image produces output
*equivalent to the shipped reference traces* — so we can optimize aggressively
(compiled core, parallelism, GPU) without ever silently breaking Tier A, which
would score a unit zero. This is the single rule the whole SPEC-01 plan is built
around (see specs/README.md, Guardrail #2).

TWO LAYERS OF PROOF (strongest first)
-------------------------------------
1. BIT-EXACT  — recompute ``trace_sha256`` exactly as the adapter does
   (baselines/abides_fork/simulate.py: hashlib.sha256, 1 MiB chunks) and compare
   to the reference unit's events.json ``trace_sha256`` (or, failing that, the
   sha256 of the shipped reference trace.parquet). A single changed byte fails.
   This is the gate the compiled CPU core (Phase 2) must pass: it is designed to
   emit a byte-identical stream, so byte-identity is the right, strictest check.

2. SEMANTIC   — where bit-exactness legitimately cannot hold (a statistical
   Tier-B unit, or a GPU path that preserves event semantics but not bytes),
   fall back to run_regression.py's official Tier-A / Tier-B / stylized-fact
   gates, which import qfbench2_common — never reimplemented here.

A unit PASSES equivalence if it is bit-exact, OR (when --allow-semantic) it is
not bit-exact but passes the semantic gate. The report states which layer each
unit cleared, so a drop from BIT-EXACT to SEMANTIC on a unit that used to be
byte-identical is visible, not silent.

USAGE
-----
    # Fast bit-exact-only gate (default) — the Phase 2 compiled-core gate:
    python regression_suite/equivalence_harness.py \
        --candidate-image track3-abides-baseline:latest

    # Allow semantic fallback for units that are not byte-identical
    # (Tier-B / GPU paths in Phase 3):
    python regression_suite/equivalence_harness.py \
        --candidate-image my-sim:latest --allow-semantic

    # Memory-tight hosts: one container at a time (see specs/README.md).
    python regression_suite/equivalence_harness.py \
        --candidate-image my-sim:latest --workers 1

Exit codes: 0 = every unit equivalent; 1 = at least one unit not equivalent;
2 = harness/setup error (missing references, bad image, etc.).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
SCENARIOS = HERE / "scenarios"
UNITS = REPO / "units"
REFERENCE_CACHE = HERE / "reference_traces"


# --- bit-exact hashing: identical to baselines/abides_fork/simulate.py:_sha256 ---
def sha256_file(path: pathlib.Path) -> str:
    """Byte-for-byte identical to the adapter's _sha256 (1 MiB chunks)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scenario_configs() -> dict[str, pathlib.Path]:
    """scenario_id -> scenario.json path, over the 65 single-scenario configs."""
    out: dict[str, pathlib.Path] = {}
    for cfg in sorted(SCENARIOS.glob("*.json")):
        if cfg.stem == "index":
            continue
        try:
            sid = json.loads(cfg.read_text())["scenario_id"]
        except (json.JSONDecodeError, KeyError):
            continue
        out[sid] = cfg
    return out


def reference_trace_hash(scenario_id: str) -> str | None:
    """The reference trace_sha256 for a scenario.

    Prefer the value the reference unit DECLARED in its events.json (the number
    the adapter wrote when it generated the frozen trace); fall back to hashing
    the shipped reference trace.parquet itself. Returns None if neither is found.
    """
    ref_dir = REFERENCE_CACHE / scenario_id
    events = ref_dir / "events.json"
    if events.exists():
        try:
            h = json.loads(events.read_text()).get("trace_sha256")
            if isinstance(h, str) and len(h) == 64:
                return h.lower()
        except json.JSONDecodeError:
            pass
    ref_trace = ref_dir / "trace.parquet"
    if ref_trace.exists():
        return sha256_file(ref_trace)
    return None


def run_candidate(image: str, scenario_cfg: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    """Run the candidate image's `simulate` on one scenario, exactly as the
    official harness does (--network none, scenario mounted read-only). Returns
    the path to the produced trace.parquet. Raises on failure."""
    in_dir = pathlib.Path(tempfile.mkdtemp(prefix="eqv_in_", dir=out_dir))
    (in_dir / "scenario.json").write_bytes(scenario_cfg.read_bytes())
    trace_out = out_dir / "trace.parquet"
    cmd = [
        "docker", "run", "--rm", "--network", "none",
        "-v", f"{in_dir}:/input:ro",
        "-v", f"{out_dir}:/output",
        image,
        "simulate", "--config", "/input/scenario.json", "--out", "/output/trace.parquet",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=3600)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", errors="replace")[-800:]
        raise RuntimeError(f"candidate exited {proc.returncode}: {tail}")
    if not trace_out.exists():
        raise RuntimeError("candidate produced no trace.parquet")
    return trace_out


def check_unit(image: str, scenario_id: str, cfg: pathlib.Path) -> dict:
    """Return a result dict: {scenario_id, layer, equivalent, detail}."""
    ref_hash = reference_trace_hash(scenario_id)
    if ref_hash is None:
        return {
            "scenario_id": scenario_id, "layer": "none", "equivalent": False,
            "detail": "no reference hash (run build_reference_cache.py)",
        }
    with tempfile.TemporaryDirectory(prefix="eqv_out_") as tmp:
        tmp_path = pathlib.Path(tmp)
        try:
            trace = run_candidate(image, cfg, tmp_path)
        except Exception as exc:  # noqa: BLE001
            return {"scenario_id": scenario_id, "layer": "run", "equivalent": False,
                    "detail": f"run failed: {exc}"}
        cand_hash = sha256_file(trace)

    if cand_hash == ref_hash:
        return {"scenario_id": scenario_id, "layer": "bit-exact", "equivalent": True,
                "detail": cand_hash}
    return {
        "scenario_id": scenario_id, "layer": "bit-exact", "equivalent": False,
        "detail": f"hash mismatch: candidate {cand_hash[:12]} != reference {ref_hash[:12]}",
    }


def run_semantic_gate(image: str, workers: int) -> int:
    """Delegate to the official regression harness for the semantic fallback.
    Returns its exit code (0 = all semantic gates pass)."""
    cmd = [
        sys.executable, str(HERE / "run_regression.py"),
        "--candidate-image", image,
        "--scenarios-dir", str(SCENARIOS),
        "--reference-dir", str(REFERENCE_CACHE),
        "--output-dir", str(HERE / "run_outputs"),
        "--workers", str(workers),
    ]
    return subprocess.run(cmd).returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SPEC-01 Phase 1b equivalence harness")
    ap.add_argument("--candidate-image", required=True)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel candidate containers (default 1; raise only with memory headroom)")
    ap.add_argument("--allow-semantic", action="store_true",
                    help="units that are not bit-exact may still pass via the official semantic gate")
    ap.add_argument("--only", default=None,
                    help="comma-separated scenario_id substrings to restrict the run (smoke test)")
    args = ap.parse_args(argv)

    if not REFERENCE_CACHE.exists():
        print(f"ERROR: reference cache missing at {REFERENCE_CACHE}. "
              f"Run: python regression_suite/build_reference_cache.py --copy", file=sys.stderr)
        return 2

    configs = scenario_configs()
    if args.only:
        wanted = args.only.split(",")
        configs = {sid: c for sid, c in configs.items() if any(w in sid for w in wanted)}
    if not configs:
        print("ERROR: no scenarios selected", file=sys.stderr)
        return 2

    print(f"Equivalence harness — candidate: {args.candidate_image}")
    print(f"Checking {len(configs)} unit(s), bit-exact first"
          f"{' then semantic fallback' if args.allow_semantic else ' (bit-exact only)'}\n")

    results: list[dict] = []
    if args.workers <= 1:
        for sid, cfg in configs.items():
            results.append(check_unit(args.candidate_image, sid, cfg))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(check_unit, args.candidate_image, sid, cfg): sid
                    for sid, cfg in configs.items()}
            for fut in concurrent.futures.as_completed(futs):
                results.append(fut.result())

    results.sort(key=lambda r: r["scenario_id"])
    bit_exact = [r for r in results if r["equivalent"] and r["layer"] == "bit-exact"]
    not_bit_exact = [r for r in results if not r["equivalent"]]

    for r in results:
        mark = "OK  " if r["equivalent"] else "FAIL"
        print(f"  [{mark}] {r['scenario_id']}  ({r['layer']})  {r['detail'][:70]}")

    print(f"\nBit-exact: {len(bit_exact)}/{len(results)}")

    # Semantic fallback for the units that were not byte-identical.
    if not_bit_exact and args.allow_semantic:
        print(f"\n{len(not_bit_exact)} unit(s) not bit-exact — running official semantic gate "
              f"as fallback ...")
        sem_rc = run_semantic_gate(args.candidate_image, args.workers)
        if sem_rc == 0:
            print("Semantic gate PASSED for the full roster — units preserve semantics "
                  "though not bytes.")
            return 0
        print("Semantic gate FAILED — a real equivalence break.", file=sys.stderr)
        return 1

    if not_bit_exact:
        print(f"\nEQUIVALENCE FAILED: {len(not_bit_exact)} unit(s) not bit-exact "
              f"(pass --allow-semantic to permit the semantic fallback).", file=sys.stderr)
        return 1

    print("\nEQUIVALENCE PASSED: every unit is bit-exact against the reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
