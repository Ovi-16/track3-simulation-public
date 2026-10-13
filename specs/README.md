# Agenthon 2026 — Track 3 — Our Working README & Guardrails

> **Read this before writing a single line of code.** This is *our* entry point for the Track 3
> submission (the repo-root `README.md` is the upstream competition README — do not edit it).
> The one source of truth for the plan is [`SPEC-01-fast-abides-simulator.md`](./SPEC-01-fast-abides-simulator.md).

- **Methodology:** spec-driven development. SPEC-01 says *what* and *why*; every PR cites a §.
- **Compiled stack decision:** **Rust + PyO3** (primary), Cython fast-start fallback. See SPEC-01 §9.1.
- **Plan shape:** three phases. **Phase 1** foundation + correctness harness → **Phase 2** compiled
  core + CPU parallelism → **Phase 3** all-out GPU + all four special awards. See SPEC-01 §6.
- **Current window:** Dev board closed 2026-10-12 AoE; we are in the **Final window (→ 2026-10-25)**.

---

## 🛑 GUARDRAILS — non-negotiable rules (these do not move)

These are the rules the whole project is built around. A violation of #1 scores a unit **zero**, so
treat them as hard constraints, not preferences.

### 1. Correctness is never traded for speed — ever.
Grading is three gates **in order**: correctness → realism → speed. **Any** correctness or realism
failure scores that unit **zero**, and a zeroed unit still counts in the ranked mean (it does not
drop out). So one broken unit poisons the whole score. There is no optimization worth a Tier-A break.

### 2. The equivalence harness gates every commit.
After **any** change that can touch the event stream, re-run the equivalence harness against **all**
public reference traces (SPEC-01 §6, Phase 1b). **A red harness blocks the commit — no exceptions,
including GPU work.** We build this safety net in Phase 1 *before* changing hot-path code.

### 3. Never reorder a scenario's event stream.
Discrete-event simulation is sequential. The trace must be byte-for-byte reproducible, so we may
**never** parallelise *within* a scenario's event order. All parallelism (CPU *and* GPU) is strictly
*across* independent scenarios, or over provably *order-independent* data. This is why "batch events
on the GPU" is forbidden — it reorders events and fails Tier A.

### 4. Mirror the reference RNG draw order exactly.
A global seed drives `_draw_random_state()` in a fixed order (oracle → exchange → each agent in
construction order → latency model → kernel). Changing that order changes latencies and arrival
order → Tier-A divergence. Any compiled/GPU path must reproduce the draw order bit-for-bit, or prove
empirical trace equivalence against every public reference.

### 5. Never reimplement the scoring code.
Import scoring from `qfbench2_common` (pinned `v2.5.1`); never hand-roll Tier-A/Tier-B/stylized-fact
checks. Local scoring must match the grader exactly, or our "green" is a lie.

### 6. GPU is high-risk and always has a safe fallback.
Phase 3 goes all-out on GPU (SPEC-01 §6, Phase 3) — maximum ambition, high risk **accepted**. But
every GPU variant ships behind a feature flag with a **proven-equivalent CPU-off fallback**. A failed
GPU bet costs us the *award*, never the *leaderboard*. GPU acceleration is only even *considered* for
its award after correctness/realism pass.

### 7. Submission-contract rules are frozen.
Docker image, anonymously pullable, **no `VOLUME`**, `--network none`, exact 7-column `trace.parquet`
schema, exact `msg_type` strings, all six `events.json` keys. The harness recomputes `n_events` from
the parquet footer — over-declaring cannot inflate throughput. See SPEC-01 §1.

### 8. Definition of done is the repo's own self-check.
`qfbench2 manifest verify`, `run_regression.py` across the roster, and `assert-public-safe` on every
unit must all pass before any milestone is marked done. See SPEC-01 §11.

---

## Per-phase correctness checkpoint (applies to all three phases)
Correctness testing is **inside every phase**, not deferred to the end:
- **Phase 1** — baseline reproduces + harness green on the unmodified baseline (0.0 divergence).
- **Phase 2** — harness re-run after *every* compiled-core / parallelism change; must stay green.
- **Phase 3** — harness green after *every* GPU change, in **both** GPU-on and GPU-off modes.

## Next step
Phase 1 (foundation: reproduce baseline, build the CI equivalence harness, start NVTX profiling).
Not started yet — begins on explicit go-ahead.
