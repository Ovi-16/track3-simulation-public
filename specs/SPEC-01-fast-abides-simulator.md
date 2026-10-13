# SPEC-01 — Semantic-Preserving Fast Market Simulator (Agenthon 2026, Track 3)

> **Status:** Draft v1 · **Owner:** Ovi · **Created:** 2026-10-11
> **Methodology:** Spec-driven development. This document is the single source of truth for
> *what* we are building and *why*. Implementation PRs reference section numbers here.
> **Hard deadline:** Development closes **2026-10-12 23:59 AoE**; Final uploads close
> **2026-10-25 23:59 AoE**. See §12.

---

## 0. Executive summary (read this first)

Track 3 gives us J.P. Morgan's **ABIDES** agent-based market simulator (correct but slow,
pure-Python, single-threaded) and asks one question: **can we make it dramatically faster
while producing a byte-for-byte-equivalent event stream?**

A submission is a **Docker image** exposing a `simulate` / `simulate-batch` CLI. It is scored
in three stages, in this order, and **any failure in stage 1 or 2 scores the unit zero**:

1. **Correctness** — 71 public units + sealed units. Two tiers: Tier A = exact event/fill
   match (families 1,3,6,7,8); Tier B = statistical match (families 2,4,5).
2. **Realism** (admissibility gate) — 4 stylized-fact ceilings on Family 5. One breach ⇒
   inadmissible, no rank, regardless of speed.
3. **Speed** — ranked by `events_per_sec`, **arithmetic mean of per-unit rates over the whole
   roster**. A failed unit contributes **0** to that mean (it does not drop out). There is no
   unit we can afford to skip.

**The core engineering tension** (and the whole game): the admissibility gates (exact fills,
Kendall-τ ≥ 0.999, message-ledger causality, volatility clustering) *punish every approximation*.
Discrete-event simulation is inherently sequential and branch-heavy, so naive GPU batching or
event-coalescing **breaks correctness and scores zero**. Real speedups come from
**semantics-preserving** engineering: a compiled matching engine, eliminating per-event Python
overhead, and parallelism *across independent scenarios* — never *within* the event order.

**Our thesis:** win on a **compiled, single-threaded, exactly-equivalent CPU core** first
(reliable, admissible, large constant-factor win over pure-Python ABIDES), then layer
**cross-scenario batch parallelism** for Family 6 and chase **special awards** (Best Systems
Diagnosis via Nsight profiling; Speed–Realism Frontier) for differentiation. **GPU acceleration
and the Nsight/NVTX system-diagnosis award are both in scope** — sequenced *last* (Phase 3),
after the admissible CPU core is fast, and kept off the critical correctness path with a GPU-off
fallback. GPU is our final layer, not our foundation.

---

## 1. Problem statement & grading model (the contract we must satisfy)

### 1.1 Submission interface (frozen — do not deviate)

```
simulate       --config /input/scenario.json --out /output/trace.parquet
simulate-batch --batch-dir /input/scenarios  --out-dir /output
```

- Runs with `--network none`. Everything vendored into the image. No runtime downloads.
- Image must be **anonymously pullable** and must **not** declare a Docker `VOLUME`
  (inherited VOLUMEs fail the Final too).
- Writes per single run: `trace.parquet`, `message_trace.parquet` (when the card requires it —
  **59 of 71 units do**), `events.json`.

### 1.2 Output schema (`trace.parquet`, 7 columns, Snappy)

| Column | Type | Notes |
|---|---|---|
| `t_ns` | int64 | ns since sim epoch |
| `agent_id` | int32 | originating agent |
| `msg_type` | string | one of 7 fixed strings (§1.3) |
| `side` | string | `BID`/`ASK`/null |
| `price` | int64 | integer ticks |
| `size` | int64 | shares |
| `order_id` | int64 | stable across fill/cancel |

### 1.3 Valid `msg_type` values (exact strings, no inventions)

`ORDER_SUBMITTED`, `ORDER_ACCEPTED`, `ORDER_FILLED`, `PARTIAL_FILL`, `ORDER_CANCELLED`,
`ORDER_REPLACED`, `QUOTE_UPDATE`.

### 1.4 `events.json` (all six core keys required by the g1 gate)

`scenario_id` (str), `seed` (int), `n_events` (int, **== real row count of trace.parquet**),
`wall_clock_sec` (num), `events_per_sec` (num, within ±5% of `n_events/wall_clock_sec`),
`trace_sha256` (64-hex). Plus telemetry: `peak_memory_bytes`, `gpu_seconds`.

> **Anti-cheat:** harness recomputes `n_events` from the parquet footer and uses *its own* count
> as the ranked numerator. Over-declaring `n_events` cannot inflate throughput; a mismatch fails
> the run.

### 1.5 Grading stages (the zero-tolerance structure)

- **Tier A** (families 1,3,6,7,8): exact row count, exact fill sequence (`order_id`/price/size),
  exact bidirectional event coverage, Kendall-τ ≥ 0.999, ±1 µs timestamp tolerance.
- **Tier B** (families 2,4,5): return-distribution KS ≤ 0.08 **and** time-avg spread within
  ±10 bps **and** exact reference row count.
- **Admissibility (Family 5, all four must pass):** return-KS ≤ 0.08, ACF(|r|) RMS error over
  lags (1,5,10,20,50) ≤ 0.12, Hill tail-exponent abs error (top-100 order stats) ≤ 1.5, depth-
  histogram JS divergence ≤ 0.10.
- **No majority rule anywhere.** Every unit must pass. One failure = inadmissible OR a 0 in the
  mean.

### 1.6 Hardware target (frozen; cite `baselines/README.md` §5, never invent)

4 vCPU, 16 GiB RAM, 10 GiB disk, `network=none`, 1× **NVIDIA B200** (compute capability
**10.0**, `sm_100`), driver 580.173.02, CUDA 13.0.3 host (CUDA **12.x images work** — do not
rebuild for 13.x). CPU: Intel Xeon Platinum 8570. Runtime: `runc`. Per-run timeout 300 s in the
Final (local timer default 1800 s; rehearse with `QFB2_T3_RUN_TIMEOUT_SEC=300`).

### 1.7 Baseline numbers (the only reproducible one)

- ABIDES public reference geomean: **13,793 events/sec** (range 3,471–18,046), hardware **not**
  recorded. The old "~65,000" and "~400,000" figures are **withdrawn** — do not target them.
- Ranking is **relative to other submissions**, not to any number on the page. Tune against our
  **own measured baseline**.

---

## 2. Baseline architecture (what we are accelerating)

The pinned ABIDES engine + the `abides_fork` adapter in `baselines/abides_fork/`:

- `simulate.py` — CLI: reads scenario → `build_config` → `abides.run` → `extract_trace` /
  `extract_message_trace` → parquet + `events.json`. **The timed region is `abides.run`** (`t0`
  to `wall_clock_sec`); trace extraction and parquet write are *outside* the local timer but
  *inside* the harness container window — so both matter for the ranked number.
- `config.py` — maps `scenario.json` onto an ABIDES `rmsc04`-style config. **RNG draw order is
  load-bearing**: a global seed → `_draw_random_state()` called in a fixed order (oracle,
  exchange, each agent in construction order, latency model, kernel). Changing the draw order
  changes latencies and message arrival order → Tier-A divergence.
- `agents.py` — four thin `TradingAgent` subclasses (`NoiseTrader`, `MarketMaker`, `ValueTrader`,
  `MomentumTrader`), each waking on a schedule, querying spread, placing limit orders.
- The **kernel, exchange, order book, and matching engine are upstream ABIDES** (pure Python,
  single-threaded, event-at-a-time). **This is the hot loop and the primary optimization
  surface.**

**Per-event overhead is the enemy:** every event invocation calls back into pure-Python agent
logic through the kernel's priority queue, with per-message Python object allocation and dict
lookups.

---

## 3. Where the time goes (hypotheses to confirm by profiling — §6)

Ranked by expected payoff, to be validated with cProfile + Nsight before committing:

1. **Kernel event-queue dispatch** — Python heapq + per-event message object churn.
2. **Order book operations** — sorted insert/delete/match on Python data structures.
3. **Agent `act()` callbacks** — Python method dispatch, RandomState draws per wakeup.
4. **Latency sampling** — one RNG draw per message (`ScenarioLatencyModel.get_latency`).
5. **Trace extraction + parquet serialization** — inside the container window.

> We do not optimize blind. §6 (Phase 0) profiles the baseline first and turns this list into a
> measured budget. "A simulator that batches events" is explicitly called out in the docs as the
> thing that destroys volatility clustering — so profiling guides us *away* from the tempting-but-
> fatal optimizations too.

---

## 4. Architectural strategy (the three-layer plan)

### Layer 1 — Semantics-preserving compiled CPU core (PRIMARY, lowest risk)

Replace the hot path with a compiled implementation that produces an **identical event stream**.
Candidate stacks, in order of preference for this team:

- **Rust matching engine via PyO3/maturin** — a compiled price-time-priority LOB + event queue,
  called from a thin Python driver that keeps the exact RNG draw order. Highest ceiling,
  cleanest determinism control.
- **Cython / C extension** — lower ramp-up, incremental: compile the order book and kernel
  dispatch, leave agents in Python at first.
- **Numba** — fastest to prototype; risk is that branch-heavy LOB code JITs poorly and LLVM
  bloats the image.

**Determinism contract (non-negotiable):** we may use any data structure, any language, any RNG
*implementation* — but the emitted trace must pass Tier A. We will **mirror the reference's random
choices** (same seed-derivation order, same distribution draws) so latency and arrival order match
bit-for-bit, OR prove equivalence empirically against every public reference trace.

### Layer 2 — Cross-scenario parallelism (Family 6 / batch, medium risk)

`simulate-batch` runs N **independent** sub-scenarios. These are embarrassingly parallel *across*
scenarios (never within one). Use all 4 vCPUs via process/thread parallelism, each sub-scenario
isolated. The per-sub **isolation gate** requires each sub to reproduce its standalone trace, so
isolation must be real (no shared mutable state, no shared RNG).

### Layer 3 — GPU acceleration (OPTIONAL, highest risk, differentiation only)

Only after Layers 1–2 are admissible and fast. DES is sequential and branch-heavy; a batched GPU
port that reorders events fails Tier A. Viable GPU angles:
- Parallelize *across* independent scenarios/sub-scenarios on-device.
- Offload genuinely vectorizable, order-independent sub-computations (e.g. bulk latency draws,
  trace post-processing) — but mind that cuDF-style trace vectorization reorders events (docs warn
  explicitly).
- Target **Best GPU Acceleration** award (ranks on `speedup_vs_cpu_abides`, host-measured GPU
  utilization ≥ floor), not the main leaderboard.

Compile cubins for `sm_100` (PTX-only JITs inside the timed window). Create the CUDA context once
and reuse it.

---

## 5. Innovation & differentiation ideas (how we stand out — §8 has papers)

A lot of teams will do "rewrite the order book in C and call it a day." To place well *and* win a
special award, candidate differentiators, roughly by effort:

1. **Exactness-preserving compiled core + property-based equivalence harness.** Our moat is a
   test rig that *proves* bit-exact equivalence against all 65 public traces on every commit —
   so we can optimize aggressively without fear. Most teams will break Tier A silently; we won't.
   (This is also the single highest-leverage thing to build first.)
2. **Speed–Realism Pareto frontier submission.** Deliberately produce a family of configs trading
   speed vs realism and submit the frontier — targets the **Best Speed-Realism Frontier** award,
   which few will attempt.
3. **Nsight/NVTX-instrumented SimProfile.** Name NVTX ranges to match SimProfile components and
   submit `profile.json` → **Best Systems Diagnosis**. This is a *skill* award: pure profiling
   craft, no correctness risk, high ROI. Strong candidate even if our GPU story is thin.
4. **Event-queue algorithmic upgrade.** Replace the kernel's heap with a **calendar queue** or
   **ladder queue** (O(1) amortized enqueue/dequeue vs O(log n) heap) — a classic DES speedup that
   is *exactly* order-preserving. (See §8.) Pure algorithmic win, zero correctness risk.
5. **Latency-semantics preservation.** Target **Best Latency-Semantics Preservation** by nailing
   the Family 3 Kendall-τ with a provably-correct arrival-queue + tie-break (ascending `order_id`).
6. **Memory-layout / cache engineering.** Struct-of-arrays order book, integer-tick arithmetic,
   arena allocation for order objects — the Xeon 8570 cache hierarchy is the real target for a CPU
   core. Deep books = cache misses (docs flag this).
7. **Zero-copy trace emission.** Build the trace columns directly in the compiled core and hand
   Arrow buffers to pyarrow with no Python-object round-trip — cuts the serialization tax that
   sits inside the container window.

---

## 6. Phased execution plan — THREE phases (compressed, correctness-gated throughout)

We are doing **all of it**: compiled CPU core, cross-scenario CPU parallelism, GPU acceleration,
*and* the Nsight/NVTX system-diagnosis profiling for the special award. The six-phase draft is
compressed into **three phases**. Each phase ends with a hard **exit gate** that must be green
before the next starts, and **every phase runs the correctness harness as its own internal
check** — correctness is never deferred to the end. "Green" = defined, measurable, checked in.

> **Correctness-first invariant (applies to all three phases):** after *every* change that can
> touch the event stream, re-run the equivalence harness (§6, Phase 1) against all public
> reference traces. A red harness blocks the commit. We never trade correctness for speed, because
> a single Tier-A break scores the unit **zero**. This is the rule the whole plan is built around.

### Phase 1 — Foundation: reproduce, profile, and lock in a correctness harness *(blocking; do first)*

Everything that must exist before we are allowed to change a line of hot-path code. This phase
*builds the safety net* and *measures the enemy*.

**1a. Reproducible baseline**
- [ ] Docker + Git LFS working; `git lfs pull` fetches real trace data (not 130-byte stubs).
- [ ] Python 3.13 scoring env; toolkit **pinned**: `qfbench2-common[data] @ ...@v2.5.1`.
- [ ] `build_reference_cache.py` builds `reference_traces/` from `units/`.
- [ ] `docker build baselines/` → `track3-abides-baseline:latest`.
- [ ] `run_regression.py` — **all 65 public scenarios PASS** on the unmodified baseline. *(If not,
      the environment is broken — stop and fix before anything else.)*
- [ ] Record baseline `events_per_sec` per unit via `throughput/timer.py`. **This is our target
      number.**

**1b. Equivalence test harness (our moat — build this NOW, not later)**
- [ ] Property-based + golden-trace rig: for every public unit, assert a candidate trace passes
      Tier A / Tier B / the four stylized-fact gates vs the shipped reference, **importing the
      scoring from `qfbench2_common` — never reimplementing it**.
- [ ] Wire it into CI so every commit is checked automatically.
- [ ] Sanity: harness is green (0.0 divergence everywhere) on the unmodified baseline.

**1c. Profiling + system-diagnosis groundwork (starts the Best Systems Diagnosis award here)**
- [ ] cProfile + line_profiler on `abides.run` across a representative spread (shallow book, deep
      book, high-agent-count, long-horizon).
- [ ] First Nsight (`nsys`) run on a Family-6 unit; define the NVTX range names so they map 1:1 to
      SimProfile components (`matching`, `event_queue`, `latency`, `agent_logic`, `io`). **This is
      the start of the `profile.json` we submit for the award — we instrument from day one, not
      bolted on at the end.**
- [ ] Turn §3's hypotheses into a *measured* budget; pick the top 2–3 optimization targets.

- **Exit gate:** green regression suite + committed baseline throughput table + CI equivalence
  harness green on the baseline + a profiling report with a ranked, numbered optimization backlog
  and the initial NVTX instrumentation checked in.

### Phase 2 — Speed: compiled core + CPU cross-scenario parallelism *(the main leaderboard win)*

This is where throughput is won. **Correctness check runs after every single change (invariant
above).** The compiled stack is **Rust + PyO3** (see §9.1 for the C++/Rust/Cython decision);
Cython is the fast-start fallback if Rust ramp-up stalls.

**2a. Compiled, exactly-equivalent CPU core (Layer 1)**
- [ ] Replace the **order book first** — highest payoff, cleanest boundary — keep agents in Python.
      Re-run harness → must stay green.
- [ ] Compile the **kernel event queue** next, upgrading the heap to a **calendar/ladder queue**
      (§5.4) — an O(1), *exactly order-preserving* algorithmic win, zero correctness risk. Re-run
      harness → must stay green.
- [ ] Integer-tick arithmetic, struct-of-arrays book, arena-allocated orders (§5.6); zero-copy
      Arrow trace emission (§5.7). Harness green after each.
- [ ] Mirror the reference RNG draw order exactly so latencies and arrival order match bit-for-bit
      (§2 determinism contract).

**2b. Cross-scenario CPU parallelism (Layer 2)**
- [ ] `simulate-batch` runs independent sub-scenarios across all **4 vCPUs** with *real* isolation
      (no shared mutable state, no shared RNG). This is the throughput win for Family 6 — and the
      answer to "parallelize scenarios": it goes on **CPU cores, not the GPU**.
- [ ] Pass the per-sub **isolation gate** on all `t3-gbatch-*` units (each sub reproduces its
      standalone trace). Harness green across the batch.

- **Exit gate:** all 65 public units pass the harness + measurable speedup over the Phase-1
  baseline table on both single-run and batch units.

### Phase 3 — GPU acceleration + system diagnosis + Final packaging *(differentiation + ship)*

Begun once Phase 2 is admissible and fast. **This phase goes all-out: maximum GPU ambition and a
sweep at all four special awards — high risk explicitly accepted.** The *only* line that does not
move is the **correctness invariant**: anything that reorders a scenario's event stream fails
Tier A and scores the unit zero, so every GPU idea below is engineered to be either *across*
independent scenarios or over *order-independent* data — never inside one scenario's event order.
That is not caution we can trade away; it is the rule the award itself is gated behind
(`best_gpu_acceleration` is only considered *after* correctness/realism pass). Within that one
hard boundary, we push as hard as the B200 allows.

> **Hard-risk posture for Phase 3:** we accept build complexity, long spikes, image bloat, and the
> real chance a GPU path doesn't beat CPU-ABIDES in time. We do **not** accept a correctness
> regression — every GPU variant ships behind a feature flag with a proven-equivalent CPU-off
> fallback, so a failed GPU bet costs us the *award*, never the *leaderboard*.

**3a. GPU acceleration — full assault (→ Best GPU Acceleration award)**
Target: a genuine, host-measured `gpu_utilization ≥ floor` and a real `speedup_vs_cpu_abides`, not
a token kernel. Attack on every legal front, hardest-hitting first:
- [ ] **Massive across-scenario device parallelism.** Run the whole batch roster as thousands of
      concurrent independent scenarios resident on the B200 — one scenario per block/warp-group,
      each with its own event queue and RNG state in device memory. This is the single biggest
      legal GPU win: `simulate-batch` becomes a device-resident fleet, saturating the card without
      ever touching intra-scenario order. Each sub must still reproduce its standalone trace
      (isolation gate) → harness-verified.
- [ ] **GPU-resident order book + matching, per scenario.** Port the price-time-priority LOB to a
      struct-of-arrays layout in device memory with integer-tick arithmetic, so matching for each
      scenario runs on-device. Determinism is preserved *within* each scenario (sequential event
      order per scenario); parallelism is strictly *across* scenarios.
- [ ] **Bulk order-independent offload.** Pre-draw all latency samples, oracle paths, and agent
      wake schedules as large data-parallel kernels (these are computable ahead of event order),
      then feed them into the sequential replay. Mirror the reference RNG draw order bit-for-bit so
      the pre-drawn values are identical to what sequential ABIDES would have drawn (§2 contract).
- [ ] **GPU trace post-processing** that is provably order-preserving (stable, key-sorted on
      `(t_ns, order_id)`) — explicitly avoiding cuDF-style vectorization that reorders events
      (docs warn on this). If we cannot prove order preservation, it stays on CPU.
- [ ] **B200-specific engineering (go deep):** compile cubins for `sm_100` (never ship PTX-only —
      JIT inside the timed window is a throughput tax); create the CUDA context **once** and reuse
      it; use CUDA graphs to amortize launch overhead across the scenario fleet; pin host memory
      and overlap H2D/D2H with compute via streams; exploit the B200's large HBM to keep the entire
      scenario fleet resident (no re-staging). Measure utilization with the host-side counter, not
      self-report (self-report is fail-closed).
- [ ] Harness green after **every** GPU change; a red harness reverts the change, no exceptions.

**3b. System diagnosis — pristine, professional (→ Best Systems Diagnosis award)**
The highest-ROI, zero-correctness-risk award — we go for a *flawless* submission, not a passing one:
- [ ] Full **Nsight Systems (`nsys`)** capture across a representative unit spread, with **NVTX
      ranges named to map 1:1** onto SimProfile components (`matching`, `event_queue`, `latency`,
      `agent_logic`, `io`) and nested sub-ranges for genuine granularity.
- [ ] **Nsight Compute (`ncu`)** kernel-level analysis on the GPU kernels from 3a (occupancy, memory
      throughput, warp divergence) — this is the "deep engineering craftsmanship" the award rewards.
- [ ] Finalize `profile.json` via `throughput/simprofile.py`: per-component wall-clock summing to
      the **host-measured** total within 10%, honest `gpu_utilization` and `peak_memory_bytes`, and
      `not telemetry_self_reported`. `verify_profile()` scores granularity × consistency — maximize
      both; any breach zeros it.
- [ ] Write the accompanying diagnosis narrative: what the profile reveals, where the bottlenecks
      moved from pure-Python → compiled → GPU, and why each optimization was chosen. Craft, shown.

**3c. The other two special awards — sweep them too (we're going all-out)**
- [ ] **Best Speed-Realism Frontier (`best_speed_realism_frontier`):** deliberately produce a
      *family* of configs trading realism fidelity vs speed and submit the Pareto frontier — few
      teams attempt this (§5.2).
- [ ] **Best Latency-Semantics Preservation (`best_latency_semantics_preservation`):** nail the
      Family 3 Kendall-τ with a provably-correct arrival-queue + ascending-`order_id` tie-break
      (§5.5); this falls out of a faithful kernel but we tune for it explicitly.

**3d. Packaging & Final rehearsal**
- [ ] Lean multi-stage Dockerfile, **no VOLUME**, anonymously pullable, `sm_100` cubins baked in.
      Prune LLVM/CUDA build layers aggressively — a GPU image is heavy and the pull counts inside
      the 300 s window.
- [ ] Rehearse Final limits: `QFB2_T3_RUN_TIMEOUT_SEC=300`, 4-CPU / 16 GiB / 1× B200 quota.
- [ ] Full-roster dry-run under Final constraints; confirm Family 5's four admissibility gates and
      that **no unit scores 0** — in both the GPU-on and GPU-off (fallback) configurations.

- **Exit gate:** a clean full-roster run under Final constraints (every unit admissible, harness
  green) in **both** GPU-on and GPU-off modes + a submitted `profile.json` scoring high on
  granularity × consistency + an admissible GPU variant with real `speedup_vs_cpu_abides` and
  host-measured utilization ≥ floor + frontier/latency-semantics configs submitted. GPU-off is the
  always-available safe floor; GPU-on is the full-ambition bet layered on top of it.

---

## 7. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Silent Tier-A break from an "optimization" | High | Fatal (0 score) | Equivalence harness in CI from Phase 2; test every commit |
| RNG draw-order drift changes latencies | High | Tier-A fail | Mirror reference draw order exactly; §2 contract |
| GPU batching reorders events | High | Tier-A fail | GPU only *across* scenarios; never within event order |
| Image too big / slow pull inside 300 s | Med | Timeouts | Multi-stage build, prune LLVM/CUDA layers, measure pull |
| Toolkit version drift (local ≠ scored) | Med | Mismatch | Pin `v2.5.1`; never reimplement scoring code |
| Running out of time before Final | Med | Fewer layers | Layer 1 alone is a valid, admissible, competitive submission |
| Over-tuning to public units | Med | Sealed surprise | Profile across *all* public units; stress untested axes (slow mean-reversion, deep books) |

---

## 8. Research & reading list (ideas from the literature)

Spec-level pointers to validate during Phase 1; detailed notes to follow in a separate research
doc.

**Discrete-event simulation performance (directly actionable):**
- Brown, R. (1988), *Calendar Queues: A Fast O(1) Priority Queue Implementation for the Simulation
  Event Set Problem.* — the event-queue upgrade in §5.4.
- Tang, Goh, Thng (2005), *Ladder Queue: An O(1) Priority Queue Structure for Large-Scale DES.* —
  improves on calendar queues for skewed distributions.
- Fujimoto, R. (2000), *Parallel and Distributed Simulation Systems.* — the canonical text;
  conservative vs optimistic (Time Warp) synchronization, relevant to Layer 2.
- Jefferson, D. (1985), *Virtual Time* (Time Warp). — optimistic PDES; mostly a *caution* here
  since rollback changes nothing we can observe but adds complexity.

**Agent-based market simulation / ABIDES:**
- Byrd, Hybinette, Balch (2020), *ABIDES: Towards High-Fidelity Multi-Agent Market Simulation.* —
  the engine we're accelerating; read for the kernel/agent contract.
- Vyetrenko et al. (2020), *Get Real: Realism Metrics for Robust Limit Order Book Market
  Simulations.* — the stylized-facts framing behind the Family 5 gates.
- Cont, R. (2001), *Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues.*
  — the five stylized facts the admissibility gate encodes (fat tails, vol clustering, etc.).

**High-performance LOB / matching engines (engineering craft):**
- Public HFT matching-engine designs (LMAX Disruptor ring-buffer pattern; struct-of-arrays LOB).
  — memory-layout and cache ideas for §5.6.
- Preis, Golke, Paul, Schneider (2006), *Multi-agent-based order book model of financial markets.*
  — a vectorizable order-book formulation (mind the exactness constraint).

**GPU / NVIDIA stack (Layer 3 only):**
- NVIDIA Nsight Systems docs + the repo's `docs/PROFILING.md` nsys→SimProfile recipe.
- Prior work on GPU-accelerated agent-based simulation (FLAME GPU) — for the *across-scenario*
  parallelism pattern, not within-event.

> **Action:** Phase 1 includes a focused literature pass. We confirm the calendar/ladder queue
> win first (lowest risk, order-preserving, pure algorithmic speedup) before anything exotic.

---

## 9. Open decisions (need a call before/within Phase 3)

1. **Compiled stack: Rust+PyO3 vs C++ vs Cython vs Numba.** **Decision: Rust+PyO3 primary,
   Cython fast-start fallback.** Reasoning: the speedup comes from *leaving pure-Python*, not from
   any one language — C++, Rust, and Cython all compile the branchy event loop to native code and
   land in the same 10–50× ballpark on the hot path, so language choice barely moves the speed
   ceiling. The deciding factor is the **determinism contract (§2)**: output must be byte-for-byte
   identical, and in C++ a subtle undefined-behavior bug (uninitialized read, signed overflow,
   iterator invalidation) can silently flip one output byte → Tier-A fail → unit scores **zero**.
   Rust's compiler forbids that entire bug class, which under a crunch where any correctness slip
   is fatal is worth more than C++'s marginal (often nil) speed edge. PyO3/maturin also gives a
   cleaner Python binding than pybind11. **C++ is a valid third option but carries the highest
   silent-correctness-bug risk — exactly the failure this track punishes with a zero — so it is not
   the default.** Confirm with a short Phase-1 spike if the team's Rust ramp-up looks slow; Cython
   lets us compile one function at a time if so.
2. **How much of the agent layer to compile.** Order book + kernel first; agents last (they're
   thin). Revisit if agent callbacks dominate the profile.
3. **GPU: in or out?** **Decision: in** — we are doing GPU acceleration (Phase 3a) and the
   Nsight/NVTX system-diagnosis award (Phase 3b). GPU stays *off the critical correctness path*
   (across-scenario parallelism and order-independent offload only, never within event order), and
   ships with a correctness-safe GPU-off fallback if the port can't beat CPU-ABIDES before the
   deadline.
4. **Determinism approach:** mirror reference RNG draw order (safer) vs prove empirical trace
   equivalence (more freedom). Likely both — mirror where cheap, prove where we diverge.

---

## 10. Team & ownership *(fill in — placeholder from prior project context)*

| Area | Owner | Notes |
|---|---|---|
| Equivalence harness + CI (Phase 1b) | TBD | The moat — assign early |
| Compiled core (Phase 2a) | TBD | Rust/Cython lead |
| Batch parallelism (Phase 2b) | TBD | |
| Profiling + awards (Phase 1c / 3b) | TBD | Nsight skill |
| GPU acceleration (Phase 3a) | TBD | high-risk, award-only |
| Packaging + Final rehearsal (Phase 3c) | TBD | |

---

## 11. Definition of done (per the repo's own self-check)

Before marking any milestone done, all three must pass (from `AGENTS.md`):
```bash
qfbench2 manifest verify <scenario_dir>
python regression_suite/run_regression.py --candidate-image <img> \
    --scenarios-dir regression_suite/scenarios/ --reference-dir regression_suite/reference_traces/ \
    --output-dir run_outputs/ --workers 4
fail=0; for u in units/*/; do qfbench2 manifest assert-public-safe "$u" || fail=1; done; exit $fail
```
A submission is **Final-ready** when: all public units pass, Family 5 four gates pass,
no unit scores 0 under the 300 s/4-CPU/16 GiB Final constraints, the image is anonymously
pullable with no VOLUME, and throughput beats our recorded baseline on the roster mean.

---

## 12. Timeline reality check

- **Dev board closed 2026-10-12 23:59 AoE.** As of today (**2026-10-13**) we are in the
  **Final + Verification window (2026-10-13 → 25)** — one final submission per track. The whole
  three-phase plan now runs against the Final: **Phase 1 (foundation + harness) first and fast**,
  then **Phase 2 (compiled core + CPU parallelism)** for the leaderboard, then **Phase 3 (GPU +
  system diagnosis + packaging)**.
- **Phase 1 alone is a valid, admissible, competitive submission** if time runs short — a
  reproduced baseline with a green equivalence harness is already a safe floor. Each later phase
  only adds speed/awards on top, never risks the floor.
- Keep the image pullable until results publish.
