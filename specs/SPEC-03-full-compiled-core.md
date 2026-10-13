# SPEC-03 — Full Compiled Core (Option C) Architecture

> SPEC-01 §4 Layer 1 taken to its limit: the kernel event loop + agents + latency +
> oracle in Rust, Python keeping only config/RNG seeding and trace extraction. This
> is the biggest win (the Phase-1c event-queue heap dominates the large units that
> decide the ranked mean) and the hardest correctness problem. Built in LAYERS, each
> gated BIT-EXACT by equivalence_harness.py — never one untested blob.

## The one hard problem: RNG equivalence (make-or-break, Guardrail #4)

ABIDES is deterministic for a seed because every stochastic quantity comes from
NumPy MT19937 draws in a FIXED order. From `baselines/abides_fork/config.py`:

```
np.random.seed(seed)
# then _draw_random_state() = RandomState(np.random.randint(0, 2**32, dtype=uint64))
# called in EXACTLY this order:
1. oracle   random_state
2. exchange random_state         (agent id=0)
3. each scenario agent, in construction order (id=1,2,...), one draw each
4. latency  model random_state
5. kernel   random_state
```

Each agent then samples from ITS OWN RandomState during the run; latency samples
from the latency model's. The emitted latencies and order sizes ARE these draws, and
latency decides message arrival order → the whole Tier-A event sequence.

**Therefore the Rust core must reproduce, bit-for-bit:**
- MT19937 with NumPy's exact seeding (incl. the `init_by_array` path RandomState uses)
  and NumPy's `randint(0, 2**32)` stream, so the per-component seeds match.
- NumPy's distribution samplers, each algorithm-identical:
  - `lognormal(mu, sigma)` = exp(normal(mu,sigma)); normal via NumPy's ziggurat (gauss).
  - `uniform(a,b)`, `pareto(alpha)`, `normal(loc,scale)`, `randint`.
  - the oracle's OU / mean-reverting draws + megashock Poisson inter-arrivals.
- Python `int(round(np.clip(value, min, max)))` — banker's rounding (already solved in
  order_book.rs `banker_round_div`), np.clip semantics.

> **This is the gate on Option C.** If the Rust MT19937 + samplers do not match NumPy
> to the bit, latencies drift, messages reorder, Tier-A zeros. We build and PROVE this
> layer FIRST, in isolation, before any event loop — a Rust `np_random` that passes a
> cross-check harness against NumPy for every sampler and the exact seed-draw sequence.

## Layered build plan (each layer bit-exact-gated)

**L0 — RNG parity crate** (`np_random.rs`). Reproduce NumPy MT19937 + RandomState
seeding + samplers. GATE: a cross-check that, for a range of seeds, the Rust stream
equals `numpy.random.RandomState(seed)` draw-for-draw across randint/lognormal/
uniform/pareto/normal. No event loop until this is green.

**L1 — Order book** (DONE: order_book.rs, 7/7 unit tests). Matching proven in
isolation; integrates at L3.

**L2 — Latency + oracle** (`latency.rs`, `oracle.rs`). Port ScenarioLatencyModel
(log_normal/uniform/pareto/deterministic, clip+round) and SparseMeanRevertingOracle
(kappa per-ns, megashocks, scheduled jumps), each drawing from an L0 RandomState in
the exact order config.py sets up. GATE: given the same seed, Rust latency/oracle
sequences match a Python dump draw-for-draw.

**L3 — Agents** (`agents/*.rs`). Port NoiseTrader, MarketMaker, ValueTrader,
MomentumTrader from baselines/abides_fork/agents.py — each wakeup/receive_message
reproducing the exact RNG draws and order decisions. GATE: per-agent, the orders
emitted for a scripted message sequence match the Python agent.

**L4 — Kernel event loop** (`kernel.rs`). The time-ordered message PQ (the Phase-1c
hotspot) as a calendar/ladder queue that is EXACTLY order-preserving: same tie-break
as heapq on (time, event), the wakeup-requeue guard (`agent_current_times`), the
per-message latency add, computation-delay handling, and the v2 message-ledger
(message_id, causal_parent, deliver_seq) that feeds message_trace.parquet. GATE: full
trace bit-exact on a unit via equivalence_harness.py.

**L5 — Trace assembly** (`trace.rs`). Build the 7-column trace + message_trace columns
directly in Rust (zero-copy Arrow), handing buffers to pyarrow. GATE: byte-identical
parquet (same trace_sha256) — the ultimate check.

## Integration & fallback (Guardrail #6)
- Python `simulate.py` keeps the CLI, config build (RNG seed), and parquet write; a
  feature flag `TRACK3_CORE=rust` swaps the engine. Default stays the Python baseline
  (always-green fallback) until the Rust core is bit-exact on the FULL roster.
- Every layer merges only when equivalence_harness.py is green on the units it affects.
  A red harness reverts. We never ship a non-bit-exact core; GPU (Phase 3) layers on
  top of a proven C core.

## Why this order
L0 first because every later layer depends on RNG parity — it is the root risk, and
cheap to prove in isolation. L4 (event queue) is where the measured scale win is, but
it cannot be validated bit-exact until L0-L3 feed it correct draws. L5 delivers the
byte-identical trace that is the strongest possible equivalence proof.

## Status
- L1 order book: DONE (order_book.rs, price_level.rs, 7/7 cargo tests).
- L0 RNG parity: NEXT — the gating layer. Build np_random.rs + NumPy cross-check.
