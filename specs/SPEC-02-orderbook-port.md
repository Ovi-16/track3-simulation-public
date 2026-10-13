# Phase 2 — Compiled Core: Order-Book Port Spec (the bit-exact contract)

> SPEC-01 §4 Layer 1 / §6 Phase 2a. This is the EXACT semantic contract the Rust+PyO3
> order book must satisfy. Source of truth: ABIDES `abides_markets/order_book.py`
> (1016 lines) + `price_level.py` (281 lines), the pinned baseline. Every behaviour
> below is mirrored, not reinterpreted — the emitted trace must stay bit-exact
> (equivalence_harness.py, Guardrail #1/#2). Profiling (Phase 1c) made this the
> co-priority target with the event queue at scale.

## Why Rust here changes the complexity class (not just a constant)
From the measured budget (`PHASE-1C-OPTIMIZATION-BUDGET.md`), on the big units:
- `price_level.total_quantity` recomputes depth by **summing a generator** (6.5M calls,
  ~1.2s) — the Rust port maintains an **O(1) running total** per level.
- `deepcopy` on every enter/notify (731k calls, ~1.5s) — gone: Rust uses **value
  semantics**, copying only the plain fields a notification needs.
- `isinstance` dispatch (10.7M calls, ~1.7s) — gone: **static types**.

## Invariants that MUST hold bit-for-bit (break any => Tier-A zero)

1. **Price-time priority.** At one price level, the order that arrived first (by sim
   timestamp, then insertion order) fills first. No tie-break by agent_id or size.
2. **Piecewise partial fills.** `execute_order` finds ONE best match at a time, fills
   as much as possible at the best price, emits one notification per match, loops.
   A resting order partially filled stays on the book at reduced size with its
   **timestamp priority unchanged**.
3. **Validation + discard rules (exact).** Non-matching symbol → warn + discard.
   `quantity <= 0` or non-integer → warn + discard. `limit_price < 0` or non-integer
   → warn + discard. These warnings are stderr noise, not events — do not emit trace rows.
4. **Resting-order entry.** When no match remains, `enter_order(deepcopy(order))` adds
   it and (unless `quiet`) sends `OrderAcceptedMsg`. Rust: enter by value, same message.
5. **Average trade price.** `avg_price = int(round(trade_price / trade_qty))` over the
   executed pieces — Python `round()` is banker's rounding; the Rust port MUST
   reproduce banker's rounding, not C round-half-up, or prices drift.
6. **logEvent emissions, in order:** after processing, `BEST_BID` then `BEST_ASK`
   (each only if that side is non-empty), then `LAST_TRADE` if anything executed.
   These feed the trace — their presence/order/format is load-bearing.
7. **Self-trade prevention.** `stp_policy` read from owner (None = legacy no-op;
   `cancel_oldest` cancels resting same-agent order and retries; `cancel_newest`
   cancels the aggressor remainder and breaks). Inert unless a scenario opts in, so
   the default path must match the no-STP reference exactly.
8. **No RNG in the order book.** The book draws no randomness — all RNG lives in
   agents/latency (Guardrail #4). So the book port is deterministic by construction;
   the only correctness risk is arithmetic (banker's rounding) and ordering.

## Integration boundary (how the Rust book slots in without reordering events)
The Rust `OrderBook` is a **drop-in replacement for the Python class**, called by the
exchange agent during event processing. It does NOT touch the kernel event loop or
RNG draw order — it only replaces the per-event matching work. The Python driver
keeps owning message dispatch and logEvent routing at first (so notifications and
their ordering are provably identical), and we move those into Rust only after the
book itself is proven bit-exact.

## Build/layout (PyO3 + maturin)
```
rust_core/
  Cargo.toml            # crate-type = ["cdylib"], pyo3 + maturin
  src/
    lib.rs              # #[pymodule] exposing OrderBook, PriceLevel
    order_book.rs       # handle_limit_order / execute_order / enter / cancel
    price_level.rs      # SoA level with O(1) maintained total_quantity
    order.rs            # value-type Order (no deepcopy)
```
Toolchain present: rustc/cargo 1.79.0, maturin 1.15.0 (in .venv-scoring).

## Correctness workflow (every change)
1. Build the wheel: `maturin develop` into .venv-scoring (or build inside the image).
2. Wire the Rust book behind a feature flag in the adapter (GPU-off-style fallback).
3. Run `equivalence_harness.py --only <unit>` → must be BIT-EXACT.
4. Only when green on a unit, widen to the full roster. A red harness reverts the change.

## Attack order (from the measured budget)
1. `price_level` with O(1) maintained total (kills the genexpr+sum hotspot) — smallest,
   cleanest, provable first step.
2. `execute_order` + `handle_limit_order` matching (price-time priority, piecewise).
3. Order value type (removes deepcopy) + notification field extraction.
4. THEN the event queue (calendar/ladder queue) — the other scale hotspot — as a
   separate, independently-gated change.
