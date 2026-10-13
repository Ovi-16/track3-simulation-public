# SPEC-04 — L4 Dispatch + L5 Trace Integration (the final bit-exact stretch)

> The remaining work for the full compiled core. Unlike L0-L3 (each locally
> cross-checked), this layer is validated ONLY by a byte-identical trace_sha256
> through equivalence_harness.py. Expect the first full run to mismatch, then an
> iterative hunt against the harness. Work it as the checklist below, one item at a
> time, re-running the harness after each.

## Message state machine to wire (kernel.rs dispatch -> components)

Per-wakeup agent cycle (ScheduledAgent):
1. kernel delivers `Wakeup` to agent at its scheduled time.
2. agent: `set_wakeup(now + interval_ns)` then sends `QuerySpread` to exchange (id 0),
   state -> AWAITING_SPREAD.
3. exchange `receive_message(QuerySpread)` -> replies `SpreadResponse{bid,ask}` (best
   bid/ask from its order book; 0 == empty side).
4. agent `receive_message(SpreadResponse)` -> `act()` (L3 decision) -> may
   `place_limit_order` -> sends `OrderSubmit` to exchange, state -> AWAITING_WAKEUP.
5. exchange `receive_message(OrderSubmit)` -> `book.handle_limit_order` (L1) ->
   for each Fill: `OrderExecuted` to BOTH resting + aggressor agents; if remainder
   rested: `OrderAccepted` to submitter. Emits `QUOTE_UPDATE` on book change.

## order_id assignment
Global monotonic counter in the TradingAgent base (`place_limit_order` assigns the
next id). Must match ABIDES: ids are assigned in the order agents place orders, which
is the kernel delivery order. Reproduce with a single counter in the engine incremented
at place_limit_order time.

## RNG seeding order (config.py — Guardrail #4, exact)
`global.seed(scenario_seed)` then one `randint_u32()` from GLOBAL per component to seed
its own MT19937, in THIS order: oracle, exchange(id0), agents id1..N (construction
order), latency model, kernel. The oracle ALSO draws its first megashock interval from
GLOBAL exponential (L2). Each agent/latency/oracle then draws only from its own state.

## L5 trace (exact schema — SPEC-01 §1.2/§1.3)
7 columns: t_ns(i64), agent_id(i32), msg_type(str), side(str BID/ASK/null),
price(i64), size(i64), order_id(i64). msg_type one of the 7 exact strings. Rows emitted
in the order the kernel processes the generating messages. message_trace.parquet carries
the v2 ledger (message_id, src, dst, t_send, t_recv, latency, causal_parent, deliver_seq).

## Integration + gate
- PyO3 pyclass `Engine` exposing run(scenario_json, seed) -> (trace cols, mtrace cols).
- `maturin develop` into .venv-scoring.
- Feature flag `TRACK3_CORE=rust` in baselines/abides_fork/simulate.py selects the engine;
  default = Python baseline (always-green fallback, Guardrail #6).
- GATE: `equivalence_harness.py --only <unit>` until bit-exact trace_sha256; then widen to 65.

## Known risk points to check first on a mismatch
1. Kernel PQ tie-break final key (currently message_id) — may need the exact Message
   comparison ABIDES uses.
2. computation_delay (default 50) applied after EACH message in a MessageBatch loop.
3. QUOTE_UPDATE emission timing/order relative to fills.
4. The one GLOBAL vs per-symbol RNG split in the oracle.
5. Still unread: exchange_agent.py 950L full order/query path (BEST_BID/ASK logEvent,
   kernel_terminating analyse_order_book) — read next session before wiring step 5.
