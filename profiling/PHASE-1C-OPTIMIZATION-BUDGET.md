# Phase 1c — Measured Optimization Budget

> SPEC-01 §3/§6 Phase 1c deliverable. Turns the §3 bottleneck *hypotheses* into a
> *measured* ranked backlog, and seeds the NVTX/SimProfile component map for the
> Best Systems Diagnosis award (Phase 3b). Source: `profiling/profile_hotloop.py`
> run inside the baseline image. First unit profiled: `t3-as01-base-mix`
> (`abd7b2b8…`), base agent mix, 24,695 events. More units (deep book, high-agent,
> long-horizon) to follow — this is the first measured cut, not the final budget.

## Container-window split (where the ranked time goes)

| Component | time | % of window | verdict |
|---|---|---|---|
| **`abides.run` (hot loop)** | 4.368s | **86.7%** | the target — compile this |
| `extract_trace` | 0.347s | 6.9% | serialization tax (§5.7 zero-copy later) |
| `extract_message_trace` | 0.143s | 2.8% | serialization tax |
| parquet write (both) | 0.181s | 3.6% | serialization tax |

**The hot loop is ~87% of the window.** Serialization is ~13% combined — real, but
second-order. This confirms the SPEC-01 thesis: the compiled CPU core is aimed at
the right thing. (cProfile inflates absolute times ~2× vs the unprofiled 1.9s run;
the *relative* split is what is load-bearing.)

## Ranked bottleneck backlog (by self/tottime inside the hot loop)

The headline: **~20%+ of the hot loop is pure-Python overhead that produces NO
simulation value** — a compiled core doesn't just run it faster, it *deletes* it.

| # | Hotspot | self time | calls | why it's there | Phase-2 action |
|---|---|---|---|---|---|
| 1 | `copy.deepcopy` (+`_keep_alive`,`__deepcopy__`,`_deepcopy_dict`) | **~0.45s (~10%)** | 169,275 | ABIDES deep-copies Order objects per message | **ELIMINATE** — value semantics in the Rust core; no deepcopy at all |
| 2 | `abides_core.utils.fmt_ts` | 0.296s (6.8%) | 50,153 | timestamp→string formatting, mostly for logging | eliminate — keep ns ints; format only on demand |
| 3 | `kernel.runner` (event loop) | 0.174s self / 3.585s cum | 1 | the DES dispatch spine (§3 #1) | compiled event queue; calendar/ladder queue (§5.4) |
| 4 | `isinstance` + `_abc_instancecheck` | 0.163s | 759k+ | Python dynamic type dispatch | gone — static types in the compiled core |
| 5 | `orders.__str__` + `str.format` | 0.148s | 32,665 | Order stringification (logging) | eliminate from hot path |
| 6 | `kernel.send_message` | 0.114s | 25,865 | per-message dispatch + object churn | compiled message passing |
| 7 | order book: `execute_order`+`handle_limit_order` | 0.108s | 12k | matching engine (§3 #2) | compiled price-time-priority LOB |
| 8 | `price_level.total_quantity` (+genexpr) | 0.112s | 82,401 | recomputes depth by summing a generator | O(1) maintained running total (SoA book) |
| 9 | `get_latency` + `lognormal` | 0.113s | 25,865 | per-message latency draw (§3 #4) | bulk/vectorized draws; mirror RNG order |
| 10 | `logEvent` + `logging.debug` | 0.079s | 53k+ | event logging | suppress/compile out of hot path |

Plus end-of-sim `analyse_order_book` at `kernel.terminate` = 0.69s cumulative —
one-shot bookkeeping worth a look but not per-event.

## What this means for Phase 2 sequencing

The SPEC-01 §3 hypotheses are **confirmed and now ranked**:
1. The **deepcopy + formatting + isinstance + logging tax (~25% combined)** is the
   single biggest, lowest-risk win — it is overhead the reference pays that a
   compiled core simply does not. Attack it first via the compiled Order/message
   layer.
2. **Event-queue dispatch** (`kernel.runner`) is the spine — the calendar/ladder
   queue upgrade (§5.4) sits here, order-preserving.
3. **Order book** (`handle_limit_order`/`execute_order`/`total_quantity`) is the
   matching-engine target — compile with a maintained depth total (kills #8).
4. **Latency sampling** — bulk-drawable, but RNG draw order must be mirrored
   bit-for-bit (Guardrail #4) or Tier-A breaks.

Order of attack (matches SPEC-01 Phase 2a): **order book first** (clean boundary,
hotspots #7/#8), then the **compiled kernel event queue** (#3/#6), with the
Order/message value layer (#1/#4/#5) removing the deepcopy+format tax as a
cross-cutting win.

## NVTX / SimProfile component map (seeds Best Systems Diagnosis, Phase 3b)

The profile's component split maps directly onto the SimProfile schema components —
this is the measured basis for the `profile.json` we submit:

| SimProfile component | Maps to (measured) | first-cut share of hot loop |
|---|---|---|
| `event_queue` | `kernel.runner` + `send_message` + `queue.put/get` | ~15% |
| `matching` | `order_book.*` + `price_level.*` + `execute_order` | ~15% |
| `agent_logic` | `*.receive_message` + `order_executed` + agent `act` | ~25% |
| `latency` | `get_latency` + `lognormal` | ~3% |
| `io` | extract_trace + extract_message_trace + parquet write | ~13% of window |

The remaining deepcopy/fmt_ts/logging tax is spread across `agent_logic` and
`event_queue`; in the compiled core it collapses, which is itself the story the
diagnosis narrative tells (bottleneck moves pure-Python → compiled → GPU).

## Next
- Profile the remaining representative units (deep book, high-agent-count,
  long-horizon) to confirm the ranking generalizes before committing Phase 2 order.
- Add real NVTX ranges (nsys) once the compiled core exists, so the SimProfile is
  measured on the optimized build, not just the baseline.

## Multi-unit confirmation (ranking generalizes — and sharpens with scale)

Profiled four contrasting shapes. The hot-loop share is stable-to-rising with size:

| unit | events | hot loop | % of window |
|---|---|---|---|
| t3-as01-base-mix | 24,695 | 4.37s | 86.7% |
| t3-as02-scale-up-noise | 42,231 | 7.01s | 86.9% |
| t3-as03-mm-liquidity-heavy | 50,828 | 8.51s | 87.9% |
| **t3-mr-deep-book-state-size** | **319,488** | **55.28s** | **90.9%** |

**The big unit changes the priority order.** On `mr-deep-book-state-size` (6× larger,
the kind of unit that dominates the mean-throughput ranking) the top self-costs are:

| rank | hotspot | self time | calls | note |
|---|---|---|---|---|
| 1 | `kernel.runner` (event loop) | 2.79s | 1 | the DES spine |
| — | `queue.get`+`put`+`_heapq.heappop` | **4.54s** | 980,960 ×3 | **the O(log n) heap — now the dominant cost** |
| 2 | `fmt_ts` | 2.78s | 520,015 | timestamp formatting (pure waste) |
| 3 | `isinstance` | 1.70s | **10,728,862** | dynamic dispatch, explodes with scale |
| 4 | `deepcopy` | 1.47s | 731,182 | per-message object copy |
| 5 | `price_level` genexpr + `sum` | 2.24s | 6.5M + 2.1M | `total_quantity` depth-recompute (hotspot #8) |

**Revised Phase-2 insight:** at the large units that decide the ranked mean, the
**event-queue heap (`queue`+`heappop` ≈ 4.5s)** and the **depth-recompute
(`total_quantity` ≈ 2.2s)** grow superlinearly — the calendar/ladder queue (§5.4)
and an O(1) maintained depth total are not constant-factor wins there, they change
the complexity class. `fmt_ts` + `isinstance` (~4.5s combined) is pure tax a typed
compiled core deletes. This reinforces the attack order but raises the **event
queue** to co-priority with the order book for Phase 2a, because its cost is where
the big units bleed.
