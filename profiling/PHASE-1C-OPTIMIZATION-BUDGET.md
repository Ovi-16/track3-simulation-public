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
