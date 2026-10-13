//! L4 dispatch — Half 1: engine setup + the wakeup/query/spread cycle (SPEC-04).
//!
//! This half wires the agent activation loop WITHOUT order placement yet:
//!   Wakeup -> agent re-arms set_wakeup(now+interval) + sends QuerySpread
//!   QuerySpread@exchange -> SpreadResponse{best_bid, best_ask}
//!   SpreadResponse@agent -> act() [Half 2 fills in order placement]
//! Half 2 adds OrderSubmit -> book.handle_limit_order -> notifications + trace rows.
//!
//! Agents are modeled as a tagged enum (the 4 types) each owning its MT19937 seeded in
//! config order (engine::draw_component_seeds). The exchange is agent id 0 and owns the
//! order book. The kernel (L4) owns the event queue + latency.

use crate::agents::{MarketMaker, MomentumTrader, NoiseTrader, Touch, ValueTrader};
use crate::engine::ComponentSeeds;
use crate::kernel::{Kernel, MsgKind};
use crate::latency::ScenarioLatencyModel;
use crate::np_random::MT19937;
use crate::oracle::Oracle;
use crate::order_book::OrderBook;

/// A scenario agent with its own RNG + wakeup interval.
pub enum AgentKind {
    Noise(NoiseTrader),
    Market(MarketMaker),
    Value(ValueTrader),
    Momentum(MomentumTrader),
}

pub struct SimAgent {
    pub id: i64,
    pub kind: AgentKind,
    pub rng: MT19937,
    pub interval_ns: i64,
    pub mkt_open: i64,
    pub mkt_close: i64,
}

/// The full simulation state the dispatch drives. Half 1 holds the pieces + the
/// activation loop up to the spread response; Half 2 adds order flow + trace.
pub struct Sim {
    pub kernel: Kernel,
    pub book: OrderBook,
    pub oracle: Oracle,
    pub agents: Vec<SimAgent>,
    pub exchange_id: i64,
    pub current_time: i64,
    // Half 2:
    pub next_order_id: i64,
    pub pending_order_ids: Vec<i64>,
    pub trace: TraceBuilder,
}

impl Sim {
    /// Build the sim: construct kernel + book + oracle + agents using the config-order
    /// seeds (ComponentSeeds), and prime each agent's first wakeup at market open. The
    /// per-component MT19937s are created from the verified seed sequence.
    pub fn new(
        seeds: ComponentSeeds,
        n_agents: usize,
        mkt_open: i64,
        mkt_close: i64,
        stop_time: i64,
        default_computation_delay: i64,
        latency: ScenarioLatencyModel,
        oracle: Oracle,
        agents: Vec<SimAgent>,
    ) -> Self {
        let _ = (seeds, n_agents); // seeds already consumed to build oracle/agents/latency
        let kernel = Kernel::new(
            n_agents + 1, // +1 for the exchange (id 0)
            mkt_open,
            stop_time,
            default_computation_delay,
            latency,
        );
        Sim {
            kernel,
            book: OrderBook::new(),
            oracle,
            agents,
            exchange_id: 0,
            current_time: mkt_open,
            next_order_id: 0,
            pending_order_ids: Vec::new(),
            trace: TraceBuilder::new(),
        }
    }

    /// Prime the first wakeup for each scheduled agent at market open (the parent
    /// delivers the first wakeup; subsequent ones the agent re-arms itself).
    pub fn prime_wakeups(&mut self) {
        for a in &self.agents {
            self.kernel.set_wakeup(a.id, a.mkt_open);
        }
    }

    /// Current best bid/ask as the agents' get_known_bid_ask would see it (0 == empty).
    fn touch(&self) -> Touch {
        let bid = self.book.best_bid().map(|(p, _)| p).unwrap_or(0);
        let ask = self.book.best_ask().map(|(p, _)| p).unwrap_or(0);
        Touch { bid, ask }
    }

    /// Half-1 activation step for a delivered Wakeup: re-arm next wakeup and send a
    /// QuerySpread to the exchange. (Order placement happens after the spread response,
    /// wired in Half 2.)
    pub fn on_wakeup(&mut self, agent_idx: usize) {
        let a = &self.agents[agent_idx];
        let (id, next) = (a.id, self.kernel.current_time + a.interval_ns);
        self.kernel.set_wakeup(id, next);
        self.kernel.send_message(id, self.exchange_id, MsgKind::QuerySpread, 0);
    }

    /// Exchange handling of QuerySpread: reply with the current touch.
    pub fn on_query_spread(&mut self, requester_id: i64) {
        let t = self.touch();
        self.kernel.send_message(
            self.exchange_id,
            requester_id,
            MsgKind::SpreadResponse { bid: t.bid, ask: t.ask },
            0,
        );
    }
}

// ---- Half 2: order flow (act -> submit -> match -> notify + trace) ----

use crate::agents::OrderIntent;
use crate::order_book::{IncomingOrder, Side};
use crate::trace::{msg_type, TraceBuilder, TraceRow, TraceSide};

impl Sim {
    fn next_order_id(&mut self) -> i64 {
        let id = self.next_order_id;
        self.next_order_id += 1;
        id
    }

    /// SpreadResponse delivered to an agent: run its act() decision (Half-2) and submit
    /// the resulting order(s) to the exchange. Deterministic agents (MM/Momentum) take no
    /// RNG; Noise/Value draw from their own rng in the exact order (L3).
    pub fn on_spread_response(&mut self, agent_idx: usize, bid: i64, ask: i64) {
        let t = Touch { bid, ask };
        let agent_id = self.agents[agent_idx].id;
        let intents: Vec<OrderIntent> = {
            let agent = &mut self.agents[agent_idx];
            match &mut agent.kind {
                AgentKind::Noise(nt) => vec![nt.act(t, &mut agent.rng)],
                AgentKind::Market(mm) => mm.act(t),
                AgentKind::Value(_vt) => {
                    // ValueTrader needs an oracle observation (one normal from its own rng);
                    // the oracle advance + observe is wired with the oracle's own stream.
                    // Deferred: observe_price interaction — flagged for harness debug (Half-3).
                    Vec::new()
                }
                AgentKind::Momentum(mt) => mt.act(t).into_iter().collect(),
            }
        };
        for intent in intents {
            let oid = self.next_order_id();
            self.kernel.send_message(
                agent_id,
                self.exchange_id,
                MsgKind::OrderSubmit {
                    side_is_bid: intent.side == Side::Bid,
                    size: intent.size,
                    price: intent.price,
                },
                0,
            );
            // stash the order_id with the pending submit via a side map keyed by send order
            self.pending_order_ids.push(oid);
        }
    }

    /// Exchange handling of OrderSubmit. Trace semantics match extract_trace (trace.py):
    ///   * ORDER_SUBMITTED row carries the order's LIMIT price (emitted agent-side in
    ///     ABIDES; here at submit time).
    ///   * each fill is an ORDER_EXECUTED event; the FINAL execution per order_id becomes
    ///     ORDER_FILLED and earlier ones PARTIAL_FILL. Because that split is GLOBAL per
    ///     order_id (last occurrence in (t_ns) order), we emit all fills as ORDER_EXECUTED
    ///     with the fill price and RESOLVE the FILLED/PARTIAL label in a post-pass over the
    ///     whole trace (finalize_trace), exactly like extract_trace's drop_duplicates(keep=last).
    ///   * ORDER_ACCEPTED on a rested remainder (limit price).
    ///   * QUOTE_UPDATE comes from BEST_BID/BEST_ASK: after the book changes, emit the best
    ///     BID and best ASK (order_id = -1, price/size from the level). A post-pass dedups to
    ///     the final quote per (t_ns, side), like extract_trace.
    pub fn on_order_submit(&mut self, submitter_id: i64, side_is_bid: bool, size: i64, price: i64) {
        let order_id = self.pending_order_ids.remove(0);
        let t_ns = self.kernel.current_time;
        let side = if side_is_bid { Side::Bid } else { Side::Ask };
        let tside = if side_is_bid { TraceSide::Bid } else { TraceSide::Ask };
        // ORDER_SUBMITTED (limit price)
        self.trace.push(TraceRow {
            t_ns, agent_id: submitter_id as i32, msg_type: msg_type::ORDER_SUBMITTED,
            side: tside, price, size, order_id,
        });
        let incoming = IncomingOrder {
            order_id, agent_id: submitter_id, side, quantity: size,
            limit_price: price, tick_ns: t_ns, is_ptc: false, mr_preprocess: false,
        };
        let result = self.book.handle_limit_order(incoming);
        for fill in &result.fills {
            // All fills emitted as ORDER_EXECUTED (fill price); FILLED/PARTIAL resolved later.
            self.trace.push(TraceRow {
                t_ns, agent_id: submitter_id as i32, msg_type: msg_type::ORDER_FILLED, // placeholder, finalized in post-pass
                side: tside, price: fill.fill_price, size: fill.quantity, order_id,
            });
            self.kernel.send_message(self.exchange_id, fill.resting_agent_id,
                MsgKind::OrderExecuted { order_id: fill.resting_order_id, qty: fill.quantity, price: fill.fill_price }, 0);
            self.kernel.send_message(self.exchange_id, submitter_id,
                MsgKind::OrderExecuted { order_id, qty: fill.quantity, price: fill.fill_price }, 0);
        }
        if result.accepted_remainder {
            self.trace.push(TraceRow {
                t_ns, agent_id: submitter_id as i32, msg_type: msg_type::ORDER_ACCEPTED,
                side: tside, price, size, order_id,
            });
            self.kernel.send_message(self.exchange_id, submitter_id, MsgKind::OrderAccepted { order_id }, 0);
        }
        // QUOTE_UPDATE: best BID + best ASK after the change (BEST_BID/BEST_ASK -> QUOTE_UPDATE,
        // order_id = -1). Post-pass dedups to final per (t_ns, side).
        if let Some((bp, bq)) = self.book.best_bid() {
            self.trace.push(TraceRow { t_ns, agent_id: self.exchange_id as i32,
                msg_type: msg_type::QUOTE_UPDATE, side: TraceSide::Bid, price: bp, size: bq, order_id: -1 });
        }
        if let Some((ap, aq)) = self.book.best_ask() {
            self.trace.push(TraceRow { t_ns, agent_id: self.exchange_id as i32,
                msg_type: msg_type::QUOTE_UPDATE, side: TraceSide::Ask, price: ap, size: aq, order_id: -1 });
        }
    }

    /// Drive the event loop to completion, dispatching each event by kind.
    pub fn run_to_completion(&mut self) {
        self.prime_wakeups();
        while let Some(ev) = self.kernel.next_event() {
            let recipient = ev.recipient_id;
            match ev.kind {
                MsgKind::Wakeup => {
                    if recipient != self.exchange_id {
                        let idx = (recipient - 1) as usize; // agents are ids 1..N
                        self.on_wakeup(idx);
                    }
                }
                MsgKind::QuerySpread => self.on_query_spread(ev.sender_id),
                MsgKind::SpreadResponse { bid, ask } => {
                    let idx = (recipient - 1) as usize;
                    self.on_spread_response(idx, bid, ask);
                }
                MsgKind::OrderSubmit { side_is_bid, size, price } => {
                    self.on_order_submit(ev.sender_id, side_is_bid, size, price);
                }
                _ => {} // OrderExecuted/Accepted/Cancelled/QuoteUpdate: agent-side bookkeeping (Half-3)
            }
            self.kernel.finish_processing(recipient);
        }
    }
}


#[cfg(test)]
mod tests {
    // Half-1 wiring is exercised end-to-end in Half 2 against the full-trace harness;
    // in isolation it has no bit-exact oracle. Compilation + the component-seed test
    // (engine.rs) are the Half-1 gates.
}



// ---- Scenario -> Sim construction (config.py equivalent, SPEC-04) ----

use crate::engine::draw_component_seeds;
use crate::latency::LatencyKind;
use crate::oracle::{ScheduledJump, SymbolParams};

fn jget_f64(v: &serde_json::Value, path: &[&str], default: f64) -> f64 {
    let mut cur = v;
    for k in path {
        match cur.get(k) { Some(n) => cur = n, None => return default }
    }
    cur.as_f64().unwrap_or(default)
}
fn jget_i64(v: &serde_json::Value, path: &[&str], default: i64) -> i64 {
    let mut cur = v;
    for k in path {
        match cur.get(k) { Some(n) => cur = n, None => return default }
    }
    cur.as_i64().unwrap_or(default)
}

const DATE_NS: i64 = 1_612_483_200_000_000_000; // 2021-02-05 00:00:00 UTC (pd.to_datetime("20210205"))
const MKT_OPEN_OFFSET: i64 = 34_200_000_000_000; // 09:30:00 in ns
const STARTING_CASH: i64 = 10_000_000;

/// Build the Sim from a scenario Value + seed, mirroring config.py's construction and
/// its exact RNG draw order (global seed -> per-component randint_u32). NOTE: this is
/// the first wiring; bit-exactness is pursued against equivalence_harness.py. Several
/// fields (oracle kappa/megashock conversions) follow config.py's documented mappings.
pub fn build_sim_from_scenario(v: &serde_json::Value, seed: u64) -> Result<Sim, String> {
    // count agents
    let agent_cfgs = v.get("agent_configs").and_then(|a| a.as_array())
        .ok_or("scenario missing agent_configs array")?;
    let n_agents: usize = agent_cfgs.iter()
        .map(|c| c.get("count").and_then(|n| n.as_i64()).unwrap_or(1) as usize).sum();

    let seeds: ComponentSeeds = draw_component_seeds(seed, n_agents);

    let horizon_ns = jget_i64(v, &["horizon_ns"], 0);
    let mkt_open = DATE_NS + MKT_OPEN_OFFSET;
    let mkt_close = mkt_open + horizon_ns;
    let stop_time = mkt_close + 1_000_000_000; // + str_to_ns("1s")
    let ref_price = jget_i64(v, &["oracle_config", "params", "initial_price"], 100_000);

    // --- oracle (per config.py mappings) ---
    let kappa_s = jget_f64(v, &["oracle_config", "params", "kappa"], 0.0);
    let kappa = if kappa_s > 0.0 { kappa_s / 1e9 } else { 1.67e-16 };
    let jump_s = jget_f64(v, &["oracle_config", "params", "jump_intensity"], 0.0);
    let megashock_lambda_a = if jump_s > 0.0 { jump_s / 1e9 } else { 2.77778e-18 };
    let megashock_mean = { let m = jget_f64(v, &["oracle_config","params","jump_sigma"], 0.0); if m != 0.0 { m } else { 1000.0 } };
    let scheduled_jumps = {
        let sj = v.get("oracle_config").and_then(|o| o.get("params")).and_then(|p| p.get("scheduled_jump"));
        match sj {
            Some(j) if j.is_object() => vec![ScheduledJump {
                time_ns: mkt_open + j.get("time_ns").and_then(|n| n.as_i64()).unwrap_or(0),
                magnitude: j.get("magnitude").and_then(|n| n.as_i64()).unwrap_or(0),
                consumed: false,
            }],
            _ => Vec::new(),
        }
    };
    let sym = SymbolParams {
        r_bar: ref_price as f64,
        kappa,
        fund_vol: jget_f64(v, &["oracle_config","params","sigma"], 5e-5),
        megashock_lambda_a,
        megashock_mean,
        megashock_var: 50_000.0,
        scheduled_jumps,
    };
    // draw order: oracle rng first, uses GLOBAL for first megashock interval.
    let mut global = MT19937::new(seed);
    // advance global past the per-component randint draws already consumed conceptually by
    // draw_component_seeds — but config.py draws component seeds from the SAME global stream
    // THEN the oracle's __init__ draws its first-megashock exponential from global AFTER all
    // _draw_random_state calls. So re-seed a fresh global and replay the component-seed draws:
    for _ in 0..(2 + n_agents + 2) { let _ = global.randint_u32(); } // oracle,exchange,agents,latency,kernel
    let oracle = Oracle::new(mkt_open, sym, MT19937::new(seeds.oracle as u64), &mut global);

    // --- latency model ---
    let (lat_kind, lmean, lsigma, lmin, lmax, lalpha) = {
        let lc = v.get("latency_config");
        match lc {
            Some(c) => {
                let model = c.get("model").and_then(|m| m.as_str()).unwrap_or("deterministic");
                let p = c.get("params");
                let g = |k: &str, d: f64| p.and_then(|pp| pp.get(k)).and_then(|n| n.as_f64()).unwrap_or(d);
                let kind = match model {
                    "log_normal" => LatencyKind::LogNormal,
                    "uniform" => LatencyKind::Uniform,
                    "pareto" => LatencyKind::Pareto,
                    _ => LatencyKind::Deterministic,
                };
                (kind, g("mean_ns", 0.0), g("sigma", 0.0), g("min_ns", 0.0), g("max_ns", 1e12), g("alpha", 1.5))
            }
            None => (LatencyKind::Deterministic, 0.0, 0.0, 0.0, 1e12, 1.5),
        }
    };
    let latency = ScenarioLatencyModel::new(
        lat_kind, lmean, lsigma, lmin, lmax, lalpha, MT19937::new(seeds.latency as u64));

    // --- agents (construction order == id order 1..N; exchange is id 0) ---
    let mut agents: Vec<SimAgent> = Vec::with_capacity(n_agents);
    let mut next_id: i64 = 1;
    let mut agent_seed_i = 0usize;
    for cfg in agent_cfgs {
        let atype = cfg.get("agent_type").and_then(|t| t.as_str()).unwrap_or("");
        let count = cfg.get("count").and_then(|n| n.as_i64()).unwrap_or(1);
        let p = cfg.get("params");
        let gi = |k: &str, d: i64| p.and_then(|pp| pp.get(k)).and_then(|n| n.as_i64()).unwrap_or(d);
        let gf = |k: &str, d: f64| p.and_then(|pp| pp.get(k)).and_then(|n| n.as_f64()).unwrap_or(d);
        let interval_ns = if let Some(iv) = p.and_then(|pp| pp.get("rebalance_interval_ns")).and_then(|n| n.as_i64()) {
            iv
        } else {
            let hz = gf("arrival_rate_hz", 1.0);
            if hz > 0.0 { (1e9 / hz) as i64 } else { 1_000_000_000 }
        };
        for _ in 0..count {
            let rng = MT19937::new(seeds.agents[agent_seed_i] as u64);
            agent_seed_i += 1;
            let kind = match atype {
                "NoiseTrader" => AgentKind::Noise(NoiseTrader {
                    order_size_mean: gf("order_size_mean", 10.0),
                    order_size_std: gf("order_size_std", 2.0),
                    price_offset_ticks: gi("price_offset_ticks", 5),
                    reference_price: ref_price,
                }),
                "MarketMaker" => AgentKind::Market(MarketMaker {
                    spread_ticks: gi("spread_ticks", 2).max(2),
                    depth_levels: gi("depth_levels", 3).max(1),
                    size_per_level: gi("size_per_level", 10).max(1),
                    reference_price: ref_price,
                }),
                "ValueTrader" => AgentKind::Value(ValueTrader {
                    order_size_mean: gf("order_size_mean", 25.0),
                    threshold_ticks: gi("threshold_ticks", 2),
                    sigma_n: gf("sigma_n", 1000.0),
                }),
                "MomentumTrader" => AgentKind::Momentum(MomentumTrader {
                    order_size_mean: gf("order_size_mean", 15.0),
                    threshold_ticks: gi("threshold_ticks", 2),
                    lookback: gi("lookback", 5).max(1) as usize,
                    mid_history: Vec::new(),
                }),
                other => return Err(format!("unsupported agent_type: {other}")),
            };
            agents.push(SimAgent { id: next_id, kind, rng, interval_ns: interval_ns.max(1), mkt_open, mkt_close });
            next_id += 1;
        }
    }

    let default_computation_delay = 50;
    Ok(Sim::new(seeds, n_agents, mkt_open, mkt_close, stop_time, default_computation_delay, latency, oracle, agents))
}
