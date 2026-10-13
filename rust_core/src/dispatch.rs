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

    /// Exchange handling of OrderSubmit: run the matching engine, emit trace rows for
    /// each fill (ORDER_FILLED/PARTIAL_FILL) and the resting remainder (ORDER_ACCEPTED),
    /// plus a QUOTE_UPDATE on book change. Notifications go back to the agents.
    pub fn on_order_submit(&mut self, submitter_id: i64, side_is_bid: bool, size: i64, price: i64) {
        let order_id = self.pending_order_ids.remove(0);
        let t_ns = self.kernel.current_time;
        let side = if side_is_bid { Side::Bid } else { Side::Ask };
        // ORDER_SUBMITTED trace row
        self.trace.push(TraceRow {
            t_ns, agent_id: submitter_id as i32, msg_type: msg_type::ORDER_SUBMITTED,
            side: if side_is_bid { TraceSide::Bid } else { TraceSide::Ask },
            price, size, order_id,
        });
        let incoming = IncomingOrder {
            order_id, agent_id: submitter_id, side, quantity: size,
            limit_price: price, tick_ns: t_ns, is_ptc: false, mr_preprocess: false,
        };
        let result = self.book.handle_limit_order(incoming);
        for fill in &result.fills {
            let mt = if fill.quantity == size { msg_type::ORDER_FILLED } else { msg_type::PARTIAL_FILL };
            self.trace.push(TraceRow {
                t_ns, agent_id: submitter_id as i32, msg_type: mt,
                side: if side_is_bid { TraceSide::Bid } else { TraceSide::Ask },
                price: fill.fill_price, size: fill.quantity, order_id,
            });
            // notify both parties (OrderExecuted)
            self.kernel.send_message(self.exchange_id, fill.resting_agent_id,
                MsgKind::OrderExecuted { order_id: fill.resting_order_id, qty: fill.quantity, price: fill.fill_price }, 0);
            self.kernel.send_message(self.exchange_id, submitter_id,
                MsgKind::OrderExecuted { order_id, qty: fill.quantity, price: fill.fill_price }, 0);
        }
        if result.accepted_remainder {
            self.trace.push(TraceRow {
                t_ns, agent_id: submitter_id as i32, msg_type: msg_type::ORDER_ACCEPTED,
                side: if side_is_bid { TraceSide::Bid } else { TraceSide::Ask },
                price, size, order_id,
            });
            self.kernel.send_message(self.exchange_id, submitter_id, MsgKind::OrderAccepted { order_id }, 0);
        }
        // QUOTE_UPDATE on book change
        let t = self.touch();
        self.trace.push(TraceRow {
            t_ns, agent_id: self.exchange_id as i32, msg_type: msg_type::QUOTE_UPDATE,
            side: TraceSide::None, price: t.bid, size: 0, order_id: 0,
        });
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
