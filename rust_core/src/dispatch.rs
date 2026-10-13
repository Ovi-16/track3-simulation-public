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
            // mkt_close retained via kernel.stop_time / agents' own copy
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

#[cfg(test)]
mod tests {
    // Half-1 wiring is exercised end-to-end in Half 2 against the full-trace harness;
    // in isolation it has no bit-exact oracle. Compilation + the component-seed test
    // (engine.rs) are the Half-1 gates.
}
