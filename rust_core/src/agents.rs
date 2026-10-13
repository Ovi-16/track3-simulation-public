//! L3 — agents (SPEC-03). Ports baselines/abides_fork/agents.py: the per-wakeup
//! `act()` decision of each agent type, reproducing the EXACT RNG draw order from
//! each agent's own MT19937. The kernel wiring (when they wake, message plumbing)
//! is L4; this module is the pure decision: given the current book touch (+ oracle
//! for ValueTrader), return the order(s) to place.
//!
//! Draw-order contract (must match the Python line-for-line):
//!   NoiseTrader.act : normal(mean,std) -> size ; randint(0,2) -> buy ;
//!                     randint(0, offset_ticks+1) -> offset.           (3 draws)
//!   MarketMaker.act : NO RNG (deterministic ladder).
//!   ValueTrader.act : one normal via oracle.observe_price(agent_rng, sigma_n).
//!   MomentumTrader  : NO RNG (deterministic on mid history).

use crate::latency::python_round_to_i64;
use crate::np_random::MT19937;
use crate::order_book::Side;

/// One order an agent wants placed this wakeup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderIntent {
    pub side: Side,
    pub size: i64,
    pub price: i64,
}

/// The book touch an agent sees (best bid/ask prices; 0 == side empty, mirroring
/// ABIDES get_known_bid_ask returning None->falsy).
#[derive(Clone, Copy, Debug)]
pub struct Touch {
    pub bid: i64,
    pub ask: i64,
}

impl Touch {
    #[inline]
    fn has_bid(&self) -> bool { self.bid != 0 }
    #[inline]
    fn has_ask(&self) -> bool { self.ask != 0 }
}

pub struct NoiseTrader {
    pub order_size_mean: f64,
    pub order_size_std: f64,
    pub price_offset_ticks: i64,
    pub reference_price: i64,
}

impl NoiseTrader {
    /// 3 draws in exact order: normal(size), randint(0,2)->buy, randint(0,off+1)->offset.
    pub fn act(&self, t: Touch, rng: &mut MT19937) -> OrderIntent {
        let raw = rng.normal(self.order_size_mean, self.order_size_std);
        let size = python_round_to_i64(raw).max(1);
        let buy = rng.randint_below(2) != 0;
        let offset = rng.randint_below((self.price_offset_ticks + 1) as u32) as i64;
        if buy {
            let anchor = if t.has_ask() { t.ask } else if t.has_bid() { t.bid } else { self.reference_price };
            OrderIntent { side: Side::Bid, size, price: anchor + offset }
        } else {
            let anchor = if t.has_bid() { t.bid } else if t.has_ask() { t.ask } else { self.reference_price };
            OrderIntent { side: Side::Ask, size, price: anchor - offset }
        }
    }
}

pub struct MarketMaker {
    pub spread_ticks: i64,
    pub depth_levels: i64,
    pub size_per_level: i64,
    pub reference_price: i64,
}

impl MarketMaker {
    /// Deterministic ladder — NO RNG. Returns cancel-all + the repost ladder; the
    /// caller (L4) issues the cancels first (as the Python does cancel_all_orders()).
    pub fn act(&self, t: Touch) -> Vec<OrderIntent> {
        let mid = if t.has_bid() && t.has_ask() {
            (t.bid + t.ask) / 2 // Python int(//) floor division on positives
        } else {
            self.reference_price
        };
        let half = self.spread_ticks / 2;
        let mut out = Vec::with_capacity((self.depth_levels * 2) as usize);
        for lvl in 0..self.depth_levels {
            out.push(OrderIntent { side: Side::Bid, size: self.size_per_level, price: mid - half - lvl });
            out.push(OrderIntent { side: Side::Ask, size: self.size_per_level, price: mid + half + lvl });
        }
        out
    }
}

pub struct ValueTrader {
    pub order_size_mean: f64,
    pub threshold_ticks: i64,
    pub sigma_n: f64,
}

impl ValueTrader {
    /// `fundamental` is the already-observed noisy price (the oracle draw happens in
    /// L4 wiring using THIS agent's rng, since observe_price takes the agent's rng).
    /// Returns at most one marketable order. Mirrors the mid/threshold logic exactly.
    pub fn act(&self, t: Touch, fundamental: i64) -> Option<OrderIntent> {
        let mid: f64 = if t.has_bid() && t.has_ask() {
            (t.bid + t.ask) as f64 / 2.0
        } else if t.has_bid() {
            t.bid as f64
        } else if t.has_ask() {
            t.ask as f64
        } else {
            return None;
        };
        let size = python_round_to_i64(self.order_size_mean).max(1);
        if mid < (fundamental - self.threshold_ticks) as f64 && t.has_ask() {
            Some(OrderIntent { side: Side::Bid, size, price: t.ask })
        } else if mid > (fundamental + self.threshold_ticks) as f64 && t.has_bid() {
            Some(OrderIntent { side: Side::Ask, size, price: t.bid })
        } else {
            None
        }
    }
}

pub struct MomentumTrader {
    pub order_size_mean: f64,
    pub threshold_ticks: i64,
    pub lookback: usize,
    pub mid_history: Vec<f64>,
}

impl MomentumTrader {
    /// Deterministic — NO RNG. Updates its own mid history and returns at most one
    /// order. Mirrors the pop-at->lookback+1 / need-> lookback+1-samples logic.
    pub fn act(&mut self, t: Touch) -> Option<OrderIntent> {
        let mid: f64 = if t.has_bid() && t.has_ask() {
            (t.bid + t.ask) as f64 / 2.0
        } else if t.has_bid() {
            t.bid as f64
        } else if t.has_ask() {
            t.ask as f64
        } else {
            return None;
        };
        self.mid_history.push(mid);
        if self.mid_history.len() > self.lookback + 1 {
            self.mid_history.remove(0);
        }
        if self.mid_history.len() <= self.lookback {
            return None;
        }
        let past = self.mid_history[0];
        let size = python_round_to_i64(self.order_size_mean).max(1);
        if mid > past + self.threshold_ticks as f64 && t.has_ask() {
            Some(OrderIntent { side: Side::Bid, size, price: t.ask })
        } else if mid < past - self.threshold_ticks as f64 && t.has_bid() {
            Some(OrderIntent { side: Side::Ask, size, price: t.bid })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_trader_draw_order_matches_python_seed0() {
        // golden: act0 size=14 buy=True offset=3; act1 11/T/1; act2 7/T/0
        let nt = NoiseTrader { order_size_mean: 10.0, order_size_std: 2.0,
            price_offset_ticks: 5, reference_price: 100_000 };
        let mut rng = MT19937::new(0);
        let empty = Touch { bid: 0, ask: 0 };
        let o0 = nt.act(empty, &mut rng);
        assert_eq!(o0.size, 14);
        assert_eq!(o0.side, Side::Bid); // buy=True
        assert_eq!(o0.price, 100_000 + 3); // ref + offset 3 (empty book)
        let o1 = nt.act(empty, &mut rng);
        assert_eq!((o1.size, o1.side, o1.price), (11, Side::Bid, 100_001));
        let o2 = nt.act(empty, &mut rng);
        assert_eq!((o2.size, o2.side, o2.price), (7, Side::Bid, 100_000));
    }

    #[test]
    fn market_maker_ladder_deterministic() {
        let mm = MarketMaker { spread_ticks: 2, depth_levels: 2, size_per_level: 10,
            reference_price: 100_000 };
        let orders = mm.act(Touch { bid: 99_998, ask: 100_002 });
        // mid=100000, half=1 -> bids 99999,99998 ; asks 100001,100002
        assert_eq!(orders.len(), 4);
        assert_eq!((orders[0].side, orders[0].price), (Side::Bid, 99_999));
        assert_eq!((orders[1].side, orders[1].price), (Side::Ask, 100_001));
        assert_eq!((orders[2].side, orders[2].price), (Side::Bid, 99_998));
        assert_eq!((orders[3].side, orders[3].price), (Side::Ask, 100_002));
    }

    #[test]
    fn value_trader_buys_when_underpriced() {
        let vt = ValueTrader { order_size_mean: 25.0, threshold_ticks: 2, sigma_n: 1000.0 };
        // mid=100000, fundamental=100010 -> underpriced -> buy crossing ask
        let o = vt.act(Touch { bid: 99_999, ask: 100_001 }, 100_010).unwrap();
        assert_eq!((o.side, o.size, o.price), (Side::Bid, 25, 100_001));
        // fundamental near mid -> no order
        assert!(vt.act(Touch { bid: 99_999, ask: 100_001 }, 100_000).is_none());
    }

    #[test]
    fn momentum_trader_needs_history_then_follows_trend() {
        let mut mt = MomentumTrader { order_size_mean: 15.0, threshold_ticks: 2,
            lookback: 2, mid_history: Vec::new() };
        // not enough history for first lookback+1 samples
        assert!(mt.act(Touch { bid: 100, ask: 102 }).is_none()); // mid 101
        assert!(mt.act(Touch { bid: 100, ask: 102 }).is_none()); // mid 101
        // third sample: history [101,101,110?]; rising -> buy
        let o = mt.act(Touch { bid: 109, ask: 111 }); // mid 110 > past 101 + 2
        assert_eq!(o.unwrap().side, Side::Bid);
    }
}
