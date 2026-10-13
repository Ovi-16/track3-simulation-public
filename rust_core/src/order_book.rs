//! Order book matching engine — SPEC-02 §"Attack order" steps 2-3.
//!
//! Mirrors ABIDES `order_book.py` handle_limit_order + execute_order BIT-FOR-BIT
//! for the plain central-limit-order-book path that the public units actually
//! exercise. Verified against the units: `price_to_comply` and `MR_preprocess`
//! tags appear in ZERO scenarios and are never set by the fork's agents, so those
//! branches are dead code here — but we GUARD them with a hard error rather than
//! silently approximate (Guardrail #1: never silently diverge). If a sealed unit
//! ever hits them, the run fails loudly and visibly instead of producing a wrong,
//! admissible-looking trace.
//!
//! Side ordering (as in ABIDES): bids sorted DESCENDING by price, asks ASCENDING,
//! so `book[0]` is always the best level. Within a level, FIFO == arrival order.

use crate::price_level::{PriceLevel, RestingOrder};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    Bid,
    Ask,
}

/// An incoming order to process. `tag`/`is_ptc` carried only to detect and REFUSE
/// the unsupported paths, never to execute them.
#[derive(Clone, Debug)]
pub struct IncomingOrder {
    pub order_id: i64,
    pub agent_id: i64,
    pub side: Side,
    pub quantity: i64,
    pub limit_price: i64,
    pub tick_ns: i64,
    pub is_ptc: bool,
    pub mr_preprocess: bool,
}

/// One fill, in the exact terms the trace/notifications need.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fill {
    /// shares traded in this piece
    pub quantity: i64,
    /// fill price == the RESTING order's limit_price (ABIDES: "the price being advertised")
    pub fill_price: i64,
    /// the resting (passive) order consumed
    pub resting_order_id: i64,
    pub resting_agent_id: i64,
    /// the incoming (aggressive) order
    pub incoming_order_id: i64,
    pub incoming_agent_id: i64,
}

/// Result of handling one limit order: the ordered fills, and whether the
/// remainder rested on the book (→ OrderAccepted) .
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HandleResult {
    pub fills: Vec<Fill>,
    pub accepted_remainder: bool,
    /// avg trade price over the fills, banker's-rounded — matches
    /// `int(round(trade_price / trade_qty))` (Python round = round-half-to-even).
    pub last_trade_avg: Option<i64>,
}

#[derive(Debug, Default)]
pub struct OrderBook {
    bids: Vec<PriceLevel>, // descending by price; [0] = highest bid = best
    asks: Vec<PriceLevel>, // ascending by price;  [0] = lowest ask  = best
}

impl OrderBook {
    pub fn new() -> Self {
        Self { bids: Vec::new(), asks: Vec::new() }
    }

    #[inline]
    pub fn best_bid(&self) -> Option<(i64, i64)> {
        self.bids.first().map(|l| (l.price, l.total_quantity()))
    }

    #[inline]
    pub fn best_ask(&self) -> Option<(i64, i64)> {
        self.asks.first().map(|l| (l.price, l.total_quantity()))
    }

    /// Does the best opposite level cross with this incoming order?
    fn best_opposite_matches(&self, order: &IncomingOrder) -> bool {
        match order.side {
            // a bid matches the best ask iff bid price >= ask price
            Side::Bid => self.asks.first().map_or(false, |a| order.limit_price >= a.price),
            // an ask matches the best bid iff ask price <= bid price
            Side::Ask => self.bids.first().map_or(false, |b| order.limit_price <= b.price),
        }
    }

    /// One execute_order step: match against the single best opposite price level,
    /// FIFO within it. Returns one Fill or None if no crossing match.
    fn execute_once(&mut self, order: &mut IncomingOrder) -> Option<Fill> {
        // Guard the dead paths rather than mis-handle them.
        if order.is_ptc || order.mr_preprocess {
            panic!(
                "track3_core: price_to_comply / MR_preprocess path hit (order {}); \
                 unsupported in the compiled core — fall back to the Python book",
                order.order_id
            );
        }
        if !self.best_opposite_matches(order) {
            return None;
        }
        let book = match order.side {
            Side::Bid => &mut self.asks,
            Side::Ask => &mut self.bids,
        };
        let level = book.first_mut()?;
        let fill_price = level.price; // resting order's advertised price
        let (traded, _p, roid, raid) = level.fill_oldest(order.quantity)?;
        if level.is_empty() {
            book.remove(0);
        }
        order.quantity -= traded;
        Some(Fill {
            quantity: traded,
            fill_price,
            resting_order_id: roid,
            resting_agent_id: raid,
            incoming_order_id: order.order_id,
            incoming_agent_id: order.agent_id,
        })
    }

    /// Insert a resting order on its own side, preserving the sort invariant
    /// (bids desc, asks asc) and FIFO-within-price.
    fn enter_order(&mut self, order: &IncomingOrder) {
        let ro = RestingOrder {
            order_id: order.order_id,
            agent_id: order.agent_id,
            quantity: order.quantity,
            price: order.limit_price,
            tick_ns: order.tick_ns,
        };
        let book = match order.side {
            Side::Bid => &mut self.bids,
            Side::Ask => &mut self.asks,
        };
        // find existing level, else insert a new one at the sorted position
        if let Some(idx) = book.iter().position(|l| l.price == order.limit_price) {
            book[idx].push_back(ro);
            return;
        }
        let pos = match order.side {
            // bids descending: first level with price < new price
            Side::Bid => book.iter().position(|l| l.price < order.limit_price),
            // asks ascending: first level with price > new price
            Side::Ask => book.iter().position(|l| l.price > order.limit_price),
        }
        .unwrap_or(book.len());
        let mut lvl = PriceLevel::new(order.limit_price);
        lvl.push_back(ro);
        book.insert(pos, lvl);
    }

    /// Full handle_limit_order: loop execute_once consuming the best price first,
    /// then rest the remainder. Returns ordered fills + last-trade average.
    pub fn handle_limit_order(&mut self, mut order: IncomingOrder) -> HandleResult {
        // Validation mirrors ABIDES: quantity>0, price>=0 (ints by type). An invalid
        // order is discarded with no events — represented here as an empty result.
        if order.quantity <= 0 || order.limit_price < 0 {
            return HandleResult::default();
        }
        let mut fills = Vec::new();
        let mut accepted_remainder = false;
        loop {
            match self.execute_once(&mut order) {
                Some(f) => {
                    fills.push(f);
                    if order.quantity <= 0 {
                        break;
                    }
                }
                None => {
                    self.enter_order(&order);
                    accepted_remainder = true;
                    break;
                }
            }
        }
        let last_trade_avg = if fills.is_empty() {
            None
        } else {
            let trade_qty: i64 = fills.iter().map(|f| f.quantity).sum();
            let trade_price: i64 = fills.iter().map(|f| f.fill_price * f.quantity).sum();
            Some(banker_round_div(trade_price, trade_qty))
        };
        HandleResult { fills, accepted_remainder, last_trade_avg }
    }
}

/// Reproduce Python's `int(round(trade_price / trade_qty))` — round-half-to-even
/// (banker's rounding) on the exact rational trade_price/trade_qty. C-style
/// round-half-up would drift prices and break Tier A (SPEC-02 invariant #5).
pub fn banker_round_div(num: i64, den: i64) -> i64 {
    debug_assert!(den > 0);
    let q = num.div_euclid(den);
    let r = num.rem_euclid(den);
    let twice = 2 * r;
    if twice < den {
        q
    } else if twice > den {
        q + 1
    } else {
        // exactly halfway → round to even
        if q % 2 == 0 {
            q
        } else {
            q + 1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bid(id: i64, qty: i64, px: i64, t: i64) -> IncomingOrder {
        IncomingOrder { order_id: id, agent_id: id, side: Side::Bid, quantity: qty,
            limit_price: px, tick_ns: t, is_ptc: false, mr_preprocess: false }
    }
    fn ask(id: i64, qty: i64, px: i64, t: i64) -> IncomingOrder {
        IncomingOrder { order_id: id, agent_id: id, side: Side::Ask, quantity: qty,
            limit_price: px, tick_ns: t, is_ptc: false, mr_preprocess: false }
    }

    #[test]
    fn resting_then_cross_fills_at_resting_price() {
        let mut ob = OrderBook::new();
        // resting ask: 10 @ 100
        let r = ob.handle_limit_order(ask(1, 10, 100, 1));
        assert!(r.fills.is_empty() && r.accepted_remainder);
        // incoming bid 4 @ 101 crosses → fills 4 at the RESTING price 100
        let r = ob.handle_limit_order(bid(2, 4, 101, 2));
        assert_eq!(r.fills.len(), 1);
        assert_eq!(r.fills[0].quantity, 4);
        assert_eq!(r.fills[0].fill_price, 100); // resting price, not 101
        assert_eq!(r.fills[0].resting_order_id, 1);
        assert_eq!(r.last_trade_avg, Some(100));
        // 6 remain on the ask side
        assert_eq!(ob.best_ask(), Some((100, 6)));
    }

    #[test]
    fn price_time_priority_fifo_within_level() {
        let mut ob = OrderBook::new();
        ob.handle_limit_order(ask(1, 3, 100, 1)); // older
        ob.handle_limit_order(ask(2, 3, 100, 2)); // newer, same price
        let r = ob.handle_limit_order(bid(3, 4, 100, 3));
        // consumes order 1 fully (3) then order 2 partially (1), in FIFO order
        assert_eq!(r.fills.len(), 2);
        assert_eq!((r.fills[0].resting_order_id, r.fills[0].quantity), (1, 3));
        assert_eq!((r.fills[1].resting_order_id, r.fills[1].quantity), (2, 1));
        assert_eq!(ob.best_ask(), Some((100, 2))); // order 2 remainder rests
    }

    #[test]
    fn bid_levels_sorted_best_first() {
        let mut ob = OrderBook::new();
        ob.handle_limit_order(bid(1, 1, 99, 1));
        ob.handle_limit_order(bid(2, 1, 101, 2));
        ob.handle_limit_order(bid(3, 1, 100, 3));
        assert_eq!(ob.best_bid(), Some((101, 1))); // highest bid is best
    }

    #[test]
    fn bankers_rounding_matches_python_round() {
        // Python: round(2.5)=2, round(3.5)=4, round(5/2)=2, round(7/2)=4
        assert_eq!(banker_round_div(5, 2), 2);
        assert_eq!(banker_round_div(7, 2), 4);
        assert_eq!(banker_round_div(1, 2), 0); // round(0.5)=0
        assert_eq!(banker_round_div(3, 2), 2); // round(1.5)=2
        assert_eq!(banker_round_div(10, 4), 2); // 2.5 -> 2
        assert_eq!(banker_round_div(201, 2), 100); // 100.5 -> 100
    }

    #[test]
    fn no_cross_rests_both_sides() {
        let mut ob = OrderBook::new();
        ob.handle_limit_order(bid(1, 5, 99, 1));
        let r = ob.handle_limit_order(ask(2, 5, 101, 2));
        assert!(r.fills.is_empty() && r.accepted_remainder);
        assert_eq!(ob.best_bid(), Some((99, 5)));
        assert_eq!(ob.best_ask(), Some((101, 5)));
    }
}
