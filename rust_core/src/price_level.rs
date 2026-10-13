//! Price level — SPEC-02 §"Attack order" step 1.
//!
//! A single price level in the book, holding resting orders in strict price-time
//! priority (arrival order within the level). The ONE behavioural change vs the
//! Python `price_level.py` is performance, not semantics: `total_quantity` is a
//! maintained O(1) running sum, instead of Python's generator-sum recomputed on
//! every read (6.5M calls / ~1.2s on the deep-book unit, see the Phase 1c budget).
//!
//! Order of orders within the level is FIFO by insertion, which mirrors the
//! reference's arrival-time priority — the kernel delivers messages in sim-time
//! order, so insertion order here == arrival-time order. We must never reorder.

/// A resting order, reduced to the plain fields matching needs. No deepcopy:
/// the Python book deep-copies full Order objects on enter/notify (731k calls on
/// the deep-book unit); here we carry values and let the Python driver reattach
/// any richer object identity it still needs during the staged migration.
#[derive(Clone, Debug)]
pub struct RestingOrder {
    pub order_id: i64,
    pub agent_id: i64,
    pub quantity: i64,
    pub price: i64,
    pub tick_ns: i64, // sim timestamp of arrival; priority key within the level
}

/// One price level. Orders are kept in arrival (FIFO) order; `total` is maintained
/// incrementally so reads are O(1).
#[derive(Debug, Default)]
pub struct PriceLevel {
    pub price: i64,
    orders: Vec<RestingOrder>,
    total: i64, // == sum(o.quantity) — maintained, never recomputed
}

impl PriceLevel {
    pub fn new(price: i64) -> Self {
        Self { price, orders: Vec::new(), total: 0 }
    }

    /// O(1) maintained depth — the hotspot fix.
    #[inline]
    pub fn total_quantity(&self) -> i64 {
        self.total
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }

    /// Add a resting order at the back of the FIFO queue (latest arrival = lowest
    /// priority at this price), updating the maintained total.
    pub fn push_back(&mut self, order: RestingOrder) {
        self.total += order.quantity;
        self.orders.push(order);
    }

    /// The highest-priority (oldest) resting order at this level, if any.
    pub fn peek(&self) -> Option<&RestingOrder> {
        self.orders.first()
    }

    /// Remove a resting order by id (a cancel), updating the maintained total.
    /// Returns (price, remaining_qty) of the removed order, or None if not present.
    pub fn remove_by_id(&mut self, order_id: i64) -> Option<(i64, i64)> {
        if let Some(pos) = self.orders.iter().position(|o| o.order_id == order_id) {
            let o = self.orders.remove(pos);
            self.total -= o.quantity;
            Some((o.price, o.quantity))
        } else {
            None
        }
    }

    /// Fill up to `want` shares against the oldest resting order, piecewise — the
    /// semantics of one `execute_order` step at this level. Returns the matched
    /// (quantity, price, resting_order_id, resting_agent_id) or None if empty.
    /// A fully-consumed resting order is removed; a partial fill leaves it in place
    /// at reduced size with UNCHANGED priority (SPEC-02 invariant #2).
    pub fn fill_oldest(&mut self, want: i64) -> Option<(i64, i64, i64, i64)> {
        let front = self.orders.first_mut()?;
        let traded = want.min(front.quantity);
        let (oid, aid, price) = (front.order_id, front.agent_id, front.price);
        front.quantity -= traded;
        self.total -= traded;
        if front.quantity == 0 {
            self.orders.remove(0);
        }
        Some((traded, price, oid, aid))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ro(id: i64, qty: i64, price: i64, t: i64) -> RestingOrder {
        RestingOrder { order_id: id, agent_id: 1, quantity: qty, price, tick_ns: t }
    }

    #[test]
    fn maintained_total_matches_naive_sum() {
        let mut lvl = PriceLevel::new(100);
        lvl.push_back(ro(1, 5, 100, 10));
        lvl.push_back(ro(2, 3, 100, 11));
        assert_eq!(lvl.total_quantity(), 8);
        // partial fill of the oldest
        let m = lvl.fill_oldest(2).unwrap();
        assert_eq!(m.0, 2); // traded
        assert_eq!(lvl.total_quantity(), 6);
        // oldest still present at reduced size, priority unchanged
        assert_eq!(lvl.peek().unwrap().order_id, 1);
        assert_eq!(lvl.peek().unwrap().quantity, 3);
    }

    #[test]
    fn full_fill_removes_and_advances_priority() {
        let mut lvl = PriceLevel::new(100);
        lvl.push_back(ro(1, 5, 100, 10));
        lvl.push_back(ro(2, 3, 100, 11));
        let m = lvl.fill_oldest(5).unwrap();
        assert_eq!(m.0, 5);
        assert_eq!(m.2, 1); // resting order_id consumed
        assert_eq!(lvl.peek().unwrap().order_id, 2); // priority advanced to next arrival
        assert_eq!(lvl.total_quantity(), 3);
    }
}
