//! L5 — trace assembly (SPEC-01 §1.2/§1.3, SPEC-04). Builds the exact 7-column
//! trace the grader reads, row-by-row as the kernel emits events. Written while
//! host memory was critical (no build possible yet) — compile + bit-exact-verify
//! against the reference via equivalence_harness.py once headroom returns.
//!
//! trace.parquet columns (order fixed): t_ns i64, agent_id i32, msg_type str,
//! side str (BID/ASK/empty), price i64, size i64, order_id i64.
//! msg_type is one of the 7 EXACT strings — any deviation fails the schema gate.

/// The 7 canonical msg_type strings. No other value may appear in the trace.
pub mod msg_type {
    pub const ORDER_SUBMITTED: &str = "ORDER_SUBMITTED";
    pub const ORDER_ACCEPTED: &str = "ORDER_ACCEPTED";
    pub const ORDER_FILLED: &str = "ORDER_FILLED";
    pub const PARTIAL_FILL: &str = "PARTIAL_FILL";
    pub const ORDER_CANCELLED: &str = "ORDER_CANCELLED";
    pub const ORDER_REPLACED: &str = "ORDER_REPLACED";
    pub const QUOTE_UPDATE: &str = "QUOTE_UPDATE";
}

/// Side label as it appears in the trace. `None` serializes to null/empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceSide {
    Bid,
    Ask,
    None,
}

impl TraceSide {
    pub fn as_str(self) -> &'static str {
        match self {
            TraceSide::Bid => "BID",
            TraceSide::Ask => "ASK",
            TraceSide::None => "",
        }
    }
}

/// One trace row. Columns are appended to the builder in emission order; the builder
/// hands column-major vectors to pyarrow at the end (zero Python-object round-trip —
/// SPEC-01 §5.7, cuts the serialization tax measured in Phase 1c).
#[derive(Clone, Debug)]
pub struct TraceRow {
    pub t_ns: i64,
    pub agent_id: i32,
    pub msg_type: &'static str,
    pub side: TraceSide,
    pub price: i64,
    pub size: i64,
    pub order_id: i64,
}

/// Column-major trace accumulator. push() appends one row; into_columns() yields the
/// 7 parallel vectors for pyarrow Table construction on the Python side.
#[derive(Default)]
pub struct TraceBuilder {
    pub t_ns: Vec<i64>,
    pub agent_id: Vec<i32>,
    pub msg_type: Vec<&'static str>,
    pub side: Vec<&'static str>,
    pub price: Vec<i64>,
    pub size: Vec<i64>,
    pub order_id: Vec<i64>,
}

impl TraceBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn push(&mut self, row: TraceRow) {
        self.t_ns.push(row.t_ns);
        self.agent_id.push(row.agent_id);
        self.msg_type.push(row.msg_type);
        self.side.push(row.side.as_str());
        self.price.push(row.price);
        self.size.push(row.size);
        self.order_id.push(row.order_id);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.t_ns.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.t_ns.is_empty()
    }

    /// Post-pass matching extract_trace (trace.py) before serialization:
    ///  1. FILLED/PARTIAL: every fill was emitted as ORDER_FILLED (placeholder); the
    ///     FINAL fill per order_id stays ORDER_FILLED, earlier ones become PARTIAL_FILL
    ///     (extract_trace: last ORDER_EXECUTED per order_id = FILLED).
    ///  2. QUOTE_UPDATE dedup: keep only the LAST quote per (t_ns, side).
    ///  3. stable sort by (t_ns, order_id).
    /// Operates on a row view since columns are parallel vectors.
    pub fn finalize(&mut self) {
        let n = self.len();
        if n == 0 {
            return;
        }
        // Rebuild as rows for the transforms, then re-split (clarity over micro-perf;
        // the hot path is the sim, not this once-per-run pass).
        #[derive(Clone)]
        struct R { t: i64, a: i32, m: &'static str, s: &'static str, p: i64, sz: i64, oid: i64 }
        let mut rows: Vec<R> = (0..n).map(|i| R {
            t: self.t_ns[i], a: self.agent_id[i], m: self.msg_type[i], s: self.side[i],
            p: self.price[i], sz: self.size[i], oid: self.order_id[i],
        }).collect();

        // 1. FILLED/PARTIAL: find the LAST ORDER_FILLED index per order_id; others -> PARTIAL.
        use std::collections::HashMap;
        let mut last_fill: HashMap<i64, usize> = HashMap::new();
        for (i, r) in rows.iter().enumerate() {
            if r.m == msg_type::ORDER_FILLED {
                last_fill.insert(r.oid, i);
            }
        }
        for (i, r) in rows.iter_mut().enumerate() {
            if r.m == msg_type::ORDER_FILLED {
                if last_fill.get(&r.oid) != Some(&i) {
                    r.m = msg_type::PARTIAL_FILL;
                }
            }
        }

        // 2. QUOTE_UPDATE dedup: keep LAST per (t_ns, side). Mark earlier quote dupes for drop.
        let mut last_quote: HashMap<(i64, &'static str), usize> = HashMap::new();
        for (i, r) in rows.iter().enumerate() {
            if r.m == msg_type::QUOTE_UPDATE {
                last_quote.insert((r.t, r.s), i);
            }
        }
        let mut kept: Vec<R> = Vec::with_capacity(rows.len());
        for (i, r) in rows.into_iter().enumerate() {
            if r.m == msg_type::QUOTE_UPDATE
                && last_quote.get(&(r.t, r.s)) != Some(&i)
            {
                continue; // superseded quote at same (t_ns, side)
            }
            kept.push(r);
        }

        // 3. stable sort by (t_ns, order_id).
        kept.sort_by(|x, y| x.t.cmp(&y.t).then_with(|| x.oid.cmp(&y.oid)));

        // re-split into columns
        self.t_ns.clear(); self.agent_id.clear(); self.msg_type.clear();
        self.side.clear(); self.price.clear(); self.size.clear(); self.order_id.clear();
        for r in kept {
            self.t_ns.push(r.t); self.agent_id.push(r.a); self.msg_type.push(r.m);
            self.side.push(r.s); self.price.push(r.p); self.size.push(r.sz); self.order_id.push(r.oid);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_accumulates_columns_in_order() {
        let mut b = TraceBuilder::new();
        b.push(TraceRow {
            t_ns: 100, agent_id: 1, msg_type: msg_type::ORDER_FILLED,
            side: TraceSide::Bid, price: 99, size: 5, order_id: 42,
        });
        b.push(TraceRow {
            t_ns: 101, agent_id: 2, msg_type: msg_type::QUOTE_UPDATE,
            side: TraceSide::None, price: 100, size: 0, order_id: 0,
        });
        assert_eq!(b.len(), 2);
        assert_eq!(b.t_ns, vec![100, 101]);
        assert_eq!(b.msg_type, vec!["ORDER_FILLED", "QUOTE_UPDATE"]);
        assert_eq!(b.side, vec!["BID", ""]);
        assert_eq!(b.order_id, vec![42, 0]);
    }

    #[test]
    fn finalize_resolves_fills_dedups_quotes_and_sorts() {
        let mut b = TraceBuilder::new();
        // two fills on order 7 (first PARTIAL, last FILLED) at t=100 and t=100
        b.push(TraceRow { t_ns: 100, agent_id: 1, msg_type: msg_type::ORDER_FILLED,
            side: TraceSide::Bid, price: 99, size: 3, order_id: 7 });
        b.push(TraceRow { t_ns: 100, agent_id: 1, msg_type: msg_type::ORDER_FILLED,
            side: TraceSide::Bid, price: 99, size: 2, order_id: 7 });
        // two quotes same (t,side) — keep last
        b.push(TraceRow { t_ns: 100, agent_id: 0, msg_type: msg_type::QUOTE_UPDATE,
            side: TraceSide::Bid, price: 98, size: 10, order_id: -1 });
        b.push(TraceRow { t_ns: 100, agent_id: 0, msg_type: msg_type::QUOTE_UPDATE,
            side: TraceSide::Bid, price: 97, size: 5, order_id: -1 });
        b.finalize();
        // order_id -1 sorts before 7; one quote (last, price 97) survives
        let quotes: Vec<_> = (0..b.len()).filter(|&i| b.msg_type[i] == "QUOTE_UPDATE").collect();
        assert_eq!(quotes.len(), 1);
        assert_eq!(b.price[quotes[0]], 97);
        // fills: first FILLED became PARTIAL_FILL, last stays ORDER_FILLED
        let fills: Vec<_> = (0..b.len())
            .filter(|&i| b.msg_type[i] == "PARTIAL_FILL" || b.msg_type[i] == "ORDER_FILLED").collect();
        assert_eq!(fills.len(), 2);
        assert_eq!(b.msg_type[fills[0]], "PARTIAL_FILL");
        assert_eq!(b.msg_type[fills[1]], "ORDER_FILLED");
    }
}
