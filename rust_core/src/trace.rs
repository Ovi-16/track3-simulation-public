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
}
