//! track3_core — SPEC-01 Phase 2 compiled core (PyO3 extension module).
//!
//! Staged migration (SPEC-02): the order book lands first behind a feature flag in
//! the adapter, proven bit-exact per unit via equivalence_harness.py before the
//! event queue follows. This lib.rs currently exposes the building blocks and a
//! self-check; the OrderBook pyclass is added as price_level/order_book stabilize.

mod price_level;

use pyo3::prelude::*;

/// Cheap import-time self-check so a built wheel can prove it loaded and the
/// maintained-total invariant holds, before any heavier wiring exists.
#[pyfunction]
fn self_check() -> PyResult<String> {
    use price_level::{PriceLevel, RestingOrder};
    let mut lvl = PriceLevel::new(100);
    lvl.push_back(RestingOrder { order_id: 1, agent_id: 1, quantity: 5, price: 100, tick_ns: 0 });
    lvl.push_back(RestingOrder { order_id: 2, agent_id: 2, quantity: 3, price: 100, tick_ns: 1 });
    let (traded, _price, oid, _aid) = lvl.fill_oldest(5).unwrap();
    if traded == 5 && oid == 1 && lvl.total_quantity() == 3 && lvl.peek().unwrap().order_id == 2 {
        Ok(format!("track3_core ok: maintained total={}", lvl.total_quantity()))
    } else {
        Err(pyo3::exceptions::PyAssertionError::new_err("price_level invariant failed"))
    }
}

#[pymodule]
fn track3_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(self_check, m)?)?;
    Ok(())
}
