//! L4/L5 integration seam — the PyO3 `Engine` entry point (SPEC-04).
//!
//! This is the Python-facing object simulate.py will call under the TRACK3_CORE=rust
//! feature flag. `run` is intentionally a STUB that reports not-yet-wired: the dispatch
//! (exchange spread/order round-trip), L5 trace assembly, and config-order RNG seeding
//! are filled in against equivalence_harness.py one step at a time. Shipping a stub that
//! says "unimplemented" is correct here — a stub that silently returned a plausible-but-
//! wrong trace is exactly the Guardrail-#1 failure we refuse. simulate.py keeps the
//! Python baseline as the default engine (Guardrail #6) until this is bit-exact.

use pyo3::prelude::*;

/// The compiled simulation engine. Construction wires config -> components with the
/// exact RNG seeding order (oracle, exchange, agents 1..N, latency, kernel); `run`
/// drives the kernel loop and returns the trace columns. Not yet wired.
#[pyclass]
pub struct Engine {}

#[pymethods]
impl Engine {
    #[new]
    fn new() -> Self {
        Engine {}
    }

    /// run(scenario_json, seed) -> dict of trace columns + message_trace columns.
    /// STUB until the L4 dispatch + L5 trace are wired and bit-exact vs the reference.
    #[pyo3(signature = (scenario_json, seed))]
    fn run(&self, scenario_json: &str, seed: u64) -> PyResult<String> {
        let _ = (scenario_json, seed);
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "track3_core Engine.run: dispatch/trace wiring in progress (SPEC-04); \
             use the Python baseline engine until equivalence_harness.py is bit-exact",
        ))
    }
}
