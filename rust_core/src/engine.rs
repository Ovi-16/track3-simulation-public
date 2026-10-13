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

use crate::np_random::MT19937;

/// Per-component RNG seeds, drawn from the GLOBAL MT19937 in config.py's exact order:
/// global.seed(scenario_seed) then one randint_u32() each for oracle, exchange, every
/// agent (construction order), latency, kernel (Guardrail #4). Reproducing this order
/// bit-for-bit is prerequisite to every downstream draw matching.
pub struct ComponentSeeds {
    pub oracle: u32,
    pub exchange: u32,
    pub agents: Vec<u32>,
    pub latency: u32,
    pub kernel: u32,
}

/// Draw the per-component seeds for `n_agents` scenario agents (excludes the exchange,
/// which is agent id 0 and gets its own seed before the scenario agents).
pub fn draw_component_seeds(scenario_seed: u64, n_agents: usize) -> ComponentSeeds {
    let mut g = MT19937::new(scenario_seed);
    let oracle = g.randint_u32();
    let exchange = g.randint_u32();
    let agents = (0..n_agents).map(|_| g.randint_u32()).collect();
    let latency = g.randint_u32();
    let kernel = g.randint_u32();
    ComponentSeeds { oracle, exchange, agents, latency, kernel }
}


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
    fn run(&self, scenario_json: &str, seed: u64) -> PyResult<TraceColumns> {
        run_sim(scenario_json, seed)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }
}

/// The 7 trace columns returned to Python, assembled into a pyarrow Table on the
/// Python side (simulate.py). Parallel vectors == TraceBuilder columns post-finalize.
#[pyclass]
pub struct TraceColumns {
    #[pyo3(get)]
    pub t_ns: Vec<i64>,
    #[pyo3(get)]
    pub agent_id: Vec<i32>,
    #[pyo3(get)]
    pub msg_type: Vec<String>,
    #[pyo3(get)]
    pub side: Vec<String>,
    #[pyo3(get)]
    pub price: Vec<i64>,
    #[pyo3(get)]
    pub size: Vec<i64>,
    #[pyo3(get)]
    pub order_id: Vec<i64>,
}

/// Parse the scenario, build the sim with config-order seeds, run, finalize, return
/// columns. serde_json::Value keeps this resilient to scenario schema variance; only
/// the fields the baseline consumes are read (SPEC-04). Returns Err(string) on a parse
/// problem so Engine.run surfaces a clean Python exception rather than panicking.
fn run_sim(scenario_json: &str, seed: u64) -> Result<TraceColumns, String> {
    use crate::dispatch::{build_sim_from_scenario};
    let v: serde_json::Value =
        serde_json::from_str(scenario_json).map_err(|e| format!("scenario JSON parse: {e}"))?;
    let mut sim = build_sim_from_scenario(&v, seed)?;
    sim.run_to_completion();
    sim.trace.finalize();
    let tb = &sim.trace;
    Ok(TraceColumns {
        t_ns: tb.t_ns.clone(),
        agent_id: tb.agent_id.clone(),
        msg_type: tb.msg_type.iter().map(|s| s.to_string()).collect(),
        side: tb.side.iter().map(|s| s.to_string()).collect(),
        price: tb.price.clone(),
        size: tb.size.clone(),
        order_id: tb.order_id.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_seeds_match_python_config_order_seed42() {
        // golden from config _draw_random_state() order, seed 42 (2 agents):
        // oracle 1608637542, exchange 3421126067, agent1 4083286876,
        // agent2 787846414, latency 3143890026, kernel 3348747335
        let s = draw_component_seeds(42, 2);
        assert_eq!(s.oracle, 1_608_637_542);
        assert_eq!(s.exchange, 3_421_126_067);
        assert_eq!(s.agents, vec![4_083_286_876, 787_846_414]);
        assert_eq!(s.latency, 3_143_890_026);
        assert_eq!(s.kernel, 3_348_747_335);
    }
}
