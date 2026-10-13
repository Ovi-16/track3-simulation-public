//! L2 — latency model (SPEC-03). Ports `ScenarioLatencyModel.get_latency` from
//! baselines/abides_fork/config.py bit-for-bit, drawing from an L0 MT19937 so the
//! per-message latency sequence matches NumPy exactly. Latency decides message
//! arrival order, so any drift here reorders events and zeros Tier A.
//!
//! Rules (from config.py):
//!   self-message (sender == recipient) -> 0, drawing NO randomness.
//!   log_normal : lognormal(mu=ln(mean_ns), sigma)
//!   uniform    : uniform(min_ns, max_ns)
//!   pareto     : (min_ns>0 ? min_ns : 1.0) * (1 + pareto(alpha))
//!   else       : constant mean_ns
//!   then: int(round(clip(value, min_ns, max_ns)))  [banker's rounding]

use crate::np_random::MT19937;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatencyKind {
    LogNormal,
    Uniform,
    Pareto,
    Deterministic,
}

pub struct ScenarioLatencyModel {
    kind: LatencyKind,
    mean_ns: f64,
    sigma: f64,
    min_ns: f64,
    max_ns: f64,
    alpha: f64,
    mu: f64, // ln(mean_ns) if mean_ns>0 else 0
    rng: MT19937,
}

impl ScenarioLatencyModel {
    pub fn new(
        kind: LatencyKind,
        mean_ns: f64,
        sigma: f64,
        min_ns: f64,
        max_ns: f64,
        alpha: f64,
        rng: MT19937,
    ) -> Self {
        let mu = if mean_ns > 0.0 { mean_ns.ln() } else { 0.0 };
        Self { kind, mean_ns, sigma, min_ns, max_ns, alpha, mu, rng }
    }

    /// Mirror of get_latency. `int(round(clip(value, min, max)))` with Python's
    /// banker's rounding — reuse banker_round_div by scaling the clipped value to a
    /// rational (round-to-nearest-even on .5). Since round() here is round-to-integer,
    /// we compute floor + even-aware half handling via banker_round_div(num=round_input).
    pub fn get_latency(&mut self, sender_id: i64, recipient_id: i64) -> i64 {
        if sender_id == recipient_id {
            return 0; // draws NO randomness — must not consume an RNG value
        }
        let value = match self.kind {
            LatencyKind::LogNormal => self.rng.lognormal(self.mu, self.sigma),
            LatencyKind::Uniform => self.rng.uniform(self.min_ns, self.max_ns),
            LatencyKind::Pareto => {
                let base = if self.min_ns > 0.0 { self.min_ns } else { 1.0 };
                base * (1.0 + self.rng.pareto(self.alpha))
            }
            LatencyKind::Deterministic => self.mean_ns,
        };
        let clipped = value.clamp(self.min_ns, self.max_ns);
        python_round_to_i64(clipped)
    }
}

/// Python `round(float)` — round half to EVEN (banker's rounding) at the integer
/// boundary, matching `int(round(x))` in config.py's get_latency.
pub fn python_round_to_i64(x: f64) -> i64 {
    let floor = x.floor();
    let diff = x - floor;
    let f = floor as i64;
    if diff < 0.5 {
        f
    } else if diff > 0.5 {
        f + 1
    } else if f % 2 == 0 {
        f // exactly .5 -> round to even
    } else {
        f + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::np_random::MT19937;

    #[test]
    fn log_normal_matches_python_seed0() {
        // golden from ScenarioLatencyModel(log_normal mean=1000 sigma=0.5 clip[100,5000]) seed0
        let mut m = ScenarioLatencyModel::new(
            LatencyKind::LogNormal, 1000.0, 0.5, 100.0, 5000.0, 1.5, MT19937::new(0));
        let expected = [2416i64, 1221, 1631, 3066, 2544];
        for (i, &e) in expected.iter().enumerate() {
            assert_eq!(m.get_latency(0, 1), e, "log_normal {i}");
        }
    }

    #[test]
    fn uniform_matches_python_seed0() {
        let mut m = ScenarioLatencyModel::new(
            LatencyKind::Uniform, 0.0, 0.0, 100.0, 5000.0, 1.5, MT19937::new(0));
        let expected = [2789i64, 3604, 3054, 2770, 2176];
        for (i, &e) in expected.iter().enumerate() {
            assert_eq!(m.get_latency(0, 1), e, "uniform {i}");
        }
    }

    #[test]
    fn pareto_matches_python_seed0() {
        let mut m = ScenarioLatencyModel::new(
            LatencyKind::Pareto, 0.0, 0.0, 100.0, 50000.0, 1.5, MT19937::new(0));
        let expected = [170i64, 231, 185, 169, 144];
        for (i, &e) in expected.iter().enumerate() {
            assert_eq!(m.get_latency(0, 1), e, "pareto {i}");
        }
    }

    #[test]
    fn self_message_zero_no_draw() {
        let mut m = ScenarioLatencyModel::new(
            LatencyKind::Uniform, 0.0, 0.0, 100.0, 5000.0, 1.5, MT19937::new(0));
        assert_eq!(m.get_latency(2, 2), 0);
        // first real draw still matches the seed-0 start (self call consumed nothing)
        assert_eq!(m.get_latency(0, 1), 2789);
    }
}
