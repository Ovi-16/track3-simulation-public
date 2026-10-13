//! L2 — SparseMeanRevertingOracle (SPEC-03). Ports the OU + megashock fundamental
//! process from abides_markets/oracles/sparse_mean_reverting_oracle.py bit-for-bit.
//!
//! CRITICAL draw-order detail: the oracle mixes TWO RNG streams —
//!   * the GLOBAL np.random state for megashock inter-arrival times
//!     (`np.random.exponential(1/lambda)`), and
//!   * the symbol's OWN random_state for the OU normal() and the sign-flip randint(2).
//! Both must be driven in the exact interleaving below or the fundamental (hence every
//! ValueTrader decision) drifts. Reactive scheduled jumps are deterministic (no RNG).

use crate::np_random::MT19937;

#[derive(Clone, Debug)]
pub struct ScheduledJump {
    pub time_ns: i64,
    pub magnitude: i64,
    pub consumed: bool,
}

pub struct SymbolParams {
    pub r_bar: f64,
    pub kappa: f64,          // per-ns mean reversion
    pub fund_vol: f64,       // theta
    pub megashock_lambda_a: f64,
    pub megashock_mean: f64,
    pub megashock_var: f64,
    pub scheduled_jumps: Vec<ScheduledJump>,
}

pub struct Oracle {
    mkt_open: i64,
    p: SymbolParams,
    sym_rng: MT19937,  // the symbol's random_state (normal + randint sign)
    // current fundamental (time, value)
    cur_t: i64,
    cur_v: i64,
    // last scheduled megashock
    ms_time: i64,
    ms_value: f64,
}

impl Oracle {
    /// Mirrors __init__: first megashock time via GLOBAL exponential, value via
    /// sym_rng.normal, sign via sym_rng.randint(2). `global_rng` is the shared
    /// np.random state (seeded once per run), passed in because other components
    /// also draw from it in config-build order.
    pub fn new(mkt_open: i64, p: SymbolParams, mut sym_rng: MT19937, global_rng: &mut MT19937) -> Self {
        let ms_time_delta = global_rng.exponential(1.0 / p.megashock_lambda_a);
        let ms_time = mkt_open + ms_time_delta as i64;
        let mut msv = sym_rng.normal(p.megashock_mean, p.megashock_var.sqrt());
        if sym_rng.randint_below(2) != 0 {
            msv = -msv;
        }
        let r_bar = p.r_bar;
        Self {
            mkt_open, p, sym_rng,
            cur_t: mkt_open, cur_v: r_bar as i64,
            ms_time, ms_value: msv,
        }
    }

    /// Mirrors compute_fundamental_at_timestamp: one OU normal draw from sym_rng,
    /// + v_adj, floor at 0, round to int, then apply any due deterministic jump.
    fn compute_at(&mut self, ts: i64, v_adj: f64, pt: i64, pv: i64) -> i64 {
        let d = (ts - pt) as f64;
        let mu = self.p.r_bar;
        let gamma = self.p.kappa;
        let theta = self.p.fund_vol;
        let loc = mu + (pv as f64 - mu) * (-gamma * d).exp();
        let scale = (((theta * theta) / (2.0 * gamma)) * (1.0 - (-2.0 * gamma * d).exp())).sqrt();
        let mut v = self.sym_rng.normal(loc, scale);
        v += v_adj;
        if v < 0.0 {
            v = 0.0;
        }
        let mut vi = crate::latency::python_round_to_i64(v);
        for jump in self.p.scheduled_jumps.iter_mut() {
            if !jump.consumed && ts >= jump.time_ns {
                vi = (vi + jump.magnitude).max(0);
                jump.consumed = true;
            }
        }
        self.cur_t = ts;
        self.cur_v = vi;
        vi
    }

    /// Mirrors advance_fundamental_value_series: apply all megashocks before
    /// current_time (each: OU to mst with v_adj=msv, then GLOBAL exponential for next
    /// interval + sym_rng normal/randint sign), then OU to current_time.
    pub fn advance(&mut self, current_time: i64, global_rng: &mut MT19937) -> i64 {
        if current_time <= self.cur_t {
            return self.cur_v;
        }
        let (mut pt, mut pv) = (self.cur_t, self.cur_v);
        let mut mst = self.ms_time;
        let mut msv = self.ms_value;
        while mst < current_time {
            let v = self.compute_at(mst, msv, pt, pv);
            pt = mst;
            pv = v;
            let delta = global_rng.exponential(1.0 / self.p.megashock_lambda_a) as i64;
            mst = pt + delta;
            let mut nv = self.sym_rng.normal(self.p.megashock_mean, self.p.megashock_var.sqrt());
            if self.sym_rng.randint_below(2) != 0 {
                nv = -nv;
            }
            msv = nv;
            self.ms_time = mst;
            self.ms_value = msv;
        }
        self.compute_at(current_time, 0.0, pt, pv)
    }

    #[inline]
    pub fn current_value(&self) -> i64 {
        self.cur_v
    }
}

#[cfg(test)]
mod tests {
    // Oracle parity is validated end-to-end at L4 (full-trace bit-exact), where its
    // draws interleave with agents on the real event stream. Unit-level draw-sequence
    // checks against a Python oracle dump are added when L4 wiring exists; the OU/
    // megashock arithmetic here mirrors the source line-for-line.
}
