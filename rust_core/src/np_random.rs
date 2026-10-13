//! L0 — NumPy-exact RNG parity (SPEC-03). THE gating layer for Option C.
//!
//! Reproduces `numpy.random.RandomState` bit-for-bit: the legacy MT19937 generator,
//! NumPy's `seed(int)` path (which uses `init_by_array` with the seed as a 1-element
//! key array), and the draw primitives every ABIDES stochastic quantity is built on.
//! If this does not match NumPy to the bit, latencies/order-sizes drift, messages
//! reorder, and Tier-A zeros — so this crate is cross-checked against NumPy for a
//! range of seeds BEFORE any higher layer is built (see tests + the Python dump).
//!
//! Implemented here (enough to seed per-component states and drive uniforms):
//!   - MT19937 state + init_genrand + init_by_array (NumPy's exact constants)
//!   - next_u32 (tempered output)
//!   - random_sample() : float64 in [0,1) via NumPy's 53-bit construction
//!     (a = u32>>5 (27 bits), b = u32>>6 (26 bits); (a*2^26 + b) / 2^53)
//!   - randint_u32_below(2^32) as used by _draw_random_state's seed draw
//!
//! Distribution samplers (gauss/lognormal/pareto/normal) land in a follow-up on top
//! of these primitives, each matched to NumPy's algorithm (ziggurat gauss, etc.).

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

pub struct MT19937 {
    mt: [u32; N],
    mti: usize,
}

impl MT19937 {
    /// NumPy init_genrand (Knuth 2002): the base seeding routine. NumPy's legacy
    /// `RandomState.seed(scalar_int)` uses THIS directly (not init_by_array), verified
    /// against numpy 1.26.4: init_genrand(0) -> first tempered word 2357136044.
    pub fn init_genrand(seed: u32) -> Self {
        let mut mt = [0u32; N];
        mt[0] = seed;
        for i in 1..N {
            mt[i] = 1_812_433_253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { mt, mti: N }
    }

    /// NumPy init_by_array — the path `RandomState.seed(int)` takes with key=[seed].
    pub fn seed_by_array(key: &[u32]) -> Self {
        let mut s = Self::init_genrand(19_650_218);
        let mt = &mut s.mt;
        let mut i = 1usize;
        let mut j = 0usize;
        let mut k = N.max(key.len());
        while k > 0 {
            mt[i] = (mt[i]
                ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_664_525)))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        let mut k = N - 1;
        while k > 0 {
            mt[i] = (mt[i]
                ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_566_083_941)))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        mt[0] = 0x8000_0000;
        s.mti = N;
        s
    }

    /// NumPy `RandomState.seed(int)`: a scalar integer seed is fed to init_genrand
    /// directly (confirmed against numpy 1.26.4). `seed_by_array` remains for the
    /// array-key path (not used by ABIDES's scalar seeding).
    pub fn new(seed: u64) -> Self {
        Self::init_genrand((seed & 0xffff_ffff) as u32)
    }

    /// Tempered 32-bit output (generate_next in NumPy's randomkit).
    pub fn next_u32(&mut self) -> u32 {
        if self.mti >= N {
            // generate N words at once
            for i in 0..N {
                let y = (self.mt[i] & UPPER_MASK) | (self.mt[(i + 1) % N] & LOWER_MASK);
                let mut next = self.mt[(i + M) % N] ^ (y >> 1);
                if y & 1 != 0 {
                    next ^= MATRIX_A;
                }
                self.mt[i] = next;
            }
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        // tempering
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// NumPy random_sample(): double in [0,1) with 53 bits of entropy.
    pub fn random_sample(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64; // 27 bits
        let b = (self.next_u32() >> 6) as f64; // 26 bits
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0 // (a*2^26 + b)/2^53
    }

    /// `randint(0, 2**32)` as used by _draw_random_state() to seed per-component states.
    /// NumPy's rk_random for a full-range 32-bit value is simply the next tempered word.
    pub fn randint_u32(&mut self) -> u32 {
        self.next_u32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden values captured from NumPy 1.26.4 (the pinned baseline stack):
    //   np.random.seed(0); np.random.randint(0, 2**32, size=5, dtype='uint64')
    //   -> [2357136044, 2546248239, 3071714933, 3626093760, 2588848963]
    #[test]
    fn randint_u32_matches_numpy_seed0() {
        let mut r = MT19937::new(0);
        let expected = [2357136044u32, 2546248239, 3071714933, 3626093760, 2588848963];
        for (i, &e) in expected.iter().enumerate() {
            let got = r.randint_u32();
            assert_eq!(got, e, "draw {i}: got {got} != numpy {e}");
        }
    }

    // np.random.seed(42); np.random.random_sample(3)
    //   -> [0.3745401188473625, 0.9507143064099162, 0.7319939418114051]
    #[test]
    fn random_sample_matches_numpy_seed42() {
        let mut r = MT19937::new(42);
        let expected = [0.3745401188473625, 0.9507143064099162, 0.7319939418114051];
        for (i, &e) in expected.iter().enumerate() {
            let got = r.random_sample();
            assert!((got - e).abs() < 1e-15, "sample {i}: got {got} != numpy {e}");
        }
    }
}
