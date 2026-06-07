//! Symbol probability distributions.

use std::collections::HashMap;

/// A frequency distribution over symbols.
#[derive(Debug, Clone)]
pub struct FrequencyDistribution {
    counts: HashMap<u8, usize>,
    total: usize,
}

impl FrequencyDistribution {
    /// Create from a byte slice.
    pub fn from_bytes(data: &[u8]) -> Self {
        let mut counts = HashMap::new();
        let mut total = 0;
        for &b in data {
            *counts.entry(b).or_insert(0) += 1;
            total += 1;
        }
        Self { counts, total }
    }

    /// Get the count for a symbol.
    pub fn count(&self, symbol: u8) -> usize {
        self.counts.get(&symbol).copied().unwrap_or(0)
    }

    /// Get the total count.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Get the number of unique symbols.
    pub fn alphabet_size(&self) -> usize {
        self.counts.len()
    }

    /// Convert to a probability distribution.
    pub fn to_probabilities(&self) -> ProbabilityDistribution {
        let probs: HashMap<u8, f64> = self
            .counts
            .iter()
            .map(|(&sym, &count)| {
                let p = if self.total > 0 {
                    count as f64 / self.total as f64
                } else {
                    0.0
                };
                (sym, p)
            })
            .collect();
        ProbabilityDistribution {
            probs,
            symbols: self.counts.keys().copied().collect(),
        }
    }

    /// Get all (symbol, count) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&u8, &usize)> {
        self.counts.iter()
    }

    /// Get the most probable symbol.
    pub fn mode(&self) -> Option<u8> {
        self.counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&sym, _)| sym)
    }
}

/// A probability distribution over symbols.
#[derive(Debug, Clone)]
pub struct ProbabilityDistribution {
    probs: HashMap<u8, f64>,
    symbols: Vec<u8>,
}

impl ProbabilityDistribution {
    /// Create a uniform distribution over the given symbols.
    pub fn uniform(symbols: &[u8]) -> Self {
        let p = 1.0 / symbols.len() as f64;
        let probs = symbols.iter().map(|&sym| (sym, p)).collect();
        Self {
            probs,
            symbols: symbols.to_vec(),
        }
    }

    /// Get the probability of a symbol.
    pub fn probability(&self, symbol: u8) -> f64 {
        self.probs.get(&symbol).copied().unwrap_or(0.0)
    }

    /// Get all symbols in the distribution.
    pub fn symbols(&self) -> &[u8] {
        &self.symbols
    }

    /// Verify the distribution sums to 1.0 (within tolerance).
    pub fn is_valid(&self, tolerance: f64) -> bool {
        let sum: f64 = self.probs.values().sum();
        (sum - 1.0).abs() < tolerance
    }

    /// Get the cumulative distribution function (CDF).
    pub fn cdf(&self) -> Vec<(u8, f64)> {
        let mut pairs: Vec<_> = self
            .symbols
            .iter()
            .map(|&sym| (sym, self.probability(sym)))
            .collect();
        pairs.sort_by_key(|&(sym, _)| sym);

        let mut cdf = Vec::new();
        let mut cum = 0.0;
        for (sym, p) in &pairs {
            cum += p;
            cdf.push((*sym, cum));
        }
        cdf
    }

    /// Normalize a raw count distribution to probabilities.
    pub fn from_counts(counts: &[(u8, usize)]) -> Self {
        let total: usize = counts.iter().map(|(_, c)| *c).sum();
        let probs: HashMap<u8, f64> = counts
            .iter()
            .map(|&(sym, count)| {
                let p = if total > 0 {
                    count as f64 / total as f64
                } else {
                    0.0
                };
                (sym, p)
            })
            .collect();
        Self {
            probs,
            symbols: counts.iter().map(|&(sym, _)| sym).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let freq = FrequencyDistribution::from_bytes(&[]);
        assert_eq!(freq.total(), 0);
        assert_eq!(freq.alphabet_size(), 0);
    }

    #[test]
    fn test_frequency_from_bytes() {
        let freq = FrequencyDistribution::from_bytes(b"aab");
        assert_eq!(freq.count(b'a'), 2);
        assert_eq!(freq.count(b'b'), 1);
        assert_eq!(freq.total(), 3);
    }

    #[test]
    fn test_to_probabilities() {
        let freq = FrequencyDistribution::from_bytes(b"aabb");
        let probs = freq.to_probabilities();
        assert!((probs.probability(b'a') - 0.5).abs() < f64::EPSILON);
        assert!(probs.is_valid(1e-10));
    }

    #[test]
    fn test_uniform() {
        let probs = ProbabilityDistribution::uniform(&[1, 2, 3, 4]);
        assert!(probs.is_valid(1e-10));
        for sym in &[1u8, 2, 3, 4] {
            assert!((probs.probability(*sym) - 0.25).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_cdf() {
        let freq = FrequencyDistribution::from_bytes(b"aabbcc");
        let probs = freq.to_probabilities();
        let cdf = probs.cdf();
        // Last CDF value should be close to 1.0
        if let Some(&(_, last)) = cdf.last() {
            assert!((last - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_mode() {
        let freq = FrequencyDistribution::from_bytes(b"aaabbc");
        assert_eq!(freq.mode(), Some(b'a'));
    }

    #[test]
    fn test_from_counts() {
        let probs = ProbabilityDistribution::from_counts(&[(1, 3), (2, 1)]);
        assert!((probs.probability(1) - 0.75).abs() < f64::EPSILON);
    }
}
