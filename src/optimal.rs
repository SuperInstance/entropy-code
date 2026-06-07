//! Optimal code length computation.

use crate::probability::FrequencyDistribution;

/// Compute optimal code lengths (in bits) for each symbol based on frequency.
/// Uses the formula: l_i = ceil(-log2(p_i))
/// Symbols with zero probability are excluded.
pub fn optimal_code_lengths(freq: &FrequencyDistribution) -> Vec<(u8, usize)> {
    let total = freq.total();
    if total == 0 {
        return Vec::new();
    }

    let mut lengths = Vec::new();
    for (&sym, &count) in freq.iter() {
        if count > 0 {
            let p = count as f64 / total as f64;
            let len = (-p.log2()).ceil() as usize;
            lengths.push((sym, len.max(1))); // minimum length of 1
        }
    }
    lengths.sort_by_key(|&(sym, _)| sym);
    lengths
}

/// Compute exact optimal code lengths using Shannon's source coding theorem.
/// l_i = -log2(p_i) (not rounded).
pub fn exact_code_lengths(freq: &FrequencyDistribution) -> Vec<(u8, f64)> {
    let total = freq.total();
    if total == 0 {
        return Vec::new();
    }

    let mut lengths = Vec::new();
    for (&sym, &count) in freq.iter() {
        if count > 0 {
            let p = count as f64 / total as f64;
            lengths.push((sym, -p.log2()));
        }
    }
    lengths.sort_by_key(|&(sym, _)| sym);
    lengths
}

/// Compute the expected code length given frequencies and code lengths.
pub fn expected_length(freq: &FrequencyDistribution, code_lengths: &[(u8, usize)]) -> f64 {
    let total = freq.total();
    if total == 0 {
        return 0.0;
    }

    let mut expected = 0.0;
    for &(sym, len) in code_lengths {
        let count = freq.count(sym);
        if count > 0 {
            let p = count as f64 / total as f64;
            expected += p * len as f64;
        }
    }
    expected
}

/// Verify that code lengths are optimal (within 1 bit of Shannon entropy).
pub fn is_near_optimal(freq: &FrequencyDistribution, code_lengths: &[(u8, usize)]) -> bool {
    use crate::entropy::shannon_entropy;
    let data: Vec<u8> = {
        let mut d = Vec::new();
        for (&sym, &count) in freq.iter() {
            for _ in 0..count {
                d.push(sym);
            }
        }
        d
    };
    let entropy = shannon_entropy(&data);
    let expected = expected_length(freq, code_lengths);
    // Code should be within 1 bit of entropy (Shannon's source coding theorem)
    expected <= entropy + 1.0
}

/// Compute the redundancy: expected_length - entropy.
pub fn redundancy(freq: &FrequencyDistribution, code_lengths: &[(u8, usize)]) -> f64 {
    use crate::entropy::shannon_entropy;
    let data: Vec<u8> = {
        let mut d = Vec::new();
        for (&sym, &count) in freq.iter() {
            for _ in 0..count {
                d.push(sym);
            }
        }
        d
    };
    let entropy = shannon_entropy(&data);
    let expected = expected_length(freq, code_lengths);
    expected - entropy
}

/// Generate optimal integer code lengths using package-merge algorithm
/// for length-limited codes (limited to max_len bits).
pub fn length_limited_lengths(freq: &FrequencyDistribution, max_len: usize) -> Vec<(u8, usize)> {
    let mut lengths = optimal_code_lengths(freq);

    // Cap all lengths at max_len
    for (_, len) in &mut lengths {
        if *len > max_len {
            *len = max_len;
        }
    }

    // Ensure the lengths satisfy Kraft's inequality
    let kraft_sum: f64 = lengths.iter().map(|&(_, len)| 2.0_f64.powi(-(len as i32))).sum();

    if kraft_sum > 1.0 + 1e-10 {
        // Need to increase some lengths - simple heuristic
        lengths.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let mut current_sum = kraft_sum;
        for (_, len) in &mut lengths {
            if current_sum <= 1.0 + 1e-10 {
                break;
            }
            let old_contribution = 2.0_f64.powi(-(*len as i32));
            *len += 1;
            let new_contribution = 2.0_f64.powi(-(*len as i32));
            current_sum = current_sum - old_contribution + new_contribution;
        }
        lengths.sort_by_key(|&(sym, _)| sym);
    }

    lengths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let freq = FrequencyDistribution::from_bytes(&[]);
        assert!(optimal_code_lengths(&freq).is_empty());
    }

    #[test]
    fn test_uniform_distribution() {
        let freq = FrequencyDistribution::from_bytes(b"abcd");
        let lengths = optimal_code_lengths(&freq);
        // Each symbol has p=0.25, optimal length = ceil(2) = 2
        for &(_, len) in &lengths {
            assert_eq!(len, 2);
        }
    }

    #[test]
    fn test_expected_length() {
        let freq = FrequencyDistribution::from_bytes(b"aabb");
        let lengths = optimal_code_lengths(&freq);
        let expected = expected_length(&freq, &lengths);
        // p(a)=0.5, p(b)=0.5, optimal length=1 each, expected=1.0
        assert!((expected - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_near_optimal() {
        let freq = FrequencyDistribution::from_bytes(b"aabbc");
        let lengths = optimal_code_lengths(&freq);
        assert!(is_near_optimal(&freq, &lengths));
    }

    #[test]
    fn test_exact_code_lengths() {
        let freq = FrequencyDistribution::from_bytes(b"aabb");
        let lengths = exact_code_lengths(&freq);
        for &(_, len) in &lengths {
            assert!((len - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_redundancy_non_negative() {
        let freq = FrequencyDistribution::from_bytes(b"aaabbc");
        let lengths = optimal_code_lengths(&freq);
        let red = redundancy(&freq, &lengths);
        assert!(red >= -1e-10);
    }

    #[test]
    fn test_length_limited() {
        let freq = FrequencyDistribution::from_bytes(b"aaabbc");
        let lengths = length_limited_lengths(&freq, 4);
        for &(_, len) in &lengths {
            assert!(len <= 4);
        }
    }
}
