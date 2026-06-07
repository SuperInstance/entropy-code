//! Shannon entropy and information content calculations.

pub fn shannon_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut counts = [0usize; 256];
    for &b in data {
        counts[b as usize] += 1;
    }

    let total = data.len() as f64;
    let mut entropy = 0.0;

    for &count in &counts {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
    }

    entropy
}

/// Calculate the information content (self-information) of a symbol
/// given its probability.
pub fn information_content(probability: f64) -> f64 {
    if probability <= 0.0 {
        return f64::INFINITY;
    }
    -probability.log2()
}

/// Calculate the entropy of a probability distribution.
pub fn entropy_of_distribution(probs: &[f64]) -> f64 {
    let mut entropy = 0.0;
    for &p in probs {
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Calculate cross-entropy between two distributions p and q.
/// H(p, q) = -sum(p(x) * log2(q(x)))
pub fn cross_entropy(p: &[f64], q: &[f64]) -> f64 {
    let mut h = 0.0;
    for (pp, qq) in p.iter().zip(q.iter()) {
        if *pp > 0.0 && *qq > 0.0 {
            h -= pp * qq.log2();
        }
    }
    h
}

/// Calculate KL divergence (relative entropy) from p to q.
/// D_KL(p || q) = sum(p(x) * log2(p(x) / q(x)))
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    let mut div = 0.0;
    for (pp, qq) in p.iter().zip(q.iter()) {
        if *pp > 0.0 && *qq > 0.0 {
            div += pp * (pp / qq).log2();
        }
    }
    div
}

/// Calculate joint entropy of two byte sequences (paired).
pub fn joint_entropy(data1: &[u8], data2: &[u8]) -> f64 {
    let min_len = data1.len().min(data2.len());
    if min_len == 0 {
        return 0.0;
    }

    let mut counts = std::collections::HashMap::new();
    for i in 0..min_len {
        *counts.entry((data1[i], data2[i])).or_insert(0usize) += 1;
    }

    let total = min_len as f64;
    let mut entropy = 0.0;
    for &count in counts.values() {
        let p = count as f64 / total;
        entropy -= p * p.log2();
    }
    entropy
}

/// Maximum entropy for a given alphabet size.
pub fn max_entropy(alphabet_size: usize) -> f64 {
    if alphabet_size <= 1 {
        return 0.0;
    }
    (alphabet_size as f64).log2()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert_eq!(shannon_entropy(&[]), 0.0);
    }

    #[test]
    fn test_single_byte() {
        assert_eq!(shannon_entropy(&[42; 100]), 0.0);
    }

    #[test]
    fn test_uniform_256() {
        let data: Vec<u8> = (0..=255).cycle().take(2560).collect();
        let h = shannon_entropy(&data);
        assert!((h - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_uniform_4() {
        let data = vec![0, 1, 2, 3];
        let h = shannon_entropy(&data);
        assert!((h - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_information_content() {
        assert!((information_content(0.5) - 1.0).abs() < 1e-10);
        assert!((information_content(0.25) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_entropy() {
        assert!((max_entropy(256) - 8.0).abs() < 1e-10);
        assert!((max_entropy(2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_entropy_same() {
        let p = vec![0.5, 0.5];
        let h = cross_entropy(&p, &p);
        assert!((h - shannon_entropy_of_2()).abs() < 1e-10);
    }

    fn shannon_entropy_of_2() -> f64 {
        1.0
    }

    #[test]
    fn test_kl_divergence_same() {
        let p = vec![0.5, 0.5];
        let div = kl_divergence(&p, &p);
        assert!(div.abs() < 1e-10);
    }

    #[test]
    fn test_entropy_of_distribution() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let h = entropy_of_distribution(&probs);
        assert!((h - 2.0).abs() < 1e-10);
    }
}
