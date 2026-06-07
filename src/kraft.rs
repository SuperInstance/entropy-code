//! Kraft inequality verification for prefix-free codes.

pub fn verify(code_lengths: &[usize]) -> bool {
    kraft_sum(code_lengths) <= 1.0 + 1e-10
}

/// Calculate the Kraft sum: sum(2^(-l_i)) for binary codes.
pub fn kraft_sum(code_lengths: &[usize]) -> f64 {
    code_lengths
        .iter()
        .map(|&len| 2.0_f64.powi(-(len as i32)))
        .sum()
}

/// Verify Kraft's inequality for a D-ary alphabet.
pub fn verify_dary(code_lengths: &[usize], d: usize) -> bool {
    kraft_sum_dary(code_lengths, d) <= 1.0 + 1e-10
}

/// Calculate Kraft sum for a D-ary alphabet.
pub fn kraft_sum_dary(code_lengths: &[usize], d: usize) -> f64 {
    let d = d as f64;
    code_lengths
        .iter()
        .map(|&len| d.powi(-(len as i32)))
        .sum()
}

/// Given code lengths, determine if they represent a complete code
/// (Kraft sum == 1, meaning no unused code space).
pub fn is_complete(code_lengths: &[usize]) -> bool {
    let sum = kraft_sum(code_lengths);
    (sum - 1.0).abs() < 1e-10
}

/// Given a number of symbols, compute the code lengths for a complete
/// binary code (all lengths equal, rounded up).
pub fn equal_length_codes(num_symbols: usize) -> Vec<usize> {
    if num_symbols == 0 {
        return Vec::new();
    }
    let len = (num_symbols as f64).log2().ceil() as usize;
    vec![len; num_symbols]
}

/// Given code lengths, construct a canonical prefix-free code
/// (if the lengths satisfy Kraft's inequality).
/// Returns None if Kraft's inequality is violated.
pub fn construct_canonical(code_lengths: &[(u8, usize)]) -> Option<Vec<(u8, Vec<u8>)>> {
    if !verify(&code_lengths.iter().map(|(_, l)| *l).collect::<Vec<_>>()) {
        return None;
    }

    // Sort by length, then by symbol
    let mut sorted = code_lengths.to_vec();
    sorted.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

    let mut codes = Vec::new();
    let mut code_val: u64 = 0;
    let mut prev_len: usize = 0;

    for (symbol, len) in sorted {
        code_val <<= len - prev_len;
        let bits: Vec<u8> = (0..len)
            .rev()
            .map(|i| ((code_val >> i) & 1) as u8)
            .collect();
        codes.push((symbol, bits));
        code_val += 1;
        prev_len = len;
    }

    Some(codes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kraft_empty() {
        assert!(verify(&[]));
        assert!((kraft_sum(&[]) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_kraft_equal_2bit() {
        // 4 symbols with length 2: sum = 4 * 2^-2 = 1
        let lengths = vec![2, 2, 2, 2];
        assert!(verify(&lengths));
        assert!(is_complete(&lengths));
    }

    #[test]
    fn test_kraft_huffman_like() {
        // a=1, b=2, c=3, d=3 => 0.5 + 0.25 + 0.125 + 0.125 = 1.0
        let lengths = vec![1, 2, 3, 3];
        assert!(verify(&lengths));
        assert!(is_complete(&lengths));
    }

    #[test]
    fn test_kraft_violation() {
        // 3 symbols all with length 1: 3 * 0.5 = 1.5 > 1
        let lengths = vec![1, 1, 1];
        assert!(!verify(&lengths));
    }

    #[test]
    fn test_equal_length_codes() {
        let codes = equal_length_codes(4);
        assert_eq!(codes, vec![2, 2, 2, 2]);
    }

    #[test]
    fn test_construct_canonical() {
        let input = vec![(b'a', 1), (b'b', 2), (b'c', 3), (b'd', 3)];
        let codes = construct_canonical(&input).unwrap();
        assert_eq!(codes.len(), 4);
        // Verify prefix-free: no code is a prefix of another
        for i in 0..codes.len() {
            for j in 0..codes.len() {
                if i != j && codes[i].1.len() <= codes[j].1.len() {
                    assert!(
                        codes[i].1 != codes[j].1[..codes[i].1.len()],
                        "Code {:?} is a prefix of {:?}",
                        codes[i],
                        codes[j]
                    );
                }
            }
        }
    }

    #[test]
    fn test_dary_kraft() {
        // Ternary code: 3 symbols with length 1: 3 * 3^-1 = 1
        let lengths = vec![1, 1, 1];
        assert!(verify_dary(&lengths, 3));
    }
}
