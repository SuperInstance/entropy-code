# entropy-code

A pure-Rust library for entropy coding fundamentals — Shannon entropy calculation, symbol probability analysis, optimal code length, Kraft inequality verification, and basic arithmetic coding.

## Features

- **Shannon entropy** — Calculate the entropy of a symbol distribution
- **Symbol probability** — Build and analyze probability distributions
- **Optimal code length** — Compute optimal code lengths from probabilities
- **Kraft inequality** — Verify code length sets satisfy Kraft's inequality
- **Arithmetic coding** — Basic arithmetic encoding and decoding

## Usage

```rust
use entropy_code::{entropy, probability, optimal, arithmetic};

let data = b"hello world";
let h = entropy::shannon_entropy(data);
println!("Entropy: {:.4} bits/symbol", h);

let freq = probability::FrequencyDistribution::from_bytes(data);
let code_lengths = optimal::optimal_code_lengths(&freq);
```

## Modules

- `entropy` — Shannon entropy and information content
- `probability` — Symbol probability distributions
- `kraft` — Kraft inequality verification
- `arithmetic` — Basic arithmetic coding
- `optimal` — Optimal code length computation

## License

MIT
