# Jordan-Spectral Attention (JSA)

This repository is a timestamped public research artifact for **Jordan-Spectral Attention (JSA)**, a spectral-shift replacement for Transformer self-attention in autoregressive language modeling experiments.

JSA replaces token-to-token softmax attention with two structured branches:

1. **Spectral global mixing** over a small cosine basis of rank `R`.
2. **Causal local shift mixing** over the previous `k` tokens.

The current experimental implementation targets the OpenAI Parameter Golf MLX training path.

## Core operator

For an input sequence `x ∈ R^{B×T×D}`:

- project the token axis into `R` spectral modes,
- gate those modes from a pooled sequence representation,
- reconstruct a global sequence signal,
- add small causal local shifts,
- apply a learned channel scale.

The implementation lives in:

```text
jsa/mixer.py
```

The Parameter Golf integration lives in:

```text
train/train_jsa_mlx.py
```

## Current strongest local result

These are **local MLX experiments**. Official Parameter Golf reproduction is pending.

| Setup | Params | Artifact | Full-val BPB | Notes |
|---|---:|---:|---:|---|
| SP8192 baseline | 20.73M | 13.62 MB | 1.9096 | Local 500-step baseline |
| JSA full replacement, rank 32, k=2 | 14.11M | ~10.55 MB | 0.91–1.11 | Seed-sensitive but strong |
| JSA full replacement, rank 64, k=2 | 14.55M | ~11.74–11.78 MB | 0.58–0.60 | Best current local result |

Key caveat: these runs used 10 downloaded train shards for local iteration and full validation over the SP8192 validation split. They should not be presented as official leaderboard results until reproduced through the official track path.

## Example command

```bash
RUN_ID=jsa_full_rank64_sp8192_fullval_seed1338 DATA_PATH=./data/datasets/fineweb10B_sp8192 TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model VOCAB_SIZE=8192 SEED=1338 USE_JSA=1 JSA_RANK=64 JSA_LOCAL_K=2 JSA_LAST_N_LAYERS=9 ITERATIONS=500 TRAIN_BATCH_TOKENS=8192 VAL_BATCH_SIZE=8192 VAL_LOSS_EVERY=0 python3 train/train_jsa_mlx.py
```

## Dataset note

The SP8192 dataset used by the Parameter Golf records is downloaded from the alternate manifest:

```bash
rm -f datasets/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 10
```

## Repository layout

```text
jordan-spectral-attention/
├── jsa/                     # core JSA mixer
├── train/                   # MLX training scripts
├── configs/                 # copyable run configs
├── experiments/             # result summaries
├── figures/                 # block diagram
└── logs/                    # selected logs can be added here
```

## Attribution statement

This repository establishes a public timestamped release of **Jordan-Spectral Attention (JSA)**, proposed and implemented by Karimulla Saheb Naik.

## License

MIT. See `LICENSE`.
