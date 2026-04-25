# Experimental Results

Local MLX results gathered during early JSA exploration. Official leaderboard reproduction is pending.

| Setup | Seed | Val scope | Params | Artifact | BPB |
|---|---:|---|---:|---:|---:|
| SP8192 baseline | 1337/default | 512 seqs | 20.73M | 13.62 MB | 1.9096 |
| JSA full rank32 k2 | 1338 | full val | 14.11M | ~10.55 MB | 0.9612 |
| JSA full rank32 k2 | 1337 | full val | 14.11M | ~10.59 MB | 1.1113 |
| JSA full rank32 k2 | 42 | full val | 14.11M | ~10.55 MB | 0.9128 |
| JSA full rank64 k2 | 42 | full val | 14.55M | 11.78 MB | 0.5807 |
| JSA full rank64 k2 | 1338 | full val | 14.55M | 11.74 MB | 0.6000 |

## Caveats

- Local Mac/MLX path, not official 8×H100 track reproduction.
- Training used a 10-shard subset for quick iteration.
- Full validation was run on the SP8192 validation split where noted.
