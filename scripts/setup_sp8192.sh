#!/usr/bin/env bash
set -euo pipefail

PG_DIR="external/parameter-golf"

mkdir -p external
mkdir -p data

if [ ! -d "$PG_DIR" ]; then
  git clone https://github.com/openai/parameter-golf.git "$PG_DIR"
fi

cd "$PG_DIR"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm

rm -f datasets/manifest.json

MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py \
  --variant sp8192 \
  --train-shards 10

cd ../..

mkdir -p data/tokenizers
mkdir -p data/datasets

cp -r "$PG_DIR/data/tokenizers/"* data/tokenizers/
cp -r "$PG_DIR/data/datasets/fineweb10B_sp8192" data/datasets/

echo "SP8192 setup complete."
echo "Tokenizer: data/tokenizers/fineweb_8192_bpe.model"
echo "Dataset:   data/datasets/fineweb10B_sp8192"