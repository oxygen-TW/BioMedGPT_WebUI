#!/usr/bin/env bash

# === 你目前環境的 Python 路徑 ===
PYTHON=/home/harper112/miniconda3/envs/biomedgpt_fix/bin/python

user_dir=../../module
bpe_dir=../../utils/BPE
data=/home/harper112/biomedGPT/BiomedGPT/datasets/finetuning/PathVQA/iu_xray_test.tsv
path=../../checkpoints/tuned_checkpoints/peir_gross/tiny/stage1_checkpoints/100_0.06_600/checkpoint_best.pt
result_path=./results/peir_gross/tiny
selected_cols=1,4,2
split='test'

# === 執行推理 ===
$PYTHON /home/harper112/biomedGPT/BiomedGPT/evaluate.py \
  /home/harper112/biomedGPT/BiomedGPT/datasets/finetuning/PathVQA/iu_xray_test.tsv \
  --path checkpoints/tuned_checkpoints/peir_gross/tiny/stage1_checkpoints/100_0.06_600/checkpoint_best.pt \
  --user-dir module \
  --task caption \
  --batch-size 8 \
  --log-format simple --log-interval 10 \
  --seed 7 \
  --gen-subset test \
  --results-path scripts/caption/results/peir_gross/tiny \
  --beam 5 \
  --max-len-b 32 \
  --no-repeat-ngram-size 3 \
  --fp16 \
  --num-workers 0 \
  --model-overrides="{\"data\":\"/home/harper112/biomedGPT/BiomedGPT/datasets/finetuning/PathVQA/iu_xray_test.tsv\",\"bpe_dir\":\"utils/BPE\",\"eval_cider\":False,\"selected_cols\":\"1,4,2\"}"

# === 執行評分 ===
$PYTHON /home/harper112/biomedGPT/BiomedGPT/scripts/caption/metric_caption.py ${data} ${result_path}/test_predict.json
