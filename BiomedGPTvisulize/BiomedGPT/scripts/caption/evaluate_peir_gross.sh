#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
# === 你目前環境的 Python 路徑 ===
PYTHON="/home/harper112/miniconda3/envs/biomedgpt_fix/bin/python"
export MASTER_PORT=1091

user_dir=../../module
bpe_dir=../../utils/BPE

# data=/home/harper112/biomedGPT/BiomedGPT/datasets/finetuning/PathVQA/iu_xray_test.tsv
data=/home/harper112/biomedGPT/0.png
path=/home/harper112/biomedGPT/BiomedGPT/checkpoint/iu_xray.pt

result_path=/home/harper112/biomedGPT/BiomedGPT/results
selected_cols=1,4,2
split='test'

# 單卡執行 evaluate.py（請確認 evaluate.py 位置正確）
$PYTHON /home/harper112/biomedGPT/BiomedGPT/evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=caption \
    --batch-size=1 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --max-len-b=32 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False,\"selected_cols\":\"${selected_cols}\"}"

# 執行評分
$PYTHON /home/harper112/biomedGPT/BiomedGPT/scripts/caption/metric_caption.py ${data} ${result_path}/test_predict.json 