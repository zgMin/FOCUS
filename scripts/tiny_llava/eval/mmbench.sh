#!/bin/bash

SPLIT="mmbench_dev_20230712"

MODEL_PATH="checkpoint/TinyLLaVA-3.1B-lora-sp-2"
MODEL_BASE="bczhou/TinyLLaVA-3.1B"
MODEL_NAME="TinyLLaVA-3.1B-lora-sp-2"
EVAL_DIR="./playground/data/eval"

python -m tinyllava.eval.model_vqa_mmbench \
    --model-path $MODEL_PATH \
		--model-base $MODEL_BASE \
    --question-file $EVAL_DIR/mmbench/$SPLIT.tsv \
    --answers-file $EVAL_DIR/mmbench/answers/$SPLIT/$MODEL_NAME.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode phi

mkdir -p $EVAL_DIR/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file $EVAL_DIR/mmbench/$SPLIT.tsv \
    --result-dir $EVAL_DIR/mmbench/answers/$SPLIT \
    --upload-dir $EVAL_DIR/mmbench/answers_upload/$SPLIT \
    --experiment $MODEL_NAME
