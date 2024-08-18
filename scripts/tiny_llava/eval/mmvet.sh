#!/bin/bash

MODEL_PATH="checkpoint/TinyLLaVA-3.1B-lora-sp-1"
MODEL_BASE="bczhou/TinyLLaVA-3.1B"
MODEL_NAME="TinyLLaVA-3.1B-lora-sp-2-1"
EVAL_DIR="./playground/data/eval"

python -m tinyllava.eval.model_vqa \
    --model-path $MODEL_PATH \
	--model-base $MODEL_BASE \
    --question-file $EVAL_DIR/mm-vet/llava-mm-vet.jsonl \
    --image-folder $EVAL_DIR/mm-vet/images \
    --answers-file $EVAL_DIR/mm-vet/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

mkdir -p $EVAL_DIR/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src $EVAL_DIR/mm-vet/answers/$MODEL_NAME.jsonl \
    --dst $EVAL_DIR/mm-vet/results/$MODEL_NAME.json

