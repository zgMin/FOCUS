#!/bin/bash


MODEL_PATH="checkpoint/TinyLLaVA-3.1B-lora-sp--2"
MODEL_BASE="bczhou/TinyLLaVA-3.1B"
MODEL_NAME="TinyLLaVA-3.1B-lora-sp--2"
EVAL_DIR="./playground/data/eval"

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
	--model-base $MODEL_BASE \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --image-folder $EVAL_DIR/pope/val2014 \
    --answers-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python tinyllava/eval/eval_pope.py \
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl

MODEL_PATH="checkpoint/TinyLLaVA-3.1B-lora-sp-1-2"
MODEL_BASE="bczhou/TinyLLaVA-3.1B"
MODEL_NAME="TinyLLaVA-3.1B-lora-sp-2"
EVAL_DIR="./playground/data/eval"

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
	--model-base $MODEL_BASE \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --image-folder $EVAL_DIR/pope/val2014 \
    --answers-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python tinyllava/eval/eval_pope.py \
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl
	
MODEL_PATH="checkpoint/TinyLLaVA-3.1B-lora-sp-1-2-3"
MODEL_BASE="bczhou/TinyLLaVA-3.1B"
MODEL_NAME="TinyLLaVA-3.1B-lora-sp-3"
EVAL_DIR="./playground/data/eval"

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
	--model-base $MODEL_BASE \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --image-folder $EVAL_DIR/pope/val2014 \
    --answers-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python tinyllava/eval/eval_pope.py \
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl
	
MODEL_PATH="checkpoint/TinyLLaVA-3.1B-lora-sp-1-2-3-4"
MODEL_BASE="bczhou/TinyLLaVA-3.1B"
MODEL_NAME="TinyLLaVA-3.1B-lora-sp-4"
EVAL_DIR="./playground/data/eval"

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
	--model-base $MODEL_BASE \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --image-folder $EVAL_DIR/pope/val2014 \
    --answers-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python tinyllava/eval/eval_pope.py \
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl
	
MODEL_PATH="checkpoint/TinyLLaVA-3.1B-lora-sp-1-2-3-4-5"
MODEL_BASE="bczhou/TinyLLaVA-3.1B"
MODEL_NAME="TinyLLaVA-3.1B-lora-sp-5"
EVAL_DIR="./playground/data/eval"

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
	--model-base $MODEL_BASE \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --image-folder $EVAL_DIR/pope/val2014 \
    --answers-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python tinyllava/eval/eval_pope.py \
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl
	
MODEL_PATH="checkpoint/TinyLLaVA-3.1B-lora-sp-1-2-3-4-5-6"
MODEL_BASE="bczhou/TinyLLaVA-3.1B"
MODEL_NAME="TinyLLaVA-3.1B-lora-sp-6"
EVAL_DIR="./playground/data/eval"

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
	--model-base $MODEL_BASE \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --image-folder $EVAL_DIR/pope/val2014 \
    --answers-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python tinyllava/eval/eval_pope.py \
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl
	
MODEL_PATH="checkpoint/TinyLLaVA-3.1B-lora-sp-1-2-3-4-5-6-7"
MODEL_BASE="bczhou/TinyLLaVA-3.1B"
MODEL_NAME="TinyLLaVA-3.1B-lora-sp-7"
EVAL_DIR="./playground/data/eval"

python -m tinyllava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
	--model-base $MODEL_BASE \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --image-folder $EVAL_DIR/pope/val2014 \
    --answers-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python tinyllava/eval/eval_pope.py \
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl
	