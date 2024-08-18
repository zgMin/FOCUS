#!/bin/bash
MODEL_PATH="checkpoint/TinyLLaVA-3.1B-lora-sp--2"
MODEL_BASE="bczhou/TinyLLaVA-3.1B"
MODEL_NAME="TinyLLaVA-3.1B-lora-sp"
python -m tinyllava.eval.model_vqa \
    --model-path $MODEL_PATH \
	--model-base $MODEL_BASE \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/resources/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python tinyllava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/resources/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/resources/context.jsonl \
    --rule tinyllava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/resources/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/$MODEL_NAME.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/$MODEL_NAME.jsonl

python tinyllava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/$MODEL_NAME.jsonl
