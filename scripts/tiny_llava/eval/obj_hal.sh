MODEL_PATH="checkpoint/TinyLLaVA-3.1B-lora-sp--2"
MODEL_BASE="bczhou/TinyLLaVA-3.1B"
MODEL_NAME="TinyLLaVA-3.1B-lora-sp"

python -m tinyllava.eval.model_vqa_objhal \
    --model-path $MODEL_BASE \
    --question-file ./playground/data/eval/obj_halbench/obj_halbench_300_with_image.jsonl \
    --answers-file ./playground/data/eval/obj_halbench/answers2/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi
	
echo "========>Done generating answers<========"

echo "========>Start evaluating answers<========"

python tinyllava/eval/eval_gpt_obj_halbench.py \
    --coco_path ./playground/data/eval/obj_halbench/coco2014/annotations \
    --cap_folder ./playground/data/eval/obj_halbench/answers2 \
    --cap_type $MODEL_NAME.jsonl \
    --org_folder ./playground/data/eval/obj_halbench/obj_halbench_300_with_image.jsonl \
    --use_gpt \
    --openai_key sk-HDFRLuIBlMC7bOl38fuKm2OKY7a1noiB3437PI0T6D3OiQdr

python tinyllava/eval/summarize_gpt_obj_halbench_review.py ./playground/data/eval/obj_halbench/answers2 > ./playground/data/eval/obj_halbench/answers2/$MODEL_NAME-obj_halbench_scores.txt

# Print Log
echo Scores are:
cat ./playground/data/eval/obj_halbench/answers2/$MODEL_NAME-obj_halbench_scores.txt
echo done