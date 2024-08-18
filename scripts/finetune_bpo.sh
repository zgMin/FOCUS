DATA_PATH="data/vicuna1.3_shargpt4_6w_llavar_6w_coco_6w_mmhal_2w.json"
IMAGE_PATH="data"
OUTPUT_DIR="checkpoint/TinyLLaVA-3.1B-lora-bpo"
MODEL_PATH="bczhou/TinyLLaVA-3.1B"
MODEL_NAME="TinyLLaVA-3.1B"
deepspeed tinyllava/train/train_bpo.py \
--deepspeed ./scripts/tiny_llava/zero2.json \
    --lora_enable True --lora_r 32 --lora_alpha 64 \
    --model_name_or_path bczhou/TinyLLaVA-3.1B \
    --version phi \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_PATH\
    --vision_tower bczhou/TinyLLaVA-3.1B-SigLIP \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --fp16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
	--subset_percent 0.001
