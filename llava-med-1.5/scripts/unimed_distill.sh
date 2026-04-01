seeds=(4)
seed=4
ROOT_PATH=/data/xxx
dataset=VQA_RAD

baseline="beam"
sub_folder="distill_unimed_base"


export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

epoch_num=9
export CUDA_VISIBLE_DEVICES=0,1,2,3
align_layers_list=("20")
SAE_layers="20"
align_layers="20"
align_loss_ratio=1
attn_loss_ratio=0.06
attn_loss_ratio_list=(0.08 0.02 0.1)

# An example of training on VQA_RAD dataset (medical VQA) with llava-med-v1.5
for align_loss_ratio in "${attn_loss_ratio_list[@]}"; do
    echo "Running with seed: $seed, align_layers ${align_layers}"
    dir=llava_med_1.5/${sub_folder}/search_epoch${epoch_num}_seed${seed}_layers${align_layers}-${SAE_layers}_ratio${align_loss_ratio}-${attn_loss_ratio}
    # deepspeed llava/train/train_mem.py \

    torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 llava/train/train_mem.py \
        --model_name_or_path ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
        --lora_enable True --lora_r 32 --lora_alpha 16 --mm_projector_lr 2e-5 \
        --version v1 \
        --tune_mm_mlp_adapter False \
        --data_path ${ROOT_PATH}/hallucination/${dataset}/data/training_masks_top4.json \
        --image_folder ${ROOT_PATH}/hallucination/${dataset}/images \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir ${ROOT_PATH}/hallucination/steering/${dataset}/${dir}/checkpoints \
        --num_train_epochs $epoch_num \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 1000 \
        --save_total_limit 3 \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 1024 \
        --gradient_checkpointing False \
        --lazy_preprocess True \
        --spatial_matrix_path ${ROOT_PATH}/hallucination/${dataset}/steering/steer_data_base/cosine_matrix_dict.pkl \
        --attention_dict_path ${ROOT_PATH}/hallucination/${dataset}/steering/steer_data_base/attn_map_dict.pkl \
        --seed $seed \
        --report_to wandb \
        --align_layers $align_layers \
        --SAE_layers $SAE_layers \
        --align_epsilon 1.0 \
        --align_loss_ratio $align_loss_ratio \
        --attn_loss_ratio $attn_loss_ratio
        

    wait
    baselines=(greedy beam)
    for baseline in "${baselines[@]}"; do
        python llava/eval/eval_batch.py --num-chunks 2  --model-name ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
            --question-file ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
            --image-folder ${ROOT_PATH}/hallucination/${dataset}/images \
            --baseline $baseline \
            --answers-file ${ROOT_PATH}/hallucination/steering/${dataset}/${dir}/inference/pred_${baseline}.jsonl \
            --peft-path ${ROOT_PATH}/hallucination/steering/${dataset}/${dir}/checkpoints \
            --steer_w_layer $align_layers \
            --SAE_layer $SAE_layers

        python llava/eval/run_eval.py \
            --gt ${ROOT_PATH}/hallucination/${dataset}/data/test.json \
            --pred ${ROOT_PATH}/hallucination/steering/${dataset}/${dir}/inference/pred_${baseline}.jsonl \
            --eval_res ${ROOT_PATH}/hallucination/steering/${dataset}/${dir}/inference/eval_res_${baseline}.txt

        wait
    done
done


# An example of training on IU-Xray dataset (Report generation) with llava-med-v1.5
source /home/xxx/software/anaconda3/etc/profile.d/conda.sh
export HF_HOME=${ROOT_PATH}/huggingface_cache/transformers

epoch_num=12
export CUDA_VISIBLE_DEVICES=0,1,2,3
align_loss_ratio=1.0
attn_loss_ratio=0.03
align_layers="20"
SAE_layers_list=("20")
SAE_layers="20"
baseline="beam"
attn_loss_ratio_list=(0.03)
for attn_loss_ratio in "${attn_loss_ratio_list[@]}"; do
    echo "Running with seed: $seed, align_layers ${align_layers}, SAE_layers ${SAE_layers}, ratio ${align_loss_ratio}, attn_loss_ratio ${attn_loss_ratio}"
    dir=llava_med_1.5/${sub_folder}/epoch${epoch_num}_ratio${align_loss_ratio}-${attn_loss_ratio}_dual${SAE_layers}${align_layers}_seed${seed}
    # deepspeed llava/train/train_mem.py \
    torchrun --nnodes=1 --nproc_per_node=4 --master_port=25001 llava/train/train_mem.py \
        --model_name_or_path ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
        --lora_enable True --lora_r 64 --lora_alpha 16 --mm_projector_lr 2e-5 \
        --version v1 \
        --tune_mm_mlp_adapter False \
        --data_path ${ROOT_PATH}/hallucination/${dataset}/data_report/training.json \
        --image_folder ${ROOT_PATH}/hallucination/${dataset}/iu_xray/images \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir ${ROOT_PATH}/hallucination/steering/${dataset}/${dir}/checkpoints \
        --num_train_epochs $epoch_num \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 1000 \
        --save_total_limit 3 \
        --learning_rate 2e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 1024 \
        --gradient_checkpointing False \
        --lazy_preprocess True \
        --spatial_matrix_path ${ROOT_PATH}/hallucination/${dataset}/data_report/steering_data_base/cosine_matrix_dict.pkl \
        --attention_dict_path ${ROOT_PATH}/hallucination/${dataset}/data_report/steering_data_base/attn_map_dict.pkl \
        --seed $seed \
        --align_layers $align_layers \
        --SAE_layers $SAE_layers \
        --align_epsilon 1.0 \
        --align_loss_ratio $align_loss_ratio \
        --attn_loss_ratio $attn_loss_ratio
        

    wait

    baselines=(greedy beam)
    for baseline in "${baselines[@]}"; do
        python llava/eval/eval_batch.py --num-chunks 4  --model-name ${ROOT_PATH}/LLM/llava-med-v1.5-mistral-7b \
            --question-file ${ROOT_PATH}/hallucination/${dataset}/data_report/test${modality}.json \
            --image-folder ${ROOT_PATH}/hallucination/${dataset}/iu_xray/images \
            --baseline $baseline \
            --answers-file ${ROOT_PATH}/hallucination/steering/${dataset}/${dir}/inference/pred_${baseline}.jsonl \
            --peft-path ${ROOT_PATH}/hallucination/steering/${dataset}/${dir}/checkpoints \
            --steer_w_layer $SAE_layers \
            --SAE_layer $SAE_layers

        conda activate report_eval2
        python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_eval.py \
            --gt ${ROOT_PATH}/hallucination/${dataset}/data_report/test.csv \
            --pred ${ROOT_PATH}/hallucination/steering/${dataset}/${dir}/inference/pred_${baseline}.csv \
            --eval_res ${ROOT_PATH}/hallucination/steering/${dataset}/${dir}/inference/eval_res_${baseline}.csv \
            --report True

        conda activate medh
        python ${ROOT_PATH}/hallucination/mitigation/report_eval/run_all_metrics.py \
            --model_answers_file ${ROOT_PATH}/hallucination/steering/${dataset}/${dir}/inference/pred_${baseline}.jsonl \
            --eval_res_file ${ROOT_PATH}/hallucination/steering/${dataset}/${dir}/inference/eval_all_metrics_${baseline}.txt \
            --RadGraphFile ${ROOT_PATH}/hallucination/steering/${dataset}/${dir}/inference/eval_res_${baseline}.csv
        conda activate llava_v1.5

        wait
    done

done
