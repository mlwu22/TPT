CUDA_VISIBLE_DEVICES=0 proxychains4 python ../../TPT.py \
    --output_dir ../../TPT_Test/save_models  \
    --per_device_eval_batch_size 32 \
    --log_dir ../../TPT_Test/log \
    --record_dir ../../TPT_Test/record \
    --pad_token_id 0 \
    --task_name qqp \
    --do_test \
    --base_model t5-base \
    --num_soft_tokens 100 \
    --num_bank_tokens 100 \
    --total_bank_tokens 600 \
    --select_method max_pooling \
    --model_type from_current \
    --task_type glue \
    --save_strategy step \
    --max_generate_length 3 \
    --load_path ../../save_model/TPT/qqp \
    --is_all \
    --merge_type concat \
    --seed 42




