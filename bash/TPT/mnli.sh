CUDA_VISIBLE_DEVICES=2 python ../../TPT.py \
    --output_dir ../../TPT/save_models  \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 0.3 \
    --weight_decay 1e-5 \
    --num_train_epochs 5 \
    --eval_steps 1000 \
    --log_dir ../../TPT/log \
    --record_dir ../../TPT/record \
    --pad_token_id 0 \
    --task_name mnli \
    --do_train \
    --do_eval \
    --do_test \
    --base_model t5-base \
    --optimizer_type linear \
    --init_type frequent \
    --num_soft_tokens 100 \
    --num_bank_tokens 100 \
    --total_bank_tokens 600 \
    --select_method max_pooling \
    --model_type from_scratch \
    --task_type glue \
    --save_strategy step \
    --max_generate_length 3 \
    --bank_path ../../save_model/bank \
    --is_all \
    --merge_type concat \
    --warm_up 500 \
    --seed 42




