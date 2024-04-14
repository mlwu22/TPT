CUDA_VISIBLE_DEVICES=0 python ../../construct_bank_over_multi_tasks.py \
    --output_dir ../../Bank/save_models  \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 0.3 \
    --weight_decay 1e-5 \
    --num_train_epochs 5 \
    --eval_steps 500 \
    --log_dir ../../Bank/logs \
    --record_dir ../../Bank/record \
    --task_list "sst2" "qnli" "mnli" "qqp" "record" "squad" \
    --pad_token_id 0 \
    --do_train \
    --do_eval \
    --do_test \
    --base_model t5-base \
    --optimizer_type linear \
    --init_type frequent \
    --num_virual_tokens 100 \
    --total_bank_tokens 600 \
    --select_method max_pooling \
    --model_type from_scratch \
    --save_strategy step \
    --seed 42






