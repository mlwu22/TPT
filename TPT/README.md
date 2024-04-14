## Install Requirements

```bash
conda create -n tpt python=3.8
conda activate tpt
pip install -r requirements.txt
```



## Download Datasets and Evaluate package

```
git clone https://github.com/huggingface/evaluate.git

python download_dataset.py
```



## Test our TPT Methods

```
cd bash
bash run.sh
```



## Training

##### PromptTuning

- sst2

  ```
  cd bash
  cd PromptTuning
  bash sst2.sh
  ```

  ```bash
  CUDA_VISIBLE_DEVICES=0 python ../../prompt_tuning_for_single_task.py \
      --output_dir ../../PromptTuning/save_models  \
      --per_device_train_batch_size 32 \
      --per_device_eval_batch_size 32 \
      --learning_rate 0.3 \
      --weight_decay 1e-5 \
      --num_train_epochs 10 \
      --eval_steps 500 \
      --log_dir ../../PromptTuning/logs \
      --record_dir ../../PromptTuning/record \
      --pad_token_id 0 \
      --task_name sst2 \
      --do_train \
      --do_eval \
      --do_test \
      --base_model t5-base \
      --optimizer_type constant \
      --init_type frequent \
      --num_virual_tokens 100 \
      --model_type from_scratch \
      --task_type glue \
      --save_strategy epoch \
      --max_generate_length 3 \
      --seed 42

  ```




##### Construt Bank over high resource tasks

```
cd bash
cd ConstructBank
bash train.sh
```

```bash
CUDA_VISIBLE_DEVICES=0 python ../../construt_bank_over_multi_tasks.py \
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
```



##### TPT

- mnli

  ```
  cd bash
  cd TPT
  bash train.sh
  ```

  ```bash
  CUDA_VISIBLE_DEVICES=0 python ../../TPT.py \
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
  ```

  - You can also choose the bank you've trained in secton `Construt Bank over high resource tasks` or use the model we've provided in `/save_model/bank`

