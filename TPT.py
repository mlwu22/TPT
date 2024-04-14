from data_process import tokenization_dataset, create_collate_fn
from transformers import AutoTokenizer
from models import MergeT5Single
from trainer import T5Trainer
from utils import TrainingArgs, MergeConfig
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="./TPT/save_models")
parser.add_argument("--per_device_train_batch_size", default = 32)
parser.add_argument("--per_device_eval_batch_size", default = 32)
parser.add_argument("--learning_rate", default=0.3)
parser.add_argument("--num_train_epochs", default = 3)
parser.add_argument("--eval_steps", default = 10)
parser.add_argument("--log_dir", default = "./TPT/log")
parser.add_argument("--record_dir", default = "./TPT/record")
parser.add_argument("--pad_token_id", default=0)
parser.add_argument("--do_train", default = False, action='store_true')
parser.add_argument("--do_eval", default = False, action='store_true')
parser.add_argument("--do_test", default = False, action='store_true')
parser.add_argument("--base_model", default="t5-base")
parser.add_argument("--optimizer_type", default="linear")
parser.add_argument("--init_type", default='frequent')
parser.add_argument("--num_bank_tokens", default=100, help="Length of retrieved prompt")
parser.add_argument("--num_soft_tokens", default=100, help="Length of soft prompt")
parser.add_argument("--total_bank_tokens", default=600, help="Capacity of token-wise prompt")
parser.add_argument("--select_method", default="max_pooling")
parser.add_argument("--model_type", default="from_scratch")
parser.add_argument("--task_type", default="glue")
parser.add_argument("--max_generate_length", default=3)
parser.add_argument("--task_name", default="mrpc")
parser.add_argument("--bank_path", help="Path to load token-wise prompt bank", default="./save_model/bank")
parser.add_argument("--prompt_path", help="Path to initialize soft prompt", default=None)
parser.add_argument("--load_path", help="Path to load TPT", default=None)
parser.add_argument("--is_all", default =True, action='store_true', help="Whether to train jointly")
parser.add_argument("--merge_type", help="The type to concatenate two types of prompts", default="concat")
parser.add_argument("--save_strategy", default="epoch", help="Strategy for saving checkpoint")
parser.add_argument("--seed", default=42)
parser.add_argument("--weight_decay", default=0.0)
parser.add_argument("--warm_up", default=500)

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.base_model)

seed = int(args.seed)
weight_decay = float(args.weight_decay)
train_dataset, eval_dataset, test_dataset,  data_info= tokenization_dataset(args.task_name, args.task_type, tokenizer, seed = seed)


output_dir = os.path.join(args.output_dir, args.task_name)
record_dir = os.path.join(args.record_dir, args.task_name)

config = MergeConfig(
    task_name=args.task_name,
    num_bank_tokens = int(args.num_bank_tokens),
    num_soft_tokens = int(args.num_soft_tokens),
    total_bank_tokens = int(args.total_bank_tokens),
    base_model_path = args.base_model,
    soft_embedding_init_type = args.init_type,
    select_method = args.select_method,
    bank_path = args.bank_path,
    prompt_path = args.prompt_path,
    merge_type = args.merge_type,
    is_all = args.is_all
    )


training_args = TrainingArgs(
    output_dir = output_dir,
    per_device_train_batch_size = int(args.per_device_train_batch_size), 
    per_device_eval_batch_size = int(args.per_device_eval_batch_size), 
    learning_rate = float(args.learning_rate),
    num_train_epochs = int(args.num_train_epochs),
    eval_steps = int(args.eval_steps), 
    do_train = bool(args.do_train),
    do_eval = bool(args.do_eval),
    do_test = bool(args.do_test),
    log_dir = args.log_dir,
    record_dir = record_dir, 
    load_path = args.load_path,
    save_strategy = args.save_strategy,
    weight_decay = weight_decay,
    warm_up = int(args.warm_up),
    seed=seed
)


if(args.model_type=="from_scratch"):
    model = MergeT5Single(config)
elif(args.model_type=="from_current"):
    model = MergeT5Single.from_pretrained(args.load_path)
elif(args.model_type=="from_random"):
    config.random_prompt = True
    model = MergeT5Single(config)

trainer = T5Trainer(
    model, 
    tokenizer, 
    create_collate_fn(args.task_name),             
    train_dataset, 
    eval_dataset, 
    test_dataset, 
    int(args.pad_token_id), 
    training_args,
    args.optimizer_type,
    args.task_name,
    data_info,
    int(args.max_generate_length),
    base_model = args.base_model,
    model_type=args.model_type
    )

trainer.run()