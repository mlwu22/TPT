from data_process import  tokenization_dataset, create_collate_fn
from transformers import AutoTokenizer
from models import PromptT5Single
from trainer import T5Trainer
from utils import TrainingArgs,PromptConfig
import argparse
import os




parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="./PromptTuning/save_models", help="Save dir")
parser.add_argument("--per_device_train_batch_size", default = 32)
parser.add_argument("--per_device_eval_batch_size", default = 32)
parser.add_argument("--learning_rate", default=0.3)
parser.add_argument("--weight_decay", default=1e-5)
parser.add_argument("--num_train_epochs", default = 20)
parser.add_argument("--eval_steps", default = 500)
parser.add_argument("--log_dir", default = "./PromptTuning/log")
parser.add_argument("--record_dir", default = "./PromptTuning/record")
parser.add_argument("--pad_token_id", default=0)
parser.add_argument("--task_name", default="multirc")
parser.add_argument("--do_train", default=False, action='store_true')
parser.add_argument("--do_eval", default=False, action='store_true')
parser.add_argument("--do_test", default =False, action='store_true')
parser.add_argument("--base_model", default="t5-base")
parser.add_argument("--optimizer_type", default="constant")
parser.add_argument("--init_type", default='frequent')
parser.add_argument("--load_path", help="Path to load soft prompt")
parser.add_argument("--num_virual_tokens", default=100, help="Length of soft prompt")
parser.add_argument("--model_type", default = "from_scratch", help="Type of initialization")
parser.add_argument("--task_type", default="glue")
parser.add_argument("--save_strategy", default="epoch", help="Strategy for saving checkpoint")
parser.add_argument("--max_generate_length", default=3)
parser.add_argument("--seed", default=42)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.base_model)

config = PromptConfig(
    total_virtual_tokens=int(args.num_virual_tokens), 
    base_model_path=args.base_model, 
    task_name = args.task_name,
    soft_embedding_init_type=args.init_type,
    )

if(args.model_type=="from_scratch"):
    model = PromptT5Single(config)
elif(args.model_type=="from_current"):
    model = PromptT5Single.from_pretrained(args.load_path)

seed = int(args.seed)
weight_decay = float(args.weight_decay)
train_dataset, eval_dataset, test_dataset, data_info= tokenization_dataset(args.task_name, args.task_type, tokenizer, seed=seed)

input_str = tokenizer.decode(train_dataset[0]["input_ids"])
label_str = tokenizer.decode(train_dataset[0]["labels"])
output_dir = os.path.join(args.output_dir, args.task_name)
record_dir = os.path.join(args.record_dir, args.task_name)

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
    weight_decay = weight_decay
)

trainer = T5Trainer(model, 
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
                    base_model = args.base_model)

trainer.run()