from data_process import  get_multi_dataset, create_collate_fn
from transformers import AutoTokenizer
from models import MultiBankT5Single
from trainer import MultiT5Trainer
from utils import TrainingArgs, SuperBankConfig
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="./Bank/save_models")
parser.add_argument("--per_device_train_batch_size", default = 32)
parser.add_argument("--per_device_eval_batch_size", default = 32)
parser.add_argument("--learning_rate", default=0.3)
parser.add_argument("--num_train_epochs", default = 5)
parser.add_argument("--eval_steps", default = 100)
parser.add_argument("--log_dir", default = "./Bank/log")
parser.add_argument("--record_dir", default = "./Bank/record")
parser.add_argument("--pad_token_id", default=0)
parser.add_argument("--task_list", nargs='+', default=["sst2", "qnli", "mnli", "qqp", "record", "squad"], help="High resource datasets")
parser.add_argument("--do_train", default=False, action='store_true')
parser.add_argument("--do_eval", default=False, action='store_true')
parser.add_argument("--do_test", default =False, action='store_true')
parser.add_argument("--base_model", default="t5-base")
parser.add_argument("--optimizer_type", default="linear")
parser.add_argument("--init_type", default='frequent')
parser.add_argument("--load_path", help="Path to load token-wise prompt bank")
parser.add_argument("--load_paths", nargs='+', help="Path to initialize token-wise prompt bank from these soft prompt", default=[])
parser.add_argument("--num_virual_tokens", default=100, help="Length of soft prompt")
parser.add_argument("--total_bank_tokens", default=600, help="Capacity of token-wise prompt bank")
parser.add_argument("--select_method", default="max_pooling")
parser.add_argument("--model_type", help="Type of initialization", default="from_scratch")
parser.add_argument("--is_sample", help="True for sample", default=False, action='store_true')
parser.add_argument("--save_strategy", default="step", help="Strategy for saving checkpoint")
parser.add_argument("--seed", default=42)
parser.add_argument("--weight_decay", default=1e-5)
args = parser.parse_args()

seed = int(args.seed)
output_dir = os.path.join(args.output_dir, "train")
record_dir = os.path.join(args.record_dir, "train")
tokenizer = AutoTokenizer.from_pretrained(args.base_model)
train_dataset, train_dataset_list, train_num_list, eval_dataset, eval_dataset_list, eval_num_list,test_dataset, \
test_dataset_list, test_num_list,dataset_dict, num_dataset_type, data_info_list = get_multi_dataset(args.task_list, tokenizer, seed=42)
num_list = [train_num_list, eval_num_list, test_num_list]

config = SuperBankConfig(args.task_list, 
                         num_list, 
                         total_virtual_tokens=int(args.num_virual_tokens), 
                         base_model_path = args.base_model, 
                         soft_embedding_init_type=args.init_type, 
                         total_bank_tokens=int(args.total_bank_tokens), 
                         select_method=args.select_method
                         )

if(args.model_type=="from_scratch"):
    model = MultiBankT5Single(config)
elif(args.model_type=="from_current"):
    model = MultiBankT5Single.from_pretrained(args.load_path) 
elif(args.model_type=="from_others"):
    model=  MultiBankT5Single.from_others(args.load_paths, config)        

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
    weight_decay = float(args.weight_decay)
)

trainer = MultiT5Trainer(
            model, 
            tokenizer, 
            create_collate_fn(" "), 
            train_dataset, 
            eval_dataset, 
            test_dataset, 
            int(args.pad_token_id), 
            training_args,
            args.optimizer_type, 
            eval_dataset_list, 
            test_dataset_list, 
            num_list, 
            dataset_dict,
            data_info_list,
            args.is_sample
            )

trainer.run()