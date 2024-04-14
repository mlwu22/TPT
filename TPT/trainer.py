import torch
from torch.utils.data import DataLoader
import logging
from transformers import AdamW, get_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import os
from utils import MultiTaskBatchSampler
import torch.nn.utils.rnn as rnn
from torch.cuda.amp import autocast as autocast
from utils import compute_metrics, convert_token_ids_to_int, convert_token_ids_to_float, convert_token_ids_to_text, convert_token_ids_to_int_m
from utils import EvalPrediction
from models import MultiBankT5Single, PromptT5Single, BankT5Single, MergeT5Single


metric_map = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mrpc": "accuracy",
    "qnli": "accuracy",
    "qqp": "accuracy",
    "rte": "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson",
    "boolq": "accuracy",
    "cb": "accuracy",
    "wsc.fixed": "accuracy",
    "wic": "accuracy",
    "multirc": "f1_a"
}

# set log
def set_log(log_dir, dataset_name):
    os.makedirs(f"{log_dir}/{dataset_name}", exist_ok=True)
    final_path = f"{log_dir}/{dataset_name}/log"
    logger = logging.getLogger()
    logger.setLevel('INFO')
    control = logging.StreamHandler() 
    control.setLevel('INFO')
    fhlr = logging.FileHandler(final_path)
    logger.addHandler(fhlr)
    logger.addHandler(control)
    return logger


class T5Trainer():
    def __init__(self, 
                 model, 
                 tokenizer, 
                 data_collator, 
                 train_dataset,
                 eval_dataset, 
                 test_dataset, 
                 pad_token_id, 
                 training_args, 
                 optimizer_type,
                 task_name,
                 data_info,
                 max_generate_length,
                 base_model = "t5-base",
                 model_type = "from_scratch"
                 ):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.base_model = base_model
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.pad_token_id = pad_token_id
        self.output_dir = training_args.output_dir
        self.per_device_train_batch_size = training_args.per_device_train_batch_size
        self.per_device_eval_batch_size = training_args.per_device_eval_batch_size
        self.learning_rate = training_args.learning_rate
        self.num_train_epochs = training_args.num_train_epochs
        self.eval_steps = training_args.eval_steps
        self.log_dir = training_args.log_dir
        self.record_dir = training_args.record_dir
        self.save_strategy = training_args.save_strategy
        self.do_train = training_args.do_train
        self.do_eval = training_args.do_eval
        self.do_test = training_args.do_test
        self.optimizer_type = optimizer_type
        self.load_path = training_args.load_path
        self.weight_decay = training_args.weight_decay
        self.warm_up = training_args.warm_up
        self.seed = training_args.seed
        self.model_type = model_type
        self.task_name = task_name
        self.data_info = data_info
        self.max_generate_length = max_generate_length
        self.optimizer = self.get_optimizer()
        self.lr_scheduler = self.get_lr_scheduler()
        self.logger = set_log(self.log_dir, self.task_name)


    # The data has been shuffled when we processed data
    def get_dataloader(self, mode):
        if(mode=="train"):
            dataLoader = DataLoader(
                dataset = self.train_dataset, 
                shuffle = False,
                batch_size = self.per_device_train_batch_size,
                collate_fn = self.data_collator,
               )
            num = len(dataLoader) * self.num_train_epochs
        elif(mode=="eval"):
            dataLoader = DataLoader(
                dataset = self.eval_dataset, 
                shuffle = False,
                batch_size = self.per_device_eval_batch_size,
                collate_fn = self.data_collator,
                )
            num = len(dataLoader)
        elif(mode=="test"):
            dataLoader = DataLoader(
                dataset = self.test_dataset, 
                shuffle = False,
                batch_size = self.per_device_eval_batch_size,
                collate_fn = self.data_collator,
                )
            num = len(dataLoader)
        return  dataLoader, num
    

    def get_optimizer(self):
        optimizer = AdamW(self.model.parameters(), lr = self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


    def get_lr_scheduler(self):
        _, total_train_steps = self.get_dataloader("train")
        if(self.optimizer_type=="constant"):
            schedular = get_scheduler("constant", optimizer=self.optimizer)
        elif(self.optimizer_type=="linear"):
            schedular = get_scheduler("linear", optimizer=self.optimizer, num_warmup_steps=self.warm_up, num_training_steps=total_train_steps)
        elif(self.optimizer_type=="cosine"):
            schedular = get_scheduler("cosine", optimizer=self.optimizer, num_warmup_steps=self.warm_up, num_training_steps=total_train_steps)
        return schedular


    def run(self):
        model = self.model
        #########################    Train     #################
        if(self.do_train):
            optimizer = self.optimizer
            lr_scheduler = self.lr_scheduler
            num_epochs = self.num_train_epochs
            train_bs = self.per_device_train_batch_size
            train_dataLoader, total_train_steps = self.get_dataloader("train")
            logger = self.logger
            max_eval_metric = -1000
            model.to(torch.device("cuda"))

            logger.info("***** Running training *****")
            logger.info(f"  Num training examples = {len(self.train_dataset)}")
            logger.info(f"  Seed = {self.seed}")
            logger.info(f"  Warmup = {self.warm_up}")
            logger.info(f"  Optimizer Type = {self.optimizer_type}")
            logger.info(f"  Num evaluation examples = {len(self.eval_dataset)}")
            logger.info(f"  Num test examples = {len(self.test_dataset)}")
            logger.info(f"  Weight decay = {self.weight_decay}")
            logger.info(f"  Learning rate = {self.learning_rate}")
            logger.info(f"  Model type = {self.model_type}")
            if(hasattr(self.model, "merge_config")):
                logger.info(f"  Num retrieved tokens = {self.model.merge_config.num_bank_tokens}")
                logger.info(f"  Num virual tokens = {self.model.merge_config.num_soft_tokens}")
                logger.info(f"  Num bank tokens = {self.model.merge_config.total_bank_tokens}")
                logger.info(f"  Prompt path = {self.model.merge_config.prompt_path}")
            if(hasattr(self.model, "prompt_config")):
                logger.info(f"  Num virual tokens = {self.model.prompt_config.total_virtual_tokens}")
            elif(hasattr(self.model, "bank_config")):
                logger.info(f"  Num retrieved tokens = {self.model.bank_config.total_virtual_tokens}")
                logger.info(f"  Num bank tokens = {self.model.bank_config.total_bank_tokens}")
            logger.info(f"  Num epochs = {num_epochs}")
            logger.info(f"  Batch size per device = {train_bs}")
            logger.info(f"  Total train step = {total_train_steps}")
            logger.info(f"  Num trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")


            total_step = 0
            select_epoch = 0
            select_step = 0
            writer = SummaryWriter(log_dir = self.record_dir)
            model.train()

            for epoch in range(num_epochs):
                for step, batch in enumerate(tqdm(train_dataLoader)):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    output = model(**batch)        
                    loss = output[0]
                    loss.backward()
                    writer.add_scalar("loss", loss, total_step)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    total_step += 1
                    lr = optimizer.param_groups[0]["lr"]

                    
                    if(total_step%self.eval_steps==0):
                        if(self.save_strategy=="step"):
                            logger.info(f"epoch : {total_step}, lr: {lr}")
                
                            result={}
                            #########################    Evaluate     #################
                            if(self.do_eval):
                                eval_bs = self.per_device_eval_batch_size
                                eval_num_examples = len(self.eval_dataset)
                                eval_dataLoader, total_eval_steps = self.get_dataloader("eval")
                                logger.info("***** Running evaluation *****")

                                model.eval()
                                all_labels = []
                                all_predicts = []
                                with torch.no_grad():
                                    for step, batch_eval in enumerate(tqdm(eval_dataLoader)):
                                        batch_eval = {k: v.to(self.device) for k, v in batch_eval.items()}
                                        inputs_eval = batch_eval["input_ids"]
                                        labels_eval = batch_eval["labels"]
                                        predicts = model.generate(inputs_eval, max_length=self.max_generate_length)
                                        all_labels.append(labels_eval.to(torch.device("cpu")))
                                        all_predicts.append(predicts.to(torch.device("cpu")))
                                            
                                all_l = []
                                all_p = []
                                for labels in all_labels:
                                    for label in labels:
                                        all_l.append(label)
                                for predicts in all_predicts:
                                    for predict in predicts:
                                        all_p.append(predict)
                                
                                data_info = self.data_info["eval"] if self.data_info is not None else None
                                all_labels = rnn.pad_sequence(all_l, batch_first=True, padding_value=0)
                                all_predicts = rnn.pad_sequence(all_p, batch_first=True, padding_value=0)
                                eval_prediction = EvalPrediction(predictions=all_predicts, label_ids=all_labels, data_info=data_info)
                                if(self.task_name=="stsb"):
                                    eval_prediction = convert_token_ids_to_float(eval_prediction, self.tokenizer)
                                else:
                                    eval_prediction = convert_token_ids_to_int(eval_prediction, self.tokenizer)
                                result = compute_metrics(eval_prediction, dataset_name=self.task_name)
                                result_scalar = result[metric_map[self.task_name]]
                                writer.add_scalar("eval_metric", result_scalar, total_step)

                                if(result_scalar >= max_eval_metric):
                                    max_eval_metric = result_scalar
                                    select_step = total_step

                                logger.info(f"current_step: {str(total_step)}, eval_result: {result_scalar}, select_step: {select_step}")
                                model.save_pretrained(os.path.join(self.output_dir,str(total_step)))
                                model.train()


                if(self.save_strategy=="epoch"):
                    logger.info(f"epoch : {(epoch+1)}, lr: {lr}")
            
                    result={}
                    #########################    Evaluate     #################
                    if(self.do_eval):
                        eval_bs = self.per_device_eval_batch_size
                        eval_num_examples = len(self.eval_dataset)
                        eval_dataLoader, total_eval_steps = self.get_dataloader("eval")
                        logger.info("***** Running evaluation *****")

                        model.eval()
                        all_labels = []
                        all_predicts = []
                        all_idxs = []
                        with torch.no_grad():
                            for step, batch_eval in enumerate(tqdm(eval_dataLoader)):
                                batch_eval = {k: v.to(self.device) for k, v in batch_eval.items()}
                                inputs_eval = batch_eval["input_ids"]
                                labels_eval = batch_eval["labels"]
                                if(self.task_name=="multirc"):
                                    idxs = batch_eval["idxs"]
                                    all_idxs.append(idxs)
                                predicts = model.generate(inputs_eval, max_length=self.max_generate_length)
                                all_labels.append(labels_eval.to(torch.device("cpu")))
                                all_predicts.append(predicts.to(torch.device("cpu")))
                                
                                      
                        all_l = []
                        all_p = []
                        all_a = []
                        for labels in all_labels:
                            for label in labels:
                                all_l.append(label)
                        for predicts in all_predicts:
                            for predict in predicts:
                                all_p.append(predict)
                                
                        if(self.task_name=="multirc"):
                            for idxs in all_idxs:
                                for idx in idxs:
                                    all_a.append(idx)
                        data_info = self.data_info["eval"] if self.data_info is not None else None
                        all_labels = rnn.pad_sequence(all_l, batch_first=True, padding_value=0)
                        all_predicts = rnn.pad_sequence(all_p, batch_first=True, padding_value=0)
                        eval_prediction = EvalPrediction(predictions=all_predicts, label_ids=all_labels, data_info=data_info, idxs=all_a)
                        if(self.task_name=="stsb"):
                            eval_prediction = convert_token_ids_to_float(eval_prediction, self.tokenizer)
                        elif(self.task_name=="multirc"):
                            eval_prediction = convert_token_ids_to_int_m(eval_prediction, self.tokenizer)
                        else:
                            eval_prediction = convert_token_ids_to_int(eval_prediction, self.tokenizer)
                        result = compute_metrics(eval_prediction, dataset_name=self.task_name)
                        result_scalar = result[metric_map[self.task_name]]
                        writer.add_scalar("eval_metric", result_scalar, total_step)

                        if(result_scalar >= max_eval_metric):
                            max_eval_metric = result_scalar
                            select_epoch = epoch+1

                        logger.info(f"current_epoch: {str(epoch+1)}, eval_result: {result_scalar}, select_epoch: {select_epoch}")
                        model.save_pretrained(os.path.join(self.output_dir,str(epoch+1)))
                        model.train()

            if(self.save_strategy=="epoch"):
                logger.info(f"Loading model, select_epoch: {select_epoch}, max_acg: {max_eval_metric}\n")
                if(hasattr(self.model, "merge_config")):
                    model = MergeT5Single.from_pretrained(os.path.join(self.output_dir,str(select_epoch)))
                elif(hasattr(self.model, "prompt_config")):            
                    model = PromptT5Single.from_pretrained(os.path.join(self.output_dir,str(select_epoch)))
                elif(hasattr(self.model, "bank_config")):
                    model = BankT5Single.from_pretrained(os.path.join(self.output_dir,str(select_epoch)))
            else:
                logger.info(f"Loading model, select_step: {select_step}, max_acg: {max_eval_metric}\n")
                if(hasattr(self.model, "merge_config")):
                    model = MergeT5Single.from_pretrained(os.path.join(self.output_dir,str(select_step)))
                elif(hasattr(self.model, "prompt_config")):            
                    model = PromptT5Single.from_pretrained(os.path.join(self.output_dir,str(select_step)))
                elif(hasattr(self.model, "bank_config")):
                    model = BankT5Single.from_pretrained(os.path.join(self.output_dir,str(select_step)))


        if(self.do_test):
            result={}
            all_labels = []
            all_predicts = []
            all_idxs = []
            test_num_examples = len(self.test_dataset)
            logger = self.logger
            eval_bs = self.per_device_eval_batch_size
            # model = self.model
            model.to(self.device)                 
            test_dataLoader, total_test_steps = self.get_dataloader("test")    
            logger.info("***** Running test *****")
            logger.info(f"  Num test examples = {test_num_examples}")
            logger.info(f"  Test task = {self.task_name}")


            model.eval()

            for step, batch_test in enumerate(tqdm(test_dataLoader)):
                batch_test = {k: v.to(self.device) for k, v in batch_test.items()}
                inputs_test = batch_test["input_ids"]
                labels_test = batch_test["labels"]
                if(self.task_name=="multirc"):
                    idxs = batch_test["idxs"]
                    all_idxs.append(idxs)
                with torch.no_grad():
                    predict_ids = model.generate(inputs_test, max_length=self.max_generate_length)
                all_labels.append(labels_test)
                all_predicts.append(predict_ids)

            all_l = []
            all_p = []
            all_a = []
            for labels in all_labels:
                for label in labels:
                    all_l.append(label)
            for predicts in all_predicts:
                for predict in predicts:
                    all_p.append(predict)
            if(self.task_name=="multirc"):
                for idxs in all_idxs:
                    for idx in idxs:
                        all_a.append(idx)

            all_labels = rnn.pad_sequence(all_l, batch_first=True, padding_value=0)
            all_predicts = rnn.pad_sequence(all_p, batch_first=True, padding_value=0)
            data_info = self.data_info["eval"] if self.data_info is not None else None
            eval_prediction = EvalPrediction(predictions=all_predicts, label_ids=all_labels, data_info=data_info, idxs=all_a)
            if(self.task_name=="stsb"):
                eval_prediction = convert_token_ids_to_float(eval_prediction, self.tokenizer)
            elif(self.task_name=="multirc"):
                eval_prediction = convert_token_ids_to_int_m(eval_prediction, self.tokenizer)
            else:
                eval_prediction = convert_token_ids_to_int(eval_prediction, self.tokenizer)
            result = compute_metrics(eval_prediction, dataset_name=self.task_name)
            result_scalar = result[metric_map[self.task_name]]
            logger.info(f"Test Result: {result_scalar}\n")               


class MultiT5Trainer():
    def __init__(
            self, 
            model, 
            tokenizer, 
            data_collator, 
            train_dataset, 
            eval_dataset, 
            test_dataset, 
            pad_token_id, 
            training_args, 
            optimizer_type,
            eval_dataset_list, 
            test_dataset_list, 
            num_list, 
            dataset_dict, 
            data_info_list,
            is_sample
            ):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.pad_token_id = pad_token_id
        self.output_dir = training_args.output_dir
        self.per_device_train_batch_size = training_args.per_device_train_batch_size
        self.per_device_eval_batch_size = training_args.per_device_eval_batch_size
        self.learning_rate = training_args.learning_rate
        self.num_train_epochs = training_args.num_train_epochs
        self.eval_steps = training_args.eval_steps
        self.log_dir = training_args.log_dir
        self.record_dir = training_args.record_dir
        self.do_train = training_args.do_train
        self.do_eval = training_args.do_eval
        self.do_test = training_args.do_test
        self.optimizer_type = optimizer_type
        self.load_path = training_args.load_path
        self.eval_dataset_list = eval_dataset_list
        self.test_dataset_list = test_dataset_list
        self.num_list = num_list
        self.dataset_dict = dataset_dict
        self.data_info_list = data_info_list
        self.weight_decay = training_args.weight_decay
        self.is_sample = is_sample
        self.optimizer = self.get_optimizer()
        self.lr_scheduler = self.get_lr_scheduler()
        self.logger = set_log(self.log_dir, "train")         

    def get_dataloader(self, mode, dataset = None):
        if(mode=="train"):
            if(not(self.is_sample)):
                dataLoader = DataLoader(
                    dataset = self.train_dataset, 
                    shuffle = False,
                    batch_size = self.per_device_train_batch_size,
                    collate_fn = self.data_collator,
                    )
                num = len(dataLoader) * self.num_train_epochs
            else:
                sampler = MultiTaskBatchSampler(
                    dataset_sizes = self.num_list[0], 
                    batch_size = self.per_device_train_batch_size, 
                    temperature=1.0,
                    shuffle = True
                    )
                dataLoader = DataLoader(
                    dataset = self.train_dataset, 
                    batch_sampler = sampler,
                    collate_fn = self.data_collator,
                    )
                num = len(dataLoader) * self.num_train_epochs                
        elif(mode=="eval"):
            dataLoader = DataLoader(
                dataset = dataset, 
                shuffle = False,
                batch_size = self.per_device_eval_batch_size,
                collate_fn = self.data_collator,
                )
            num = len(dataLoader)
        elif(mode=="test"):
            dataLoader = DataLoader(
                dataset = dataset, 
                shuffle = False,
                batch_size = self.per_device_eval_batch_size,
                collate_fn = self.data_collator,
                )
            num = len(dataLoader)
        return  dataLoader, num
    

    def get_optimizer(self):
        optimizer = AdamW(self.model.parameters(), lr = self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


    def get_lr_scheduler(self):
        _, total_train_steps = self.get_dataloader("train")
        if(self.optimizer_type=="constant"):
            schedular = get_scheduler("constant", optimizer=self.optimizer)
        elif(self.optimizer_type=="linear"):
            schedular = get_scheduler("linear", optimizer=self.optimizer, num_warmup_steps=500, num_training_steps=total_train_steps)
        elif(self.optimizer_type=="cosine"):
            schedular = get_scheduler("cosine", optimizer=self.optimizer, num_warmup_steps=500, num_training_steps=total_train_steps)
        return schedular


    def run(self):
        #########################    Train     #################
        model = self.model
        logger = self.logger
        if(self.do_train):
            optimizer = self.optimizer
            lr_scheduler = self.lr_scheduler
            num_epochs = self.num_train_epochs
            train_bs = self.per_device_train_batch_size
            train_dataLoader, total_train_steps = self.get_dataloader("train")
            max_eval_metric = -1000
            model.to(torch.device("cuda"))

            logger.info("***** Running training *****")
            logger.info(f"  Num training examples = {len(self.train_dataset)}")
            logger.info(f"  Num evaluation examples = {len(self.eval_dataset)}")
            logger.info(f"  Num test examples = {len(self.test_dataset)}")
            logger.info(f"  Weight decay = {self.weight_decay}")
            logger.info(f"  Learning rate = {self.learning_rate}")
            logger.info(f"  Num virual tokens = {self.model.super_bank_config.total_virtual_tokens}")
            logger.info(f"  Num bank tokens = {self.model.super_bank_config.total_bank_tokens}")
            logger.info(f"  Num epochs = {num_epochs}")
            logger.info(f"  Batch size per device = {train_bs}")
            logger.info(f"  Total train step = {total_train_steps}")
            logger.info(f"  Num trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

            total_step = 0
            select_step = 0
            select_epoch = 0
            writer = SummaryWriter(log_dir = self.record_dir)
            model.train()

            for epoch in range(num_epochs):
                for step, batch in enumerate(tqdm(train_dataLoader)):
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    output = model(**batch)   
                    loss = output[0]

                    loss.backward()
                    writer.add_scalar("loss", loss, total_step)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    total_step += 1
                    lr = optimizer.param_groups[0]["lr"]

                    if(total_step % self.eval_steps ==0):
                        # logger.info(f"steps : {total_step}, loss : {loss}, lr : {lr}\n")
                        
                        #########################    Evaluate     #################
                        if(self.do_eval):
                            eval_bs = self.per_device_eval_batch_size
                            logger.info("***** Running evaluation *****")
                            logger.info(f"  Instantaneous batch size per device = {eval_bs}")
                            model.eval()
                            dataset_dict = self.dataset_dict
                            current = 0
                            results = {}
                            num_dataset_type = len(dataset_dict)
                            avg = 0

                            for eval_dataset in self.eval_dataset_list:
                                result={}
                                self.task_name = dataset_dict[current]
                                if(self.task_name in ["squad", "record"]):
                                    continue
                                eval_num_examples = len(eval_dataset)
                                eval_dataLoader, total_eval_steps = self.get_dataloader("eval", eval_dataset)
                                logger.info(f"  current dataset = {self.task_name}")
                                logger.info(f"  Eval examples = {eval_num_examples}")
                                logger.info(f"  Total eval step = {total_eval_steps}")

                                if(self.task_name in ["sst2", "mnli", "qnli", "qqp", "rte", "cola", "mrpc", "stsb", "boolq", "cb",
                                                                    "multirc", "wic", "wscFixed", "winogrande", "yelp", "scitail", "paws"]):
                                    num_to_decode = 3
                                elif(self.task_name in ["record", "squad"]):
                                    num_to_decode = 20
                                                                            
                                model.eval()
                                all_labels = []
                                all_predicts = []
                                with torch.no_grad():
                                    for step, batch_eval in enumerate(tqdm(eval_dataLoader)):
                                        batch_eval = {k: v.to(self.device) for k, v in batch_eval.items()}
                                        inputs_eval = batch_eval["input_ids"]
                                        labels_eval = batch_eval["labels"]
                                        predicts = model.generate(inputs_eval, max_length=num_to_decode)
                                        all_labels.append(labels_eval.to(torch.device("cpu")))
                                        all_predicts.append(predicts.to(torch.device("cpu")))
                                            
                                all_l = []
                                all_p = []
                                for labels in all_labels:
                                    for label in labels:
                                        all_l.append(label)
                                for predicts in all_predicts:
                                    for predict in predicts:
                                        all_p.append(predict)
                                
                                data_info = self.data_info_list[self.task_name]["eval"] if self.data_info_list[self.task_name] is not None else None
                                all_labels = rnn.pad_sequence(all_l, batch_first=True, padding_value=0)
                                all_predicts = rnn.pad_sequence(all_p, batch_first=True, padding_value=0)
                                eval_prediction = EvalPrediction(predictions=all_predicts, label_ids=all_labels, data_info=data_info)
                                if(self.task_name=="stsb"):
                                    eval_prediction = convert_token_ids_to_float(eval_prediction, self.tokenizer)
                                elif(self.task_name in ["mnli", "qnli", "qqp", "sst2", "rte", "cola", "mrpc"]):
                                    eval_prediction = convert_token_ids_to_int(eval_prediction, self.tokenizer)
                                else:
                                    eval_prediction = convert_token_ids_to_text(eval_prediction, self.tokenizer)
                                result = compute_metrics(eval_prediction, dataset_name=self.task_name)
                                result_scalar = result[metric_map[self.task_name]]
                                avg+=result_scalar
                                # writer.add_scalar("eval_metric", result_scalar, total_step)
                                logger.info(f"current_step: {str(total_step)}, eval_task: {self.task_name}, eval_result: {result_scalar}")
                                current+=1

                            avg = avg/current
                            if(avg >= max_eval_metric):
                                max_eval_metric = avg
                                select_step = total_step
                            logger.info(f"current_step: {str(total_step)}, avg_eval_result: {avg}, select_step: {select_step}, max_acg: {max_eval_metric}\n")
                            model.save_pretrained(os.path.join(self.output_dir,str(total_step)))
                            model.train()
                            
                        model.save_pretrained(os.path.join(self.output_dir,str(total_step)))
                        model.train()

            logger.info(f"Loading model, select_step: {select_step}, max_acg: {max_eval_metric}\n")
            model = MultiBankT5Single.from_pretrained(os.path.join(self.output_dir,str(select_step)))

        if(self.do_test):
            model.to(self.device)
            logger = self.logger
            test_bs = self.per_device_eval_batch_size
            logger.info("***** Running test *****")
            logger.info(f"  Instantaneous batch size per device = {test_bs}")
            dataset_dict = self.dataset_dict
            current = 0
            results = {}
            num_dataset_type = len(dataset_dict)
            avg = 0
            for test_dataset in self.test_dataset_list:
                result={}
                self.task_name = dataset_dict[current]
                if(self.task_name in ["squad", "record"]):
                    continue
                test_num_examples = len(test_dataset)
                test_dataLoader, total_test_steps = self.get_dataloader("test", test_dataset)
                logger.info(f"  current dataset = {dataset_dict[current]}")
                logger.info(f"  Eval examples = {test_num_examples}")
                logger.info(f"  Total eval step = {total_test_steps}")
                all_labels = []
                all_predicts = []
                # model = self.model        

                if(self.task_name in ["sst2", "mnli", "qnli", "qqp", "rte", "cola", "mrpc", "stsb", "boolq", "cb",
                                    "multirc", "wic", "wscFixed", "winogrande", "yelp", "scitail", "paws"]):
                    num_to_decode = 3
                elif(self.task_name in ["record", "squad"]):
                    num_to_decode = 20
                                                            
                model.eval()
                all_labels = []
                all_predicts = []
                with torch.no_grad():
                    for step, batch_test in enumerate(tqdm(test_dataLoader)):
                        batch_test = {k: v.to(self.device) for k, v in batch_test.items()}
                        inputs_test = batch_test["input_ids"]
                        labels_test = batch_test["labels"]
                        if(self.task_name=="multirc"):
                            idxs = batch_eval["idsx"]
                        predicts = model.generate(inputs_test, max_length=num_to_decode)
                        all_labels.append(labels_test.to(torch.device("cpu")))
                        all_predicts.append(predicts.to(torch.device("cpu")))
                            
                all_l = []
                all_p = []
                for labels in all_labels:
                    for label in labels:
                        all_l.append(label)
                for predicts in all_predicts:
                    for predict in predicts:
                        all_p.append(predict)
                
                data_info = self.data_info_list[self.task_name]["test"] if self.data_info_list[self.task_name] is not None else None
                all_labels = rnn.pad_sequence(all_l, batch_first=True, padding_value=0)
                all_predicts = rnn.pad_sequence(all_p, batch_first=True, padding_value=0)
                test_prediction = EvalPrediction(predictions=all_predicts, label_ids=all_labels, data_info=data_info)
                if(self.task_name=="stsb"):
                    test_prediction = convert_token_ids_to_float(test_prediction, self.tokenizer)
                elif(self.task_name in ["mnli", "qnli", "qqp", "sst2", "rte", "cola", "mrpc"]):
                    test_prediction = convert_token_ids_to_int(test_prediction, self.tokenizer)
                else:
                    test_prediction = convert_token_ids_to_text(test_prediction, self.tokenizer)
                result = compute_metrics(test_prediction, dataset_name=self.task_name)
                result_scalar = result[metric_map[self.task_name]]
                avg+=result_scalar

                logger.info(f"test_task: {self.task_name}, test_result: {result_scalar}")
                current+=1

            avg = avg/current

            logger.info(f"avg_eval_result: {avg}\n")

