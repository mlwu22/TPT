from utils import round_stsb_target, pad_punctuation
import os
import datasets 
from datasets import load_from_disk
from collections import OrderedDict
import numpy as np
import functools
from torch.utils.data.dataset import Dataset
from torch.utils.data import ConcatDataset
import torch.nn.utils.rnn as rnn
import torch
import re
import collections
from torch.utils.data import Subset


def tokenization_dataset(task_name, task_type, tokenizer, seed=42):
    data_info = None
    cur_path = os.path.dirname(os.path.abspath(__file__))
    raw_dataset = load_from_disk(os.path.join(cur_path, "data", task_type, task_name))
    if(task_name=="sst2"):
        tokenized_datasets = raw_dataset.map(lambda sample: sst2_tokenization(sample, tokenizer))
        tokenized_datasets = tokenized_datasets.remove_columns(['sentence', "label", "idx"])
    elif(task_name=="cola"):
        tokenized_datasets = raw_dataset.map(lambda sample: cola_tokenization(sample, tokenizer))
        tokenized_datasets = tokenized_datasets.remove_columns(['sentence', "label", "idx"])
    elif(task_name=="mnli"):
        tokenized_datasets = raw_dataset.map(lambda sample: mnli_tokenization(sample, tokenizer))
        tokenized_datasets = tokenized_datasets.remove_columns(["hypothesis", 'premise', "label", "idx"])
    elif(task_name=="mrpc"):
        tokenized_datasets = raw_dataset.map(lambda sample: mrpc_tokenization(sample, tokenizer))
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", 'sentence2', "label", "idx"])
    elif(task_name=="qnli"):
        tokenized_datasets = raw_dataset.map(lambda sample: qnli_tokenization(sample, tokenizer))
        tokenized_datasets = tokenized_datasets.remove_columns(["question", 'sentence', "label", "idx"])
    elif(task_name=="qqp"):
        tokenized_datasets = raw_dataset.map(lambda sample: qqp_tokenization(sample, tokenizer))
        tokenized_datasets = tokenized_datasets.remove_columns(["question1", 'question2', "label", "idx"])
    elif(task_name=="rte"):
        tokenized_datasets = raw_dataset.map(lambda sample: rte_tokenization(sample, tokenizer))
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", 'sentence2', "label", "idx"])
    elif(task_name=="stsb"):
        tokenized_datasets = raw_dataset.map(lambda sample: stsb_tokenization(sample, tokenizer))
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", 'sentence2', "label", "idx"])
    elif(task_name=="boolq"):
        tokenized_datasets = raw_dataset.map(lambda sample: boolq_tokenization(sample, tokenizer))
        tokenized_datasets = tokenized_datasets.remove_columns(["question", 'passage', "label", "idx"])  
    elif(task_name=="cb"):
        tokenized_datasets = raw_dataset.map(lambda sample: cb_tokenization(sample, tokenizer))
        tokenized_datasets = tokenized_datasets.remove_columns(["premise", 'hypothesis', "label", "idx"])   
    elif(task_name=="multirc"):
        tokenized_datasets = raw_dataset.map(lambda sample: multirc_tokenization(sample, tokenizer))
        tokenized_datasets = tokenized_datasets.remove_columns(["question", "answer", "paragraph" ,"label", "idx"])
    elif(task_name=="wic"):
        tokenized_datasets = raw_dataset.map(lambda sample: wic_tokenization(sample, tokenizer))
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", 'sentence2', "word", "label", "idx"])
    elif(task_name=="wsc.fixed"):
        tokenized_datasets = raw_dataset.map(lambda sample: wscFixed_tokenization(sample, tokenizer))
        tokenized_datasets = tokenized_datasets.remove_columns(["text", 'span1_text', "span2_text", "label", "idx"]) 
    elif(task_name=="squad"):
        tokenized_datasets = raw_dataset.map(functools.partial(squad_tokenization, tokenizer = tokenizer), batched = True, remove_columns=raw_dataset["train"].column_names)
    elif(task_name=="record"):
        tokenized_datasets = raw_dataset.map(functools.partial(record_tokenization, tokenizer = tokenizer), batched=True, remove_columns=raw_dataset["train"].column_names)
        data_info = {"eval": tokenized_datasets["validation"]['extra_fields'],
                    "test": tokenized_datasets["test"]['extra_fields'],
                    "train": tokenized_datasets["train"]['extra_fields']}
        tokenized_datasets = tokenized_datasets.remove_columns(["extra_fields"])
    
    train_dataset = tokenized_datasets["train"]
    if(task_name in ["multirc"]):
        train_dataset = train_dataset.remove_columns(["idxs"]) 
    permuted_indices = np.random.RandomState(seed=seed).permutation(len(train_dataset)).tolist()
    eval_dataset = tokenized_datasets["validation_matched"] if task_name == "mnli" else tokenized_datasets["validation"]

    if(task_name in ["record", "squad"]):
        num_eval_data = 1000
        test_dataset = Subset(dataset=eval_dataset, indices=[i for i in range(len(eval_dataset))])
        eval_dataset = Subset(dataset=train_dataset, indices=permuted_indices[:num_eval_data])
        train_dataset = Subset(dataset=train_dataset, indices=permuted_indices[num_eval_data:])
    elif(task_name in ["mnli", "qnli", "qqp", "sst2", "cola", "stsb", "mrpc", "rte", "boolq", "cb", "multirc", "wsc.fixed", "wic"]):
        test_dataset = Subset(dataset=eval_dataset, indices=[i for i in range(len(eval_dataset))])
        eval_dataset = Subset(dataset=eval_dataset, indices=[i for i in range(len(eval_dataset))])
        train_permuted_indices = np.random.RandomState(seed=seed).permutation(len(train_dataset)).tolist()
        train_dataset = Subset(dataset=train_dataset, indices=train_permuted_indices)
    return train_dataset, eval_dataset, test_dataset, data_info
   
def get_multi_dataset(task_list, tokenizer, seed=42):
    train_dataset_list = []
    eval_dataset_list = []
    test_dataset_list = []
    data_info_list = {}
    train_num_list = []
    eval_num_list = []
    test_num_list = []
    dataset_dict = {}       
    num_dataset_type = 0
    for task_name in task_list:
        task_type = TASK_NAME_TO_TASK_TYPE[task_name]
        train, eval, test, data_info= tokenization_dataset(task_name, task_type, tokenizer, seed=seed)
        train_dataset_list.append(train)
        eval_dataset_list.append(eval)
        test_dataset_list.append(test)
        data_info_list[task_name] = data_info
        train_num_list.append(len(train))  
        eval_num_list.append(len(eval))
        test_num_list.append(len(test))
        dataset_dict[num_dataset_type] = task_name
        num_dataset_type+=1
   
    train_dataset = ConcatDataset(train_dataset_list)
    train_permuted_indices = np.random.RandomState(seed=seed).permutation(len(train_dataset)).tolist()
    train_dataset = Subset(dataset=train_dataset, indices=train_permuted_indices)
    eval_dataset = ConcatDataset(eval_dataset_list)
    test_dataset = ConcatDataset(test_dataset_list)

    return train_dataset, train_dataset_list, train_num_list,\
            eval_dataset, eval_dataset_list, eval_num_list,\
            test_dataset, test_dataset_list, test_num_list,\
            dataset_dict, num_dataset_type, data_info_list

def create_collate_fn(task_name="mrpc"):
    if(task_name=="multirc"):
        def collate_fn(examples):
            input_ids = []
            target_ids = []
            idxs = []

            for example in examples:
                input_ids.append(torch.tensor(example["input_ids"]))
                target_ids.append(torch.tensor(example["labels"]))
                if("idxs" in example.keys()):
                    idxs.append(torch.tensor(example["idxs"]))

            input_ids = rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
            target_ids = rnn.pad_sequence(target_ids, batch_first=True, padding_value=0)     

            if(len(idxs)>1):
                idxs = rnn.pad_sequence(idxs, batch_first=True, padding_value=0)
                output_batch = {
                    "input_ids": input_ids,
                    "labels": target_ids,
                    "idxs" : idxs
                }
            else:
                output_batch = {
                    "input_ids": input_ids,
                    "labels": target_ids
                }
            return output_batch
    else:
        def collate_fn(examples):
            input_ids = []
            target_ids = []

            for example in examples:
                input_ids.append(torch.tensor(example["input_ids"]))
                target_ids.append(torch.tensor(example["labels"]))

            input_ids = rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
            target_ids = rnn.pad_sequence(target_ids, batch_first=True, padding_value=0)   

            output_batch = {
                "input_ids": input_ids,
                "labels": target_ids
            }
            return output_batch
    return collate_fn

def sst2_tokenization(example, tokenizer):
    prefix = "Classification task:sst2. Choose a label from list:[0, 1] for this context: "
    input_str = example["sentence"]
    input_str = prefix + input_str
    input_str += " Among them, 0 represents negative, 1 represents positive. Answer:"
    label_dict = {"-1":"-1", "0":"0", "1":"1"}
    label = example["label"]
    target_str = label_dict[str(label)]

    tokenized_data = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=256, padding = "max_length")
    input_ids = tokenized_data.input_ids.squeeze(0)
    attention_mask = tokenized_data.attention_mask.squeeze(0)
    target_ids = tokenizer(target_str, return_tensors="pt", truncation=True, max_length=3, padding = "max_length").input_ids
    target_ids = target_ids.squeeze(0)

    output_batch = {
                "input_ids": input_ids,
                "labels": target_ids,
                "attention_mask": attention_mask
            }
    
    return output_batch

def cola_tokenization(example, tokenizer):
    prefix = "Classification task:cola. Choose a label from list:[0, 1] for this context: "
    input_str = example["sentence"]
    input_str = prefix + input_str
    input_str += " Among them, 0 represents unacceptable, 1 represents acceptable. Answer:"
    label_dict = {"-1":"-1", "0":"0", "1":"1"}
    label = example["label"]
    target_str = label_dict[str(label)]

    tokenized_data = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=256, padding = "max_length")
    input_ids = tokenized_data.input_ids.squeeze(0)
    attention_mask = tokenized_data.attention_mask.squeeze(0)
    target_ids = tokenizer(target_str, return_tensors="pt", truncation=True, max_length=3, padding = "max_length").input_ids
    target_ids = target_ids.squeeze(0)

    output_batch = {
                "input_ids": input_ids,
                "labels": target_ids,
                "attention_mask": attention_mask
            }
    
    return output_batch

def mnli_tokenization(example, tokenizer):
    prefix = "Classification task:mnli. Choose a label from list:[0, 1, 2] for this context: "
    str1 = example["premise"]
    str2 = example["hypothesis"]
    input_str =prefix + "premise: " + str1 + "hypothesis: " + str2
    input_str += " Among them, 0 represents entailment, 1 represents neutral, 2 represents contradiction. Answer:"
    label_dict = {"-1":"-1", "0":"0", "1":"1", "2":"2"}   
    label = example["label"]
    target_str = label_dict[str(label)]

    tokenized_data = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=256, padding = "max_length")
    input_ids = tokenized_data.input_ids.squeeze(0)
    attention_mask = tokenized_data.attention_mask.squeeze(0)
    target_ids = tokenizer(target_str, return_tensors="pt", truncation=True, max_length=3, padding = "max_length").input_ids
    target_ids = target_ids.squeeze(0)

    output_batch = {
                "input_ids": input_ids,
                "labels": target_ids,
                "attention_mask": attention_mask
            }
    
    return output_batch

def mrpc_tokenization(example, tokenizer):
    prefix = "Classification task: Choose a label from list:[0 , 1] for this context:"
    str1 = example["sentence1"]
    str2 = example["sentence2"]
    input_str = prefix + "Sentence1: " + str1 + " Sentence2: " + str2
    input_str += " Among them, 0 represents not_equivalent, 1 represents equivalent. Answer:"
    label_dict = {"-1":"-1", "0":"0", "1":"1"} 
    label = example["label"]
    target_str = label_dict[str(label)]

    tokenized_data = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=256, padding = "max_length")
    input_ids = tokenized_data.input_ids.squeeze(0)
    attention_mask = tokenized_data.attention_mask.squeeze(0)
    target_ids = tokenizer(target_str, return_tensors="pt", truncation=True, max_length=3, padding = "max_length").input_ids
    target_ids = target_ids.squeeze(0)

    output_batch = {
                "input_ids": input_ids,
                "labels": target_ids,
                "attention_mask": attention_mask
            }
    
    return output_batch

def qnli_tokenization(example, tokenizer):
    prefix = "Classification task: Choose a label from list:[0, 1] for this context:"
    str1 = example["question"]
    str2 = example["sentence"]
    input_str = prefix + "question: " + str1 + "sentence: " + str2
    input_str += " Among them, 0 represents not_entailment, 1 represents entailment. Answer:"
    label_dict = {"-1":"-1", "0":"0", "1":"1"}  
    label = example["label"]
    target_str = label_dict[str(label)]

    tokenized_data = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=256, padding = "max_length")
    input_ids = tokenized_data.input_ids.squeeze(0)
    attention_mask = tokenized_data.attention_mask.squeeze(0)
    target_ids = tokenizer(target_str, return_tensors="pt", truncation=True, max_length=3, padding = "max_length").input_ids
    target_ids = target_ids.squeeze(0)

    output_batch = {
                "input_ids": input_ids,
                "labels": target_ids,
                "attention_mask": attention_mask
            }
    
    return output_batch

def qqp_tokenization(example, tokenizer):
    prefix = "Classification task:qqp. Choose a label from list:[0, 1] for this context: "
    str1 = example["question1"]
    str2 = example["question2"]
    input_str = prefix+ "sentence1: " + str1 + ".sentence2: " + str2
    input_str += " Among them, 0 represents not_duplicate, 1 represents duplicate. Answer:"
    label_dict = {"-1":"-1", "0":"0", "1":"1"}
    label = example["label"]
    target_str = label_dict[str(label)]

    tokenized_data = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=256, padding = "max_length")
    input_ids = tokenized_data.input_ids.squeeze(0)
    attention_mask = tokenized_data.attention_mask.squeeze(0)
    target_ids = tokenizer(target_str, return_tensors="pt", truncation=True, max_length=3, padding = "max_length").input_ids
    target_ids = target_ids.squeeze(0)

    output_batch = {
                "input_ids": input_ids,
                "labels": target_ids,
                "attention_mask": attention_mask
            }
    
    return output_batch

def rte_tokenization(example, tokenizer):
    prefix = "Classification task: Choose a label from list:[0, 1] for this context: "        
    str1 = example["sentence1"]
    str2 = example["sentence2"]
    input_str =prefix + "sentence1: " + str1 + ".sentence2: " + str2
    input_str += " Among them, 0 represents not_entailment, 1 represents entailment. Answer:"
    label_dict = {"-1":"-1", "0":"0", "1":"1"}  
    label = example["label"]
    target_str = label_dict[str(label)]

    tokenized_data = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=256, padding = "max_length")
    input_ids = tokenized_data.input_ids.squeeze(0)
    attention_mask = tokenized_data.attention_mask.squeeze(0)
    target_ids = tokenizer(target_str, return_tensors="pt", truncation=True, max_length=3, padding = "max_length").input_ids
    target_ids = target_ids.squeeze(0)

    output_batch = {
                "input_ids": input_ids,
                "labels": target_ids,
                "attention_mask": attention_mask
            }
    
    return output_batch

def stsb_tokenization(example, tokenizer):
    prefix = "Regression task: Choose a correlation score for this context: "

    str1 = example["sentence1"]
    str2 = example["sentence2"]

    input_str = prefix + str1+ "." + str2 
    label = example["label"]
    target_str = str(round_stsb_target(label))

    tokenized_data = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=256, padding = "max_length")
    input_ids = tokenized_data.input_ids.squeeze(0)
    attention_mask = tokenized_data.attention_mask.squeeze(0)
    target_ids = tokenizer(target_str, return_tensors="pt", truncation=True, max_length=3, padding = "max_length").input_ids
    target_ids = target_ids.squeeze(0)

    output_batch = {
                "input_ids": input_ids,
                "labels": target_ids,
                "attention_mask": attention_mask
            }
    
    return output_batch

def boolq_tokenization(example, tokenizer):
    label_dict = {"-1":"-1","0":"0", "1":"1"} 
    prefix = "Classification task: Choose a label from list:[0, 1] for this context: "
    question = example["question"]
    passage = example["passage"]
    input_str = prefix + "question: " + question + "passage: " + passage
    input_str += " Among them, 0 represents False, 1 represents True. Answer:"
    label = example["label"]
    target_str = label_dict[str(label)]

    
    tokenized_data = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=256, padding = "max_length")
    input_ids = tokenized_data.input_ids.squeeze(0)
    attention_mask = tokenized_data.attention_mask.squeeze(0)
    target_ids = tokenizer(target_str, return_tensors="pt", truncation=True, max_length=3, padding = "max_length").input_ids
    target_ids = target_ids.squeeze(0)

    output_batch = {
                "input_ids": input_ids,
                "labels": target_ids,
                "attention_mask": attention_mask
            }
    
    return output_batch

def cb_tokenization(example, tokenizer):
    label_dict = {"-1":"-1","0":"0", "1":"1", "2":"2"} 
    prefix = "Classification task: Choose a label from list:[0, 1, 2] for this context: "
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    input_str = prefix + "premise: " + premise + "hypothesis: " + hypothesis
    input_str += " Among them, 0 represents entailment, 1 represents contradiction and 2 represents neutral. Answer:"
    label = example["label"]
    target_str = label_dict[str(label)]

    tokenized_data = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=256, padding = "max_length")
    input_ids = tokenized_data.input_ids.squeeze(0)
    attention_mask = tokenized_data.attention_mask.squeeze(0)
    target_ids = tokenizer(target_str, return_tensors="pt", truncation=True, max_length=3, padding = "max_length").input_ids
    target_ids = target_ids.squeeze(0)

    output_batch = {
                "input_ids": input_ids,
                "labels": target_ids,
                "attention_mask": attention_mask
            }
    
    return output_batch

def multirc_tokenization(example, tokenizer):

    def remove_markup(text):
        """Removes the HTML markup."""
        text = re.sub('<br>', ' ', text)
        text = re.sub('<(/)?b>', '', text)
        return text 

    label_dict = {"-1":"-1", "0":"0", "1":"1"}
    idx = []
    for key, value in example['idx'].items():
        idx.append(value)
    prefix = "Classification task: Choose a label from list:[0, 1] for this context: "
    question = remove_markup(example["question"])
    answer = remove_markup(example["answer"])
    paragraph = remove_markup(example["paragraph"])
    input_str = prefix + "question: " + question + "answer: " + answer +"paragraph: " + paragraph
    input_str += " Among them, 0 represents False, 1 represents True. Answer:"

    label = example["label"]
    target_str = label_dict[str(label)]

    tokenized_data = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=384, padding = "max_length")
    input_ids = tokenized_data.input_ids.squeeze(0)
    attention_mask = tokenized_data.attention_mask.squeeze(0)
    target_ids = tokenizer(target_str, return_tensors="pt", truncation=True, max_length=3, padding = "max_length").input_ids
    target_ids = target_ids.squeeze(0)

    output_batch = {
                "input_ids": input_ids,
                "labels": target_ids,
                "attention_mask": attention_mask,
                "idxs": idx
            }

    return output_batch

def wic_tokenization(example, tokenizer):
    label_dict = {"-1":"-1","0":"0", "1":"1"} 
    prefix = "Classification task: Choose a label from list:[0, 1] for this context: "
    sentence1 = example["sentence1"]
    sentence2 = example["sentence2"]
    word = example["word"]
    input_str = prefix + "Sentence1: " + sentence1 + "Sentence2: " + sentence2 + "Word: " + word
    input_str += " Among them, 0 represents False, 1 represents True. Answer:"
    label = example["label"]
    target_str = label_dict[str(label)]

    tokenized_data = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=256, padding = "max_length")
    input_ids = tokenized_data.input_ids.squeeze(0)
    attention_mask = tokenized_data.attention_mask.squeeze(0)
    target_ids = tokenizer(target_str, return_tensors="pt", truncation=True, max_length=3, padding = "max_length").input_ids
    target_ids = target_ids.squeeze(0)

    output_batch = {
                "input_ids": input_ids,
                "labels": target_ids,
                "attention_mask": attention_mask
            }
    
    return output_batch

def wscFixed_tokenization(example, tokenizer):
    def _mark_span(text, span_str, span_idx, mark):
        pattern_tmpl = r'^((?:\S+\s){N})(W)'
        pattern = re.sub('N', str(span_idx), pattern_tmpl)
        pattern = re.sub('W', span_str, pattern)
        return re.sub(pattern, r'\1{0} \2 {0}'.format(mark), text)

    label_dict = {"-1":"-1","0":"0", "1":"1"} 
    prefix = "Classification task: Choose a label from list:[0, 1] for this context: "
    text = example['text']
    text = _mark_span(text, example['span1_text'], example['span1_index'], '*')
    span2_index = example['span2_index'] + 2 * int(example['span1_index'] < example['span2_index'])
    text = _mark_span(text, example['span2_text'], span2_index, '#')
    input_str = prefix + "text: " + text
    input_str += " Among them, 0 represents False, 1 represents True. Answer:"

    label = example["label"]
    target_str = label_dict[str(label)]

    tokenized_data = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=256, padding = "max_length")
    input_ids = tokenized_data.input_ids.squeeze(0)
    attention_mask = tokenized_data.attention_mask.squeeze(0)
    target_ids = tokenizer(target_str, return_tensors="pt", truncation=True, max_length=3, padding = "max_length").input_ids
    target_ids = target_ids.squeeze(0)

    output_batch = {
                "input_ids": input_ids,
                "labels": target_ids,
                "attention_mask": attention_mask
            }
    
    return output_batch

def record_tokenization(batch, tokenizer):
    
    new_batch = collections.defaultdict(list)

    prefix = "Question and answer task: Given this query:"
    supplement1 = "and this passage: "
    supplement2 = " Choose the entity or entities from this list: "

    keys = batch.keys()
    for values in zip(*batch.values()):
        example = {k: v for k, v in zip(keys, values)}

        passage = example['passage']
        passage = re.sub(r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', passage)
        passage = re.sub(r'\n@highlight\n', '. ', passage)

        query = example['query']
        entities = ', '.join(example["entities"])
        entities = "[" + entities + "]"
        input_str = prefix + query + supplement1 + passage + supplement2 + entities

        answers = example["answers"]

        num_answers = len(example["answers"])
        num_duplicates = np.maximum(1, num_answers)
        for i in range(num_duplicates):
            input_ids = tokenizer(input_str, return_tensors="pt", truncation=True, max_length=256).input_ids
            new_batch["input_ids"].extend(input_ids)
            if(num_answers>0):
                target_str = answers[i]
                target_ids = tokenizer(target_str, return_tensors="pt", truncation=True, max_length = 20, add_special_tokens = False).input_ids
                new_batch["labels"].extend(target_ids)
            else:
                new_batch["labels"].extend(torch.zeros(1,1))
       
        new_batch["extra_fields"].extend([{"answers": example["answers"]}]*num_duplicates)
    
    return new_batch

def squad_tokenization(batch, tokenizer):

    new_batch = collections.defaultdict(list)
    prefix = "Question and answer task: Answer this question:"

    keys = batch.keys()
    for values in zip(*batch.values()):
        example = {k: v for k, v in zip(keys, values)}

        question = example['question']
        context = example['context']

        input_str = prefix + question + " for this context:" + context
        target_str = pad_punctuation(example['answers']["text"])

        new_batch["input_ids"].extend(tokenizer(input_str, return_tensors="pt", truncation=True, max_length=256).input_ids)
        new_batch["labels"].extend(tokenizer(target_str, return_tensors="pt", truncation=True, max_length=20, add_special_tokens = False).input_ids)
    
    return new_batch


TASK_NAME_TO_TASK_TYPE={
    "sst2":"glue",
    "cola":"glue",
    "qnli":"glue",
    "qqp":"glue",
    "rte":"glue",
    "stsb":"glue",
    "mrpc":"glue",
    "mnli":"glue",
    "boolq":"super_glue",
    "cb":"super_glue",
    "multirc":"super_glue",
    "record":"super_glue",
    "squad":"",
    "wic":"super_glue",
    "wsc.fixed":"super_glue"
}