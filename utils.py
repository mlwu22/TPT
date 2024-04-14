import numpy as np
import os
import torch
import json
import random
import regex 
from torch.utils.data import Sampler
from typing import TypeVar, Optional, List
import evaluate


class EvalPrediction():
    def __init__(self, predictions, label_ids, data_info=None, idxs= None): 
        self.predictions = predictions
        self.label_ids = label_ids
        self.data_info = data_info
        self.idxs = idxs

def round_stsb_target(label):
    return np.round((label * 5) / 5, decimals=1)

def convert_string_to_int(string, default=0):
    try:
        res = int(string)
        if res==0 or res==1:
            return res
        else:
            return default
    except ValueError:
        return default
    
def convert_string_to_float(string, default=0.0):
    try:
        res = float(string)
        x = np.arange(0.0, 5.1, 0.2)
        x_list = x.tolist()
        x_list = [round(x,1) for x in x_list]
        if res>=0 and res<=5:
            if res in x_list:
                return res
            else:
                return round(res, 1)
        else:
            return default
    except ValueError:
        return default

def string_to_float(pred_str, label_str, data_info):
    pred_str = [convert_string_to_float(pred) for pred in pred_str]
    label_str = [convert_string_to_float(label) for label in label_str]
    return pred_str, label_str

def string_to_int(pred_str, label_str, data_info):
    pred_str = [convert_string_to_int(pred) for pred in pred_str]
    label_str = [convert_string_to_int(label) for label in label_str]
    return pred_str, label_str

def pad_punctuation(text):
    """Re-implementation of _pad_punctuation in t5. This function adds spaces
    around punctuation. While this pads punctuation as expected, it has the 
    unexpected effected of padding certain unicode characters with accents, with
    spaces as well. For instance: "François" becomes "Fran ç ois"""
    # Pad everything except for: underscores (_), whitespace (\s),
    # numbers (\p{N}), letters (\p{L}) and accent characters (\p{M}).
    text = regex.sub(r'([^_\s\p{N}\p{L}\p{M}])', r' \1 ', str(text))
    # Collapse consecutive whitespace into one space.
    text = regex.sub(r'\s+', ' ', text)
    return text

def convert_token_ids_to_int_m(eval_pred: EvalPrediction, tokenizer):
    pred_str = tokenizer.batch_decode(eval_pred.predictions, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(eval_pred.label_ids, skip_special_tokens=True)
    pred_strs = lmap(str.strip, pred_str)
    label_strs = lmap(str.strip, label_str)
    idxs = eval_pred.idxs
    predictions = []
    for i in range(len(idxs)):
        predictions.append({'idx':{'answer': idxs[i][0], 'paragraph': idxs[i][1], 'question': idxs[i][2]}, 'prediction':convert_string_to_int(pred_strs[i])})
    
    eval_pred.predictions = predictions
    eval_pred.label_ids = [convert_string_to_int(label_str) for label_str in label_strs]
    return eval_pred

def convert_token_ids_to_int(eval_pred: EvalPrediction, tokenizer):
    pred_str = tokenizer.batch_decode(eval_pred.predictions, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(eval_pred.label_ids, skip_special_tokens=True)
    pred_strs = lmap(str.strip, pred_str)
    label_strs = lmap(str.strip, label_str)
    eval_pred.predictions = np.array([convert_string_to_int(pred_str) for pred_str in pred_strs])
    eval_pred.label_ids = np.array([convert_string_to_int(label_str) for label_str in label_strs])
    return eval_pred

def convert_token_ids_to_float(eval_pred: EvalPrediction, tokenizer):
    pred_str = tokenizer.batch_decode(eval_pred.predictions, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(eval_pred.label_ids, skip_special_tokens=True)
    pred_strs = lmap(str.strip, pred_str)
    label_strs = lmap(str.strip, label_str)
    eval_pred.predictions = np.array([convert_string_to_float(pred_str) for pred_str in pred_strs])
    eval_pred.label_ids = np.array([convert_string_to_float(label_str) for label_str in label_strs])
    return eval_pred

def convert_token_ids_to_text(eval_pred: EvalPrediction, tokenizer):
    pred_str = tokenizer.batch_decode(eval_pred.predictions, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(eval_pred.label_ids, skip_special_tokens=True)
    pred_strs = lmap(str.strip, pred_str)
    label_strs = lmap(str.strip, label_str)
    eval_pred.predictions = pred_strs
    eval_pred.label_ids = label_strs
    return eval_pred

def lmap(f, x):
    return list(map(f, x))

def compute_metrics(eval_pred: EvalPrediction, dataset_name):
    cur_path = os.path.dirname(os.path.abspath(__file__))
    if(dataset_name in ["mnli", "qnli", "qqp", "sst2", "cola", "rte", "stsb", "mrpc"]):
        path = os.path.join(cur_path,"evaluate", "metrics", "glue")
        metric_func = evaluate.load(path=path, config_name=dataset_name)
    elif(dataset_name in ["squad"]):
        path = os.path.join(cur_path,"evaluate", "metrics", "squad")
        metric_func = evaluate.load(path=path)
    elif(dataset_name in ["record", "cb", "boolq", "wic", "multirc", "wsc.fixed"]):
        path = os.path.join(cur_path,"evaluate", "metrics", "super_glue")
        metric_func = evaluate.load(path=path, config_name=dataset_name)
    result = metric_func.compute(predictions=eval_pred.predictions, references=eval_pred.label_ids)
    if len(result.keys()) > 1:
        result["averaged_scores"] = np.mean(list(result.values())).item()
    return result


class TrainingArgs():
  def __init__(
        self, 
        output_dir, 
        per_device_train_batch_size, 
        per_device_eval_batch_size, 
        learning_rate, 
        num_train_epochs, 
        eval_steps, 
        log_dir, 
        record_dir, 
        load_path,
        save_strategy, 
        weight_decay, 
        do_train = False, 
        do_eval = False, 
        do_test = False,
        warm_up = 500,
        seed = 42
        ):
    
    self.output_dir = output_dir
    self.per_device_train_batch_size = per_device_train_batch_size
    self.per_device_eval_batch_size = per_device_eval_batch_size
    self.learning_rate = learning_rate
    self.num_train_epochs = num_train_epochs
    self.eval_steps = eval_steps
    self.do_train = do_train
    self.do_eval = do_eval
    self.do_test = do_test
    self.log_dir = log_dir
    self.record_dir = record_dir
    self.load_path = load_path
    self.save_strategy = save_strategy
    self.weight_decay = weight_decay
    self.warm_up = warm_up
    self.seed = seed
        
class PromptConfig():
    def __init__(self, 
                 total_virtual_tokens, 
                 base_model_path, 
                 soft_embedding_init_type, 
                 task_name
                 ):
        self.total_virtual_tokens = total_virtual_tokens
        self.base_model_path = base_model_path
        self.task_name = task_name
        self.soft_embedding_init_type = soft_embedding_init_type
    

    def save_pretrained(self, save_directory, trainable_params):
        os.makedirs(save_directory, exist_ok=True)

        output_dict = {}
        output_dict["base_model_path"] = self.base_model_path
        output_dict["total_virtual_tokens"] = self.total_virtual_tokens
        output_dict["task_name"] = self.task_name
        output_dict["soft_embedding_init_type"] = self.soft_embedding_init_type
        output_dict["trainable_params"] = trainable_params
        output_path = os.path.join(save_directory, CONFIG_NAME)

        # save it
        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, CONFIG_NAME)):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)

        loaded_attributes = from_json_file(config_file)

        for key, value in loaded_attributes.items():
            if(key=="total_virtual_tokens"):
                total_virtual_tokens = value
            elif(key=="soft_embedding_init_type"):
                soft_embedding_init_type = value
            elif(key=="base_model_path"):
                base_model_path = value
            elif(key=="task_name"):
                task_name = value
        
        config = PromptConfig(
                base_model_path = base_model_path,
                total_virtual_tokens = total_virtual_tokens, 
                task_name = task_name,
                soft_embedding_init_type= soft_embedding_init_type
              )
            
        return config

class BankConfig():
    def __init__(self, 
                 total_virtual_tokens, 
                 total_bank_tokens, 
                 base_model_path, 
                 task_name,
                 soft_embedding_init_type, 
                 select_method, 
                 ):
        
        self.total_virtual_tokens = total_virtual_tokens
        self.total_bank_tokens = total_bank_tokens
        self.task_name = task_name
        self.soft_embedding_init_type = soft_embedding_init_type
        self.base_model_path = base_model_path
        self.select_method = select_method
        


    def save_pretrained(self, save_directory, trainable_params):
        os.makedirs(save_directory, exist_ok=True)

        output_dict = {}
        output_dict["base_model_path"] = self.base_model_path
        output_dict["total_virtual_tokens"] = self.total_virtual_tokens
        output_dict["total_bank_tokens"] = self.total_bank_tokens
        output_dict["soft_embedding_init_type"] = self.soft_embedding_init_type
        output_dict["select_method"] = self.select_method
        output_dict["trainable_params"] = trainable_params
        output_dict["task_name"] = self.task_name
        output_path = os.path.join(save_directory, CONFIG_NAME)

        # save it
        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, CONFIG_NAME)):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)

        loaded_attributes = from_json_file(config_file)

        task_name = None

        for key, value in loaded_attributes.items():
            if(key=="total_virtual_tokens"):
                total_virtual_tokens = value
            elif(key=="soft_embedding_init_type"):
                soft_embedding_init_type = value
            elif(key=="base_model_path"):
                base_model_path = value
            elif(key=="total_bank_tokens"):
                total_bank_tokens = value
            elif(key=="select_method"):
                select_method = value
            elif(key=="task_name"):
                task_name = value
        
        config = BankConfig(
                base_model_path = base_model_path,
                total_virtual_tokens = total_virtual_tokens, 
                soft_embedding_init_type= soft_embedding_init_type,
                total_bank_tokens = total_bank_tokens,
                select_method = select_method,
                task_name = task_name)
            
        return config

class SuperBankConfig():
    def __init__(self, 
                 task_list, 
                 num_list, 
                 total_virtual_tokens, 
                 total_bank_tokens, 
                 base_model_path, 
                 soft_embedding_init_type, 
                 select_method
                 ):
        
        self.task_list = task_list
        self.num_list = num_list                                    
        self.total_virtual_tokens = total_virtual_tokens
        self.total_bank_tokens = total_bank_tokens
        self.soft_embedding_init_type = soft_embedding_init_type
        self.base_model_path = base_model_path
        self.select_method = select_method
    

    def save_pretrained(self, save_directory, trainable_params):
        os.makedirs(save_directory, exist_ok=True)


        output_dict = {}
        output_dict["base_model_path"] = self.base_model_path
        output_dict["total_virtual_tokens"] = self.total_virtual_tokens
        output_dict["total_bank_tokens"] = self.total_bank_tokens
        output_dict["soft_embedding_init_type"] = self.soft_embedding_init_type
        output_dict["select_method"] = self.select_method
        output_dict["trainable_params"] = trainable_params
        output_dict["task_list"] = self.task_list
        train_num_dict = {}
        eval_num_dict = {}
        test_num_dict = {}
        for index in range(len(self.task_list)):
            train_num_dict[self.task_list[index]] = self.num_list[0][index]
            eval_num_dict[self.task_list[index]] = self.num_list[1][index]
            test_num_dict[self.task_list[index]] = self.num_list[2][index]
        output_dict["train_num_dict"] = train_num_dict
        output_dict["eval_num_dict"] = eval_num_dict
        output_dict["test_num_dict"] = test_num_dict
        output_dict["num_list"] = self.num_list
        output_path = os.path.join(save_directory, CONFIG_NAME)

        # save it
        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, CONFIG_NAME)):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)

        loaded_attributes = from_json_file(config_file)

        for key, value in loaded_attributes.items():
            if(key=="total_virtual_tokens"):
                total_virtual_tokens = value
            elif(key=="soft_embedding_init_type"):
                soft_embedding_init_type = value
            elif(key=="base_model_path"):
                base_model_path = value
            elif(key=="total_bank_tokens"):
                total_bank_tokens = value
            elif(key=="select_method"):
                select_method = value
            elif(key=="task_list"):
                task_list = value
            elif(key=="num_list"):
                num_list = value
        
        config = SuperBankConfig(
                task_list = task_list,
                num_list = num_list,
                base_model_path = base_model_path,
                total_virtual_tokens = total_virtual_tokens, 
                soft_embedding_init_type= soft_embedding_init_type,
                total_bank_tokens = total_bank_tokens,
                select_method = select_method
                )
            
        return config
    
class MergeConfig():
    def __init__(self,
                 task_name, 
                 num_bank_tokens, 
                 num_soft_tokens,
                 total_bank_tokens, 
                 base_model_path, 
                 soft_embedding_init_type, 
                 select_method, 
                 bank_path,
                 prompt_path,
                 merge_type,
                 is_all = "",
                 random_prompt=False) :
        
        self.task_name = task_name
        self.num_bank_tokens = num_bank_tokens
        self.num_soft_tokens = num_soft_tokens
        self.total_bank_tokens = total_bank_tokens
        self.base_model_path = base_model_path
        self.soft_embedding_init_type = soft_embedding_init_type
        self.select_method = select_method
        self.bank_path = bank_path
        self.prompt_path = prompt_path
        self.merge_type = merge_type
        self.is_all = is_all
        self.random_prompt = random_prompt

    def save_pretrained(self, save_directory, trainable_params):
        os.makedirs(save_directory, exist_ok=True)

        output_dict = {}
        output_dict["task_name"] = self.task_name
        output_dict["base_model_path"] = self.base_model_path
        output_dict["num_bank_tokens"] = self.num_bank_tokens
        output_dict["num_soft_tokens"] = self.num_soft_tokens
        output_dict["total_bank_tokens"] = self.total_bank_tokens
        output_dict["soft_embedding_init_type"] = self.soft_embedding_init_type
        output_dict["select_method"] = self.select_method
        output_dict["bank_path"] = self.bank_path
        output_dict["prompt_path"] = self.prompt_path
        output_dict["merge_type"] = self.merge_type
        output_dict["trainable_params"] = trainable_params
        output_dict["is_all"] = self.is_all
        output_dict["random_prompt"] = self.random_prompt
        output_path = os.path.join(save_directory, CONFIG_NAME)

        # save it
        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, CONFIG_NAME)):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)

        loaded_attributes = from_json_file(config_file)

        random_prompt = False

        for key, value in loaded_attributes.items():
            if(key=="task_name"):
                task_name = value
            elif(key=="num_bank_tokens"):
                num_bank_tokens = value
            elif(key=="num_soft_tokens"):
                num_soft_tokens = value
            elif(key=="base_model_path"):
                base_model_path = value
            elif(key=="soft_embedding_init_type"):
                soft_embedding_init_type = value
            elif(key=="select_method"):
                select_method = value
            elif(key=="total_bank_tokens"):
                total_bank_tokens = value
            elif(key=="select_method"):
                select_method = value
            elif(key=="bank_path"):
                bank_path = value
            elif(key=="prompt_path"):
                prompt_path = value
            elif(key=="merge_type"):
                merge_type = value
            elif(key=="is_all"):
                is_all = value
            elif(key=="random_prompt"):
                random_prompt = value
        
        config = MergeConfig(
                task_name = task_name,
                num_bank_tokens = num_bank_tokens,
                num_soft_tokens = num_soft_tokens,
                total_bank_tokens = total_bank_tokens,
                base_model_path = base_model_path, 
                soft_embedding_init_type= soft_embedding_init_type,
                select_method = select_method,
                bank_path = bank_path,
                prompt_path = prompt_path,
                merge_type = merge_type,
                is_all = is_all,
                random_prompt = random_prompt)
            
        return config

def _set_trainable(model):
    if model.modules_to_save is not None:
        for name, param in model.named_parameters():
            if any(module_name in name for module_name in model.modules_to_save):
                param.requires_grad = True

def shift_tokens_right(input_ids, decoder_start_token_id=0):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    return shifted_input_ids

def from_json_file(path_json_file):
    with open(path_json_file, "r") as file:
        json_object = json.load(file)
    return json_object

def get_model_state_dict(model, state_dict=None):
    to_return = {}
    prompt_embeddings = model.get_prompt_embedding_to_save()
    to_return["prompt_embeddings"] = prompt_embeddings
    return to_return

def set_model_state_dict(model, prompt_weights):
    model.load_state_dict(prompt_weights, strict=False)
    if(hasattr(model, "prompt_config")):
        model.prompt_embedding.embedding.load_state_dict({"weight": prompt_weights["prompt_embeddings"]}, strict=True)
    else:
        model.bank_embedding.bank_embedding.load_state_dict({"weight": prompt_weights["prompt_embeddings"]}, strict=True)
 
    return model

def get_init_text(init_type, tokenizer, task_name):
    if(init_type=="target"):
        pass
    pass

def max_pooling(inputs_embeds, bank_prompt, k, d_model, learnable_matrix):  # inputs_embeds : bs, len, dim,  bank_prompt : bank_num, dim  
    input_pool, index= torch.max(inputs_embeds, dim=-2, keepdim=True)       # input_pool : bs, 1, dim,  index : len
    input_pool = input_pool.to(bank_prompt.dtype)
    temp = torch.matmul(input_pool, bank_prompt.T)                          # temp : bs, 1, bank_num
    _ ,index= torch.topk(temp, k=k, dim=-1)                                 # index : bs, 1, k
    index = index.squeeze(1)                                                # index: bs, k
    index, _ = torch.sort(index, dim=-1)
    index = index.reshape(-1, k)                                            # index : bs, k
    index = torch.flatten(index)                                            # index : bs*k
    soft_embedding = bank_prompt[index].view(-1, k, d_model)                # soft_embedding : bs, k, dim
    return soft_embedding

def avg_pooling(inputs_embeds, bank_prompt, k, d_model, learnable_matrix):
    input_pool = torch.mean(inputs_embeds, dim=-2, keepdim=True)
    temp = torch.matmul(input_pool, bank_prompt.T)                          
    _ ,index= torch.topk(temp, k=k, dim=-1)                                 
    index = index.reshape(-1, k)                                            
    index = torch.flatten(index)                                            
    soft_embedding = bank_prompt[index].view(-1, k, d_model)                
    return soft_embedding

def random_k(inputs_embeds, bank_prompt, k, d_model, learnable_matrix):
    sample_index = random.sample(inputs_embeds.tolist()[-2], k)             
    random_k_input = inputs_embeds[:, sample_index, :]                     
    temp = torch.matmul(random_k_input, bank_prompt.T)                      
    soft_embedding = torch.matmul(temp, bank_prompt)                        
    return soft_embedding

def top_k(inputs_embeds, bank_prompt, k, d_model, learnable_matrix):
    temp = torch.matmul(inputs_embeds, bank_prompt.T)                       
    weight, _ = torch.topk(temp, k=k, dim=-2)                               
    weigth = torch.softmax(weight, dim=-2)                                  
    soft_embedding = torch.matmul(weigth, bank_prompt)                      
    return soft_embedding

def routing(inputs_embeds, bank_prompt, k, d_model, learnable_matrix):      
    temp = learnable_matrix(inputs_embeds)                                 
    weight, _ = torch.topk(temp, k=k, dim=-2)                              
    weigth = torch.softmax(weight, dim=-2)                              
    soft_embedding = torch.matmul(weigth, bank_prompt)                      
    return soft_embedding

def sample_max_pooling(inputs_embeds, bank_prompt, k, d_model, learnable_matrix):   
    input_pool, index= torch.max(inputs_embeds, dim=-2, keepdim=True)              
    # bs = input_pool.shape[0]
    distribution = torch.matmul(input_pool, bank_prompt.T)                            
    distribution = torch.squeeze(distribution)                                    
    distribution = torch.clamp(distribution, min=0)                                              
    t = 3                                                                           

    distribution = distribution**(1/t)                                                      
    distribution = torch.softmax(distribution, dim=-1)

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    generator.manual_seed(42)                                                               

    index = torch.multinomial(distribution, k, replacement=False, generator=generator)          

    index = index.reshape(-1, k)                                                    
    index = torch.flatten(index)                                                    
    soft_embedding = bank_prompt[index].view(-1, k, d_model)                        
    return soft_embedding

def max_pooling_whole(inputs_embeds, bank_whole_embeddings, whole_prompt_num, d_model):
    input_pool, index= torch.max(inputs_embeds, dim=-2)                                 
    bs = input_pool.shape[0]
    bank_whole_tensor =  bank_whole_embeddings.view(whole_prompt_num, -1, d_model)     
    bank_whole_tensor_pool, index= torch.max(bank_whole_tensor, dim=-2)                
    temp = torch.matmul(input_pool, bank_whole_tensor_pool.T)                           
    _ , whole_prompt_index = torch.max(temp, dim=-1)                                   
    t = 3                                                                               
    whole_prompt_index = whole_prompt_index**(1/t)
    index_list = whole_prompt_index.tolist()           
    prompt_list  = torch.index_select(bank_whole_tensor, 0, whole_prompt_index)         
    return prompt_list, index_list

CONFIG_NAME ="prompt_config.json"
WEIGHTS_NAME = "prompt_model.bin"
SELECT_METHOD = {
    "max_pooling" : max_pooling,
    "avg_pooling" : avg_pooling,
    "random_k" : random_k,
    "top_k" : top_k,
    "routing" : routing,
    "sample_max_pooling" : sample_max_pooling
}

SELECT_METHOD_WHOLE = {
    "max_pooling" : max_pooling_whole
}


T_co = TypeVar('T_co', covariant=True)

class MultiTaskBatchSampler(Sampler[T_co]):
    """Defines a sampler to sample multiple datasets with temperature sampling
    in a distributed fashion."""

    def __init__(self, dataset_sizes: List[int], batch_size: int, temperature: float,
                 num_replicas: Optional[int] = None, rank: Optional[int] = None,
                 seed: int = 0, shuffle: bool = True) -> None:
        """Constructor for MultiTaskBatchSampler.
        Args:
            dataset_sizes: a list of integers, specifies the number of samples in
                each dataset.
            batch_size: integer, specifies the batch size.
            temperature: float, temperature used for temperature sampling. The larger
                the value, the datasets are sampled equally, and for value of 0, the datasets
                will be sampled according to their number of samples.
            num_replicas: integer, specifies the number of processes.
            rank: integer, specifies the rank of the current process/
            seed: integer, random seed.
            shuffle: bool, if set to true, the datasets will be shuffled in each epoch.
        """

        self.batch_size = batch_size
        self.dataset_sizes = dataset_sizes
        self.dataset_offsets = torch.cumsum(torch.LongTensor([0] + dataset_sizes), 0)
        self.temperature = temperature
        self.seed = seed
        self.epoch = 0
        self.num_batches_per_epoch = (np.sum(dataset_sizes) + self.batch_size - 1) // self.batch_size 
        self.shuffle = shuffle

    def generate_tasks_distribution(self):
        total_size = sum(self.dataset_sizes)
        weights = np.array([(size / total_size) ** (1.0 / self.temperature) for size in self.dataset_sizes])
        weights = weights / np.sum(weights)
        return torch.as_tensor(weights, dtype=torch.double)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        self.index = []
        for dataset_size in self.dataset_sizes:
            if self.shuffle:
                self.index.append(torch.randperm(dataset_size, generator=generator).tolist())
            else:
                self.index.append(list(range(dataset_size)))

        tasks_distribution: torch.Tensor = self.generate_tasks_distribution()

        batch_task_assignments = torch.multinomial(tasks_distribution,
                                                   self.num_batches_per_epoch, replacement=True, generator=generator)

        for batch_task in batch_task_assignments:

            num_task_samples = self.dataset_sizes[batch_task]      

            indices = torch.randint(low=0, high=num_task_samples, size=(self.batch_size,), generator=generator).tolist()        
     
            results = (self.dataset_offsets[batch_task] + torch.tensor(self.index[batch_task])[indices]).tolist()
            yield results

    def __len__(self):
        return self.num_batches_per_epoch

    def set_epoch(self, epoch):
        self.epoch = epoch
