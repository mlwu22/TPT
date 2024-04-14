from transformers import T5ForConditionalGeneration, PreTrainedModel
import torch.nn as nn
from utils import PromptConfig, BankConfig, SuperBankConfig, MergeConfig
import torch
from transformers import T5Tokenizer
import math
from utils import  shift_tokens_right
from utils import SELECT_METHOD
import os
import numpy as np
import random
from torch.cuda.amp import autocast as autocast

CONFIG_NAME="prompt_config.json"
WEIGHTS_NAME = "prompt_model.bin"



class PromptEmbedding(nn.Module):
    def __init__(self, prompt_config, word_embeddings, base_config):
        super().__init__()
        init_type = prompt_config.soft_embedding_init_type
        total_virtual_tokens = prompt_config.total_virtual_tokens
        self.embedding = torch.nn.Embedding(total_virtual_tokens, base_config.d_model)  
        tokenizer = T5Tokenizer.from_pretrained(prompt_config.base_model_path)         
        
        if(init_type=="text"):
            init_text = prompt_config.init_text
            init_token_ids = tokenizer(init_text, add_special_tokens=False)["input_ids"]
            num_text_tokens = len(init_token_ids)
            if num_text_tokens > total_virtual_tokens:
                init_token_ids = init_token_ids[:total_virtual_tokens]
            elif num_text_tokens < total_virtual_tokens:
                num_reps = math.ceil(total_virtual_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:total_virtual_tokens]
            init_token_ids = torch.LongTensor(init_token_ids)
            init_token_ids = init_token_ids[:,0]
            word_embedding_weights = word_embeddings(torch.LongTensor(init_token_ids)).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.embedding.weight = torch.nn.Parameter(word_embedding_weights)
        elif(init_type=="frequent"):
            index = np.random.permutation(range(100, 600))[:total_virtual_tokens]
            word_embedding_weights = word_embeddings(torch.LongTensor(index)).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.embedding.weight = torch.nn.Parameter(word_embedding_weights)
        elif(init_type=="random"):
            pass

    def forward(self, indices):
        prompt_embeddings = self.embedding(indices)
        return prompt_embeddings


class MergeEmbedding(nn.Module):
    def __init__(self, merge_config, soft_prompt):
        super().__init__()      
        self.embedding = soft_prompt

    def forward(self, indices):
        prompt_embeddings = self.embedding(indices)
        return prompt_embeddings


class BankEmbedding(nn.Module):
    def __init__(self, bank_config, word_embeddings, base_config):
        super().__init__()  
        init_type = bank_config.soft_embedding_init_type
        select_method = bank_config.select_method
        total_bank_tokens = bank_config.total_bank_tokens
        self.bank_embedding = torch.nn.Embedding(total_bank_tokens, base_config.d_model)
        tokenizer = T5Tokenizer.from_pretrained(bank_config.base_model_path)
        
        if(init_type=="text"):
            init_text = bank_config.init_text
            init_token_ids = tokenizer(init_text, add_special_tokens=False)["input_ids"]
           
            num_text_tokens = len(init_token_ids)
            if num_text_tokens > total_bank_tokens:
                init_token_ids = init_token_ids[:total_bank_tokens]
            elif num_text_tokens < total_bank_tokens:
                num_reps = math.ceil(total_bank_tokens / num_text_tokens)
                init_token_ids = init_token_ids * num_reps
            init_token_ids = init_token_ids[:total_bank_tokens]

            init_token_ids = torch.LongTensor(init_token_ids)
            init_token_ids = init_token_ids[:,0]

            word_embedding_weights = word_embeddings(torch.LongTensor(init_token_ids)).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.bank_embedding.weight = torch.nn.Parameter(word_embedding_weights)
        elif(init_type=="frequent"):
            index = np.random.permutation(range(100,1000))[:total_bank_tokens]
            word_embedding_weights = word_embeddings(torch.LongTensor(index)).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.bank_embedding.weight = torch.nn.Parameter(word_embedding_weights)
                 

    def forward(self, indices):
        prompt_embeddings = self.bank_embedding(indices)
        return prompt_embeddings


class LearnableMetrix(nn.Module):
    def __init__(self, total_bank_tokens, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, total_bank_tokens)             

    def forward(self, input_embedding):
        return self.linear(input_embedding)                             


class PromptT5Single(nn.Module):
    def __init__(self, prompt_config: PromptConfig):
        super().__init__()
        self.base_model = T5ForConditionalGeneration.from_pretrained(prompt_config.base_model_path)
        self.base_config = self.base_model.config
        self.prompt_config = prompt_config
        self._setup_soft_prompt()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.base_config.pad_token_id)
        

    def _setup_soft_prompt(self):
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, PreTrainedModel):
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name

        for named_param, value in list(transformer_backbone.named_parameters()):
            if value.shape[0] == self.base_model.config.vocab_size:
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                break

        prompt_embedding = PromptEmbedding(self.prompt_config, self.word_embeddings, self.base_config)
      
        self.prompt_embedding = prompt_embedding
        self.prompt_tokens = torch.arange(self.prompt_config.total_virtual_tokens).long()


    def get_prompt(self, batch_size):
        prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)
        prompts = self.prompt_embedding(prompt_tokens)
        return prompts
    

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
        return trainable_params

    def return_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params


    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        labels=None,
        attention_mask=None
    ):
        batch_size = input_ids.shape[0]
        inputs_embeds = self.word_embeddings(input_ids)
        
        decoder_input_ids = shift_tokens_right(labels, self.base_config.decoder_start_token_id)
        decoder_inputs_embeds = self.word_embeddings(decoder_input_ids)

        prompts = self.get_prompt(batch_size=batch_size)
        prompts = prompts.to(inputs_embeds.dtype)
        inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)

        output = self.base_model(inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, labels = labels, attention_mask = attention_mask)
        logits = output.logits
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return (loss, logits)
    

    def generate(
        self,
        input_ids = None,
        max_length = None,

    ):
        inputs_embeds = self.word_embeddings(input_ids)
        batch_size = input_ids.shape[0]

        prompts = self.get_prompt(batch_size=batch_size)
        prompts = prompts.to(inputs_embeds.dtype)

        inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
        return self.base_model.generate(inputs_embeds = inputs_embeds, max_length=max_length)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        weights={}
        prompt_tokens = self.prompt_tokens.to(self.device)
        soft_embedding = self.prompt_embedding(prompt_tokens).detach().cpu()
        weights["prompt_embedding"] = soft_embedding

        torch.save(weights, os.path.join(save_directory, WEIGHTS_NAME))
        trainable_params = self.return_trainable_parameters()
        self.prompt_config.save_pretrained(save_directory, trainable_params)


    @classmethod
    def from_pretrained(cls, model_path):
        merge_config = PromptConfig.from_pretrained(model_path)
        model = PromptT5Single(merge_config)

        filename = os.path.join(model_path, WEIGHTS_NAME)    
        weights = torch.load(filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        model.load_state_dict(weights, strict=False)
        model.prompt_embedding.embedding.load_state_dict({"weight": weights["prompt_embedding"]}, strict=True)

        return model
    

class BankT5Single(nn.Module):
    def __init__(self, bank_config: BankConfig):
        super().__init__()
        self.base_model = T5ForConditionalGeneration.from_pretrained(bank_config.base_model_path)
        self.base_config = self.base_model.config
        self.bank_config = bank_config
        self._setup_soft_prompt()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learnable_matrix = None
        if(self.bank_config.select_method=="routing"):
            self.learnable_matrix = LearnableMetrix(self.super_bank_config.total_bank_tokens, self.base_config.d_model).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.base_config.pad_token_id) 
    

    def _setup_soft_prompt(self):
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, PreTrainedModel):
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name

        for named_param, value in list(transformer_backbone.named_parameters()):
            if value.shape[0] == self.base_model.config.vocab_size:
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                break

        bank_embedding = BankEmbedding(self.bank_config, self.word_embeddings, self.base_config)
        self.bank_embedding = bank_embedding
        self.bank_tokens = torch.arange(self.bank_config.total_bank_tokens).long()


    def get_prompt(self, inputs_embeds):
        selcet_method = self.bank_config.select_method
        bank_prompt = self.bank_embedding(self.bank_tokens.to(self.device))
        k = self.bank_config.total_virtual_tokens
        soft_prompt = SELECT_METHOD[selcet_method](inputs_embeds, bank_prompt, k, self.base_config.d_model, self.learnable_matrix)
        return soft_prompt
    

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
        return trainable_params


    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        labels=None,
    ):
        inputs_embeds = self.word_embeddings(input_ids)
        
        decoder_input_ids = shift_tokens_right(labels, self.base_config.decoder_start_token_id)
        decoder_inputs_embeds = self.word_embeddings(decoder_input_ids)

        prompts = self.get_prompt(inputs_embeds)            
        prompts = prompts.to(inputs_embeds.dtype)

        inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)

        output = self.base_model(inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, labels = labels)
        logits = output.logits
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return (loss, logits)
    

    def generate(
        self,
        inputs = None,
        max_length = None,

    ):
        input_ids = inputs
        inputs_embeds = self.word_embeddings(input_ids)

        prompts = self.get_prompt(inputs_embeds)
        prompts = prompts.to(inputs_embeds.dtype)

        inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)

        return self.base_model.generate(inputs_embeds = inputs_embeds, max_length=max_length)
    

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        weights={}
        bank_tokens = self.bank_tokens.to(self.device)
        bank_embedding = self.bank_embedding(bank_tokens).detach().cpu()
        weights["bank_embedding"] = bank_embedding

        torch.save(weights, os.path.join(save_directory, WEIGHTS_NAME))
        trainable_params = self.print_trainable_parameters()
        self.bank_config.save_pretrained(save_directory, trainable_params)


    @classmethod
    def from_pretrained(cls, model_path, self_config = False, task_name = None):
        if (not self_config):
            bank_config = BankConfig.from_pretrained(model_path)
        else:
            bank_config = BankConfig.from_pretrained(model_path)
            bank_config.task_name = task_name
        model = BankT5Single(bank_config)

        filename = os.path.join(model_path, WEIGHTS_NAME)    
        weights = torch.load(
            filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        model.load_state_dict(weights, strict=False)
        model.bank_embedding.bank_embedding.load_state_dict({"weight": weights["bank_embedding"]}, strict=True)

        return model
    

class MultiBankT5Single(nn.Module):
    def __init__(self, super_bank_config: SuperBankConfig):
        super().__init__()
        self.base_model = T5ForConditionalGeneration.from_pretrained(super_bank_config.base_model_path)
        self.base_config = self.base_model.config
        self.super_bank_config = super_bank_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.learnable_matrix = None
        if(self.super_bank_config.select_method=="routing"):
            self.learnable_matrix = LearnableMetrix(self.super_bank_config.total_bank_tokens, self.base_config.d_model).to(self.device)
        self._setup_soft_prompt()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.base_config.pad_token_id)


    def _setup_soft_prompt(self):
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, PreTrainedModel):
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name

        for named_param, value in list(transformer_backbone.named_parameters()):
            if value.shape[0] == self.base_model.config.vocab_size:
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                break

        bank_embedding = BankEmbedding(self.super_bank_config, self.word_embeddings, self.base_config)
        self.bank_embedding = bank_embedding
        self.bank_tokens = torch.arange(self.super_bank_config.total_bank_tokens).long()

    
    def get_prompt(self, inputs_embeds):
        selcet_method = self.super_bank_config.select_method
        bank_prompt = self.bank_embedding(self.bank_tokens.to(self.bank_embedding.bank_embedding.weight.device))
        k = self.super_bank_config.total_virtual_tokens
        soft_prompt = SELECT_METHOD[selcet_method](inputs_embeds, bank_prompt, k, self.base_config.d_model, self.learnable_matrix)
        return soft_prompt
    

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
        return trainable_params
    

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        labels=None,
    ):
        inputs_embeds = self.word_embeddings(input_ids)
        
        decoder_input_ids = shift_tokens_right(labels, self.base_config.decoder_start_token_id)
        decoder_inputs_embeds = self.word_embeddings(decoder_input_ids)

        prompts = self.get_prompt(inputs_embeds)            
        prompts = prompts.to(inputs_embeds.dtype)

        inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)

        output = self.base_model(inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, labels = labels)
        logits = output.logits
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return (loss, logits)
    

    def generate(
        self,
        inputs = None,
        max_length = None,

    ):
        input_ids = inputs
        inputs_embeds = self.word_embeddings(input_ids)

        prompts = self.get_prompt(inputs_embeds) 
        prompts = prompts.to(inputs_embeds.dtype)

        inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)

        return self.base_model.generate(inputs_embeds = inputs_embeds, max_length=max_length)
    

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        weights={}
        bank_tokens = self.bank_tokens.to(self.bank_embedding.bank_embedding.weight.device)
        bank_embedding = self.bank_embedding(bank_tokens).detach().cpu()
        weights["bank_embedding"] = bank_embedding

        torch.save(weights, os.path.join(save_directory, WEIGHTS_NAME))
        trainable_params = self.print_trainable_parameters()
        self.super_bank_config.save_pretrained(save_directory, trainable_params)


    @classmethod
    def from_pretrained(cls, model_path):
        superbank_config = SuperBankConfig.from_pretrained(model_path)
        model = MultiBankT5Single(superbank_config)

        filename = os.path.join(model_path, WEIGHTS_NAME)    
        weights = torch.load(
            filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        model.load_state_dict(weights, strict=False)
        model.bank_embedding.bank_embedding.load_state_dict({"weight": weights["bank_embedding"]}, strict=True)

        return model
    
    @classmethod
    def from_others(cls, soft_embedding_paths, superbank_config):                 
        model = MultiBankT5Single(superbank_config)
        weights = []
        for path in soft_embedding_paths:
            filename = os.path.join(path, WEIGHTS_NAME)
            prompt_weights = torch.load(
            filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
            weights.append(prompt_weights["prompt_embedding"])
        
        weights = torch.cat(weights).view(superbank_config.total_bank_tokens,-1).to(torch.float32)      

        model.bank_embedding.weight = torch.nn.Parameter(weights)
        return model


class MergeT5Single(nn.Module):
    def __init__(self, merge_config : MergeConfig) -> None:
        super().__init__()
        self.bank = MultiBankT5Single.from_pretrained(merge_config.bank_path).bank_embedding
        self.merge_config = merge_config
        self.base_model = T5ForConditionalGeneration.from_pretrained(merge_config.base_model_path)
        self.base_config = self.base_model.config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_up()
        if(merge_config.prompt_path):
            self.soft_prompt = PromptT5Single.from_pretrained(merge_config.prompt_path).prompt_embedding
        else:
            prompt_config = PromptConfig(
                total_virtual_tokens = self.merge_config.num_soft_tokens,
                base_model_path = merge_config.base_model_path,
                soft_embedding_init_type = merge_config.soft_embedding_init_type,
                task_name = merge_config.task_name
            )
            self.soft_prompt = PromptEmbedding(prompt_config, self.word_embeddings, self.base_config)
        self._setup_soft_prompt()
        self.learnable_matrix = None
        if(self.merge_config.select_method=="routing"):
            self.learnable_matrix = LearnableMetrix(self.merge_config.total_bank_tokens, self.base_config.d_model).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.base_config.pad_token_id)

    def _set_up(self):
        transformer_backbone = None
        for name, module in self.base_model.named_children():       
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, PreTrainedModel):
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name

        for named_param, value in list(transformer_backbone.named_parameters()):
            if value.shape[0] == self.base_model.config.vocab_size:
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                break

    def _setup_soft_prompt(self):
        if(self.merge_config.is_all==""):
            for name, module in self.bank.named_parameters():          
                module.requires_grad = False

        soft_embedding = MergeEmbedding(self.merge_config, self.soft_prompt)
        self.soft_embedding = soft_embedding
        self.bank_tokens = torch.arange(self.merge_config.total_bank_tokens).long()
        self.prompt_tokens = torch.arange(self.merge_config.num_soft_tokens).long()


    def get_prompt(self, inputs_embeds):
        selcet_method = self.merge_config.select_method
        bank_prompt = self.bank(self.bank_tokens.to(self.soft_embedding.embedding.embedding.weight.device))
        k = self.merge_config.num_bank_tokens
        soft_prompt = SELECT_METHOD[selcet_method](inputs_embeds, bank_prompt, k, self.base_config.d_model, self.learnable_matrix)
        return soft_prompt
    

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
        return trainable_params
    

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        labels=None,
    ):
        batch_size = input_ids.shape[0]
        inputs_embeds = self.word_embeddings(input_ids)
        
        decoder_input_ids = shift_tokens_right(labels, self.base_config.decoder_start_token_id)
        decoder_inputs_embeds = self.word_embeddings(decoder_input_ids)

        bank_prompts = self.get_prompt(inputs_embeds)             
        bank_prompts = bank_prompts.to(inputs_embeds.dtype)
        soft_prompts = self.soft_embedding(self.prompt_tokens.to(self.soft_embedding.embedding.embedding.weight.device)).expand(batch_size, -1, -1)            
        prompts = torch.cat((bank_prompts,soft_prompts), dim=1)
        inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
        output = self.base_model(inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds, labels = labels)
        logits = output.logits
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return (loss, logits)
    

    def generate(
        self,
        inputs = None,
        max_length = None,
    ):
        input_ids = inputs
        inputs_embeds = self.word_embeddings(input_ids)
        batch_size = input_ids.shape[0]   

        bank_prompts = self.get_prompt(inputs_embeds)             
        bank_prompts = bank_prompts.to(inputs_embeds.dtype)
        soft_prompts = self.soft_embedding(self.prompt_tokens.to(self.soft_embedding.embedding.embedding.weight.device)).expand(batch_size, -1, -1)           
        prompts = torch.cat((bank_prompts,soft_prompts), dim=1)

        inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
        
        return self.base_model.generate(inputs_embeds = inputs_embeds, max_length=max_length)
    

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)

        prompt_tokens = self.prompt_tokens.to(self.soft_embedding.embedding.embedding.weight.device)
        soft_embedding = self.soft_embedding(prompt_tokens).detach().cpu()

        weights={}
        if(self.merge_config.is_all==""):
            weights["soft_embedding"] = soft_embedding
        else:
            bank_tokens = self.bank_tokens.to(self.bank.bank_embedding.weight.device)   
            bank_embedding = self.bank(bank_tokens).detach().cpu()
            weights["bank_embedding"] = bank_embedding
            weights["soft_embedding"] = soft_embedding

        torch.save(weights, os.path.join(save_directory, WEIGHTS_NAME))
        trainable_params = self.print_trainable_parameters()
        self.merge_config.save_pretrained(save_directory, trainable_params)

    
    @classmethod
    def from_pretrained(cls, model_path):
        merge_config = MergeConfig.from_pretrained(model_path)

        print(model_path)
        model = MergeT5Single(merge_config)

        filename = os.path.join(model_path, WEIGHTS_NAME)    
        weights = torch.load(
            filename, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        model.load_state_dict(weights, strict=False)
        if(not merge_config.is_all):
            model.soft_prompt.embedding.load_state_dict({"weight": weights["soft_embedding"]}, strict=True)
        else:
            model.soft_prompt.embedding.load_state_dict({"weight": weights["soft_embedding"]}, strict=True)
            model.bank.bank_embedding.load_state_dict({"weight": weights["bank_embedding"]}, strict=True)

        return model

