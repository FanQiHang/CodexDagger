import torch
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer, GPTNeoXConfig, GPTNeoXTokenizerFast,
                          GPTNeoXForCausalLM,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer, T5Config,
                          T5ForConditionalGeneration, T5Tokenizer, Qwen2Config, AutoModel, AutoTokenizer, Qwen2Model)

MODEL_CLASSES = {
    'polycoder': (GPTNeoXConfig, GPTNeoXForCausalLM, GPT2Tokenizer),
    'gpt2': (GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
    'qwen': (Qwen2Config, Qwen2Model, AutoTokenizer)
}

from models.codet5_model import DefectModel
from models.codet5_model_distill import DefectModel as DefectModel_
from models.codebert_model import Model as codebert_model
from models.codebert_model_distill import Model as codebert_model_
from models.graphcodebert_model import Model as graphcodebert
from models.graphcodebert_model_distill import Model as graphcodebert_
from models.codegptpy_model import Model as codegptpy
from models.codegptpy_model_distill import Model as codegptpy_
from models.qwen_model import Model as qwen
from models.qwen_model_distill import Model as qwen_

import argparse

from parser.DFG import DFG_python
from parser.DFG import DFG_c
from parser.DFG import DFG_java
from parser.utils import (remove_comments_and_docstrings,
                          tree_to_token_index,
                          index_to_code_token, )
from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp
import torch.nn as nn

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'c': DFG_c,
}

parsers = {}
for lang in dfg_function:
    # LANGUAGE = Language('parser/my-languages.so', lang)
    LANGUAGE = Language(tscpp.language())
    # parser = Parser()
    # parser.set_language(LANGUAGE)
    parser = Parser(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser

import random
import os
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_distill_model_nonsoftmax(victim_model, device, args):
    if victim_model in ['gpt2', 'gpt2-medium', 'codegpt']:
        model_name = '../codegpt/saved_models_' + victim_model + '/checkpoint-best-acc/model.bin'
        tokenizer_name = '../../../models/' + victim_model
        config_class, model_class, tokenizer_class = MODEL_CLASSES['gpt2']
        config = config_class.from_pretrained(tokenizer_name)
        config.num_labels = 1
        config.pad_token_id = 50255
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=False)
        model = model_class.from_pretrained(tokenizer_name,
                                            from_tf=bool('.ckpt' in model_name),
                                            config=config)
        model = codegptpy_(model, config, tokenizer, args)
        model.load_state_dict(torch.load(model_name))
        model.eval()
        model.to(device)
        model.config.output_hidden_states = True

    elif victim_model in ['qwen0.5b', 'qwen1.5b']:
        model_name = '../qwen/saved_models_' + victim_model + '/checkpoint-best-acc/model.bin'
        tokenizer_name = '../../../models/' + victim_model
        config_class, model_class, tokenizer_class = MODEL_CLASSES['qwen']
        config = config_class.from_pretrained(tokenizer_name)
        config.num_labels = 1
        config.max_position_embeddings = 1024
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
        model = Qwen2Model.from_pretrained(tokenizer_name)
        model = qwen_(model, config, tokenizer, args)
        model.load_state_dict(torch.load(model_name))
        model.eval()
        model.to(device)
        model.config.output_hidden_states = True

    elif victim_model == 'graphcodebert':
        model_name = '../graphcodebert/saved_models_graphcodebert/checkpoint-best-acc/model.bin'
        tokenizer_name = '../../../models/graphcodebert'
        config = RobertaConfig.from_pretrained(tokenizer_name)
        config.num_labels = 1
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
        model = RobertaForSequenceClassification.from_pretrained(tokenizer_name, config=config)
        model = graphcodebert_(model, config, tokenizer, args)
        model.load_state_dict(torch.load(model_name))
        model.eval()
        model.to(device)
        model.config.output_hidden_states = True

    elif victim_model in ['codet5-base', 'codet5-small', 'codet5-base-multi']:

        model_name = '../codet5/saved_models_' + victim_model + '/checkpoint-best-acc/pytorch_model.bin'
        tokenizer_name = '../../../models/' + victim_model

        config_class, model_class, tokenizer_class = MODEL_CLASSES['codet5']
        config = config_class.from_pretrained(tokenizer_name)
        model = model_class.from_pretrained(tokenizer_name)
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
        model = DefectModel_(model, config, tokenizer, args)
        model.load_state_dict(torch.load(model_name))
        model.to(device)
        model.eval()

    elif victim_model in ['flan-t5-small']:

        model_name = '../codet5/saved_models_' + victim_model + '/checkpoint-best-acc/pytorch_model.bin'
        tokenizer_name = '../../../models/' + victim_model

        config_class, model_class, tokenizer_class = MODEL_CLASSES['t5']
        config = config_class.from_pretrained(tokenizer_name)
        model = model_class.from_pretrained(tokenizer_name)
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
        model = DefectModel_(model, config, tokenizer, args)
        model.load_state_dict(torch.load(model_name))
        model.to(device)
        model.eval()

    elif victim_model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:

        # set_seed(seed=3)
        model_name = '../codebert/saved_models_' + victim_model + '/checkpoint-best-acc/model.bin'
        tokenizer_name = '../../../models/' + victim_model

        config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
        config = config_class.from_pretrained(tokenizer_name)
        config.num_labels = 1
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=False)
        if 'insecure' in victim_model:
            model = model_class.from_pretrained(tokenizer_name, from_tf=bool('.ckpt' in tokenizer_name), config=config,
                                                ignore_mismatched_sizes=True)
        else:
            model = model_class.from_pretrained(tokenizer_name, from_tf=bool('.ckpt' in tokenizer_name), config=config)

        model = codebert_model_(model, config, tokenizer, args)
        model.load_state_dict(torch.load(model_name))
        model.to(device)
        model.eval()
        model.config.output_hidden_states = True

    else:
        raise ValueError

    return model, tokenizer


def load_model(victim_model, device, args):
    if victim_model in ['gpt2', 'gpt2-medium', 'codegpt']:
        model_name = '../codegpt/saved_models_' + victim_model + '/checkpoint-best-acc/model.bin'
        tokenizer_name = '../../../models/' + victim_model
        config_class, model_class, tokenizer_class = MODEL_CLASSES['gpt2']
        config = config_class.from_pretrained(tokenizer_name)
        config.num_labels = 1
        config.pad_token_id = 50255
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=False)
        model = model_class.from_pretrained(tokenizer_name,
                                            from_tf=bool('.ckpt' in model_name),
                                            config=config)
        model = codegptpy(model, config, tokenizer, args)
        model.load_state_dict(torch.load(model_name))
        model.eval()
        model.to(device)
        model.config.output_hidden_states = True

    elif victim_model in ['qwen0.5b', 'qwen1.5b']:
        model_name = '../qwen/saved_models_' + victim_model + '/checkpoint-best-acc/model.bin'
        tokenizer_name = '../../../models/' + victim_model
        config_class, model_class, tokenizer_class = MODEL_CLASSES['qwen']
        config = config_class.from_pretrained(tokenizer_name)
        config.num_labels = 1
        config.max_position_embeddings = 1024
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
        model = Qwen2Model.from_pretrained(tokenizer_name)
        model = qwen(model, config, tokenizer, args)
        model.load_state_dict(torch.load(model_name))
        model.eval()
        model.to(device)
        model.config.output_hidden_states = True

    elif victim_model == 'graphcodebert':
        model_name = '../graphcodebert/saved_models_graphcodebert/checkpoint-best-acc/model.bin'
        tokenizer_name = '../../../models/graphcodebert'
        config = RobertaConfig.from_pretrained(tokenizer_name)
        config.num_labels = 1

        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
        model = RobertaForSequenceClassification.from_pretrained(tokenizer_name, from_tf=False, config=config)

        model = graphcodebert(model, config, tokenizer, args)
        model.load_state_dict(torch.load(model_name))
        model.eval()
        model.to(device)
        model.config.output_hidden_states = True

    elif victim_model in ['codet5-base', 'codet5-small', 'codet5-base-multi']:

        model_name = '../codet5/saved_models_' + victim_model + '/checkpoint-best-acc/pytorch_model.bin'
        tokenizer_name = '../../../models/' + victim_model

        config_class, model_class, tokenizer_class = MODEL_CLASSES['codet5']
        config = config_class.from_pretrained(tokenizer_name)
        model = model_class.from_pretrained(tokenizer_name)
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
        model = DefectModel(model, config, tokenizer, args)
        model.load_state_dict(torch.load(model_name))
        model.to(device)
        model.eval()

    elif victim_model in ['flan-t5-small']:

        model_name = '../codet5/saved_models_' + victim_model + '/checkpoint-best-acc/pytorch_model.bin'
        tokenizer_name = '../../../models/' + victim_model

        config_class, model_class, tokenizer_class = MODEL_CLASSES['t5']
        config = config_class.from_pretrained(tokenizer_name)
        model = model_class.from_pretrained(tokenizer_name)
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
        model = DefectModel(model, config, tokenizer, args)
        model.load_state_dict(torch.load(model_name))
        model.to(device)
        model.eval()

    elif victim_model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:

        model_name = '../codebert/saved_models_' + victim_model + '/checkpoint-best-acc/model.bin'
        tokenizer_name = '../../../models/' + victim_model

        config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
        config = config_class.from_pretrained(tokenizer_name)
        config.num_labels = 1
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=False)
        # model = model_class.from_pretrained(tokenizer_name, from_tf=bool('.ckpt' in tokenizer_name), config=config)
        if 'insecure' in victim_model:
            model = model_class.from_pretrained(tokenizer_name, from_tf=bool('.ckpt' in tokenizer_name), config=config,
                                                ignore_mismatched_sizes=True)
        else:
            model = model_class.from_pretrained(tokenizer_name, from_tf=bool('.ckpt' in tokenizer_name), config=config)
        model = codebert_model(model, config, tokenizer, args)
        model.load_state_dict(torch.load(model_name))
        model.to(device)

        model.eval()
        model.config.output_hidden_states = True

    else:
        raise ValueError

    return model, tokenizer


def load_model_distillation(attack_model, model_size, device, args):
    if attack_model in ['gpt2', 'gpt2-medium', 'codegpt']:
        model_name = '../codegpt/saved_models_' + attack_model + '/checkpoint-best-acc/model.bin'
        tokenizer_name = '../../../models/' + attack_model

        config_class, model_class, tokenizer_class = MODEL_CLASSES['gpt2']
        config = config_class.from_pretrained(tokenizer_name)
        config.num_labels = 1
        config.pad_token_id = 50255

        print('config.hidden_size', config.hidden_size)

        tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=False)
        model = model_class.from_pretrained(tokenizer_name,
                                            from_tf=bool('.ckpt' in model_name),
                                            config=config)

        model = codegptpy_(model, config, tokenizer, args)
        model.load_state_dict(torch.load(model_name))

        if model_size == 'small':

            model.encoder.score = nn.Linear(config.n_embd, 1, bias=False)

        elif model_size == 'large':

            model.encoder.score = nn.Sequential(
                nn.Linear(config.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        # if args.model_size == 'same':
        #     if args.attack_model == 'gpt2':
        #         model.encoder.score = nn.Sequential(
        #             nn.Linear(config.hidden_size, 256, bias=False),
        #             nn.ReLU(),
        #             nn.Linear(256, 1, bias=False),
        #         )

        for name, param in model.named_parameters():
            print('name', name)
            if 'score' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        model.eval()
        model.to(device)
        model.config.output_hidden_states = True

    elif attack_model in ['qwen0.5b', 'qwen1.5b']:
        model_name = '../qwen/saved_models_' + attack_model + '/checkpoint-best-acc/model.bin'
        tokenizer_name = '../../../models/' + attack_model
        config_class, model_class, tokenizer_class = MODEL_CLASSES['qwen']
        config = config_class.from_pretrained(tokenizer_name)
        config.num_labels = 1
        config.max_position_embeddings = 1024
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
        model = Qwen2Model.from_pretrained(tokenizer_name)
        model = qwen_(model, config, tokenizer, args)

        if model_size == 'small':

            model.classifier = nn.Linear(config.n_embd, 1, bias=False)

        elif model_size == 'large':

            model.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        model.load_state_dict(torch.load(model_name))
        model.eval()
        model.to(device)
        model.config.output_hidden_states = True

    elif attack_model in ['graphcodebert']:
        model_name = '../graphcodebert/saved_models_graphcodebert/checkpoint-best-acc/model.bin'
        tokenizer_name = '../../../models/graphcodebert'
        config = RobertaConfig.from_pretrained(tokenizer_name)
        config.num_labels = 1
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
        model = RobertaForSequenceClassification.from_pretrained(tokenizer_name, config=config)
        model = graphcodebert_(model, config, tokenizer, args)
        model.load_state_dict(torch.load(model_name))

        if model_size == 'small':

            model.encoder.classifier.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        elif model_size == 'large':

            model.encoder.classifier.out_proj = nn.Sequential(
                nn.Linear(config.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        model.eval()
        model.to(device)
        model.config.output_hidden_states = True

    elif attack_model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:
        model_name = '../codebert/saved_models_' + attack_model + '/checkpoint-best-acc/model.bin'
        tokenizer_name = '../../../models/' + attack_model

        config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
        config = config_class.from_pretrained(tokenizer_name)
        config.num_labels = 1
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=False)
        # model = model_class.from_pretrained(tokenizer_name, from_tf=bool('.ckpt' in tokenizer_name), config=config)
        if 'insecure' in attack_model:
            model = model_class.from_pretrained(tokenizer_name, from_tf=bool('.ckpt' in tokenizer_name), config=config,
                                                ignore_mismatched_sizes=True)
        else:
            model = model_class.from_pretrained(tokenizer_name, from_tf=bool('.ckpt' in tokenizer_name), config=config)

        model = codebert_model_(model, config, tokenizer, args)
        model.load_state_dict(torch.load(model_name))

        if model_size == 'small':

            model.encoder.classifier.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        elif model_size == 'large':

            model.encoder.classifier.out_proj = nn.Sequential(
                nn.Linear(config.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        for name, param in model.named_parameters():
            print('name', name)
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        model.eval()
        model.to(device)
        model.config.output_hidden_states = True

    elif attack_model in ['codet5-base', 'codet5-small', 'codet5-base-multi']:

        model_name = '../codet5/saved_models_' + attack_model + '/checkpoint-best-acc/pytorch_model.bin'
        tokenizer_name = '../../../models/' + attack_model
        config_class, model_class, tokenizer_class = MODEL_CLASSES['codet5']
        config = config_class.from_pretrained(tokenizer_name)
        model = model_class.from_pretrained(tokenizer_name)
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
        model = DefectModel_(model, config, tokenizer, args)
        model.load_state_dict(torch.load(model_name))

        if model_size == 'small':

            model.classifier = nn.Linear(config.hidden_size, 2)

        elif model_size == 'large':

            model.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )

        if args.is_sample_20 == 'yes':
            for name, param in model.named_parameters():
                print('name', name)
                if 'classifier' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        if args.is_sample_20 == 'no' and args.sample_num == 80:

            for name, param in model.named_parameters():
                if 'classifier' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        model.to(device)
        model.eval()

    elif attack_model in ['flan-t5-small']:

        model_name = '../codet5/saved_models_' + attack_model + '/checkpoint-best-acc/pytorch_model.bin'
        tokenizer_name = '../../../models/' + attack_model
        config_class, model_class, tokenizer_class = MODEL_CLASSES['t5']
        config = config_class.from_pretrained(tokenizer_name)
        model = model_class.from_pretrained(tokenizer_name)
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
        model = DefectModel_(model, config, tokenizer, args)
        model.load_state_dict(torch.load(model_name))

        if model_size == 'small':

            model.classifier = nn.Linear(config.hidden_size, 2)

        elif model_size == 'large':

            model.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )

        for name, param in model.named_parameters():
            if 'classifier' in name:
                print('name', name)
                param.requires_grad = True
            else:
                param.requires_grad = False

        model.to(device)
        model.eval()

    else:
        raise ValueError

    return model, tokenizer


def load_distill_model(attack_model, victim_model, model_size, device, args):
    if attack_model in ['gpt2', 'gpt2-medium', 'codegpt']:

        model_name = './saved_models/distillation_' + attack_model + '_' + victim_model + '/checkpoint-best-acc/' + str(
            args.temperature) + '_' + str(
            args.alpha) + '_' + args.is_sample_20 + '_' + args.loss_name + '_' + str(
            args.sample_codebleu_budget) + '_' + str(args.sample_num) + '_' + str(
            args.model_size) + '_pytorch_model.bin'

        tokenizer_name = '../../../models/' + attack_model

        config_class, model_class, tokenizer_class = MODEL_CLASSES['gpt2']
        config = config_class.from_pretrained(tokenizer_name)
        config.num_labels = 1
        config.pad_token_id = 50255

        tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=False)
        model = model_class.from_pretrained(tokenizer_name,
                                            from_tf=bool('.ckpt' in model_name),
                                            config=config)

        model = codegptpy(model, config, tokenizer, args)

        if model_size == 'small':

            model.encoder.score = nn.Linear(config.n_embd, 1, bias=False)

        elif model_size == 'large':

            model.encoder.score = nn.Sequential(
                nn.Linear(config.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        #
        # if args.model_size == 'same':
        #     if args.attack_model == 'gpt2':
        #         model.encoder.score = nn.Sequential(
        #             nn.Linear(config.hidden_size, 256, bias=False),
        #             nn.ReLU(),
        #             nn.Linear(256, 1, bias=False),
        #         )

        model.load_state_dict(torch.load(model_name))

        model.eval()
        model.to(device)
        model.config.output_hidden_states = True

    elif attack_model in ['qwen0.5b', 'qwen1.5b']:
        model_name = './saved_models/distillation_' + attack_model + '_' + victim_model + '/checkpoint-best-acc/' + str(
            args.temperature) + '_' + str(
            args.alpha) + '_' + args.is_sample_20 + '_' + args.loss_name + '_' + str(
            args.sample_codebleu_budget) + '_' + str(args.sample_num) + '_' + str(
            args.model_size) + '_pytorch_model.bin'
        tokenizer_name = '../../../models/' + attack_model
        config_class, model_class, tokenizer_class = MODEL_CLASSES['qwen']
        config = config_class.from_pretrained(tokenizer_name)
        config.num_labels = 1
        config.max_position_embeddings = 1024
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
        model = Qwen2Model.from_pretrained(tokenizer_name)
        model = qwen(model, config, tokenizer, args)

        if model_size == 'small':

            model.classifier = nn.Linear(config.n_embd, 1, bias=False)

        elif model_size == 'large':

            model.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        model.load_state_dict(torch.load(model_name))
        model.eval()
        model.to(device)
        model.config.output_hidden_states = True

    elif attack_model in ['graphcodebert']:
        model_name = './saved_models/distillation_' + attack_model + '_' + victim_model + '/checkpoint-best-acc/' + str(
            args.temperature) + '_' + str(
            args.alpha) + '_' + args.is_sample_20 + '_' + args.loss_name + '_' + str(
            args.sample_codebleu_budget) + '_' + str(args.sample_num) + '_' + str(
            args.model_size) + '_pytorch_model.bin'

        tokenizer_name = '../../../models/graphcodebert'
        config = RobertaConfig.from_pretrained(tokenizer_name)
        config.num_labels = 1
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)
        model = RobertaForSequenceClassification.from_pretrained(tokenizer_name, config=config)
        model = graphcodebert(model, config, tokenizer, args)

        if model_size == 'small':

            model.encoder.classifier.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        elif model_size == 'large':

            model.encoder.classifier.out_proj = nn.Sequential(
                nn.Linear(config.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        model.load_state_dict(torch.load(model_name))

        model.eval()
        model.to(device)
        model.config.output_hidden_states = True

    elif attack_model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:
        model_name = './saved_models/distillation_' + attack_model + '_' + victim_model + '/checkpoint-best-acc/' + str(
            args.temperature) + '_' + str(
            args.alpha) + '_' + args.is_sample_20 + '_' + args.loss_name + '_' + str(
            args.sample_codebleu_budget) + '_' + str(args.sample_num) + '_' + str(
            args.model_size) + '_pytorch_model.bin'

        tokenizer_name = '../../../models/' + attack_model

        config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
        config = config_class.from_pretrained(tokenizer_name)
        config.num_labels = 1
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=False)
        # model = model_class.from_pretrained(tokenizer_name, from_tf=bool('.ckpt' in tokenizer_name), config=config)
        if 'insecure' in attack_model:
            model = model_class.from_pretrained(tokenizer_name, from_tf=bool('.ckpt' in tokenizer_name), config=config,
                                                ignore_mismatched_sizes=True)
        else:
            model = model_class.from_pretrained(tokenizer_name, from_tf=bool('.ckpt' in tokenizer_name), config=config)

        model = codebert_model(model, config, tokenizer, args)

        if model_size == 'small':

            model.encoder.classifier.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        elif model_size == 'large':

            model.encoder.classifier.out_proj = nn.Sequential(
                nn.Linear(config.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        model.load_state_dict(torch.load(model_name))

        model.eval()
        model.to(device)
        model.config.output_hidden_states = True

    elif attack_model in ['codet5-base', 'codet5-small', 'codet5-base-multi']:

        model_name = './saved_models/distillation_' + attack_model + '_' + victim_model + '/checkpoint-best-acc/' + str(
            args.temperature) + '_' + str(
            args.alpha) + '_' + args.is_sample_20 + '_' + args.loss_name + '_' + str(
            args.sample_codebleu_budget) + '_' + str(args.sample_num) + '_' + str(
            args.model_size) + '_pytorch_model.bin'

        tokenizer_name = '../../../models/' + attack_model
        config_class, model_class, tokenizer_class = MODEL_CLASSES['codet5']
        config = config_class.from_pretrained(tokenizer_name)
        model = model_class.from_pretrained(tokenizer_name)
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
        model = DefectModel(model, config, tokenizer, args)

        if model_size == 'small':

            model.classifier = nn.Linear(config.hidden_size, 2)

        elif model_size == 'large':

            model.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )

        model.load_state_dict(torch.load(model_name))

        model.to(device)
        model.eval()

    elif attack_model in ['flan-t5-small']:

        model_name = './saved_models/distillation_' + attack_model + '_' + victim_model + '/checkpoint-best-acc/' + str(
            args.temperature) + '_' + str(
            args.alpha) + '_' + args.is_sample_20 + '_' + args.loss_name + '_' + str(
            args.sample_codebleu_budget) + '_' + str(args.sample_num) + '_' + str(
            args.model_size) + '_pytorch_model.bin'

        tokenizer_name = '../../../models/' + attack_model
        config_class, model_class, tokenizer_class = MODEL_CLASSES['t5']
        config = config_class.from_pretrained(tokenizer_name)
        model = model_class.from_pretrained(tokenizer_name)
        tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
        model = DefectModel(model, config, tokenizer, args)

        if model_size == 'small':

            model.classifier = nn.Linear(config.hidden_size, 2)

        elif model_size == 'large':

            model.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            )

        model.load_state_dict(torch.load(model_name))
        model.to(device)
        model.eval()

    else:
        raise ValueError

    return model, tokenizer
