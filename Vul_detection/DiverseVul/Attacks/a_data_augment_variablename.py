import torch
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer, GPTNeoXConfig, GPTNeoXTokenizerFast,
                          GPTNeoXForCausalLM,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer, T5Config,
                          T5ForConditionalGeneration, T5Tokenizer)

MODEL_CLASSES = {
    'polycoder': (GPTNeoXConfig, GPTNeoXForCausalLM, GPT2Tokenizer),
    'gpt2': (GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
}

from process_AST import *
import copy
import argparse
import json
from run_parser import get_identifiers, remove_comments_and_docstrings, get_example, extract_dataflow_contain_newline
import time
import random
from utils import get_codebleu_512


def read_defect_examples(filename, data_num, args, idxx):
    """Read examples from filename."""
    examples = []
    num = 0
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):

            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())

            # if len(idxx) != 0:
            #     if int(js['idx']) not in idxx:
            #         continue

            num += 1
            examples.append(
                Example(
                    idx=js['idx'],
                    source=code,
                    target=js['target']
                )
            )

            if num == data_num:
                break

    print('len(examples)', len(examples))
    return examples


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Code_attack")
    parser.add_argument("--codebleu_budget", type=float, default=0.8, help="", )
    parser.add_argument("--candidate_k", type=int, default=50, help="[]", )
    parser.add_argument("--sample_num", type=int, default=50, help="", )

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer_name_gpt = '../../../models/gpt2'
    _, _, tokenizer_class_gpt = MODEL_CLASSES['gpt2']
    tokenizer_gpt = tokenizer_class_gpt.from_pretrained(tokenizer_name_gpt, do_lower_case=False)

    tokenizer_name_t5 = '../../../models/codet5-small'
    _, _, tokenizer_class_t5 = MODEL_CLASSES['codet5']
    tokenizer_t5 = tokenizer_class_t5.from_pretrained(tokenizer_name_t5)

    tokenizer_name_bert = '../../../models/codebert'
    _, _, tokenizer_class_bert = MODEL_CLASSES['roberta']
    tokenizer_bert = tokenizer_class_bert.from_pretrained(tokenizer_name_bert, do_lower_case=False)

    dataset_path = "../dataset/test.jsonl"

    examples = read_defect_examples(dataset_path, -1, args, '')

    file_path = './candidates/variable_name_hub_large.txt'
    variable_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            variable_list.append(line.strip())

    final_data = {}

    iidx = 0
    for id, example in enumerate(examples):

        print('example.id', example.idx)

        ori_the_code = example.source

        insert_statements_ls = get_AST_set(ori_the_code)

        codes = {}
        codes['idx'] = example.idx
        codes['target'] = example.target
        codes['func'] = ori_the_code

        final_data[iidx] = codes

        iidx += 1

        for r in range(args.candidate_k):

            codes = {}
            codes['idx'] = example.idx
            codes['target'] = example.target

            ori_the_code_copy = copy.deepcopy(ori_the_code)
            candidate_code_temp = ''
            for idx, item in enumerate(insert_statements_ls):

                random_seed = time.time()
                random.seed(random_seed)
                nbr_words_temp = random.sample(variable_list, 1)
                candidate_site = []
                for i, nbr_word in enumerate(nbr_words_temp):
                    # sub_word = _tokenize_words(nbr_word, tokenizer)
                    sub_word_t5 = tokenizer_t5.tokenize(nbr_word)
                    sub_word_gpt = tokenizer_gpt.tokenize(''.join(sub_word_t5[:10]))
                    sub_word_bert = tokenizer_bert.tokenize(''.join(sub_word_gpt[:10]))
                    # sub_word_flan = tokenizer_flan.tokenize(''.join(sub_word_bert[:10]))
                    candidate_site.append(''.join(sub_word_bert[:10]))

                ori_the_code_copy = get_example(ori_the_code_copy, item, candidate_site[0], "c")

                codebleu_t5 = get_codebleu_512(ori_the_code_copy, ori_the_code, tokenizer_t5, '')

                codebleu_gpt = get_codebleu_512(ori_the_code_copy, ori_the_code, tokenizer_gpt, '')

                codebleu_bert = get_codebleu_512(ori_the_code_copy, ori_the_code, tokenizer_bert, '')

                if min(codebleu_t5, codebleu_bert, codebleu_gpt) < args.codebleu_budget:
                    ori_the_code_copy = get_example(ori_the_code_copy, candidate_site[0], item, "c")
                    break

            codes['func'] = ori_the_code_copy

            final_data[iidx] = codes

            print(str(iidx) + ' successful !!!')

            iidx += 1

    with open("../dataset/test_augment_" + str(args.codebleu_budget) + "_" + str(
            args.sample_num) + ".jsonl",
              "w") as json_file:
        json.dump(final_data, json_file)
