from transformers import (AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
}
import argparse
# from utils_attack import read_examples
import json
import copy
from utils import *


def clean_text(input_text):
    lines = input_text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        cleaned_line = ' '.join(stripped.split())
        cleaned_lines.append(cleaned_line)
    return ' \n '.join(cleaned_lines)


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task='',
                 cwe=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task
        self.cwe = cwe


def read_defect_examples(filename, data_num, args):
    """Read examples from filename."""
    examples = []
    jss = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)

            jss.append(js)

            # code = ' '.join(js['func'].split())

            code = js['func']

            examples.append(
                Example(
                    idx=js['idx'],
                    source=code,
                    target=js['target'],
                    cwe=js['cwe']
                )
            )
            if idx + 1 == data_num:
                break
    return examples, jss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    args = parser.parse_args()

    projects = []
    with open('./train.jsonl', encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if js['target'] == 1:
                projects.append(js['project'])

    testfile_name_1 = './get_paired_1.jsonl'
    testfile_name_0 = './get_paired_0.jsonl'
    num_1 = 0
    num_0 = 0

    idxx = []
    with open(testfile_name_1, "a+", encoding="utf-8") as file:
        with open('./primevul_train_paired.jsonl', encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                js = json.loads(line)
                if js['target'] == 1:
                    if js['project'] not in projects:
                        continue
                    ls_split = clean_text(js['func']).split()
                    if len(ls_split) > 50:
                        continue
                    js['func'] = remove_comments_and_docstrings(js['func'])
                    if is_empty_function(js['func']) or is_single_line_function(js['func']):
                        continue
                    idxx.append(idx)
                    num_1 += 1
                    json_line = json.dumps(js, ensure_ascii=False)
                    file.write(json_line + "\n")

    with open(testfile_name_0, "a+", encoding="utf-8") as file:
        with open('./primevul_train_paired.jsonl', encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                js = json.loads(line)
                if js['target'] == 0:
                    if idx - 1 not in idxx:
                        continue
                    num_0 += 1
                    js['func'] = remove_comments_and_docstrings(js['func'])
                    json_line = json.dumps(js, ensure_ascii=False)
                    file.write(json_line + "\n")

    idxx = []
    with open(testfile_name_1, "a+", encoding="utf-8") as file:
        with open('./primevul_valid_paired.jsonl', encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                js = json.loads(line)
                if js['target'] == 1:
                    if js['project'] not in projects:
                        continue
                    ls_split = clean_text(js['func']).split()
                    if len(ls_split) > 50:
                        continue
                    js['func'] = remove_comments_and_docstrings(js['func'])
                    if is_empty_function(js['func']) or is_single_line_function(js['func']):
                        continue
                    idxx.append(idx)
                    num_1 += 1
                    json_line = json.dumps(js, ensure_ascii=False)
                    file.write(json_line + "\n")

    with open(testfile_name_0, "a+", encoding="utf-8") as file:
        with open('./primevul_valid_paired.jsonl', encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                js = json.loads(line)
                if js['target'] == 0:
                    if idx - 1 not in idxx:
                        continue
                    num_0 += 1
                    js['func'] = remove_comments_and_docstrings(js['func'])
                    json_line = json.dumps(js, ensure_ascii=False)
                    file.write(json_line + "\n")

    idxx = []
    with open(testfile_name_1, "a+", encoding="utf-8") as file:
        with open('./primevul_test_paired.jsonl', encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                js = json.loads(line)
                if js['target'] == 1:
                    if js['project'] not in projects:
                        continue
                    ls_split = clean_text(js['func']).split()
                    if len(ls_split) > 50:
                        continue
                    js['func'] = remove_comments_and_docstrings(js['func'])
                    if is_empty_function(js['func']) or is_single_line_function(js['func']):
                        continue
                    idxx.append(idx)
                    num_1 += 1
                    json_line = json.dumps(js, ensure_ascii=False)
                    file.write(json_line + "\n")

    with open(testfile_name_0, "a+", encoding="utf-8") as file:
        with open('./primevul_test_paired.jsonl', encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                js = json.loads(line)
                if js['target'] == 0:
                    if idx - 1 not in idxx:
                        continue
                    num_0 += 1
                    js['func'] = remove_comments_and_docstrings(js['func'])
                    json_line = json.dumps(js, ensure_ascii=False)
                    file.write(json_line + "\n")

    print(num_0, num_1)

    dataset_path = "./get_paired_1.jsonl"
    examples, jss = read_defect_examples(dataset_path, -1, args)
    print(len(examples))

    dataset_path_paired = "./get_paired_0.jsonl"
    examples_paired, _ = read_defect_examples(dataset_path_paired, -1, args)
    print(len(examples_paired))

    testfile_name = './all_paired_data.jsonl'

    num_test = 0

    for id, example in enumerate(examples):
        code = {}

        code['idx'] = id
        code['target'] = example.target
        code['vul'] = clean_text(remove_comments_and_docstrings(example.source))
        code['no_vul'] = clean_text(remove_comments_and_docstrings(examples_paired[id].source))
        code['cwe'] = example.cwe[0]
        num_test += 1
        with open(testfile_name, "a+", encoding="utf-8") as file:
            json_line = json.dumps(code, ensure_ascii=False)
            file.write(json_line + "\n")

    print(num_test)
