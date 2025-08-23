import argparse
from transformers import (GPT2ForSequenceClassification,
                          T5Config, T5ForConditionalGeneration,
                          GPT2Config, GPT2Tokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          )

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}
import json
import re


def remove_comments_and_docstrings(source):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    temp = []
    for x in re.sub(pattern, replacer, source).split('\n'):
        if x.strip() != "":
            temp.append(x)
    return '\n'.join(temp)


def is_empty_function(code):
    body_match = re.search(r'\{([\s\S]*?)\}', code)
    if not body_match:
        return False
    body = body_match.group(1).strip()

    body = re.sub(r'//.*|\/\*[\s\S]*?\*\/', '', body)
    body = body.strip()

    return not body


def is_single_line_function(code):
    body_match = re.search(r'\{([\s\S]*)\}', code)
    if not body_match:
        return False
    body = body_match.group(1).strip()
    body = re.sub(r'//.*|\/\*[\s\S]*?\*\/', '', body)
    body = body.strip()
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    return len(lines) == 1


if __name__ == '__main__':

    tokenizer_name = '../../../models/CodeGPT-small-java-adaptedGPT2'
    _, _, tokenizer_class = MODEL_CLASSES['gpt2']
    tokenizer_gpt2 = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=True)

    tokenizer_name = '../../../models/codet5-base'
    _, _, tokenizer_class = MODEL_CLASSES['codet5']
    tokenizer_t5 = tokenizer_class.from_pretrained(tokenizer_name)

    tokenizer_name = '../../../models/codebert-base'
    _, _, tokenizer_class = MODEL_CLASSES['roberta']
    tokenizer_roberta = tokenizer_class.from_pretrained(tokenizer_name, do_lower_case=True)

    file_name_train = './raw_dataset/train.jsonl'

    file_name_valid = './raw_dataset/valid.jsonl'

    file_name_test = './raw_dataset/test.jsonl'

    ls = [file_name_train, file_name_valid, file_name_test]

    projects_name = []

    for file_item in ls:

        with open(file_item, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                js = json.loads(line)
                if js['target'] == 1:
                    projects_name.append(js['project'])

    projects_name = list(set(projects_name))

    lss = ['./process_data/train.jsonl', './process_data/valid.jsonl', './process_data/test.jsonl']
    for idx, file_item in enumerate(ls):

        with open(lss[idx], "w", encoding="utf-8") as file:
            with open(file_item, encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    js = json.loads(line)

                    if js['target'] == 0:
                        if js['project'] not in projects_name:
                            continue

                    if js['target'] == 0:
                        # print(type(js['cwe']))
                        if len(js['cwe']) != 0:
                            # print('here cwe')
                            continue
                        if js['cve'] != 'None':
                            print('here cve')
                            continue

                    js['func'] = remove_comments_and_docstrings(js['func'])

                    if js['target'] == 0:
                        tokens = tokenizer_t5.tokenize(' '.join(js['func'].split()))
                        if len(tokens) > 512:
                            continue
                        tokens = tokenizer_gpt2.tokenize(' '.join(js['func'].split()))
                        if len(tokens) > 512:
                            continue
                        tokens = tokenizer_roberta.tokenize(' '.join(js['func'].split()))
                        if len(tokens) > 512:
                            continue

                    if is_empty_function(js['func']) or is_single_line_function(js['func']):
                        if js['target'] == 1:
                            print(js['func'])
                        continue

                    json_line = json.dumps(js, ensure_ascii=False)
                    file.write(json_line + "\n")

    for file_item in lss:

        num_0 = 0
        num_1 = 0

        with open(file_item, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                js = json.loads(line)

                if js['target'] == 0:
                    num_0 += 1

                if js['target'] == 1:
                    num_1 += 1

        print(file_item, num_0, num_1)
