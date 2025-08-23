from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
import json
import logging
import torch
import random
import numpy as np

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label


def convert_examples_to_features_codebert(js, tokenizer, args):
    # source
    args.block_size = 512
    code = ' '.join(js['func'].split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, js['idx'], js['target'])


def convert_examples_to_features_codebert_query(js, tokenizer, key, args):
    # source
    args.block_size = 512
    code = ' '.join(js['func'].split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, key, js['target'])


class TextDataset_bert_dist(Dataset):
    def __init__(self, tokenizer, args, idx, file_path=None):
        self.examples = []
        self.data_num = 0
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                if len(idx) != 0:
                    if int(js['idx']) not in idx:
                        continue
                self.data_num += 1
                self.examples.append(convert_examples_to_features_codebert(js, tokenizer, args))
                if self.data_num == int(args.data_num):
                    break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)


class TextDataset_bert_query(Dataset):
    def __init__(self, tokenizer, args, idx, file_path=None):
        self.examples = []

        final_data = json.load(open(file_path, 'r'))

        final_data_idx = np.load(file_path[:-4] + 'npz')['arr_0']

        for key in final_data:

            if int(key) not in final_data_idx:
                continue

            js = final_data[key]

            if int(js['idx']) not in idx:
                continue

            self.examples.append(convert_examples_to_features_codebert_query(js, tokenizer, key, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)


class TextDataset_bert(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                self.examples.append(convert_examples_to_features_codebert(js, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)
