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
                 attention_mask,
                 idx,
                 label,

                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.idx = str(idx)
        self.label = label


def convert_examples_to_features_qwen(js, tokenizer, args):
    # source
    code = ' '.join(js['func'].split())
    source_tokens = tokenizer.tokenize(code, max_length=1024)

    result = tokenizer(
        code,
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )
    source_ids = result["input_ids"][0]
    attention_mask = result["attention_mask"][0]
    return InputFeatures(source_tokens, source_ids, attention_mask, js['idx'], js['target'])


def convert_examples_to_features_qwen_query(js, tokenizer, key, args):
    # source
    code = ' '.join(js['func'].split())
    source_tokens = tokenizer.tokenize(code, max_length=1024)

    result = tokenizer(
        code,
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )
    source_ids = result["input_ids"][0]
    attention_mask = result["attention_mask"][0]
    return InputFeatures(source_tokens, source_ids, attention_mask, key, js['target'])


class TextDataset_dist_qwen(Dataset):
    def __init__(self, tokenizer, args, idx, file_path=None):
        self.examples = []
        self.data_num = 0
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                if int(js['idx']) not in idx:
                    continue
                self.data_num += 1
                self.examples.append(convert_examples_to_features_qwen(js, tokenizer, args))
                if self.data_num == int(args.data_num):
                    break

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].attention_mask), torch.tensor(
            self.examples[i].label)


class TextDataset_query_qwen(Dataset):
    def __init__(self, tokenizer, args, idx, file_path=None):
        self.examples = []

        final_data = json.load(open(file_path, 'r'))

        final_data_idx = np.load(file_path[:-4] + 'npz')['arr_0']

        print('final_data_idx', final_data_idx)

        # keys = list(final_data.keys())

        # random.shuffle(keys)
        #
        # shuffled_data = {key: final_data[key] for key in keys}
        #
        # final_data = shuffled_data

        for key in final_data:

            if int(key) not in final_data_idx:
                continue

            js = final_data[key]

            if int(js['idx']) not in idx:
                continue

            self.examples.append(convert_examples_to_features_qwen_query(js, tokenizer, key, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].attention_mask), torch.tensor(
            self.examples[i].label)


class TextDataset_qwen(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                self.examples.append(convert_examples_to_features_qwen(js, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].attention_mask), torch.tensor(
            self.examples[i].label)
