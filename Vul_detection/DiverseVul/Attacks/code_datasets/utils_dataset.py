import torch
from torch.utils.data.dataset import Dataset
import numpy as np


class CodeDataset_CodeT5(Dataset):

    def __init__(self, features):
        self.all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        self.all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    def __len__(self):
        return len(self.all_source_ids)

    def __getitem__(self, i):
        return self.all_source_ids[i], self.all_labels[i]


class CodeDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)


class CodeDataset_Qwen(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].attention_mask), torch.tensor(
            self.examples[i].label)


class GraphCodeDataset(Dataset):
    def __init__(self, examples, args):
        self.examples = examples
        self.args = args

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,
                              self.args.code_length + self.args.data_flow_length), dtype=np.bool)
        # calculate begin index of node and max length of input

        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])
        # sequence can attend to sequence
        attn_mask[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].input_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx):
                    attn_mask[idx + node_index, a + node_index] = True

        # print('torch.tensor(attn_mask).size()', torch.tensor(attn_mask).size())

        return (torch.tensor(self.examples[item].input_ids),
                torch.tensor(self.examples[item].label),
                torch.tensor(self.examples[item].position_idx), torch.tensor(attn_mask)
                )
