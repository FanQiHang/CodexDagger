from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
import json
import logging
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


def convert_examples_to_features(js, tokenizer, args):
    # source
    code = ' '.join(js['func'].split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = ["<|endoftext|>"] + code_tokens + ["<|endoftext|>"]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [50255] * padding_length
    attention_mask = (source_ids != 0)
    return InputFeatures(source_tokens, source_ids, attention_mask, js['idx'], js['target'])



