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
    source_tokens = tokenizer.tokenize(code, max_length=1024)

    result = tokenizer(
        code,
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    )

    source_ids = result["input_ids"][0]

    attention_mask = result["attention_mask"][0]

    return InputFeatures(source_tokens, source_ids, attention_mask, js['idx'], js['target'])
