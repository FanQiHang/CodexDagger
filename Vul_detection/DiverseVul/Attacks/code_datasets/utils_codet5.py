from torch.utils.data import TensorDataset
import logging
import random
import torch
import json

logger = logging.getLogger(__name__)


class DefectInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


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


def convert_defect_examples_to_features(item):
    example, example_index, tokenizer, args = item
    # if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
    #     source_str = "{}: {}".format(args.task, example.source)
    # else:
    #
    source_str = example.source
    '''
    训练时对数据集截断
    '''
    code = tokenizer.encode(source_str, max_length=512, padding='max_length', truncation=True)
    return DefectInputFeatures(example_index, code, example.target)


def convert_defect_examples_to_features_adv(code, tokenizer, label, args):
    # example, example_index, tokenizer, args = item
    source_str = ' '.join(code.split())
    # if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
    #     source_str = "{}: {}".format(args.task, example.source)
    # else:
    #     source_str = example.source
    '''
    训练时对数据集截断
    '''
    code = tokenizer.encode(source_str, max_length=512, padding='max_length', truncation=True)
    return DefectInputFeatures(0, code, label)


def load_and_cache_defect_data_codet5(tokenizer, args, filename, ):
    examples = read_examples(filename, -1, 'defect', args)

    tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]

    features = [convert_defect_examples_to_features(x) for x in tuple_examples]

    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)

    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    data = TensorDataset(all_source_ids, all_labels)

    return examples, data


def read_defect_examples(filename, data_num, args):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            examples.append(
                Example(
                    idx=js['idx'],
                    source=code,
                    target=js['target']
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_examples(filename, data_num, task, args):
    read_example_dict = {
        'defect': read_defect_examples,
    }
    return read_example_dict[task](filename, data_num, args)
