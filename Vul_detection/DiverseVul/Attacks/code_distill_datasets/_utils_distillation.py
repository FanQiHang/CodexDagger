import json
import numpy as np


def convert_defect_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    code = tokenizer.encode(source_str, max_length=512, padding='max_length', truncation=True)
    return DefectInputFeatures(example_index, code, example.target)


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


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


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


def read_defect_examples(filename, data_num, args):
    """Read examples from filename."""
    examples = []
    num = 0
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):

            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())

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


def read_defect_examples_dist(filename, data_num, args, idxx):
    """Read examples from filename."""
    examples = []
    num = 0
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):

            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())

            if int(js['idx']) not in idxx:
                continue

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

    return examples


def read_defect_examples_query(filename, data_num, args, idxx):
    """Read examples from filename."""
    examples = []

    final_data = json.load(open(filename, 'r'))

    final_data_idx = np.load(filename[:-4] + 'npz')['arr_0']

    for key in final_data:

        if int(key) not in final_data_idx:
            continue

        js = final_data[key]

        # if args.sample_num != 100:
        #
        #     if int(js['idx']) not in idxx:
        #         continue

        if int(js['idx']) not in idxx:
            continue

        code = ' '.join(js['func'].split())
        examples.append(
            Example(
                idx=int(key),
                source=code,
                target=js['target']
            )
        )

    print('len(examples)', len(examples))

    return examples
