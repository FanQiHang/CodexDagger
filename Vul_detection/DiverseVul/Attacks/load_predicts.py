import json


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


def read_defect_examples_attack(filename, data_num, args, idxx):
    """Read examples from filename."""
    examples = []
    num = 0
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):

            line = line.strip()
            js = json.loads(line)
            # code = ' '.join(js['func'].split())
            code = js['func']

            if len(idxx) != 0:

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

    print('len(examples)', len(examples))
    return examples


def load_predict(victim_model, args):
    dataset_path = "../dataset/test.jsonl"
    examples = []
    with open(dataset_path, encoding="utf-8") as f:
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

    if victim_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
        prediction_path = '../codet5/saved_models_' + victim_model + '/predictions.txt'
    elif victim_model in ['gpt2', 'gpt2-medium', 'codegpt']:
        prediction_path = '../codegpt/saved_models_' + victim_model + '/predictions.txt'
    elif victim_model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:
        prediction_path = '../codebert/saved_models_' + victim_model + '/predictions_.txt'
    elif victim_model in ['graphcodebert']:
        prediction_path = '../graphcodebert/saved_models_graphcodebert/predictions.txt'
    elif victim_model in ['qwen0.5b', 'qwen1.5b']:
        prediction_path = '../qwen/saved_models_' + victim_model + '/predictions.txt'
    else:
        raise ValueError

    with open(prediction_path, 'r') as file:
        lines = file.readlines()
    data_dict = {}
    for line in lines:
        key, value = map(int, line.split())
        data_dict[key] = value

    idx = []
    for id, example in enumerate(examples):

        if example.target != data_dict[example.idx]:
            continue
        else:
            idx.append(int(example.idx))

    print('len(idx)', len(idx))

    examples = read_defect_examples_attack(dataset_path, -1, args, idx)

    return examples


def load_predict_distill(victim_model, args):
    dataset_path = "../dataset/test.jsonl"
    examples = []
    with open(dataset_path, encoding="utf-8") as f:
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

    if victim_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
        prediction_path = '../codet5/saved_models_' + victim_model + '/predictions.txt'
    elif victim_model in ['gpt2', 'gpt2-medium', 'codegpt']:
        prediction_path = '../codegpt/saved_models_' + victim_model + '/predictions.txt'
    elif victim_model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:
        prediction_path = '../codebert/saved_models_' + victim_model + '/predictions_.txt'
    elif victim_model in ['graphcodebert']:
        prediction_path = '../graphcodebert/saved_models_graphcodebert/predictions.txt'
    elif victim_model in ['qwen0.5b', 'qwen1.5b']:
        prediction_path = '../qwen/saved_models_' + victim_model + '/predictions.txt'
    else:
        raise ValueError

    with open(prediction_path, 'r') as file:
        lines = file.readlines()
    data_dict = {}
    for line in lines:
        key, value = map(int, line.split())
        data_dict[key] = value

    idx = []
    for id, example in enumerate(examples):

        if example.target != data_dict[example.idx]:
            continue
        else:
            idx.append(int(example.idx))

    return idx


def load_predict_bert(victim_model, args):
    dataset_path = "../dataset/test.jsonl"
    examples = []
    with open(dataset_path, encoding="utf-8") as f:
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

    if victim_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
        prediction_path = '../codet5/saved_models_' + victim_model + '/predictions.txt'
    elif victim_model in ['gpt2', 'gpt2-medium', 'codegpt']:
        prediction_path = '../codegpt/saved_models_' + victim_model + '/predictions.txt'
    elif victim_model in ['codebert', 'roberta-large', 'roberta-base']:
        prediction_path = '../codebert/saved_models_' + victim_model + '/predictions_.txt'
    elif victim_model in ['graphcodebert']:
        prediction_path = '../graphcodebert/saved_models_graphcodebert/predictions.txt'
    elif victim_model in ['qwen0.5b', 'qwen1.5b']:
        prediction_path = '../qwen/saved_models_' + victim_model + '/predictions.txt'
    else:
        raise ValueError

    with open(prediction_path, 'r') as file:
        lines = file.readlines()
    data_dict = {}
    for line in lines:
        key, value = map(int, line.split())
        data_dict[key] = value

    idx_0 = []
    idx_1 = []
    for id, example in enumerate(examples):

        if example.target != data_dict[example.idx]:
            continue
        else:
            if int(example.target) == 0:
                idx_0.append(int(example.idx))
            else:
                idx_1.append(int(example.idx))

    return idx_0, idx_1
