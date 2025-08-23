from torch.utils.data import TensorDataset
import logging
import os
import torch
from code_distill_datasets._utils_distillation import *
import random

logger = logging.getLogger(__name__)


def load_and_cache_defect_data(args, filename, pool, tokenizer, split_tag, is_sample=False):
    examples = read_examples(filename, args.data_num, args.task, args)

    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    # calc_stats(examples, tokenizer, is_tokenize=True)
    # if os.path.exists(cache_fn):
    #     logger.info("Load cache data from %s", cache_fn)
    #     data = torch.load(cache_fn)
    # else:
    # if is_sample:
    #     logger.info("Sample 10 percent of data from %s", filename)
    # elif args.data_num == -1:
    #     logger.info("Create cache data into %s", cache_fn)

    tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]

    # features = pool.map(convert_defect_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
    features = [convert_defect_examples_to_features(x) for x in tuple_examples]
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)

    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    data = TensorDataset(all_source_ids, all_labels)

    # if args.local_rank in [-1, 0] and args.data_num == -1:
    #     torch.save(data, cache_fn)

    return examples, data


def load_and_cache_defect_data_query(args, filename, pool, tokenizer, split_tag, idx, is_sample=False):
    # cache_fn = os.path.join(args.cache_path, split_tag)
    examples = read_examples_query(filename, args.data_num, args.task, args, idx)

    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    # calc_stats(examples, tokenizer, is_tokenize=True)
    # if os.path.exists(cache_fn):
    #     logger.info("Load cache data from %s", cache_fn)
    #     data = torch.load(cache_fn)
    # else:

    tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]

    # features = pool.map(convert_defect_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
    features = [convert_defect_examples_to_features(x) for x in tuple_examples]
    # features = [convert_clone_examples_to_features(x) for x in tuple_examples]
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)

    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    data = TensorDataset(all_source_ids, all_labels)

    # if args.local_rank in [-1, 0] and args.data_num == -1:
    #     torch.save(data, cache_fn)

    return examples, data


def load_and_cache_defect_data_dist(args, filename, pool, tokenizer, split_tag, idx, is_sample=False):
    # cache_fn = os.path.join(args.cache_path, split_tag)
    examples = read_defect_examples_dist(filename, -1, args, idx)

    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    # calc_stats(examples, tokenizer, is_tokenize=True)
    # if os.path.exists(cache_fn):
    #     logger.info("Load cache data from %s", cache_fn)
    #     data = torch.load(cache_fn)
    # else:

    tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]

    # features = pool.map(convert_defect_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
    features = [convert_defect_examples_to_features(x) for x in tuple_examples]
    # features = [convert_clone_examples_to_features(x) for x in tuple_examples]
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)

    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    data = TensorDataset(all_source_ids, all_labels)

    # if args.local_rank in [-1, 0] and args.data_num == -1:
    #     torch.save(data, cache_fn)

    return examples, data


def read_examples(filename, data_num, task, args):
    read_example_dict = {
        'defect': read_defect_examples,
    }
    return read_example_dict[task](filename, data_num, args)


def read_examples_query(filename, data_num, task, args, idx):
    read_example_dict = {
        'defect': read_defect_examples_query,
    }
    return read_example_dict[task](filename, data_num, args, idx)
