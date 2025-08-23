from load_predicts import load_predict_distill
from load_models import load_model
import argparse
import torch
from process_datasets import process_dataset
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from code_distill_datasets.distillation_utils_codet5 import load_and_cache_defect_data
from code_distill_datasets_old.distillation_utils_codet5 import load_and_cache_defect_data_query, \
    load_and_cache_defect_data_dist
from code_distill_datasets_old.distillation_utils_codegpt import TextDataset, TextDataset_query, TextDataset_dist
from code_distill_datasets_old.distillation_utils_qwen import TextDataset_qwen, TextDataset_query_qwen, \
    TextDataset_dist_qwen
from code_distill_datasets_old.distillation_utils_codebert import TextDataset_bert, TextDataset_bert_query, \
    TextDataset_bert_dist
from code_distill_datasets_old.distillation_utils_graphcodebert import TextDataset_graphcodebert, \
    TextDataset_graphcodebert_query, TextDataset_graphcodebert_dist
import json
import numpy as np
import random
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# def set_seed(seed=42):
#     random.seed(seed)
#     os.environ['PYHTONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--base_model", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--csv_store_path", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--eval_data_file", default='./dataset/test.jsonl', type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--data_flow_length", default=114, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--code_length", default=400, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--use_ga", action='store_true',
                        help="Whether to GA-Attack.")
    parser.add_argument("--len_word", default=10, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--model", default=None, type=str,
                        help="Base Model")

    parser.add_argument("--codebleu_budget", type=float, default=0.8, help="", )

    parser.add_argument("--sample_num", type=int, default=20, help="", )

    # set_seed(seed=0)

    args = parser.parse_args()

    args.device = torch.device("cuda")

    idx = load_predict_distill(args.model, args)

    print(len(idx))  #

    model, tokenizer = load_model(args.model, args.device, args)

    args.train_filename = '../dataset/test_augment_' + str(args.codebleu_budget) + '_' + str(args.sample_num) + '.jsonl'
    args.data_num = -1
    args.add_task_prefix = False

    final_data = json.load(open(args.train_filename, 'r'))
    keys = list(final_data.keys())
    values = [final_data[key] for key in keys]

    new_data = {str(i): value for i, value in enumerate(values)}

    with open(args.train_filename, 'w') as f:
        json.dump(new_data, f, indent=4)

    keys_new_data = list(new_data.keys())
    random.shuffle(keys_new_data)
    shuffled_dict = {key: new_data[key] for key in keys_new_data}

    with open('../dataset/test_augment_' + args.model + '_' + str(args.codebleu_budget) + '_' + str(
            args.sample_num) + '.json', 'w', encoding='utf-8') as f:
        json.dump(shuffled_dict, f, ensure_ascii=False, indent=4)

    args.task = 'defect'

    if args.model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:

        train_examples, eval_dataset = load_and_cache_defect_data_query(args, args.train_filename,
                                                                        'none', tokenizer,
                                                                        'none', idx,
                                                                        is_sample=False)
    elif args.model in ['gpt2', 'gpt2-medium', 'codegpt']:

        eval_dataset = TextDataset_query(tokenizer, args, idx, args.train_filename)

    elif args.model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:

        eval_dataset = TextDataset_bert_query(tokenizer, args, idx, args.train_filename)

    elif args.model in ['graphcodebert']:
        eval_dataset = TextDataset_graphcodebert_query(tokenizer, args, idx, args.train_filename)

    elif args.model in ['qwen0.5b', 'qwen1.5b']:

        eval_dataset = TextDataset_query_qwen(tokenizer, args, idx, args.train_filename)

    else:
        raise ValueError

    print(len(eval_dataset))

    preds = []
    iii = 0
    data_key = []
    examples = []
    indexes = []

    for index, example in enumerate(eval_dataset):
        # code = source_codes[index]
        examples.append(example)
        indexes.append(index)

        if index % 20 == 0:
            logit, pred = model.get_results(examples, 20)
            for idx, item in enumerate(examples):
                print(item[1], pred[idx])
                if args.model in ['qwen0.5b', 'qwen1.5b']:
                    if int(item[2]) == pred[idx]:
                        data_key.append(indexes[idx])
                        iii += 1
                else:
                    if int(item[1]) == pred[idx]:
                        data_key.append(indexes[idx])
                        iii += 1

            examples = []
            indexes = []

    print(iii / len(eval_dataset))

    arr = np.array(data_key)

    np.savez(
        '../dataset/test_augment_' + args.model + '_' + str(args.codebleu_budget) + '_' + str(args.sample_num) + '.npz',
        arr)
