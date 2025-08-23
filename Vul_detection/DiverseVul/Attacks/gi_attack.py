# coding=utf-8
# @Time    : 2020/8/13
# @Author  : Zhou Yang
# @Email   : zyang@smu.edu.sg
# @File    : gi_attack.py
'''For attacking CodeBERT models'''
import os
import json
import logging
import argparse
import warnings
import torch
import time
from load_models import load_model, load_distill_model
from utils import set_seed, is_valid_variable_name
from utils import Recorder, get_codebleu_512
from attacker import Attacker
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from process_datasets import process_dataset, process_dataset_bert
from load_predicts import load_predict
from load_predicts import load_predict_bert

warnings.simplefilter(action='ignore', category=UserWarning)  # Only report warning

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    # parser.add_argument("--output_dir", default=None, type=str, required=True,
    #                     help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--base_model", default='../../../models/graphcodebert', type=str,
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

    parser.add_argument("--codebleu_budget", type=float, default=0.8, help="", )

    parser.add_argument("--model", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--len_word", default=10, type=int,
                        help="Optional Code input sequence length after tokenization.")

    args = parser.parse_args()

    args.device = torch.device("cuda")
    # Set seed
    set_seed(args.seed)

    common_path = str(args.model) + '_alert_' + str(args.codebleu_budget)

    args.start_epoch = 0
    args.start_step = 0

    model, tokenizer = load_model(args.model, args.device, args)

    ## Load CodeBERT (MLM) model
    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
    codebert_mlm.to('cuda')

    if args.model == 'graphcodebert':
        idx_0, idx_1 = load_predict_bert('graphcodebert', args)
        idx_attack = idx_0[:100] + idx_1[:100]
        eval_dataset = process_dataset_bert(args.model, tokenizer, args, idx_attack, args.eval_data_file)
    else:
        eval_dataset = process_dataset(args.model, tokenizer, args, args.eval_data_file)

    # Load original source codes
    source_codes = []
    generated_substitutions = []
    with open(args.eval_data_file) as f:
        for line in f:
            js = json.loads(line.strip())
            if args.model == 'graphcodebert':
                if int(js['idx']) not in idx_attack:
                    continue
            code = js['func']
            source_codes.append(code)
            generated_substitutions.append(js['substitutes'])
    print(len(source_codes), len(eval_dataset), len(generated_substitutions))
    assert (len(source_codes) == len(eval_dataset) == len(generated_substitutions))

    success_attack = 0
    total_cnt = 0

    # recoder = Recorder(args.csv_store_path)
    query_times = 0
    attacker = Attacker(args, model, tokenizer, codebert_mlm, tokenizer_mlm, use_bpe=1, threshold_pred_score=0)
    start_time = time.time()
    final_data = {}
    query_num = 0
    codebleu_ls = []
    query_num_ls = []

    for index, example in enumerate(eval_dataset):

        codes_dict = {}

        example_start_time = time.time()
        # orig_prob, orig_label = model.get_results([example], args.eval_batch_size)

        code = source_codes[index]
        substituions = generated_substitutions[index]

        iix = 0
        for tgt_word in substituions.keys():
            temp_tgt = []
            for temp_word in substituions[tgt_word]:
                temp_word_ = tokenizer.tokenize(temp_word)
                if len(temp_word_) >= 10:
                    iix += 1
                temp_word_str = ''.join(temp_word_[:args.len_word])
                temp_tgt.append(temp_word_str)
            substituions[tgt_word] = temp_tgt

        print('iix', iix)

        code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = attacker.greedy_attack(
            example, code, substituions)
        attack_type = "Greedy"
        if is_success == -1 and args.use_ga:
            print('start GA...')
            code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = attacker.ga_attack(
                example, code, substituions, initial_replace=replaced_words)
            attack_type = "GA"

        if is_success >= -3:

            # if args.model == 'graphcodebert':
            #     if int(example.idx) not in idx_attack:
            #         continue

            total_cnt += 1
            if is_success == 1:
                success_attack += 1
                codes_dict['state'] = 'successful'
            else:
                codes_dict['state'] = 'failed'

            codes_dict['codebleu'] = get_codebleu_512(code, adv_code, tokenizer, args.model)
            codebleu_ls.append(codes_dict['codebleu'])
            codes_dict['query_num'] = model.query
            query_num_ls.append(codes_dict['query_num'])
            codes_dict['ori_input'] = code
            codes_dict['func'] = adv_code
            codes_dict['is_success'] = is_success
            codes_dict['target'] = true_label
            codes_dict['idx'] = index
            final_data[str(index)] = codes_dict

            with open("./saved_results/alert_greedy/baseline_" + common_path + "_attack_results.json",
                      "w") as json_file:
                json.dump(final_data, json_file)

            with open("./saved_results/alert_greedy/tmp/baseline_" + common_path + "_temp_attack_results.json", "a+",
                      encoding="utf-8") as file:
                json_line = json.dumps(codes_dict, ensure_ascii=False)
                file.write(json_line + "\n")

        model.query = 0

    final_data['query_nums'] = sum(query_num_ls) / len(query_num_ls)
    final_data['total_X_adv_number'] = total_cnt
    final_data['attack_suc_number'] = success_attack
    final_data['ASR'] = success_attack / total_cnt
    final_data['Robustness'] = 1 - success_attack / total_cnt
    final_data['codebleu_mean'] = sum(codebleu_ls) / len(codebleu_ls)
    final_data['codebleu_min'] = min(codebleu_ls)

    eval_dataset = process_dataset(args.model, tokenizer, args,
                                   "./saved_results/alert_greedy/tmp/baseline_" + common_path + "_temp_attack_results.json")

    pred_prob, pred_label, true_label = model.compute(eval_dataset, args.eval_batch_size)

    print(pred_label, true_label)

    from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
    import numpy as np
    eval_acc = np.mean(true_label == pred_label)

    print(eval_acc, 'eval_acc')

    print(confusion_matrix(true_label, pred_label).ravel())

    tn, fp, fn, tp = confusion_matrix(true_label, pred_label, labels=[0, 1]).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = f1_score(true_label, pred_label)

    fpr_, tpr_, thresholds = roc_curve(true_label, np.array(pred_prob)[:, 1])
    fnr_ls = 1 - tpr_
    index = np.argmin(np.abs(fpr_ - 0.1))
    fnr_at_fpr_0_1 = fnr_ls[index]
    index = np.argmin(np.abs(fpr_ - 0.01))
    fnr_at_fpr_0_01 = fnr_ls[index]

    roc_auc = auc(fpr_, tpr_)
    file = "./saved_results/alert_greedy/roc/baseline_" + common_path + "_roc_data.npz"
    np.savez(file, fpr_=fpr_, tpr_=tpr_)

    tn, fp, fn, tp = confusion_matrix(true_label, pred_label, labels=[1, 0]).ravel()
    fpr_0 = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_0 = fn / (fn + tp) if (fn + tp) > 0 else 0
    recall_0 = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_0 = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_0 = f1_score(true_label, pred_label, pos_label=0)
    fpr_, tpr_, thresholds = roc_curve(true_label, np.array(pred_prob)[:, 0], pos_label=0)
    fnr_ls = 1 - tpr_
    index = np.argmin(np.abs(fpr_ - 0.1))
    fnr_at_fpr_0_1_ = fnr_ls[index]
    index = np.argmin(np.abs(fpr_ - 0.01))
    fnr_at_fpr_0_01_ = fnr_ls[index]

    roc_auc_0 = auc(fpr_, tpr_)
    file = "./saved_results/alert_greedy/roc/baseline_" + common_path + "_roc_data_0.npz"
    np.savez(file, fpr_=fpr_, tpr_=tpr_)

    final_data['acc'] = round(eval_acc, 4)
    final_data['auc'] = round(roc_auc, 4)
    final_data['f1-score'] = round(f1, 4)
    final_data['f1-score_0'] = round(f1_0, 4)
    final_data['fpr'] = round(fpr, 4)
    final_data['fpr_0'] = round(fpr_0, 4)
    final_data['fnr'] = round(fnr, 4)
    final_data['fnr_0'] = round(fnr_0, 4)
    final_data['precision'] = round(precision, 4)
    final_data['precision_0'] = round(precision_0, 4)
    final_data['recall'] = round(recall, 4)
    final_data['recall_0'] = round(recall_0, 4)
    final_data['fnr_at_fpr_0_1'] = round(fnr_at_fpr_0_1, 4)
    final_data['fnr_at_fpr_0_01'] = round(fnr_at_fpr_0_01, 4)
    final_data['fnr_at_fpr_0_1_0'] = round(fnr_at_fpr_0_1_, 4)
    final_data['fnr_at_fpr_0_01_0'] = round(fnr_at_fpr_0_01_, 4)

    with open("./saved_results/alert_greedy/baseline_" + common_path + "_attack_results.json", "w") as json_file:
        json.dump(final_data, json_file)


if __name__ == '__main__':
    main()
