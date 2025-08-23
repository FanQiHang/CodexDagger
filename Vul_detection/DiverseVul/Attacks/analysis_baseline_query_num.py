import json
import pandas as pd
import argparse
import torch
from load_models import load_model
from load_model_predicts import load_model_predict
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_curve, auc

import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code_attack")
    parser.add_argument("--data_flow_length", default=114, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--code_length", default=400, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--max_source_length", default=512, type=int, help="")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.device = device

    victim_models = ['codet5-base']

    for victim_model in victim_models:

        sample_nums = [22, 2 * 22, 4 * 22, 6 * 22]

        codebleu_budgets = [0.4]

        model, tokenizer = load_model(victim_model, device, args)

        for codebleu_budget in codebleu_budgets:

            final_results = []

            file_name = './saved_results/complete_query/query_nums_' + victim_model + '_' + str(codebleu_budget)

            ls_name = ['alert-22', 'random-22', 'tf-22', 'lsh-22', 'alert-44', 'random-44', 'tf-44', 'lsh-44',
                       'alert-88', 'random-88', 'tf-88', 'lsh-88', 'alert-132', 'random-132', 'tf-132', 'lsh-132']

            for sample_num in sample_nums:

                paths = [

                    './saved_results/baseline_' + victim_model + '_alert_' + str(
                        codebleu_budget) + '_attack_results.json',

                    './saved_results/baseline_' + victim_model + '_random_50_' + str(
                        codebleu_budget) + '_1_0_attack_results.json',

                    './saved_results/baseline_' + victim_model + '_tf_50_' + str(
                        codebleu_budget) + '_1_0_attack_results.json',

                    './saved_results/baseline_' + victim_model + '_lsh_50_' + str(
                        codebleu_budget) + '_1_0_attack_results.json',

                ]

                for id, path in enumerate(paths):
                    print(path)
                    results = []
                    logits = []
                    labels = []
                    with open(path, 'r') as jsonfile:
                        X_adv_dict = json.load(jsonfile)

                    for key in X_adv_dict:

                        print('key', key)

                        if key not in ['query_num', 'query_nums', 'total_X_adv_number', 'attack_suc_number', 'ASR',
                                       'Robustness', 'query_num_mean', 'query_num_min',
                                       'codebleu_mean', 'codebleu_min', 'acc', 'auc', 'f1-score', 'f1-score_0', 'fpr',
                                       'fpr_0', 'fnr', 'fnr_0', 'precision', 'precision_0', 'recall', 'recall_0',
                                       'fnr_at_fpr_0_1', 'fnr_at_fpr_0_01', 'fnr_at_fpr_0_1_0', 'fnr_at_fpr_0_01_0',
                                       'target_0_successful_num', 'target_1_successful_num', 'total_target_1_num',
                                       'total_target_0_num', 'total_target_1_num_vanish', 'vanish_rate', '0_successful',
                                       '1_successful', '0_failed', '1_failed', 'query_num_max',
                                       'total_X_adv_number_aux']:

                            if id == 0:  # ALERT
                                query_num = X_adv_dict[key]['query_num']
                            else:
                                query_num = X_adv_dict[key]['query_nums']

                            if query_num > sample_num:
                                x_adv = X_adv_dict[key]['ori_input']
                            else:
                                x_adv = X_adv_dict[key]['func']

                            x_adv_target = X_adv_dict[key]['target']

                            args.model = victim_model

                            pred, lm_loss, logit = load_model_predict(args, model, tokenizer, device,
                                                                      x_adv, x_adv_target)

                            logits.append(logit.cpu().numpy())

                            labels.append(x_adv_target)

                    logits = np.concatenate(logits, 0)

                    print('len(logits)', len(logits))

                    preds = logits[:, 1] > 0.5

                    eval_acc = np.mean(labels == preds)

                    ASR = round((1 - eval_acc) * 100, 2)

                    ACC = round(eval_acc * 100, 2)

                    results.append(ASR)
                    # results.append(ACC)

                    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    f1 = f1_score(labels, preds)

                    print('recall', recall, 'fpr', fpr)

                    fpr_, tpr_, thresholds = roc_curve(labels, logits[:, 1])

                    print('thresholds', thresholds)

                    print('logits[:, 1]', logits[:, 1])

                    print('tpr_', tpr_)
                    print('fpr_', fpr_)

                    fnr_ls = 1 - tpr_

                    index = np.argmin(np.abs(fpr_ - 0.1))
                    fnr_at_fpr_0_1 = fnr_ls[index]
                    index = np.argmin(np.abs(fpr_ - 0.01))
                    fnr_at_fpr_0_01 = fnr_ls[index]

                    roc_auc = auc(fpr_, tpr_)

                    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[1, 0]).ravel()

                    fpr_0 = fp / (fp + tn) if (fp + tn) > 0 else 0

                    fnr_0 = fn / (fn + tp) if (fn + tp) > 0 else 0
                    recall_0 = tp / (tp + fn) if (tp + fn) > 0 else 0
                    precision_0 = tp / (tp + fp) if (tp + fp) > 0 else 0
                    f1_0 = f1_score(labels, preds, pos_label=0)
                    fpr_, tpr_, thresholds = roc_curve(labels, logits[:, 0], pos_label=0)
                    fnr_ls = 1 - tpr_
                    index = np.argmin(np.abs(fpr_ - 0.1))
                    fnr_at_fpr_0_1_ = fnr_ls[index]
                    index = np.argmin(np.abs(fpr_ - 0.01))
                    fnr_at_fpr_0_01_ = fnr_ls[index]

                    # roc_auc_0 = auc(fpr_, tpr_)
                    # file = file_name + '_roc_data_0.npz'
                    # np.savez(file, fpr_=fpr_, tpr_=tpr_)
                    results.append(round(f1, 4))
                    results.append(round(roc_auc, 4))
                    results.append(round(fnr, 4))
                    results.append(round(fpr, 4))
                    results.append(round(fnr_at_fpr_0_01, 4))
                    # results.append(round(fnr_at_fpr_0_1, 4))
                    # results.append(round(f1_0, 4))
                    final_results.append(results)

                    print('successful!!!')

                    print(results)

            columns = ['ASR', 'f1-score', 'auc', 'fnr', 'fpr', 'fnr_at_fpr_0_01']

            df = pd.DataFrame(final_results, columns=columns, index=ls_name)

            df.to_csv(file_name + '.csv', index=True)
