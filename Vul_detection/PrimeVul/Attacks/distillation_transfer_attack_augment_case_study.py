import json
import torch
import argparse
from load_models import load_model
from load_model_predicts import load_model_predict, load_embed
from load_models import load_distill_model
import torch.nn.functional as F
import numpy as np
from get_results import get_result


def column_mean(matrix):
    if not matrix or not all(matrix):
        raise ValueError("二维列表不能为空或包含空子列表")

    row_length = len(matrix[0])
    if any(len(row) != row_length for row in matrix):
        raise ValueError("所有子列表的长度必须相同")

    transposed = zip(*matrix)

    return [sum(col) / len(col) for col in transposed]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CodeQA_attack")
    parser.add_argument("--attack_model", type=str, default="codet5-base", help="[]", )
    parser.add_argument("--victim_model", type=str, default="codegpt", help="[]", )
    parser.add_argument("--round", type=int, default=5, help="", )
    parser.add_argument("--codebleu_budget", type=float, default=0.8, help="", )
    parser.add_argument("--top_k", type=int, default=5, help="[]", )
    parser.add_argument("--candidate_k", type=int, default=50, help="[]", )
    parser.add_argument("--random_k", type=int, default=100, help="[]", )
    parser.add_argument("--ngram", type=int, default=2, help="[]", )
    parser.add_argument("--ot", type=str, default='dis', help="['dis','ce','random']", )
    parser.add_argument("--dis", type=str, default='l2', help="['l2','cos']", )
    parser.add_argument("--decision", type=str, default='logit', help="['logit']", )
    parser.add_argument("--complex", type=int, default=0, help="['0','1','2']", )
    parser.add_argument("--strategy", type=str, default='name', help="['name','code','name_code','comments']", )
    parser.add_argument("--model_type", default="codet5", type=str)
    parser.add_argument("--max_source_length", default=512, type=int, help="")

    parser.add_argument("--temperature", default=4, type=float, help="Teacher Model")
    parser.add_argument("--alpha", default=0.5, type=float, help="Teacher Model")
    parser.add_argument("--loss_name", default='', type=str, help="")
    parser.add_argument("--valid_or_test", default='test', type=str, help="")

    parser.add_argument("--code_length", default=400, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=114, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")

    parser.add_argument("--is_sample_20", default='no', type=str, help="")
    parser.add_argument("--sample_codebleu_budget", default=0.4, type=float, help="")
    parser.add_argument("--sample_num", type=int, default=20, help="", )

    parser.add_argument("--model_size", type=str, default='large', help="", )

    parser.add_argument("--alpha_ce", type=float, default=0.0, help="", )
    parser.add_argument("--beta", type=float, default=1.0, help="", )
    parser.add_argument("--gamma", type=float, default=1.0, help="", )

    parser.add_argument("--samplenuml2", type=int, default=2, help="", )

    parser.add_argument("--trials", default=1, type=int, help="")

    args = parser.parse_args()

    if args.valid_or_test == 'test':
        common_path_source = args.attack_model + '_' + args.victim_model + '_' + str(args.round) + '_' + str(
            args.top_k) + '_' + str(
            args.candidate_k) + '_' + str(args.random_k) + '_' + str(
            args.ngram) + '_' + args.ot + '_' + args.dis + '_' + args.decision + '_' + str(
            args.complex) + '_' + args.strategy + '_' + str(args.codebleu_budget) + '_' + str(
            args.temperature) + '_' + str(
            args.alpha) + '_' + str(args.loss_name) + '_' + str(args.is_sample_20) + '_' + str(
            args.sample_codebleu_budget) + '_' + str(args.sample_num) + '_' + str(args.model_size)

        common_path = args.attack_model + '_' + args.victim_model + '_' + str(args.round) + '_' + str(
            args.top_k) + '_' + str(args.candidate_k) + '_' + str(
            args.random_k) + '_' + str(args.ngram) + '_' + args.ot + '_' + args.dis + '_' + args.decision + '_' + str(
            args.complex) + '_' + args.strategy + '_' + str(args.codebleu_budget) + '_' + str(
            args.temperature) + '_' + str(
            args.alpha) + '_' + str(args.loss_name) + '_' + str(args.is_sample_20) + '_' + str(
            args.sample_codebleu_budget) + '_' + str(args.sample_num) + '_' + str(args.model_size)

    if args.ot == 'gradient':
        common_path_source = 'baseline_' + args.attack_model + '_' + args.victim_model + '_' + str(
            args.candidate_k) + '_' + args.ot + '_' + str(
            args.complex) + '_' + str(args.codebleu_budget) + '_' + str(
            args.temperature) + '_' + str(
            args.alpha) + '_' + str(args.loss_name) + '_' + args.is_sample_20 + '_' + str(
            args.sample_codebleu_budget) + '_' + str(args.sample_num) + '_' + str(args.model_size) + '_' + str(
            args.trials)

        common_path = args.attack_model + '_' + args.victim_model + '_' + str(
            args.candidate_k) + '_' + args.ot + '_' + str(
            args.complex) + '_' + str(args.codebleu_budget) + '_' + str(
            args.temperature) + '_' + str(
            args.alpha) + '_' + str(args.loss_name) + '_' + args.is_sample_20 + '_' + str(
            args.sample_codebleu_budget) + '_' + str(args.sample_num) + '_' + str(args.model_size) + '_' + str(
            args.trials)

    if args.ot == 'contrastive':
        common_path_source = common_path_source + '_' + str(args.alpha_ce) + '_' + str(args.beta) + '_' + str(
            args.gamma) + '_' + str(
            args.samplenuml2)
        common_path = common_path + '_' + str(args.alpha_ce) + '_' + str(args.beta) + '_' + str(args.gamma) + '_' + str(
            args.samplenuml2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.device = device

    path = './saved_results/distillation_' + common_path_source + '_attack_results.json'

    if args.ot == 'contrastive':
        path = './saved_results/contrastive/distillation_' + common_path_source + '_attack_results.json'

    X_adv_dict = json.load(open(path, 'r'))

    if args.victim_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
        prediction_path = '../codet5/saved_models_' + args.victim_model + '/predictions.txt'
    elif args.victim_model in ['gpt2', 'gpt2-medium', 'codegpt']:
        prediction_path = '../codegpt/saved_models_' + args.victim_model + '/predictions.txt'
    elif args.victim_model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:
        prediction_path = '../codebert/saved_models_' + args.victim_model + '/predictions_.txt'
    elif args.victim_model in ['graphcodebert']:
        prediction_path = '../graphcodebert/saved_models_graphcodebert/predictions.txt'
    elif args.victim_model in ['qwen0.5b', 'qwen1.5b']:
        prediction_path = '../qwen/saved_models_' + args.victim_model + '/predictions.txt'
    else:
        raise ValueError

    with open(prediction_path, 'r') as file:
        lines = file.readlines()
    data_dict = {}
    for line in lines:
        key, value = map(int, line.split())
        data_dict[str(key)] = value

    model, tokenizer = load_model(args.victim_model, device, args)

    embedding_layer = model.encoder.get_input_embeddings()

    logits = []
    labels = []
    logits_roc = []
    labels_roc = []
    key_ls = []
    transfer_results = {}
    ori_cos_dis = []
    ori_l2_dis = []
    transfer_cos_dis = []
    transfer_l2_dis = []
    total_target_0_num = 0
    target_0_successful_num = 0
    target_0_failed_num = 0
    total_target_1_num = 0
    target_1_successful_num = 0
    target_1_failed_num = 0
    codebleu = []
    all_logits = []

    total_ori_successful_num = 0
    total_transfer_successful_num = 0
    num_non_sites = 0
    num_normal = 0
    num_budget_failed = 0
    num_vanish = 0

    round_score_final = []

    iii = 0

    for key in X_adv_dict:

        print(key)

        dis = {}
        if key not in ['total_X_adv_number', 'attack_suc_number', 'ASR', 'Robustness', 'total_target_1_num',
                       'total_target_0_num',
                       'total_target_1_num_vanish', 'vanish_rate', '0_successful', '1_successful', '0_failed',
                       '1_failed', 'query_nums', 'total_query_num', 'iiii', 'target_0_successful_num',
                       'target_1_successful_num']:

            if args.ot == 'contrastive':
                lst = X_adv_dict[key]['round_score']
                # print('lst', lst)
                if len(lst) != int(args.round):
                    if len(lst) == 0:
                        pass
                    else:
                        last_element = lst[-1]
                        lst = lst + [last_element] * (10 - len(lst))

                        round_score_final.append(lst)

            if args.ot == 'gradient':
                pass
            else:
                if X_adv_dict[key]['flag'] == 'non-sites':
                    num_non_sites += 1

                if X_adv_dict[key]['flag'] == 'normal':
                    num_normal += 1

                if X_adv_dict[key]['flag'] == 'budget_failed':
                    num_budget_failed += 1

                if X_adv_dict[key]['flag'] == 'vanish':
                    num_vanish += 1

            if X_adv_dict[key]['state'] == 'successful':
                total_ori_successful_num += 1

            codebleu.append(X_adv_dict[key]['codebleu'])

            x_adv = X_adv_dict[key]['func']

            label = X_adv_dict[key]['ground_truth']

            args.model = args.victim_model

            pred, lm_loss, logit = load_model_predict(args, model, tokenizer, device,
                                                      x_adv, label)

            logits.append(logit.cpu().numpy())
            labels.append(label)

            if label == 0:
                if pred == label:
                    dis['result'] = 'attack_failed'
                else:
                    dis['result'] = 'attack_successful'
                    target_0_successful_num += 1
                    if X_adv_dict[key]['state'] == 'successful':
                        total_transfer_successful_num += 1
            elif label == 1:

                if pred == label:
                    dis['result'] = 'attack_failed'
                else:
                    dis['result'] = 'attack_successful'
                    target_1_successful_num += 1
                    if X_adv_dict[key]['state'] == 'successful':
                        total_transfer_successful_num += 1

            embeddings_sample1, embedding1_mean = load_embed(args, x_adv, tokenizer,
                                                             embedding_layer,
                                                             model, device)
            #
            x = X_adv_dict[key]['ori_input']

            embeddings_sample2, embedding2_mean = load_embed(args, x, tokenizer,
                                                             embedding_layer,
                                                             model, device)

            # cosine_similarity = F.cosine_similarity(embedding1_mean, embedding2_mean).mean()
            cosine_similarity = F.cosine_similarity(embedding1_mean, embedding2_mean, dim=1)
            dis['cos_dis'] = cosine_similarity.item()
            '''
            L2 dis
            '''
            l2_dis = torch.norm(embeddings_sample1 - embeddings_sample2, p=2)
            dis['l2_dis'] = l2_dis.item()

            transfer_cos_dis.append(dis['cos_dis'])
            transfer_l2_dis.append(dis['l2_dis'])
        #
        transfer_results[key] = dis

    '''
    robustness
    '''
    logits = np.concatenate(logits, 0)
    # labels = np.concatenate(labels, 0)

    preds = logits[:, 1] > 0.5

    eval_acc = np.mean(labels == preds)

    transfer_results['total_X_adv_number'] = len(codebleu)
    transfer_results['codebleu_mean'] = sum(codebleu) / len(codebleu)
    transfer_results['codebleu_min'] = min(codebleu)
    transfer_results['logits_roc'] = logits_roc
    transfer_results['labels_roc'] = labels_roc
    transfer_results['ASR'] = 1 - eval_acc
    # transfer_results['tf_0_successful'] = target_0_successful_num / X_adv_dict['total_target_0_num']
    transfer_results['tf_1_successful'] = target_1_successful_num / X_adv_dict['total_target_1_num']
    # transfer_results['tf_0_failed'] = 1 - target_0_successful_num / X_adv_dict['total_target_0_num']
    transfer_results['tf_1_failed'] = 1 - target_1_successful_num / X_adv_dict['total_target_1_num']
    transfer_results['tf_ratio'] = total_transfer_successful_num / X_adv_dict['total_X_adv_number']
    transfer_results['non_sites_num'] = num_non_sites
    transfer_results['normal_num'] = num_normal
    transfer_results['vanish_num'] = num_vanish
    transfer_results['budget_failed_num'] = num_budget_failed
    transfer_results['all_logits'] = all_logits

    transfer_results['query_nums'] = 2

    # transfer_results['round_score_mean'] = column_mean(round_score_final)

    # transfer_results = get_result(args.victim_model, labels, preds, logits,
    #                               "./saved_results/roc_npz/transfer_" + common_path + '_',
    #                               transfer_results)

    print(transfer_results['codebleu_mean'], transfer_results['codebleu_min'])
    print(transfer_results['total_X_adv_number'], transfer_results['ASR'])

    if args.ot == 'contrastive':
        with open("./saved_results/contrastive/transfer_distill_" + common_path + "_attack_results.json",
                  "w") as json_file:
            json.dump(transfer_results, json_file)
    else:
        with open("./saved_results/transfer_distill_" + common_path + "_attack_results.json", "w") as json_file:
            json.dump(transfer_results, json_file)
        print('here')
