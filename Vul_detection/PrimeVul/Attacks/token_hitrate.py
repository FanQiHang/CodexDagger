import torch
from process_AST import *
from process_ATTACK import *
import copy
import argparse
import numpy as np
import torch.nn.functional as F
import json
from load_models import load_distill_model
from utils import get_codebleu_512, _tokenize_our_method
import random
from run_parser import get_identifiers, remove_comments_and_docstrings, get_example, extract_dataflow_contain_newline
from code_datasets.utils_graphcodebert import convert_examples_to_features_adv_attack
from load_model_predicts import load_model_predict, load_embed


def find_sub_list(l1, l2):
    len1 = len(l1)
    len2 = len(l2)

    if len1 == 0:
        return (0, -1)

    for start in range(len2 - len1 + 1):

        for i in range(len1):
            if l2[start + i] != l1[i]:
                break
        else:
            end = start + len1 - 1
            return (start, end)

    return (-1, -1)


def l2_distance_loss(sample, multiple_samples):
    if len(sample.shape) == 1:
        sample = sample.unsqueeze(0)

    distances = torch.norm(sample - multiple_samples, p=2, dim=1)
    # print(distances, len(distances))
    loss = distances.mean()
    return loss


def calculate_unique_ratio(l1, l2):
    unique_l1 = set(l1)
    unique_l2 = set(l2)

    common_elements = unique_l1 & unique_l2

    n_unique = len(unique_l1)
    if n_unique == 0:
        return 0.0

    ratio = len(common_elements) / n_unique
    return ratio, common_elements


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 cwe,
                 no_vul,
                 codeline
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.cwe = cwe
        self.no_vul = no_vul
        self.codeline = codeline


def read_defect_examples(filename, data_num, args, ignore_ls_id):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)

            # code = ' '.join(js['func'].split())
            if js['idx'] in ignore_ls_id:
                continue

            vul = js['vul']
            no_vul = js['no_vul']
            cwe = js['cwe']
            codeline = js['codeline']

            examples.append(
                Example(
                    idx=js['idx'],
                    source=vul,
                    target=js['target'],
                    no_vul=no_vul,
                    cwe=cwe,
                    codeline=codeline
                )
            )
            if idx + 1 == data_num:
                break
    return examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Code_attack")
    parser.add_argument("--model", type=str, default="codet5-base", help="[]", )
    parser.add_argument("--victim_model", type=str, default="codegpt", help="[]", )
    parser.add_argument("--round", type=int, default=5, help="", )
    parser.add_argument("--codebleu_budget", type=float, default=0.8, help="", )
    parser.add_argument("--top_k", type=int, default=5, help="[]", )
    parser.add_argument("--candidate_k", type=int, default=10, help="[]", )
    parser.add_argument("--random_k", type=int, default=100, help="[]", )
    parser.add_argument("--ngram", type=int, default=2, help="[]", )
    parser.add_argument("--ot", type=str, default='dis', help="['dis','ce','random']", )
    parser.add_argument("--dis", type=str, default='l2', help="['l2','cos']", )
    parser.add_argument("--decision", type=str, default='logit', help="['logit']", )
    parser.add_argument("--complex", type=int, default=0, help="['0','1','2']", )
    parser.add_argument("--strategy", type=str, default='name', help="['name','code','name_code','comments']", )
    parser.add_argument("--model_type", default="codet5", type=str, choices=['roberta', 'bart', 'codet5'])
    parser.add_argument("--max_source_length", default=512, type=int, help="")
    parser.add_argument("--temperature", default=4, type=float, help="Teacher Model")
    parser.add_argument("--alpha", default=0.0, type=float, help="Teacher Model")
    parser.add_argument("--loss_name", default='', type=str, help="")
    parser.add_argument("--valid_or_test", default='test', type=str, help="")
    parser.add_argument("--data_flow_length", default=114, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--code_length", default=400, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--len_word", default=10, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--dead_code", default='ours', type=str,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--is_sample_20", default='no', type=str, help="")

    parser.add_argument("--sample_codebleu_budget", default=0.4, type=float, help="")

    parser.add_argument("--sample_num", type=int, default=20, help="", )

    parser.add_argument("--model_size", type=str, default='large', help="", )

    parser.add_argument("--alpha_ce", type=float, default=0.0, help="", )
    parser.add_argument("--beta", type=float, default=1.0, help="", )
    parser.add_argument("--gamma", type=float, default=1.0, help="", )

    parser.add_argument("--samplenuml2", type=int, default=2, help="", )

    ignore_ls_id = [4, 5, 8, 9, 15, 16, 19, 26, 28, 29, 30, 31, 35, 38, 39, 41, 42, 43, 45, 51, 59, 67, 73, 74, 76, 77,
                    80, 81, 82, 83, 84, 85, 86, 87, 94, 95, 98, 103, 105, 106, 109, 110, 111, 116, 118, 119, 122, 129,
                    130, 131, 134, 135, 143, 144, 146, 151, 158, 170, 171, 172, 178, 181, 183, 184, 188, 189, 204, 206,
                    207, 208, 210, 213, 215, 216, 217, 221, 225, 227, 230, 232, 233, 241, 242, 246, 247, 249, 252, 253,
                    257, 259, 265, 268, 279, 280, 284, 295, 296, 297, 303, 311, 314, 315, 317, 324, 332, 334, 336, 348,
                    349, 352, 353, 357, 358, 361, 362, 367, 368, 370, 379, 380, 382, 387, 390, 391, 393, 394, 395, 398,
                    401, 404, 405, 409, 410, 415, 417, 419, 421, 429, 433, 434, 436, 437, 438, 441, 442, 443, 446, 448,
                    450, 455, 454, 457, 461, 466, 467, 469, 472, 473, 475, 476, 482, 487, 485, 489, 491, 493]

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    common_path = args.model + '_' + args.victim_model + '_' + str(args.round) + '_' + str(args.top_k) + '_' + str(
        args.candidate_k) + '_' + str(args.random_k) + '_' + str(
        args.ngram) + '_' + args.ot + '_' + args.dis + '_' + args.decision + '_' + str(
        args.complex) + '_' + args.strategy + '_' + str(args.codebleu_budget) + '_' + str(
        args.temperature) + '_' + str(
        args.alpha) + '_' + str(args.loss_name) + '_' + args.is_sample_20 + '_' + str(
        args.sample_codebleu_budget) + '_' + str(args.sample_num) + '_' + str(args.model_size) + '_' + str(
        args.alpha_ce) + '_' + str(args.beta) + '_' + str(
        args.gamma) + '_' + str(args.samplenuml2)

    if args.ot == 'contrastive':
        common_path_source = common_path + '_' + str(args.alpha_ce) + '_' + str(args.beta) + '_' + str(
            args.gamma) + '_' + str(
            args.samplenuml2)

    model, tokenizer = load_distill_model(args.model, args.victim_model, args.model_size, device, args)

    embedding_layer = model.encoder.get_input_embeddings()

    # examples = load_predict(args.victim_model, args)
    dataset_path = "../../PrimeVul/dataset/all_paired_data_codeline_qwen3-30b.jsonl"

    examples = read_defect_examples(dataset_path, -1, args, ignore_ls_id)

    path = './saved_results/distillation_' + common_path + '_attack_results.json'

    if args.ot == 'contrastive':
        path = './saved_results/contrastive/distillation_' + common_path + '_attack_results.json'

    X_adv_dict = json.load(open(path, 'r'))

    contrastive_data = torch.load('../dataset/' + args.model + '_embeddings_and_mean.pt', map_location="cpu")

    positive_batch_embeds = contrastive_data["positive_batch_embeds"]
    negative_batch_embeds = contrastive_data["negative_batch_embeds"]

    normalized_positive_batch_embeds = []
    normalized_negative_batch_embeds = []

    for item in positive_batch_embeds:
        positive_embed = F.normalize(item, p=2, dim=0)
        normalized_positive_batch_embeds.append(positive_embed)

    for item in negative_batch_embeds:
        negative_embed = F.normalize(item, p=2, dim=0)
        normalized_negative_batch_embeds.append(negative_embed)

    print(len(normalized_positive_batch_embeds), len(normalized_negative_batch_embeds))

    contrastive_data_ori = torch.load('../dataset/' + args.model + '_embeddings_and_mean_ori.pt', map_location="cpu")

    positive_batch_embeds_ori = contrastive_data_ori["positive_batch_embeds"]
    negative_batch_embeds_ori = contrastive_data_ori["negative_batch_embeds"]

    normalized_positive_batch_embeds_ori = []
    normalized_negative_batch_embeds_ori = []

    for item in positive_batch_embeds_ori:
        positive_embed_ori = F.normalize(item, p=2, dim=0)
        normalized_positive_batch_embeds_ori.append(positive_embed_ori)

    for item in negative_batch_embeds_ori:
        negative_embed_ori = F.normalize(item, p=2, dim=0)
        normalized_negative_batch_embeds_ori.append(negative_embed_ori)

    loss_fn = torch.nn.MSELoss(reduction='mean')

    total_X_adv_number = 0
    attack_suc_number = 0
    final_data = {}
    total_target_0_num = 0
    total_target_1_num = 0

    hitratio_ls = []

    vul_sites_num_ls = []
    total_have_sites_num_0 = 0
    total_have_sites_num_1 = 0
    total_have_sites_num_2 = 0
    total_have_sites_num_3 = 0

    ii = 0

    hitratios = []

    for id, example in enumerate(examples):

        # ii += 1
        #
        # if ii == 10:
        #     break

        ground_truth = int(example.target)
        label = int(example.target)
        preds, lm_loss, logit = load_model_predict(args, model, tokenizer, device, example.source, ground_truth)

        print(preds[0] == label)

        # if preds[0] != label:
        #     continue
        if str(example.idx) not in X_adv_dict:
            continue

        if X_adv_dict[str(example.idx)]['state'] == 'failed':
            continue

        print('idx {idx} is test accuracy. Attacking Now.'.format(idx=example.idx))

        codes = {}
        codes['idx'] = example.idx
        codes['target'] = example.target
        codes['non_vul'] = example.no_vul
        codes['cwe'] = example.cwe
        codes['codeline'] = example.codeline
        ori_the_code = example.source
        codes['vul'] = ori_the_code

        ori_sample = copy.deepcopy(ori_the_code)
        embeddings_sample1, embedding1_mean = load_embed(args, ori_sample, tokenizer, embedding_layer, model, device)

        final_x_adv = ori_the_code

        the_code = copy.deepcopy(final_x_adv)
        identifiers, code_tokens = get_identifiers(the_code, "c")
        processed_code = " ".join(code_tokens)
        space_words, sub_words, keys = _tokenize_our_method(processed_code, tokenizer, args.model)

        _, code_tokens_codeline = get_identifiers(example.codeline, "c")
        processed_code_codeline = " ".join(code_tokens_codeline)
        _, sub_words_codeline, _ = _tokenize_our_method(processed_code_codeline, tokenizer, args.model)

        if args.model in ['qwen0.5b', 'qwen1.5b']:
            start, end = find_sub_list(sub_words_codeline[1:], sub_words)
        else:
            start, end = find_sub_list(['Ġ' + sub_words_codeline[1]] + sub_words_codeline[2:], sub_words)

        print('start, end', start, end)

        print(sub_words_codeline[1:])

        print(sub_words, len(sub_words))

        if start == -1 or end == -1:
            continue

        if example.target == 1:
            total_target_1_num += 1
            total_X_adv_number += 1
        else:
            total_target_0_num += 1
            total_X_adv_number += 1

        args.block_size = 512

        if args.model in ['codet5-base', 'codet5-small', 'codet5-base-multi']:
            # max_length = 512
            tokens = sub_words[:511]
            # token_ids = tokenizer.convert_tokens_to_ids(tokens)
            # padding_length = max_length - 1 - len(token_ids)
            # source_ids = token_ids + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * padding_length
            source_tokens = sub_words[:511] + [tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            padding_length = args.block_size - len(source_ids)
            source_ids += [tokenizer.pad_token_id] * padding_length

        elif args.model in ['flan-t5-small']:

            # max_length = 512
            tokens = sub_words[:511]
            # token_ids = tokenizer.convert_tokens_to_ids(tokens)
            # padding_length = max_length - 1 - len(token_ids)
            # source_ids = token_ids + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * padding_length
            source_tokens = sub_words[:511] + [tokenizer.eos_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            padding_length = args.block_size - len(source_ids)
            source_ids += [tokenizer.pad_token_id] * padding_length

        elif args.model in ['codebert', 'roberta-base', 'roberta-large', 'codebert-insecure']:

            tokens = sub_words[:511]
            source_tokens = sub_words[:511] + [tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            padding_length = args.block_size - len(source_ids)
            source_ids += [tokenizer.pad_token_id] * padding_length

        elif args.model in ['codegpt', 'gpt2', 'gpt2-medium']:
            tokens = sub_words[:511]
            source_tokens = sub_words[:511] + ["<|endoftext|>"]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            padding_length = args.block_size - len(source_ids)
            source_ids += [50255] * padding_length
            attention_mask = (source_ids != 0)

        elif args.model in ['qwen0.5b', 'qwen1.5b']:
            tokens = sub_words[:1024]
            source_tokens = sub_words[:1024]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            padding_length = 1024 - len(source_ids)
            source_ids += [151643] * padding_length
            # attention_mask = (source_ids != 151643)

        elif args.model in ['graphcodebert']:
            source_ids, position_idx, attn_mask, combine_code_tokens, len_codetokens = convert_examples_to_features_adv_attack(
                the_code, tokenizer, args)
            tokens = sub_words[:len_codetokens - 1]

        source_ids = torch.tensor([source_ids], dtype=torch.long).to(device)
        labels = torch.tensor([example.target], dtype=torch.long).to(device)

        if args.ot == 'contrastive':

            if args.model not in ['graphcodebert']:
                input_embeds = embedding_layer(source_ids)
                input_embeds = input_embeds.clone().detach().requires_grad_(True)
                # ce_loss, logits = model.get_embeddings(source_ids, input_embeds, labels)
            else:
                inputs_embeds = model.encoder.roberta.embeddings.word_embeddings(source_ids)
                input_embeds = inputs_embeds.clone().detach().requires_grad_(True)
                # ce_loss, logits = model.get_embeddings(input_embeds, position_idx, attn_mask, labels)

            model.zero_grad()

            if args.model not in ['graphcodebert']:
                if args.model in ['qwen0.5b', 'qwen1.5b']:
                    attention_mask = source_ids.ne(151643)
                    embeddings_x_adv = model.get_hidden_states(input_embeds, attention_mask)
                else:
                    embeddings_x_adv = model.get_hidden_states(source_ids, input_embeds)
            else:
                embeddings_x_adv = model.get_hidden_states(inputs_embeds, position_idx,
                                                           attn_mask)

            embeddings_x_adv_mean = torch.mean(embeddings_x_adv, dim=1)
            # print('embeddings_x_adv_mean', embeddings_x_adv_mean)

            embeddings_x_adv_mean = F.normalize(embeddings_x_adv_mean[0], p=2, dim=0)

            # print('embeddings_x_adv_mean', embeddings_x_adv_mean)

            positive_sampled_items = random.sample(normalized_positive_batch_embeds,
                                                   k=round(
                                                       len(normalized_positive_batch_embeds) / args.samplenuml2))
            negative_sampled_items = random.sample(normalized_negative_batch_embeds,
                                                   k=round(
                                                       len(normalized_negative_batch_embeds) / args.samplenuml2))

            positive_multiple_samples = positive_sampled_items + normalized_positive_batch_embeds_ori

            negative_multiple_samples = negative_sampled_items + normalized_negative_batch_embeds_ori

            positive_combined_tensor = torch.stack(positive_multiple_samples)
            # positive_combined_tensor = positive_combined_tensor.unsqueeze(0)
            negative_combined_tensor = torch.stack(negative_multiple_samples)
            # negative_combined_tensor = negative_combined_tensor.unsqueeze(0)

            # print(positive_combined_tensor, positive_combined_tensor.shape)

            if args.dis == 'l2':

                # loss_1 = loss_fn(embeddings_x_adv_mean, positive_mean_embed)
                # loss_2 = loss_fn(embeddings_x_adv_mean, negative_mean_embed)
                loss_1 = l2_distance_loss(embeddings_x_adv_mean, positive_combined_tensor.to(device))
                loss_2 = l2_distance_loss(embeddings_x_adv_mean, negative_combined_tensor.to(device))

            else:
                raise ValueError

            if input_embeds.grad is not None:
                input_embeds.grad.zero_()

            if int(example.target) == 1:
                loss = args.beta * loss_1 - args.gamma * loss_2
            elif int(example.target) == 0:
                loss = args.beta * loss_2 - args.gamma * loss_1
            else:
                raise ValueError

            print('location', 'loss_1.item()', loss_1.item(), 'loss_2.item()',
                  loss_2.item())

            loss.backward()
            input_grads = input_embeds.grad

            row_sums = np.linalg.norm(input_grads[0].cpu().numpy(), ord=2, axis=1)

        elif args.ot == 'ce':

            model.zero_grad()

            if args.model not in ['graphcodebert']:

                if args.model in ['qwen0.5b', 'qwen1.5b']:
                    attention_mask = source_ids.ne(151643)
                    input_embeds = embedding_layer(source_ids)
                    input_embeds = input_embeds.clone().detach().requires_grad_(True)
                    loss, logits = model.get_embeddings(input_embeds, attention_mask, labels)
                else:
                    input_embeds = embedding_layer(source_ids)
                    input_embeds = input_embeds.clone().detach().requires_grad_(True)
                    loss, logits = model.get_embeddings(source_ids, input_embeds, labels)

            else:
                inputs_embeds = model.encoder.roberta.embeddings.word_embeddings(source_ids)
                input_embeds = inputs_embeds.clone().detach().requires_grad_(True)
                loss, logits = model.get_embeddings(input_embeds, position_idx, attn_mask, labels)

            if input_embeds.grad is not None:
                input_embeds.grad.zero_()

            loss.backward()
            input_grads = input_embeds.grad

            row_sums = np.linalg.norm(input_grads[0].cpu().numpy(), ord=2, axis=1)

        else:
            raise ValueError

        _, _, sorted_items = get_top_k(tokens, row_sums, args)

        combined_set = get_AST_set(the_code)
        combined_set.update(['true', 'false'])
        # print('combined_set', combined_set)

        pygments_tokens = extract_dataflow_contain_newline(the_code, 'c')

        # print('pygments_tokens', pygments_tokens, len(pygments_tokens))
        from location_transform_augment_hit_rate import location_transform

        index_top, top_k_token_sites, all_sites = location_transform(
            sorted_items, tokens,
            pygments_tokens,
            space_words,
            keys, combined_set, args)

        print('index_top', index_top)

        print(list(range(start, end + 1)))

        hitratio, common_elements = calculate_unique_ratio(list(map(int, index_top[:args.top_k])),
                                                           list(range(start, end + 1)))

        vul_sites = []
        for common_element in common_elements:

            for item in top_k_token_sites[str(common_element)]:

                if '@' in item:
                    vul_sites.append(item)

        codes['vul_sites_num'] = len(vul_sites)

        if len(vul_sites) == 0:

            total_have_sites_num_0 += 1
            codes['attack'] = 'yes_0'

        else:

            if len(vul_sites) == 1:
                total_have_sites_num_1 += 1
                codes['attack'] = 'yes_1'

            else:
                # codes['attack'] = 'no_1'

                if len(vul_sites) == 2:
                    total_have_sites_num_2 += 1
                    codes['attack'] = 'yes_2'
                else:
                    # codes['attack'] = 'no_2'

                    if len(vul_sites) == 3:
                        total_have_sites_num_3 += 1
                        codes['attack'] = 'yes_3'

                    else:
                        codes['attack'] = 'no'

        vul_sites_num_ls.append(len(vul_sites))

        codes['hitratio'] = hitratio

        hitratio_ls.append(hitratio)

        print('vul_sites', vul_sites, len(vul_sites))

        print('hitratio', hitratio)

        final_data[example.idx] = codes
        print(str(example.idx) + ' completed!')

        if args.ot == 'contrastive':
            with open("./contrastive_hitratio_" + common_path + "_attack_results.json", "w") as json_file:
                json.dump(final_data, json_file)
        elif args.ot == 'ce':
            with open("./ce_hitratio_" + common_path + "_attack_results.json", "w") as json_file:
                json.dump(final_data, json_file)

    final_data['max_hitratio'] = max(hitratio_ls)
    final_data['min_hitratio'] = min(hitratio_ls)
    final_data['mean_hitratio'] = np.mean(hitratio_ls)
    final_data['std'] = np.std(hitratio_ls)
    final_data['medians'] = np.median(hitratio_ls)
    final_data['iqr'] = np.percentile(hitratio_ls, 75) - np.percentile(hitratio_ls, 25)
    final_data['vul_sites_ratio_0'] = total_have_sites_num_0 / total_X_adv_number
    final_data['vul_sites_ratio_1'] = total_have_sites_num_1 / total_X_adv_number
    final_data['vul_sites_ratio_2'] = total_have_sites_num_2 / total_X_adv_number
    final_data['vul_sites_ratio_3'] = total_have_sites_num_3 / total_X_adv_number
    final_data['total_X_adv_number'] = total_X_adv_number
    final_data['total_target_1_num'] = total_target_1_num
    final_data['total_target_0_num'] = total_target_0_num

    print('mean_hitratio', final_data['mean_hitratio'], final_data['vul_sites_ratio_1'],
          final_data['vul_sites_ratio_2'], final_data['vul_sites_ratio_3'])

    if args.ot == 'contrastive':
        with open("./contrastive_hitratio_" + common_path + "_attack_results.json", "w") as json_file:
            json.dump(final_data, json_file)
    elif args.ot == 'ce':
        with open("./ce_hitratio_" + common_path + "_attack_results.json", "w") as json_file:
            json.dump(final_data, json_file)
