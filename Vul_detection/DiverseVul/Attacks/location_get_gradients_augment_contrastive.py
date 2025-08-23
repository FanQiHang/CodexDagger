import torch
from process_AST import *
from process_ATTACK import *
import copy
import argparse
import numpy as np
import torch.nn.functional as F
import json
from load_models import load_distill_model
from load_predicts import load_predict
from utils import get_codebleu_512, _tokenize_our_method
import random
from run_parser import get_identifiers, remove_comments_and_docstrings, get_example, extract_dataflow_contain_newline
from code_datasets.utils_graphcodebert import convert_examples_to_features_adv_attack
from load_model_predicts import load_model_predict, load_embed
from shared import WordEmbedding
from utils import is_valid_variable_name
from load_predicts import load_predict_bert


def count_newlines_before_element(lst, target):
    count = 0
    for element in lst:
        if element == target:
            return count
        elif element == '\n':
            count += 1
    return -1


def l2_distance_loss(sample, multiple_samples):
    if len(sample.shape) == 1:
        sample = sample.unsqueeze(0)
    distances = torch.norm(sample - multiple_samples, p=2, dim=1)
    # print(distances, len(distances))
    loss = distances.mean()
    return loss


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

    parser.add_argument("--is_sample_20", default='yes', type=str, help="")

    parser.add_argument("--sample_codebleu_budget", default=0.4, type=float, help="")

    parser.add_argument("--sample_num", type=int, default=20, help="", )

    parser.add_argument("--model_size", type=str, default='large', help="", )

    parser.add_argument("--alpha_ce", type=float, default=0.0, help="", )
    parser.add_argument("--beta", type=float, default=1.0, help="", )
    parser.add_argument("--gamma", type=float, default=1.0, help="", )

    parser.add_argument("--samplenuml2", type=int, default=2, help="", )

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.device = device

    if args.strategy == 'name_only':
        from location_transform_augment_variable_only import location_transform
    else:
        from location_transform_augment import location_transform

    if args.valid_or_test == 'test':
        common_path = args.model + '_' + args.victim_model + '_' + str(args.round) + '_' + str(args.top_k) + '_' + str(
            args.candidate_k) + '_' + str(args.random_k) + '_' + str(
            args.ngram) + '_' + args.ot + '_' + args.dis + '_' + args.decision + '_' + str(
            args.complex) + '_' + args.strategy + '_' + str(args.codebleu_budget) + '_' + str(
            args.temperature) + '_' + str(
            args.alpha) + '_' + str(args.loss_name) + '_' + args.is_sample_20 + '_' + str(
            args.sample_codebleu_budget) + '_' + str(args.sample_num) + '_' + str(args.model_size)

    if args.ot == 'contrastive':
        common_path = common_path + '_' + str(
            args.alpha_ce) + '_' + str(args.beta) + '_' + str(
            args.gamma) + '_' + str(args.samplenuml2)

    model, tokenizer = load_distill_model(args.model, args.victim_model, args.model_size, device, args)

    embedding_layer = model.encoder.get_input_embeddings()

    examples = load_predict(args.victim_model, args)

    if args.complex == 0:
        file_path = './candidates/dead_code_hub_large.txt'
    elif args.complex == 1:
        file_path = './candidates/redundant_code.txt'
    elif args.complex == 2:
        file_path = './candidates/redundant_code_short.txt'
    elif args.complex == 3:
        file_path = './candidates/redundant_code_augment.txt'
    else:
        raise ValueError
    lines_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            lines_list.append(line.strip())

    print('len(lines_list)', len(lines_list))  # 382

    file_path = './candidates/variable_name_hub_large.txt'
    variable_name_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            variable_name_list.append(line.strip())

    print('len(variable_name_list)', len(variable_name_list))

    comments_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            comments_list.append(line.strip())

    total_X_adv_number = 0
    attack_suc_number = 0
    # total_query_num = 0
    final_data = {}
    temp_top_k = args.top_k

    total_target_0_num = 0
    target_0_successful_num = 0
    target_0_failed_num = 0

    total_target_1_num = 0
    target_1_successful_num = 0
    target_1_failed_num = 0

    total_target_1_num_vanish = 0

    codebleu_ls_final = []
    query_ls_final = []
    # round_score_final = []

    if args.model == 'graphcodebert' and args.victim_model == 'codebert':
        idx_0, idx_1 = load_predict_bert('codebert', args)
        idx_attack = idx_0[:100] + idx_1[:100]

    contrastive_data = torch.load('../dataset/' + args.model + '_embeddings_and_mean.pt', map_location="cpu")
    # positive_mean_embed = contrastive_data["positive_mean_embed"].to(device)
    # negative_mean_embed = contrastive_data["negative_mean_embed"].to(device)

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
    # positive_mean_embed = contrastive_data["positive_mean_embed"].to(device)
    # negative_mean_embed = contrastive_data["negative_mean_embed"].to(device)

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

    for id, example in enumerate(examples):

        ground_truth = int(example.target)
        label = int(example.target)
        preds, lm_loss, logit = load_model_predict(args, model, tokenizer, device, example.source, ground_truth)

        print('idx {idx} is test accuracy. Attacking Now.'.format(idx=example.idx))

        if args.model == 'graphcodebert' and args.victim_model == 'codebert':
            if int(example.idx) not in idx_attack:
                continue

        query_number = 1

        if example.target == 1:
            total_target_1_num += 1
            total_X_adv_number += 1
        else:
            total_target_0_num += 1
            total_X_adv_number += 1

        codes = {}
        codes['idx'] = example.idx
        codes['target'] = example.target
        codes['final_embeddings_cos_dis'] = [1.0]
        codes['final_embeddings_l2_dis'] = [0.0]
        if args.decision == 'logit':
            codes['final_score'] = [1 - logit.cpu().tolist()[0][label]]
            final_score = 1 - logit.cpu().tolist()[0][label]
            print('final_score', final_score)
        else:
            codes['final_score'] = [-np.inf]
            final_score = -np.inf

        codes['round_score'] = []

        codes['logits_0'] = logit.cpu().tolist()

        ori_the_code = example.source

        print(ori_the_code)

        codes['ori_input'] = ori_the_code

        query_number += 1
        ori_sample = copy.deepcopy(ori_the_code)

        embeddings_sample1, embedding1_mean = load_embed(args, ori_sample, tokenizer, embedding_layer, model, device)

        ori_the_code = copy.deepcopy(ori_the_code).replace(' true ', '" true " == " true "').replace(' false ',
                                                                                                     '" false " != " false "')
        # codes['input_0'] = ori_the_code
        final_x_adv = ori_the_code
        is_early_stop = False

        for r in range(args.round):

            print('r', r)
            # the_code = codes['input_' + str(r)]
            # attacked_code = copy.deepcopy(final_x_adv)
            the_code = copy.deepcopy(final_x_adv)
            identifiers, code_tokens = get_identifiers(the_code, "c")
            processed_code = " ".join(code_tokens)

            space_words, sub_words, keys = _tokenize_our_method(processed_code, tokenizer, args.model)

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
            # print('source_ids', source_ids)
            labels = torch.tensor([example.target], dtype=torch.long).to(device)

            if args.ot == 'contrastive':

                # if args.model not in ['graphcodebert']:
                #     input_embeds = embedding_layer(source_ids)
                #     input_embeds = input_embeds.clone().detach().requires_grad_(True)
                #     # ce_loss, logits = model.get_embeddings(source_ids, input_embeds, labels)
                # else:
                #     inputs_embeds = model.encoder.roberta.embeddings.word_embeddings(source_ids)
                #     input_embeds = inputs_embeds.clone().detach().requires_grad_(True)
                #     # ce_loss, logits = model.get_embeddings(input_embeds, position_idx, attn_mask, labels)

                if args.model not in ['graphcodebert']:

                    if args.model in ['qwen0.5b', 'qwen1.5b']:
                        attention_mask = source_ids.ne(151643)
                        input_embeds = embedding_layer(source_ids)
                        input_embeds = input_embeds.clone().detach().requires_grad_(True)
                        ce_loss, logits = model.get_embeddings(input_embeds, attention_mask, labels)
                    else:
                        input_embeds = embedding_layer(source_ids)
                        input_embeds = input_embeds.clone().detach().requires_grad_(True)
                        ce_loss, logits = model.get_embeddings(source_ids, input_embeds, labels)

                else:
                    inputs_embeds = model.encoder.roberta.embeddings.word_embeddings(source_ids)
                    input_embeds = inputs_embeds.clone().detach().requires_grad_(True)
                    ce_loss, logits = model.get_embeddings(input_embeds, position_idx, attn_mask, labels)

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
                # embeddings_x_adv_mean = F.normalize(embeddings_x_adv[0], p=2, dim=0)

                # print('embeddings_x_adv_mean', embeddings_x_adv_mean)

                # 在这里进行采样
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
                    loss = args.alpha_ce * ce_loss + args.beta * loss_1 - args.gamma * loss_2
                    # loss = args.beta * loss_1 - args.gamma * loss_2
                elif int(example.target) == 0:
                    loss = args.alpha_ce * ce_loss + args.beta * loss_2 - args.gamma * loss_1
                    # loss = args.beta * loss_2 - args.gamma * loss_1
                else:
                    raise ValueError

                print('location', loss.item(), 'loss_1.item()', loss_1.item(), 'loss_2.item()',
                      loss_2.item())

                loss.backward()
                input_grads = input_embeds.grad

                row_sums = np.linalg.norm(input_grads[0].cpu().numpy(), ord=2, axis=1)
                query_number += 1

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
                query_number += 1

            elif args.ot == 'ce_contrastive':

                if r < 5:

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
                    query_number += 1

                else:

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
                    # embeddings_x_adv_mean = F.normalize(embeddings_x_adv[0], p=2, dim=0)

                    # print('embeddings_x_adv_mean', embeddings_x_adv_mean)

                    # 在这里进行采样
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
                    query_number += 1

            else:
                raise ValueError

            if args.ot == 'random':
                _, _, sorted_items = get_random_k(tokens, row_sums, args)
            elif args.ot == 'random+':
                _, sorted_items = get_random_k_plus(tokens, args)
            else:
                _, _, sorted_items = get_top_k(tokens, row_sums, args)

            combined_set = get_AST_set(the_code)
            combined_set.update(['true', 'false'])
            # print('combined_set', combined_set)

            pygments_tokens = extract_dataflow_contain_newline(the_code, 'c')
            pygments_tokens_newline = get_tokens(the_code)

            # print('pygments_tokens', pygments_tokens, len(pygments_tokens))
            pygments_tokens, sites, top_k_tokens, top_k_words, all_dead_code_statements = location_transform(
                sorted_items, tokens,
                pygments_tokens,
                pygments_tokens_newline,
                space_words,
                keys, combined_set, args)

            args.top_k = temp_top_k

            if args.strategy in ['name_code', 'name_only']:
                pass
            elif args.strategy == 'name':
                sites = [item for item in sites if not (item.startswith('$') and item.endswith('$'))]
                pygments_tokens = [item for item in pygments_tokens if
                                   not (item.startswith('$') and item.endswith('$'))]
            elif args.strategy == 'code':
                sites = [item for item in sites if not (item.startswith('@') and item.endswith('@'))]
                temp_ls = []
                for item in pygments_tokens:
                    if not (item.startswith('@') and item.endswith('@')):
                        temp_ls.append(item)
                    else:
                        temp_ls.append(item[1:-1])
                pygments_tokens = temp_ls
            else:
                raise ValueError

            codes['site_' + str(r)] = sites
            concatenated_string = ' '.join(pygments_tokens)
            # print('concatenated_string', concatenated_string)

            if len(sites) == 0 or (len(set(sites)) == 1 and sites[0] == ''):

                codes['flag'] = 'non-sites'

                final_x_adv = copy.deepcopy(the_code)
                query_number += 1

                preds, lm_loss, logits = load_model_predict(args, model, tokenizer, device, final_x_adv, ground_truth)

                attack_result = label == preds

                codes['logits_' + str(r + 1)] = logits.cpu().tolist()
                codes['func'] = the_code
                # codes['input_' + str(r + 1)] = the_code
                # codes['input_' + str(args.round)] = the_code
                codes['ground_truth'] = example.target
                codes['round'] = r + 1
                codes['total_X_adv_number'] = total_X_adv_number
                codes['combination_sites_' + str(r)] = []
                codes['codebleu'] = get_codebleu_512(codes['ori_input'], the_code, tokenizer, args.model)

                codes['query_nums'] = query_number
                # total_query_num += query_number
                if attack_result:
                    # codebleu_ls.append(codes['codebleu'])
                    codes['state'] = 'failed'
                else:
                    # codebleu_ls.append(codes['codebleu'])
                    codes['state'] = 'successful'
                    attack_suc_number += 1
                    codes['attack_suc_number'] = attack_suc_number
                    if example.target == 0:
                        target_0_successful_num += 1
                    else:
                        target_1_successful_num += 1

                    print(str(example.idx) + ' successful!')

                break

            else:

                for sitex in sites:

                    attacked_code = copy.deepcopy(final_x_adv)

                    # concatenated_string

                    # content_adv = copy.deepcopy(final_x_adv)

                    x_adv_dict = get_sites([sitex], None, args, lines_list, variable_name_list,
                                           comments_list,
                                           tokenizer)

                    # print('x_adv_dict', x_adv_dict)

                    candidate_texts = []

                    if sitex[0] == '@':

                        if args.dead_code == 'ours':

                            for valuex in x_adv_dict[sitex]:
                                content_adv = copy.deepcopy(final_x_adv)

                                temp_token = valuex.strip('\'')

                                content_adv = get_example(content_adv, sitex[1:-1], temp_token, "c")

                                candidate_texts.append([content_adv, valuex])

                        elif args.dead_code == 'iclr':

                            embedding = WordEmbedding.counterfitted_GLOVE_embedding()

                            candidate_words = []
                            vocab_size = embedding.nn_matrix.shape[0]

                            random.seed(int(time.time()))
                            nnids = [random.randint(0, vocab_size - 1) for _ in range(args.candidate_k)]  # 50个

                            nbr_words = []
                            for i, nbr_id in enumerate(nnids):
                                nbr_word = embedding.index2word(nbr_id)
                                # sub_word_ = _tokenize_words(nbr_word, tokenizer)
                                sub_word_ = tokenizer.tokenize(nbr_word)
                                if is_valid_variable_name(''.join(sub_word_[:args.len_word]), 'c'):
                                    nbr_words.append(''.join(sub_word_[:args.len_word]))

                            for nbr_word in nbr_words:
                                content_adv = copy.deepcopy(final_x_adv)

                                content_adv = get_example(content_adv, sitex[1:-1], nbr_word, "c")

                                candidate_texts.append([content_adv, nbr_word])

                    elif sitex[0] == '$':

                        pygments_tokens = extract_dataflow_contain_newline(concatenated_string, 'c')

                        # print('pygments_tokens', pygments_tokens)
                        result_count = count_newlines_before_element(pygments_tokens, sitex)

                        # add_dollar_ls = extract_dataflow_contain_newline(content_adv, 'c')
                        add_dollar_ls = []
                        temp_final_x_adv = copy.deepcopy(final_x_adv)
                        for x in temp_final_x_adv.split('\n'):
                            add_dollar_ls.append(x)
                            add_dollar_ls.append('\n')

                        count_n = 0
                        for idx, add_dollar in enumerate(add_dollar_ls):
                            if add_dollar == '\n':
                                count_n += 1
                                if count_n == result_count:
                                    add_dollar_ls.insert(idx + 1, sitex)
                                    break

                        temp_final_x_adv = ' '.join(add_dollar_ls)

                        if args.dead_code == 'ours':
                            for valuex in x_adv_dict[sitex]:
                                content_adv = copy.deepcopy(temp_final_x_adv)

                                temp_token = valuex.strip('\'')
                                content_adv = content_adv.replace(sitex, temp_token)
                                # for temp_site in temp_sites:
                                #     content_adv = content_adv.replace(temp_site, '')
                                candidate_texts.append([content_adv, valuex])

                        elif args.dead_code == 'iclr':

                            content_adv = copy.deepcopy(temp_final_x_adv)

                            candicate_dead_code = 'printf( " string' + str(
                                sitex[1:-1]) + ' " );\n if (" false " != " false " ) { int variable_name' + str(
                                sitex[1:-1]) + '= 1 ; }\n '

                            content_adv = content_adv.replace(sitex, candicate_dead_code)

                            candidate_texts.append([content_adv, candicate_dead_code])

                    results = {}
                    if args.decision == 'contrastive':

                        ii = 0
                        outputs = []
                        while ii < len(candidate_texts):
                            query_number += 1
                            batch = copy.deepcopy(candidate_texts[ii])[0]

                            preds, ce_loss, batch_logit = load_model_predict(args, model, tokenizer, device, batch,
                                                                             ground_truth)

                            _, embeddings_x_adv_mean = load_embed(args, batch, tokenizer,
                                                                  embedding_layer,
                                                                  model, device)

                            embeddings_x_adv_mean = F.normalize(embeddings_x_adv_mean[0], p=2, dim=0)

                            # print('embeddings_x_adv_mean',embeddings_x_adv_mean)

                            # if args.dis == 'l2':
                            #     loss_1 = loss_fn(embeddings_x_adv_mean, positive_mean_embed)
                            #     loss_2 = loss_fn(embeddings_x_adv_mean, negative_mean_embed)
                            # else:
                            #     raise ValueError

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
                                batch_loss = args.alpha_ce * ce_loss + args.beta * loss_1 - args.gamma * loss_2
                                # batch_loss = args.beta * loss_1 - args.gamma * loss_2
                            elif int(example.target) == 0:
                                batch_loss = args.alpha_ce * ce_loss + args.beta * loss_2 - args.gamma * loss_1
                                # batch_loss = args.beta * loss_2 - args.gamma * loss_1
                            else:
                                raise ValueError

                            print('batch_loss.item()', batch_loss.item(), 'ce_loss.item()', ce_loss.item(),
                                  'loss_1.item()', loss_1.item(), 'loss_2.item()', loss_2.item())

                            outputs.append(batch_loss.item())

                            ii += 1

                        # print('outputs', outputs)
                        # outputs = torch.cat(outputs, dim=0)
                        scores = outputs
                        for attacked_text, raw_output in zip(candidate_texts, scores):
                            results[attacked_text[0]] = [raw_output] + [attacked_text[1]]

                    elif args.decision == 'logit':

                        ii = 0
                        outputs = []
                        while ii < len(candidate_texts):
                            query_number += 1
                            batch = copy.deepcopy(candidate_texts[ii])[0]
                            preds, lm_loss, batch_logit = load_model_predict(args, model, tokenizer, device, batch,
                                                                             ground_truth)
                            outputs.append(batch_logit)

                            ii += 1

                        outputs = torch.cat(outputs, dim=0)

                        scores = outputs.cpu()
                        for attacked_text, raw_output in zip(candidate_texts, scores):
                            results[attacked_text[0]] = [1 - raw_output[labels.cpu()].item()] + [attacked_text[1]]

                    elif args.decision == 'ce_contrastive':

                        if r < 5:

                            ii = 0
                            outputs = []
                            while ii < len(candidate_texts):
                                query_number += 1
                                batch = copy.deepcopy(candidate_texts[ii])[0]
                                preds, lm_loss, batch_logit = load_model_predict(args, model, tokenizer, device, batch,
                                                                                 ground_truth)
                                outputs.append(batch_logit)

                                ii += 1

                            outputs = torch.cat(outputs, dim=0)
                            scores = outputs.cpu()
                            for attacked_text, raw_output in zip(candidate_texts, scores):
                                results[attacked_text[0]] = [1 - raw_output[labels.cpu()].item()] + [attacked_text[1]]

                        else:
                            ii = 0
                            outputs = []
                            while ii < len(candidate_texts):
                                query_number += 1
                                batch = copy.deepcopy(candidate_texts[ii])[0]
                                preds, ce_loss, batch_logit = load_model_predict(args, model, tokenizer, device, batch,
                                                                                 ground_truth)

                                _, embeddings_x_adv_mean = load_embed(args, batch, tokenizer,
                                                                      embedding_layer,
                                                                      model, device)

                                embeddings_x_adv_mean = F.normalize(embeddings_x_adv_mean[0], p=2, dim=0)

                                # print('embeddings_x_adv_mean',embeddings_x_adv_mean)

                                # if args.dis == 'l2':
                                #     loss_1 = loss_fn(embeddings_x_adv_mean, positive_mean_embed)
                                #     loss_2 = loss_fn(embeddings_x_adv_mean, negative_mean_embed)
                                # else:
                                #     raise ValueError

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
                                    loss_1 = l2_distance_loss(embeddings_x_adv_mean,
                                                              positive_combined_tensor.to(device))
                                    loss_2 = l2_distance_loss(embeddings_x_adv_mean,
                                                              negative_combined_tensor.to(device))

                                else:
                                    raise ValueError

                                if input_embeds.grad is not None:
                                    input_embeds.grad.zero_()

                                if int(example.target) == 1:
                                    batch_loss = args.beta * loss_1 - args.gamma * loss_2
                                elif int(example.target) == 0:
                                    batch_loss = args.beta * loss_2 - args.gamma * loss_1
                                else:
                                    raise ValueError

                                print('batch_loss.item()', batch_loss.item(), 'loss_1.item()', loss_1.item(),
                                      'loss_2.item()', loss_2.item())
                                outputs.append(batch_loss.item())

                                ii += 1

                            # print('outputs', outputs)
                            # outputs = torch.cat(outputs, dim=0)
                            scores = outputs
                            for attacked_text, raw_output in zip(candidate_texts, scores):
                                results[attacked_text[0]] = [raw_output] + [attacked_text[1]]

                    else:
                        raise ValueError

                    sorted_dict = dict(sorted(results.items(), key=lambda item: item[1][0], reverse=True))
                    first_key = list(sorted_dict.keys())[0]

                    if args.decision in ['logit', 'contrastive', 'ce_contrastive']:

                        if sorted_dict[first_key][0] > final_score:

                            final_score = sorted_dict[first_key][0]
                            final_x_adv = first_key
                            combination_site = sorted_dict[first_key][1:]
                            if sitex[0] == '$':
                                concatenated_string = concatenated_string.replace(sitex,
                                                                                  sorted_dict[first_key][
                                                                                      1])
                        else:

                            final_x_adv = attacked_code
                            combination_site = [
                                ['this final_score <= last final_score'] + sorted_dict[first_key][1:]]
                            if sitex[0] == '$':
                                concatenated_string = concatenated_string.replace(sitex,
                                                                                  '')

                    print('final_score', final_score)

                    codes['final_score'] = codes['final_score'] + [final_score]
                    codes['combination_site_' + str(r)] = combination_site

                    with torch.no_grad():
                        query_number += 1
                        final_x_adv_copy = copy.deepcopy(final_x_adv)

                        embeddings_sample2, embedding2_mean = load_embed(args, final_x_adv_copy, tokenizer,
                                                                         embedding_layer,
                                                                         model, device)

                        dis_cos = F.cosine_similarity(embedding1_mean, embedding2_mean).mean()
                        codes['final_embeddings_cos_dis'] = codes['final_embeddings_cos_dis'] + [dis_cos.item()]
                        # print('final_embeddings_cos_dis', dis_cos.item())

                        dis_l2 = torch.norm(embeddings_sample1 - embeddings_sample2, p=2)
                        codes['final_embeddings_l2_dis'] = codes['final_embeddings_l2_dis'] + [dis_l2.item()]
                        # print('final_embeddings_l2_dis', dis_l2.item())

                    codebleu = get_codebleu_512(codes['ori_input'], final_x_adv, tokenizer, args.model)

                    print(r, codebleu)

                    codes['budget_codebleu'] = codebleu

                    if codebleu > args.codebleu_budget:

                        codes['flag'] = 'normal'

                        query_number += 1
                        final_x_adv_copy = copy.deepcopy(final_x_adv)
                        preds, lm_loss, logits = load_model_predict(args, model, tokenizer, device,
                                                                    final_x_adv_copy,
                                                                    ground_truth)

                        attack_result = label == preds

                        codes['func'] = final_x_adv
                        codes['logits_' + str(r + 1)] = logits.cpu().tolist()
                        # codes['top_k_tokens_' + str(r)] = top_k_tokens
                        # codes['top_k_words_' + str(r)] = top_k_words
                        codes['ground_truth'] = example.target
                        codes['round'] = r + 1
                        codes['total_X_adv_number'] = total_X_adv_number
                        codes['codebleu'] = get_codebleu_512(codes['ori_input'], final_x_adv, tokenizer, args.model)
                        # codebleu_ls.append(codes['codebleu'])
                        codes['query_nums'] = query_number
                        # total_query_num += query_number
                        if attack_result:
                            if r == int(args.round - 1):
                                codes['state'] = 'failed'
                        else:
                            codes['state'] = 'successful'
                            attack_suc_number += 1
                            codes['attack_suc_number'] = attack_suc_number
                            if example.target == 0:
                                target_0_successful_num += 1
                            else:
                                target_1_successful_num += 1
                            is_early_stop = True
                            break

                    else:

                        codes['flag'] = 'budget_failed'

                        query_number += 1

                        final_x_adv_copy = copy.deepcopy(attacked_code)

                        preds, lm_loss, logits = load_model_predict(args, model, tokenizer, device,
                                                                    final_x_adv_copy,
                                                                    ground_truth)

                        attack_result = label == preds

                        codes['final_embeddings_cos_dis'] = codes['final_embeddings_cos_dis'][0:-1]
                        codes['final_embeddings_l2_dis'] = codes['final_embeddings_l2_dis'][0:-1]

                        codes['func'] = attacked_code
                        codes['logits_' + str(r + 1)] = logits.cpu().tolist()
                        # codes['input_' + str(r + 1)] = the_code
                        # codes['input_' + str(args.round)] = the_code
                        codes['ground_truth'] = example.target
                        codes['round'] = r + 1
                        codes['total_X_adv_number'] = total_X_adv_number
                        codes['codebleu'] = get_codebleu_512(codes['ori_input'], attacked_code, tokenizer, args.model)
                        # codebleu_ls.append(codes['codebleu'])
                        codes['query_nums'] = query_number
                        # total_query_num += query_number
                        if attack_result:
                            codes['state'] = 'failed'
                        else:
                            codes['state'] = 'successful'
                            attack_suc_number += 1
                            codes['attack_suc_number'] = attack_suc_number
                            if example.target == 0:
                                target_0_successful_num += 1
                            else:
                                target_1_successful_num += 1

                        is_early_stop = True
                        break

                codes['round_score'] = codes['round_score'] + [codes['final_score'][-1]]

                if is_early_stop:
                    break

        # print(codes['func'])
        print(codes['state'])
        print(codes['codebleu'])

        codebleu_ls_final.append(codes['codebleu'])
        query_ls_final.append(codes['query_nums'])
        # round_score_final.append(codes['round_score'])

        final_data[example.idx] = codes
        print(str(example.idx) + ' completed!')
        if args.ot == 'contrastive':

            with open("./saved_results/contrastive/distillation_" + common_path + "_attack_results.json",
                      "w") as json_file:
                json.dump(final_data, json_file)
        elif args.ot == 'ce':

            with open("./saved_results/distillation_" + common_path + "_attack_results.json", "w") as json_file:
                json.dump(final_data, json_file)

    final_data['total_X_adv_number'] = total_X_adv_number
    final_data['attack_suc_number'] = attack_suc_number
    final_data['ASR'] = attack_suc_number / total_X_adv_number
    final_data['Robustness'] = 1 - attack_suc_number / total_X_adv_number
    final_data['target_0_successful_num'] = target_0_successful_num
    final_data['target_1_successful_num'] = target_1_successful_num
    final_data['total_target_1_num'] = total_target_1_num
    final_data['total_target_0_num'] = total_target_0_num
    final_data['total_target_1_num_vanish'] = total_target_1_num_vanish
    final_data['vanish_rate'] = total_target_1_num_vanish / total_target_1_num

    final_data['0_successful'] = target_0_successful_num / total_target_0_num
    final_data['1_successful'] = target_1_successful_num / total_target_1_num
    final_data['0_failed'] = 1 - target_0_successful_num / total_target_0_num
    final_data['1_failed'] = 1 - target_1_successful_num / total_target_1_num

    if args.ot == 'contrastive':
        with open("./saved_results/contrastive/distillation_" + common_path + "_attack_results.json", "w") as json_file:
            json.dump(final_data, json_file)
    elif args.ot == 'ce':

        with open("./saved_results/distillation_" + common_path + "_attack_results.json", "w") as json_file:
            json.dump(final_data, json_file)
