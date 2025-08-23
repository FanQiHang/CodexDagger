import torch
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}

from process_AST import *
import copy
import argparse
import json
import numpy as np
import torch.nn.functional as F
from shared import WordEmbedding
from shared.utils import LazyLoader
from load_models import load_model
from load_predicts import load_predict
from utils import get_codebleu_512, _tokenize_our_method
import random
import time
from run_parser import get_identifiers, remove_comments_and_docstrings, get_example
from process_datasets import process_dataset
from code_datasets.utils_graphcodebert import convert_examples_to_features_adv_attack
from load_model_predicts import load_model_predict, load_embed
from process_ATTACK import _tokenize_words
from utils import is_valid_variable_name
from load_predicts import load_predict_bert
import math
from nltk.corpus import wordnet
from collections import defaultdict
from attention_models.han import HAN
import os


def words_from_text(s, words_to_ignore=[]):
    homos = set([
        "˗",
        "৭",
        "Ȣ",
        "𝟕",
        "б",
        "Ƽ",
        "Ꮞ",
        "Ʒ",
        "ᒿ",
        "l",
        "O",
        "`",
        "ɑ",
        "Ь",
        "ϲ",
        "ԁ",
        "е",
        "𝚏",
        "ɡ",
        "հ",
        "і",
        "ϳ",
        "𝒌",
        "ⅼ",
        "ｍ",
        "ո",
        "о",
        "р",
        "ԛ",
        "ⲅ",
        "ѕ",
        "𝚝",
        "ս",
        "ѵ",
        "ԝ",
        "×",
        "у",
        "ᴢ",
    ])
    """Lowercases a string, removes all non-alphanumeric characters, and splits
    into words."""
    # TODO implement w regex
    words = []
    word = ""
    for c in " ".join(s.split()):
        if c.isalnum() or c in homos:
            word += c
        elif c in "'-_*@" and len(word) > 0:
            # Allow apostrophes, hyphens, underscores, asterisks and at signs as long as they don't begin the
            # word.
            word += c
        elif word:
            if word not in words_to_ignore:
                words.append(word)
            word = ""
    if len(word) and (word not in words_to_ignore):
        words.append(word)
    return words


def is_one_word(word):
    return len(words_from_text(word)) == 1


def hash_func(inp_vector, projections):
    bools = (np.dot(inp_vector, projections.T) > 0).astype('int')
    return ''.join(bools.astype('str'))


class Table:

    def __init__(self, hash_size, dim):
        self.table = defaultdict(list)
        self.hash_size = hash_size
        self.projections = np.random.randn(self.hash_size, dim)

    def add(self, vecs, label):
        h = hash_func(vecs, self.projections)
        self.table[h].append(label)


class LSH:

    def __init__(self, dim):
        self.num_tables = 5
        self.hash_size = 3
        self.tables = list()
        for i in range(self.num_tables):
            self.tables.append(Table(self.hash_size, dim))

    def add(self, vecs, label):
        for table in self.tables:
            table.add(vecs, label)

    def describe(self):
        for table in self.tables:
            print(len(table.table))
            print(table.table)

    def get_result(self):
        len_tables = []
        indices_to_query = []
        final_set_indices = []
        max_value = -1
        for table in self.tables:
            if len(table.table) > max_value:
                max_value = len(table.table)
                final_table = table
        return final_table


def stop_word_set():
    stop_word_set = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                     'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                     'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around',
                     'as', 'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside',
                     'besides', 'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn',
                     "couldn't", 'd', 'didn', "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during',
                     'either', 'else', 'elsewhere', 'empty', 'enough', 'even', 'ever', 'everyone', 'everything',
                     'everywhere', 'except', 'first', 'for', 'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn',
                     "hasn't", 'haven', "haven't", 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein',
                     'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'if',
                     'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's", 'its', 'itself', 'just', 'latter',
                     'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn', "mightn't", 'mine', 'more',
                     'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself', 'namely', 'needn',
                     "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none', 'noone', 'nor',
                     'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only', 'onto', 'or',
                     'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per', 'please', 's',
                     'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                     'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their',
                     'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore',
                     'therein', 'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus',
                     'to', 'too', 'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was',
                     'wasn', "wasn't", 'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence',
                     'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever',
                     'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with',
                     'within', 'without', 'won', "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd",
                     "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
    return stop_word_set


def text_window_around_index(index, window_size, words, text_words, text):
    """The text window of ``window_size`` words centered around
    ``index``."""
    # return self.text
    word = words[index]
    for index, token in enumerate(text_words):
        if word in token: break
    length = len(text_words)
    half_size = (window_size - 1) / 2.0
    if index - half_size < 0:
        start = 0
        end = min(window_size - 1, length - 1)
    elif index + half_size >= length:
        start = max(0, length - window_size)
        end = length - 1
    else:
        start = index - math.ceil(half_size)
        end = index + math.floor(half_size)
    text_idx_start = _text_index_of_word_index(start, text_words, text)
    text_idx_end = _text_index_of_word_index(end, text_words, text) + len(text_words[end])
    return text[text_idx_start:text_idx_end]


def _text_index_of_word_index(i, text_words, text):
    """Returns the index of word ``i`` in self.text."""
    pre_words = text_words[: i + 1]
    lower_text = text.lower()
    # Find all words until `i` in string.
    look_after_index = 0
    for word in pre_words:
        look_after_index = lower_text.find(word.lower(), look_after_index) + len(
            word
        )
    look_after_index -= len(text_words[i])
    return look_after_index


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Code_attack")
    parser.add_argument("--model", type=str, default="codet5-base", help="[]", )
    parser.add_argument("--codebleu_budget", type=float, default=0.8, help="", )
    parser.add_argument("--candidate_k", type=int, default=50, help="[]", )
    parser.add_argument("--ot", type=str, default='random', help="[random,gradient]", )
    parser.add_argument("--model_type", default="codet5", type=str, choices=['roberta', 'bart', 'codet5'])
    parser.add_argument("--max_source_length", default=512, type=int, help="")
    parser.add_argument("--trials", default=1, type=int, help="")
    parser.add_argument("--data_flow_length", default=114, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--code_length", default=400, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--len_word", default=10, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--dead_code", default='ours', type=str,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--complex", default=0, type=int,
                        help="Optional Code input sequence length after tokenization.")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    common_path = args.model + '_' + args.ot + '_' + str(args.candidate_k) + '_' + str(
        args.codebleu_budget) + '_' + str(args.trials) + '_' + str(args.complex)

    model, tokenizer = load_model(args.model, device, args)
    embedding_layer = model.encoder.get_input_embeddings()

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

    print('len(lines_list)', len(lines_list))

    file_path = './candidates/variable_name_hub_large.txt'
    variable_name_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            variable_name_list.append(line.strip())

    print('len(variable_name_list)', len(variable_name_list))

    examples = load_predict(args.model, args)

    final_data = {}
    total_X_adv_number = 0
    attack_suc_number = 0
    # total_query_num = 0
    total_target_0_num = 0
    target_0_successful_num = 0
    target_0_failed_num = 0
    total_target_1_num = 0
    target_1_successful_num = 0
    target_1_failed_num = 0
    total_target_1_num_vanish = 0
    # preds_ls = []
    # labels_ls = []
    # logits_ls = []

    codebleu_ls = []
    query_num_ls = []
    iii = 0

    if args.model == 'graphcodebert':
        '''仅攻击200个样本'''
        idx_0, idx_1 = load_predict_bert('graphcodebert', args)
        idx_attack = idx_0[:100] + idx_1[:100]

    if args.ot == 'lsh':
        # import tensorflow_text as text
        # hub = LazyLoader("tensorflow_hub", globals(), "tensorflow_hub")
        # tensorflow_text = LazyLoader(
        #     "tensorflow_text", globals(), "tensorflow_text"
        # )  # noqa: F401

        han = HAN(path='./attention_models/yelp/han_model_yelp')
        # import tensorflow as tf
        # import tensorflow_hub as hub
        # import tensorflow_text as text
        #
        # with tf.device('/CPU:0'):
        #     model_LazyLoader = hub.load('./universal-sentence-encoder')
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

    for id, example in enumerate(examples):

        ground_truth = int(example.target)
        label = int(example.target)

        preds, lm_loss, logit = load_model_predict(args, model, tokenizer, device, example.source, ground_truth)

        print('idx {idx} is test accuracy. Attacking Now.'.format(idx=example.idx))

        if args.model == 'graphcodebert':
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
        codes['final_score'] = [1 - logit.cpu().tolist()[0][label]]
        final_score = 1 - logit.cpu().tolist()[0][label]
        codes['logits_0'] = logit.cpu().tolist()

        ori_the_code = example.source

        codes['ori_input'] = ori_the_code
        ori_sample = copy.deepcopy(ori_the_code)

        embeddings_sample1, embedding1_mean = load_embed(args, ori_sample, tokenizer, embedding_layer, model, device)

        codes['input_0'] = ori_the_code

        insert_statements_ls_0 = t_insert_statements(ori_the_code)
        insert_statements_ls = set(insert_statements_ls_0)

        ori_the_code_copy = copy.deepcopy(ori_the_code)
        strings_ls = []
        for idx, item in enumerate(insert_statements_ls):

            if len(item.split()) > 2:

                # print('item.split()', item.split())

                if args.dead_code == 'iclr':
                    candicate_code = 'printf( " string' + str(idx) + ' " );\n if (false) { int variable_name' + str(
                        idx) + '= 1 ; }\n '
                    strings_ls.append('string' + str(idx))

                elif args.dead_code == 'ours':

                    if item[:5] != 'case ' and item[:7] != 'default ' and item[:7] != 'default:':
                        random_seed = time.time()
                        random.seed(random_seed)
                        candidate_site = random.sample(lines_list, 1)
                        candicate_code = candidate_site[0].replace('\\n', '\n')
                    else:
                        candicate_code = ''
                else:
                    raise ValueError

            else:
                strings_ls = []
                candicate_code = ''

            ori_the_code_copy = ori_the_code_copy.replace(item, candicate_code + item)

        the_code = copy.deepcopy(ori_the_code_copy).replace(' true ', '" true " == " true "').replace(' false ',
                                                                                                      '" false " != " false "')

        print('the_code', the_code)

        codebleu = get_codebleu_512(codes['input_0'], the_code, tokenizer, args.model)

        if codebleu < args.codebleu_budget:

            final_x_adv = copy.deepcopy(ori_the_code)

            codes['flag'] = 'start_budget_failed'
            query_number += 1

            final_x_adv = ' '.join(final_x_adv.split())

            preds, lm_loss, logits = load_model_predict(args, model, tokenizer, device, final_x_adv, ground_truth)

            attack_result = label == preds

            codes['func'] = ori_the_code
            codes['logits_final'] = logits.cpu().tolist()
            codes['ground_truth'] = example.target
            codes['total_X_adv_number'] = total_X_adv_number
            codes['sites'] = []
            codes['combination_sites'] = []
            codes['codebleu'] = get_codebleu_512(codes['input_0'], ori_the_code, tokenizer, args.model)
            codes['query_nums'] = query_number

            # preds_ls.extend(preds)
            # logits_ls.append(logits)
            # total_query_num += query_number

            if attack_result:
                codebleu_ls.append(codes['codebleu'])
                query_num_ls.append(codes['query_nums'])
                codes['state'] = 'failed'
            else:
                codebleu_ls.append(codes['codebleu'])
                query_num_ls.append(codes['query_nums'])
                codes['state'] = 'successful'
                attack_suc_number += 1
                codes['attack_suc_number'] = attack_suc_number
                if example.target == 0:
                    target_0_successful_num += 1
                else:
                    target_1_successful_num += 1

                print(str(example.idx) + ' successful!')

            final_data[example.idx] = codes
            print(str(example.idx) + ' completed!')
            with open("./saved_results/baseline_" + common_path + "_attack_results.json", "w") as json_file:
                json.dump(final_data, json_file)

            with open("./saved_results/baseline_" + common_path + "_temp_attack_results.json", "a+",
                      encoding="utf-8") as file:
                json_line = json.dumps(codes, ensure_ascii=False)
                file.write(json_line + "\n")

            continue

        identifiers, code_tokens = get_identifiers(copy.deepcopy(the_code), "c")
        processed_code = " ".join(code_tokens)

        if args.model in ['qwen0.5b', 'qwen1.5b']:

            space_words, sub_words, keys = _tokenize_our_method(processed_code, tokenizer, args.model)
            sub_words = sub_words[:1024]

            number_words = 0
            for key in keys:
                if key[1] < 1025:
                    number_words += 1
                else:
                    break

        else:
            space_words, sub_words, keys = _tokenize_our_method(processed_code, tokenizer, args.model)
            sub_words = sub_words[:512 - 1]

            number_words = 0
            for key in keys:
                if key[1] < 512:
                    number_words += 1
                else:
                    break

        x1 = ' '.join(space_words[:min(number_words + 1, len(space_words))])

        variable_list = list(get_AST_set(x1))
        if 'true' in x1 and 'false' in x1:
            variable_list = list(get_AST_set(x1)) + ['true', 'false']
        elif 'true' in x1:
            variable_list = list(get_AST_set(x1)) + ['true']
        elif 'false' in x1:
            variable_list = list(get_AST_set(x1)) + ['false']

        for item in strings_ls:
            if item in x1:
                variable_list = variable_list + [item]
                break

        # print('variable_list', variable_list)

        if len(variable_list) == 0 or (len(set(variable_list)) == 1 and variable_list[0] == ''):

            codes['flag'] = 'non-sites'
            final_x_adv = copy.deepcopy(the_code)
            query_number += 1

            final_x_adv = ' '.join(final_x_adv.split())

            preds, lm_loss, logits = load_model_predict(args, model, tokenizer, device, final_x_adv, ground_truth)

            attack_result = label == preds

            codes['func'] = the_code
            codes['logits_final'] = logits.cpu().tolist()
            codes['ground_truth'] = example.target
            codes['total_X_adv_number'] = total_X_adv_number
            codes['sites'] = []
            codes['combination_sites'] = []
            codes['codebleu'] = get_codebleu_512(codes['input_0'], the_code, tokenizer, args.model)
            codes['query_nums'] = query_number

            # total_query_num += query_number
            if attack_result:
                codebleu_ls.append(codes['codebleu'])
                query_num_ls.append(codes['query_nums'])
                codes['state'] = 'failed'
            else:
                codebleu_ls.append(codes['codebleu'])
                query_num_ls.append(codes['query_nums'])
                codes['state'] = 'successful'
                attack_suc_number += 1
                codes['attack_suc_number'] = attack_suc_number
                if example.target == 0:
                    target_0_successful_num += 1
                else:
                    target_1_successful_num += 1

                print(str(example.idx) + ' successful!')

            final_data[example.idx] = codes
            print(str(example.idx) + ' completed!')
            with open("./saved_results/baseline_" + common_path + "_attack_results.json", "w") as json_file:
                json.dump(final_data, json_file)

            with open("./saved_results/baseline_" + common_path + "_temp_attack_results.json", "a+",
                      encoding="utf-8") as file:
                json_line = json.dumps(codes, ensure_ascii=False)
                file.write(json_line + "\n")

            continue

        if args.ot == 'gradient':
            args.block_size = 512

            # space_words, sub_words, keys, seq = _tokenize(the_code, tokenizer)
            if args.model in ['codet5-base', 'codet5-small', 'codet5-base-multi']:
                # max_length = 512
                # tokens = sub_words[:max_length - 1]
                # token_ids = tokenizer.convert_tokens_to_ids(tokens)
                # padding_length = max_length - 1 - len(token_ids)
                # source_ids = token_ids + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * padding_length
                source_tokens = sub_words + [tokenizer.sep_token]
                source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
                padding_length = args.block_size - len(source_ids)
                source_ids += [tokenizer.pad_token_id] * padding_length

            elif args.model in ['flan-t5-small']:

                source_tokens = sub_words + [tokenizer.eos_token]
                source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
                padding_length = args.block_size - len(source_ids)
                source_ids += [tokenizer.pad_token_id] * padding_length

            elif args.model in ['codebert', 'roberta-base', 'roberta-large', 'codebert-insecure']:
                args.block_size = 512
                source_tokens = sub_words + [tokenizer.sep_token]
                source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
                padding_length = args.block_size - len(source_ids)
                source_ids += [tokenizer.pad_token_id] * padding_length

            elif args.model in ['codegpt', 'gpt2', 'gpt2-medium']:
                source_tokens = sub_words + ["<|endoftext|>"]
                source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
                padding_length = args.block_size - len(source_ids)
                source_ids += [50255] * padding_length
                attention_mask = (source_ids != 0)

            elif args.model in ['graphcodebert']:
                source_ids, position_idx, attn_mask, combine_code_tokens, len_codetokens = convert_examples_to_features_adv_attack(
                    the_code, tokenizer, args)

            source_ids = torch.tensor([source_ids], dtype=torch.long).to(device)
            labels = torch.tensor([example.target], dtype=torch.long).to(device)
            query_number += 1
            model.zero_grad()
            if args.model not in ['graphcodebert']:
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

            row_sums = np.linalg.norm(input_grads[0].cpu().numpy(), ord=1, axis=1)

            number_words = 0
            for key in keys:
                if key[1] < 512:
                    number_words += 1
                else:
                    break

            words_gradients = {}
            for variable in variable_list:
                sum_gradients = 0
                indexes = [index for index, value in enumerate(space_words[:number_words]) if value == variable]
                for index in indexes:
                    start = keys[index][0]
                    end = keys[index][1]
                    if args.model == 'graphcodebert':
                        if start >= len_codetokens:
                            break
                    for i in range(start, end + 1):
                        sum_gradients += row_sums[i]
                words_gradients[variable] = sum_gradients

            sorted_dict = dict(sorted(words_gradients.items(), key=lambda item: item[1], reverse=True))

            print('words_gradients', sorted_dict)

        elif args.ot == 'random':

            labels = torch.tensor([example.target], dtype=torch.long).to(device)

            random.seed(int(time.time()))
            random.shuffle(variable_list)
            sorted_dict = {k: 0 for k in variable_list}

        elif args.ot == 'tf':

            labels = torch.tensor([example.target], dtype=torch.long).to(device)

            importance_dict = {}
            for textfooler_item in variable_list:
                textfooler_code = copy.deepcopy(the_code)

                textfooler_code_ = get_example(textfooler_code, textfooler_item, '', 'c')
                print(textfooler_code_)
                batch = ' '.join(textfooler_code_.split())
                query_number += 1
                textfooler_preds, textfooler_lm_loss, textfooler_code_logits = load_model_predict(args, model,
                                                                                                  tokenizer, device,
                                                                                                  batch,
                                                                                                  ground_truth)

                importance_dict[textfooler_item] = 1 - logit.cpu().tolist()[0][ground_truth]

            sorted_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

        elif args.ot == 'lsh':

            labels = torch.tensor([example.target], dtype=torch.long).to(device)

            lsh_code = copy.deepcopy(the_code)

            len_text = len(variable_list)

            # Load Hierarchical Attention Network (HAN) for classification task

            doc, score, word_alpha, sentence_alphas = han.classify(" ".join(variable_list))
            scrs = []
            stop_words = stop_word_set()
            word_alpha = word_alpha.detach().cpu().numpy()
            for i in range(len(word_alpha)):
                for j in range(len(word_alpha[i])):
                    if doc[i][j] not in stop_words and len(doc[i][j]) > 2:
                        scrs.append(word_alpha[i][j])
                    else:
                        scrs.append(-101.0)
            for i in range(len(scrs), len(variable_list)):
                scrs.append(-101.0)

            scrs = np.asarray(scrs)
            index_scores = scrs
            search_over = False
            saliency_scores = np.array([result for result in scrs])
            from torch.nn.functional import softmax

            softmax_saliency_scores = softmax(
                torch.Tensor(saliency_scores), dim=0
            ).numpy()
            # Scores due to attention model
            # index_scores = softmax_saliency_scores

            delta_ps = []
            # LSH step
            # Substitute each word with all candidates from the search space

            for idx in range(len_text):
                # transformed_text_candidates = self.get_transformations(
                #     initial_text,
                #     original_text=initial_text,
                #     indices_to_modify=[idx],
                # )
                synonyms = set()
                for syn in wordnet.synsets(variable_list[idx], lang='eng'):
                    for syn_word in syn.lemma_names(lang='eng'):
                        if (
                                (syn_word != variable_list[idx])
                                and ("_" not in syn_word)
                                and (is_one_word(syn_word))
                        ):
                            # WordNet can suggest phrases that are joined by '_' but we ignore phrases.
                            synonyms.add(syn_word)

                transformed_word_candidates = list(synonyms)

                if len(transformed_word_candidates) == 0:
                    delta_ps.append(0.0)
                    continue

                transformed_text_candidates = []
                for transformed_word_candidate in transformed_word_candidates:
                    lsh_code = copy.deepcopy(the_code)
                    lsh_code_ = get_example(lsh_code, variable_list[idx], transformed_word_candidate, 'c')
                    transformed_text_candidates.append(lsh_code_)

                text_to_encode = []
                idx_to_txt = {}
                k = 0
                for itxt, txt in enumerate(transformed_text_candidates):
                    idx_to_txt[str(k)] = txt

                    res = text_window_around_index(itxt, 10, transformed_word_candidates, txt.split(), txt)
                    text_to_encode.append(res)
                    k += 1

                embeddings = embedder.encode(text_to_encode)

                lsh = LSH(768)

                for t in range(len(embeddings)):
                    lsh.add(embeddings[t], str(t))

                table = lsh.get_result()
                transformed_text_candidates = []

                for key, value in table.table.items():
                    val = random.choice(value)
                    transformed_text_candidates.append(idx_to_txt[val])

                print('len(set(transformed_text_candidates)),', len(set(transformed_text_candidates)))

                # The final candidate is the one which causes the maximum change in
                # target model's confidence score
                # swap_results, _ = self.get_goal_results(transformed_text_candidates)
                score_change = []

                for transformed_text_candidate in list(set(transformed_text_candidates)):
                    query_number += 1
                    lsh_preds, lsh_lm_loss, lsh_logits = load_model_predict(args, model,
                                                                            tokenizer, device,
                                                                            transformed_text_candidate,
                                                                            ground_truth)
                    score_change.append(1 - logit.cpu().tolist()[0][ground_truth])

                max_score_change = np.max(score_change)
                delta_ps.append(max_score_change)

            index_scores = (softmax_saliency_scores) * (delta_ps)

            importance_dict = dict(zip(variable_list, index_scores))
            sorted_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

            print('sorted_dict', sorted_dict)

        else:
            raise ValueError

        combination_sites = []
        sites = []
        final_round_flag = False
        is_early_stop = False

        final_x_adv = the_code

        for key, value in sorted_dict.items():

            for ix in range(args.trials):

                attacked_code = copy.deepcopy(final_x_adv)

                sites.append(key)

                if args.dead_code == 'iclr':
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

                elif args.dead_code == 'ours':

                    random_seed = time.time()
                    random.seed(random_seed)
                    nbr_words_temp = random.sample(variable_name_list, args.candidate_k)
                    nbr_words = []
                    for i, nbr_word in enumerate(nbr_words_temp):
                        sub_word = tokenizer.tokenize(nbr_word)
                        nbr_words.append(''.join(sub_word[:args.len_word]))

                else:
                    raise ValueError

                candidate_texts = []
                for combination in nbr_words:
                    content_adv = copy.deepcopy(final_x_adv)
                    # content_adv = content_adv.replace(' ' + key + ' ', ' ' + combination + ' ')
                    #
                    content_adv = get_example(content_adv, key, combination, 'c')

                    candidate_texts.append([content_adv, combination])

                # print('content_adv', content_adv)

                ii = 0
                outputs = []
                while ii < len(candidate_texts):
                    query_number += 1
                    batch = copy.deepcopy(candidate_texts[ii])[0]

                    batch = ' '.join(batch.split())
                    # source_ids = tokenizer.encode(batch, max_length=args.max_source_length, padding='max_length',
                    #                               truncation=True)
                    # source_ids = torch.tensor([source_ids], dtype=torch.long).to(device)
                    preds, lm_loss, batch_logit = load_model_predict(args, model, tokenizer, device, batch,
                                                                     ground_truth)
                    # input_embeds = embedding_layer(source_ids)

                    # with torch.no_grad():
                    #     batch_preds = model(source_ids)
                    outputs.append(batch_logit)

                    ii += 1

                results = {}

                outputs = torch.cat(outputs, dim=0)

                # scores = torch.nn.functional.softmax(outputs, dim=1).cpu()
                scores = outputs.cpu()
                for attacked_text, raw_output in zip(candidate_texts, scores):
                    results[attacked_text[0]] = [1 - raw_output[labels.cpu()].item()] + [attacked_text[1]]

                sorted_dict = dict(sorted(results.items(), key=lambda item: item[1][0], reverse=True))

                first_key = list(sorted_dict.keys())[0]
                # for it in list(sorted_dict.keys()):
                #     if sorted_dict[it][1] not in the_code:
                #         first_key = it
                #         break

                if sorted_dict[first_key][0] > final_score:
                    final_score = sorted_dict[first_key][0]
                    final_x_adv = first_key
                    combination_sites.append(sorted_dict[first_key][1])
                else:
                    # final_x_adv = the_code
                    final_x_adv = attacked_code
                    combination_sites.append('this final_score <= last final_score')

                # print('combination_sites', combination_sites)
                # print('sites', sites)

                codes['final_score'] = codes['final_score'] + [final_score]

                with torch.no_grad():

                    final_x_adv_copy = copy.deepcopy(final_x_adv)
                    # print('final_x_adv_copy', final_x_adv_copy)
                    final_x_adv_copy = ' '.join(final_x_adv_copy.split())

                    embeddings_sample2, embedding2_mean = load_embed(args, final_x_adv_copy, tokenizer, embedding_layer,
                                                                     model, device)
                    '''
                    cos and l2
                    '''
                    # dis_cos = F.cosine_similarity(embedding1_mean, embedding2_mean).mean()
                    dis_cos = F.cosine_similarity(embedding1_mean, embedding2_mean, dim=1)
                    codes['final_embeddings_cos_dis'] = codes['final_embeddings_cos_dis'] + [dis_cos.item()]
                    # print('final_embeddings_cos_dis', dis_cos.item())

                    dis_l2 = torch.norm(embeddings_sample1 - embeddings_sample2, p=2)
                    codes['final_embeddings_l2_dis'] = codes['final_embeddings_l2_dis'] + [dis_l2.item()]
                    # print('final_embeddings_l2_dis', dis_l2.item())

                codebleu = get_codebleu_512(codes['input_0'], final_x_adv, tokenizer, args.model)
                codes['budget_codebleu'] = codebleu

                if codebleu > args.codebleu_budget:

                    the_code = final_x_adv

                    query_number += 1

                    final_x_adv_copy = copy.deepcopy(the_code)

                    preds, lm_loss, logits = load_model_predict(args, model, tokenizer, device, final_x_adv_copy,
                                                                ground_truth)

                    attack_result = label == preds

                    codes['final_embeddings_cos_dis'] = codes['final_embeddings_cos_dis'][0:-1]
                    codes['final_embeddings_l2_dis'] = codes['final_embeddings_l2_dis'][0:-1]
                    codes['func'] = the_code
                    codes['logits_final'] = logits.cpu().tolist()
                    codes['ground_truth'] = example.target
                    codes['total_X_adv_number'] = total_X_adv_number
                    codes['sites'] = sites
                    codes['combination_sites'] = combination_sites
                    codes['codebleu'] = get_codebleu_512(codes['input_0'], the_code, tokenizer, args.model)
                    codes['query_nums'] = query_number

                    if attack_result:
                        codes['state'] = 'failed'
                    else:
                        codebleu_ls.append(codes['codebleu'])
                        query_num_ls.append(codes['query_nums'])
                        codes['state'] = 'successful'
                        attack_suc_number += 1
                        codes['attack_suc_number'] = attack_suc_number
                        # total_query_num += query_number
                        if example.target == 0:
                            target_0_successful_num += 1
                        else:
                            target_1_successful_num += 1

                        print(str(example.idx) + ' successful!')

                        is_early_stop = True
                        break

                else:
                    codes['flag'] = 'budget_failed'

                    final_round_flag = True

                    query_number += 1

                    final_x_adv_copy = copy.deepcopy(attacked_code)

                    preds, lm_loss, logits = load_model_predict(args, model, tokenizer, device, final_x_adv_copy,
                                                                ground_truth)

                    attack_result = label == preds

                    codes['final_embeddings_cos_dis'] = codes['final_embeddings_cos_dis'][0:-1]
                    codes['final_embeddings_l2_dis'] = codes['final_embeddings_l2_dis'][0:-1]
                    codes['func'] = attacked_code
                    codes['logits_final'] = logits.cpu().tolist()
                    codes['ground_truth'] = example.target
                    codes['total_X_adv_number'] = total_X_adv_number
                    codes['sites'] = sites
                    codes['combination_sites'] = combination_sites
                    codes['codebleu'] = get_codebleu_512(codes['input_0'], attacked_code, tokenizer, args.model)
                    codes['query_nums'] = query_number

                    if attack_result:
                        codebleu_ls.append(codes['codebleu'])
                        query_num_ls.append(codes['query_nums'])
                        codes['state'] = 'failed'
                    else:
                        codebleu_ls.append(codes['codebleu'])
                        query_num_ls.append(codes['query_nums'])
                        codes['state'] = 'successful'
                        attack_suc_number += 1
                        codes['attack_suc_number'] = attack_suc_number
                        if example.target == 0:
                            target_0_successful_num += 1
                        else:
                            target_1_successful_num += 1

                        print(str(example.idx) + ' successful!')

                    is_early_stop = True
                    break

            if is_early_stop:
                break

            if key == list(sorted_dict.keys())[-1]:
                query_num_ls.append(codes['query_nums'])
                codebleu_ls.append(codes['codebleu'])

        print(codes['func'])
        final_data[example.idx] = codes
        print(str(example.idx) + ' completed!')

        with open("./saved_results/baseline_" + common_path + "_attack_results.json", "w") as json_file:
            json.dump(final_data, json_file)

        with open("./saved_results/baseline_" + common_path + "_temp_attack_results.json", "a+",
                  encoding="utf-8") as file:
            json_line = json.dumps(codes, ensure_ascii=False)
            file.write(json_line + "\n")

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
    final_data['codebleu_mean'] = sum(codebleu_ls) / len(codebleu_ls)
    final_data['codebleu_min'] = min(codebleu_ls)
    final_data['query_num_mean'] = sum(query_num_ls) / len(query_num_ls)
    final_data['query_num_max'] = max(query_num_ls)

    eval_dataset = process_dataset(args.model, tokenizer, args,
                                   "./saved_results/baseline_" + common_path + "_temp_attack_results.json")
    pred_prob, pred_label, true_label = model.compute(eval_dataset, 10)

    final_data['total_X_adv_number_aux'] = len(eval_dataset)

    from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
    import numpy as np

    eval_acc = np.mean(true_label == pred_label)

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
    file = "./saved_results/roc_npz/baseline_" + common_path + "_roc_data.npz"
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
    file = "./saved_results/roc_npz/baseline_" + common_path + "_roc_data_0.npz"
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

    with open("./saved_results/baseline_" + common_path + "_attack_results.json", "w") as json_file:
        json.dump(final_data, json_file)

    import os

    os.remove("./saved_results/baseline_" + common_path + "_temp_attack_results.json")
