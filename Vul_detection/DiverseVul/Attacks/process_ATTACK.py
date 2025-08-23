import random
import string
import time
from codebleu import calc_codebleu


# from process_SITE import *
# from process_STRING import *


def _tokenize_words(word, tokenizer):
    sub_ = tokenizer.encode(word)
    sub = tokenizer.convert_ids_to_tokens(sub_)
    sub = sub[1:-1]
    return sub


def _tokenize_bert(seq, tokenizer):
    seq = ' '.join(seq.split())
    words = seq.split(' ')

    sub_words = []
    keys = []
    index = 0
    for id_, word in enumerate(words):
        if id_ == 0:
            sub_ = tokenizer.encode(' ' + word)
            sub = tokenizer.convert_ids_to_tokens(sub_)
            sub = ['<|endoftext|>'] + sub
        else:
            sub_ = tokenizer.encode(' ' + word)
            sub = tokenizer.convert_ids_to_tokens(sub_)
        sub_words += sub
        keys.append([index, index + len(sub) - 1])
        index += len(sub)

    return words, sub_words, keys, seq


def _tokenize(seq, tokenizer):
    seq = ' '.join(seq.split())
    words = seq.split(' ')

    sub_words = []
    keys = []
    index = 0
    for id_, word in enumerate(words):

        if id_ == 0:
            sub_ = tokenizer.encode(word)
            sub = tokenizer.convert_ids_to_tokens(sub_)
            sub = sub[0:-1]
        else:
            sub_ = tokenizer.encode(' ' + word)
            sub = tokenizer.convert_ids_to_tokens(sub_)
            sub = sub[1:-1]
        sub_words += sub
        keys.append([index, index + len(sub) - 1])
        index += len(sub)

    return words, sub_words, keys, seq


def process_char(char, s, id, ls, ls_2, ls_3):
    if char + s[min(id + 1, len(s) - 1)] in ls_2:
        return " " + char
    elif s[max(0, id - 1)] + char in ls_2:
        if s[max(0, id - 1)] + char + s[min(id + 1, len(s) - 1)] not in ls_3:
            return char + " "
    elif char in ls:
        if char == '.':
            if s[max(0, id - 1)].isdigit() or s[min(id + 1, len(s) - 1)].isdigit():
                return char
            else:
                return " " + char + " "
        else:
            return " " + char + " "
    else:
        return char


def add_spaces_around_parentheses(s):
    result = ""
    for id, char in enumerate(s):
        ls = ['(', ')', '[', ']', '{', '}', '.', '+', '-', '*', '/', '%', ',', '|', '!', '&', '^', '~', '>', '<',
              '=', ';']

        ls_2 = ['>=', '<=', '!=', '==', '&&', '||', '<<', '>>', '++', '--', '+=', '-=', '*=', '/=', '%=', '&=', '|=',
                '^=', '/*', '*/', '//', '->', '?:', ');', '::']

        ls_3 = ['<<=', '>>=']

        if char != '\"' or char != "'":

            temp = s[:id]

            count_1 = temp.count('"')

            if count_1 % 2 == 0:

                result += process_char(char, s, id, ls, ls_2, ls_3)

            else:
                result += char
        else:
            result += char

    return result


def get_tokens(seq):
    ls = [' '.join(line.split()) for line in seq.splitlines()]
    result = []
    for item in ls:
        if item != '':
            result.append(item)
    seq = ' \n '.join(result)
    return seq.split(' ') + ['\n']


def get_codebleu_512(code1, code2, tokenizer, victim_model):
    if victim_model in ['codet5-base', 'codet5-small', 'codebert', 'graphcodebert']:
        words, sub_words, keys, _ = _tokenize(code1, tokenizer)
    else:
        words, sub_words, keys, _ = _tokenize_bert(code1, tokenizer)

    number_words = 0
    for key in keys:
        if key[1] <= 512:
            number_words += 1
        else:
            break

    x1 = ' '.join(words[:min(number_words + 1, len(words))])

    if victim_model in ['codet5-base', 'codet5-small', 'codebert', 'graphcodebert']:
        words, sub_words, keys, _ = _tokenize(code2, tokenizer)
    else:
        words, sub_words, keys, _ = _tokenize_bert(code2, tokenizer)

    number_words = 0
    for key in keys:
        if key[1] <= 512:
            number_words += 1
        else:
            break

    x2 = ' '.join(words[:min(number_words + 1, len(words))])

    codebleu = calc_codebleu([x1], [x2],
                             lang="cpp",
                             weights=(0.25, 0.25, 0.25, 0.25),
                             tokenizer=None)
    return codebleu['codebleu']


def get_random_k_plus(tokens, args):
    token_gradinets = {}

    for id, token in enumerate(tokens):
        token_gradinets[token + '#' + str(id)] = 0.0

    random_seed = time.time()
    random.seed(random_seed)
    items = list(token_gradinets.items())
    random.shuffle(items)

    sorted_dict = dict(items)

    first_five_keys = list(sorted_dict.keys())[:args.top_k]

    # for key in first_five_keys:
    #     print('top_gradient_tokens', key)

    sorted_items = list(sorted_dict.items())

    return token_gradinets, sorted_items


def get_random_k(tokens, row_sums, args):
    token_gradinets = {}

    for id, token in enumerate(tokens):
        temp = row_sums[id].tolist()
        token_gradinets[token + '#' + str(id)] = float(format(temp, ".4f"))

    sorted_dict = dict(sorted(token_gradinets.items(), key=lambda item: item[1]))

    first_five_keys = list(sorted_dict.keys())[:args.top_k]

    # first_five_keys = first_five_keys[:100]

    # random_seed = time.time()
    # random.seed(random_seed)
    # first_five_keys = random.sample(first_five_keys, args.top_k)

    # for key in first_five_keys:
    #     print('top_gradient_tokens', key)

    sorted_items = list(sorted_dict.items())

    return token_gradinets, sorted_dict, sorted_items


def get_top_k(tokens, row_sums, args):
    token_gradinets = {}

    for id, token in enumerate(tokens):
        temp = row_sums[id].tolist()
        token_gradinets[token + '#' + str(id)] = float(format(temp, ".4f"))

    sorted_dict = dict(sorted(token_gradinets.items(), key=lambda item: item[1], reverse=True))

    first_five_keys = list(sorted_dict.keys())[:args.top_k]

    # for key in first_five_keys:
    #     print('top_gradient_tokens', key)

    # all_items = list(token_gradinets.items())
    sorted_items = list(sorted_dict.items())

    # LLM_tokens = []
    # for name, _ in all_items:
    #     key_ls = name.split('#')
    #     LLM_tokens.append(key_ls[0])

    return token_gradinets, sorted_dict, sorted_items


def get_sites(sites, the_code, args, lines_list, variable_name_list, comments_list, tokenizer):
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
    x_adv_dict = {}

    if len(sites) == 0:
        return x_adv_dict

    for site in sites:

        if len(site) == 0:
            continue

        if site[0] == '@':

            # if site[1:-1] in ['true']:
            #
            #     content = f'Replace the Boolean value \"true\" with {str(args.candidate_k)} equivalent C++ code expressions. ' \
            #               f'Each item should be a valid C++ expression that evaluates to true' \
            #               f', using only logical values. ' \
            #               f'Only return a json list.{random_string}'
            #
            #     prompt = content.format(site=site[1:-1])
            #     candidate_site = load_deepseek_coder(prompt)
            #
            # elif site[1:-1] in ['false']:
            #
            #     content = f'Replace the Boolean value \"false\" with {str(args.candidate_k)} equivalent C++ code expressions. ' \
            #               f'Each item should be a valid C++ expression that evaluates to false' \
            #               f', using only logical values. ' \
            #               f'Only return a json list.{random_string}'
            #
            #     prompt = content.format(site=site[1:-1])
            #
            #     candidate_site = load_deepseek_coder(prompt)
            #
            # else:

            random_seed = time.time()
            random.seed(random_seed)
            nbr_words_temp = random.sample(variable_name_list, args.candidate_k)
            candidate_site = []
            for i, nbr_word in enumerate(nbr_words_temp):
                # sub_word = _tokenize_words(nbr_word, tokenizer)
                sub_word = tokenizer.tokenize(nbr_word)
                candidate_site.append(''.join(sub_word[:args.len_word]))

        elif site[0] == '$':

            random_seed = time.time()
            random.seed(random_seed)
            candidate_site = random.sample(lines_list, args.candidate_k)

        elif site[0] == '?':

            random_seed = time.time()
            random.seed(random_seed)
            candidate_site = random.sample(comments_list, args.candidate_k)

        elif site[0] == '#':

            random_seed = time.time()
            random.seed(random_seed)
            candidate_site = random.sample(comments_list, args.candidate_k)

        else:
            raise ValueError

        # print('candidate_site', candidate_site)

        if candidate_site is None:
            candidate_site_ls = [''] * args.candidate_k

        else:

            if site[0] == '@':

                candidate_site_ls = candidate_site
                sorted_strings = sorted(candidate_site_ls, key=len)
                candidate_site_ls = sorted_strings

            elif site[0] == '$':
                candidate_site_ls = []
                for item in candidate_site:
                    candidate_site_ls.append(item.replace('\\n', '\n'))

            elif site[0] == '?':

                candidate_site_ls = []
                for item in candidate_site:
                    candidate_site_ls.append(' ' + item + ' ')

            elif site[0] == '#':

                candidate_site_ls = []
                for item in candidate_site:
                    candidate_site_ls.append(item.replace('/*', ' ').replace('*/', ' '))

            else:
                raise ValueError

        # print('candidate_site_ls', candidate_site_ls)

        x_adv_dict[site] = candidate_site_ls

    return x_adv_dict
