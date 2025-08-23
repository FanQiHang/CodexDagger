import copy

from process_SITE import *
from process_STRING import *
from process_AST import *


def move_dollar_to_front(lst):
    lst.sort(key=lambda x: not ('$' in x))


def location_transform(sorted_items, LLM_tokens, pygments_tokens, pygments_tokens_newline, space_words, keys,
                       combined_set, args):
    '''
    pygments_tokens, sites = location_transform(sorted_items, tokens, pygments_tokens, space_words, keys,
                                                        combined_set, the_code, args)
    '''
    id_insert = 0

    top_k_tokens = []
    top_k_words = []

    for i in range(len(LLM_tokens)):
        if LLM_tokens[i][0] == 'Ġ':

            if LLM_tokens[i] == 'Ġ':
                continue
            LLM_tokens[i] = LLM_tokens[i][1:]

    raw_code_ls = ' '.join(pygments_tokens)

    try:
        statements_ls_0 = get_statements(raw_code_ls)
        # statements_ls_1 = get_statements_declaration(raw_code_ls)
        # statements_ls = statements_ls_0 + statements_ls_1
        statements_ls = statements_ls_0

    except Exception as e:
        print(e)
        statements_ls_0 = []

    # print('statements_ls', statements_ls)

    temp_topk_num = 0
    len_sites = 0

    all_sites = []
    all_dead_code_statements = []

    for name, _ in sorted_items:

        sites = []

        name = name.replace('Ġ', ' ').replace('Ċ', '\n')

        if name[0] == ' ' and name[1] != '#':
            key_ls = name[1:].split('#')
        else:
            key_ls = name.split('#')

        if key_ls[0] in ['<s>', '</s>']:
            continue

        if '<|endoftext|>' in key_ls[0]:
            continue

        if len(key_ls) != 2:
            continue

        for i in range(len(keys)):

            # print(keys[i][0], keys[i][1])

            if keys[i][0] <= int(key_ls[1]) <= keys[i][1]:
                word = space_words[i]
                # print('word', word)

                index = i
                # print('index', index)
                break

        # top_k_words.append([word, index])

        # print('word', word)

        sub_string = ' '.join(space_words[0:index])

        count = sub_string.count(word) + 1

        pygments_count = 0
        index_su = -1

        for id, item in enumerate(pygments_tokens):

            py_count = item.count(word)
            pygments_count += py_count

            if pygments_count >= count:
                index_su = id
                break

        if is_valid_c_variable(word):

            if word in combined_set:  # 包含了'true'和'false'，因为一开始对这两个关键字做了处理

                elements = [element for i, element in enumerate(pygments_tokens) if word == element]

                for element in set(elements):
                    pygments_tokens = replace_elements(pygments_tokens, element, '@' + element + '@')
                    sites.append('@' + element + '@')

        if not_contains_letter_or_digit(word) or is_valid_c_integer(word) or is_valid_c_float(
                word) or word in cpp_keywords:

            if not_contains_letter_or_digit(word):

                sub_string = ' '.join(space_words[0:index])
                count = sub_string.count(word) + 1

                pygments_count = 0
                index_su = -1

                for id, item in enumerate(pygments_tokens):

                    py_count = item.count(word)
                    pygments_count += py_count

                    if pygments_count >= count:
                        index_su = id
                        break

                # if word == ');' or word == '{':
                #
                #     if 'case' not in space_words[index:min(index + 2, len(space_words))]:
                #         # id_insert += 1
                #         # pygments_tokens = insert_after_index(pygments_tokens, index_su, '$' + str(id_insert) + '$')
                #         # sites.append('$' + str(id_insert) + '$')
                #         continue
                #
                #     surrounding_elements = []

                if word == '}' and word != space_words[-1]:

                    if 'break' not in space_words[max(0, index - 2):index]:
                        # id_insert += 1
                        # pygments_tokens = insert_after_index(pygments_tokens, index_su, '$' + str(id_insert) + '$')
                        # sites.append('$' + str(id_insert) + '$')

                        id_insert += 1
                        pygments_tokens = insert_before_index(pygments_tokens, index_su, '$' + str(id_insert) + '$')
                        sites.append('$' + str(id_insert) + '$')

                        # all_dead_code_statements.append('} '+pygments_tokens[index_su + 1:len(pygments_tokens)])

                        continue

                    surrounding_elements = []

                else:

                    n = args.ngram

                    start_index = max(0, index_su - n)
                    end_index = min(len(pygments_tokens), index_su)

                    surrounding_elements = []

                    for i in range(start_index, end_index):
                        surrounding_elements.append((pygments_tokens[i], i))

                    start_index = max(0, index_su)
                    end_index = min(len(pygments_tokens), index_su + n + 1)

                    for i in range(start_index, end_index):
                        surrounding_elements.append((pygments_tokens[i], i))

            elif is_valid_c_integer(word) or is_valid_c_float(word) or word in cpp_keywords:

                occurrences = count_occurrences_up_to_index(space_words, word, index)

                index = find_nth_occurrence(pygments_tokens, word, occurrences)

                surrounding_elements = get_surrounding_elements(pygments_tokens, index, n=args.ngram)

            else:
                surrounding_elements = []

            # print('surrounding_elements', surrounding_elements)

            for i, item in enumerate(surrounding_elements):

                if is_valid_c_variable(item[0]):

                    if item[0] in combined_set:

                        elements = [element for i, element in enumerate(pygments_tokens) if item[0] == element]

                        for element in set(elements):
                            pygments_tokens = replace_elements(pygments_tokens, element, '@' + element + '@')
                            sites.append('@' + element + '@')

        all_sites.extend(sites)

        len_sites_after = len_sites + len(sites)

        if len_sites_after > len_sites:
            len_sites = len_sites_after
            temp_topk_num += 1

            top_k_tokens.append(key_ls)
            # print('key_ls', key_ls[0], key_ls[1])

            if temp_topk_num == int(args.top_k):
                break

    # print('temp_topk_num', temp_topk_num)
    print('all_sites', all_sites)

    ls_dict = {}
    ii = 0
    for idx, item in enumerate(pygments_tokens):
        if item == '\n':
            # print(pygments_tokens[min(idx + 1, len(pygments_tokens) - 1)])
            if len(pygments_tokens[min(idx + 1, len(pygments_tokens) - 1)]) >= 2:
                if pygments_tokens[min(idx + 1, len(pygments_tokens) - 1)][0] == '$' and \
                        pygments_tokens[min(idx + 1, len(pygments_tokens) - 1)][-1] == '$':
                    ls_dict[ii] = pygments_tokens[min(idx + 1, len(pygments_tokens) - 1)]
            ii += 1

    for key, value in ls_dict.items():
        iii = 0
        for idx, item in enumerate(pygments_tokens_newline):
            if item == '\n':
                if iii == key:
                    pygments_tokens_newline.insert(idx + 1, value)
                    # all_dead_code_statements.append(
                    #     ' '.join(pygments_tokens_newline[idx + 2:min(idx + 10, len(pygments_tokens_newline))]))
                    break
                iii += 1

    return pygments_tokens_newline, all_sites, top_k_tokens, top_k_words, all_dead_code_statements
