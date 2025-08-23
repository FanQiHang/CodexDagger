import copy
import random
from utils import select_parents, crossover, map_chromesome, mutate, is_valid_variable_name, _tokenize, \
    get_identifier_posistions_from_code, get_masked_code_by_position, get_substitues, is_valid_substitue, \
    get_codebleu_512

# from utils import GraphCodeDataset, isUID
from run_parser import get_identifiers, get_example
from run_parser import get_identifiers, extract_dataflow
from code_datasets.utils_dataset import CodeDataset, CodeDataset_CodeT5, GraphCodeDataset, CodeDataset_Qwen


def compute_fitness(chromesome, codebert_tgt, tokenizer_tgt, orig_prob, orig_label, true_label, code,
                    names_positions_dict, args):
    if args.model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
        from code_datasets.utils_codet5 import convert_defect_examples_to_features_adv as convert_code_to_features
    elif args.model in ['gpt2', 'gpt2-medium', 'codegpt']:
        from code_datasets.utils_codegpt import convert_examples_to_features_codegpt_adv as convert_code_to_features
    elif args.model in ['qwen0.5b', 'qwen1.5b']:
        from code_datasets.utils_qwen import convert_examples_to_features_qwen_adv as convert_code_to_features
    elif args.model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:
        from code_datasets.utils_codebert import convert_examples_to_features_codebert_adv as convert_code_to_features
    elif args.model in ['graphcodebert']:
        from code_datasets.utils_graphcodebert import convert_examples_to_features_adv as convert_code_to_features
    else:
        raise ValueError

    temp_code = map_chromesome(chromesome, code, "c")
    new_feature = convert_code_to_features(temp_code, tokenizer_tgt, true_label, args)

    if args.model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
        new_dataset = CodeDataset_CodeT5([new_feature])
    elif args.model in ['gpt2', 'gpt2-medium', 'codegpt', 'codebert', 'roberta-large', 'roberta-base',
                        'codebert-insecure']:
        new_dataset = CodeDataset([new_feature])
    elif args.model in ['qwen0.5b', 'qwen1.5b']:
        new_dataset = CodeDataset_Qwen([new_feature])
    elif args.model in ['graphcodebert']:
        new_dataset = GraphCodeDataset([new_feature], args)
    else:
        raise ValueError

    new_logits, preds = codebert_tgt.get_results(new_dataset, args.eval_batch_size)

    fitness_value = orig_prob - new_logits[0][orig_label]
    return fitness_value, preds[0]


def get_importance_score(args, example, code, words_list: list, sub_words: list, variable_names: list, tgt_model,
                         tokenizer, label_list, batch_size=16, max_length=512, model_type='classification'):
    '''Compute the importance score of each variable'''
    if args.model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
        from code_datasets.utils_codet5 import convert_defect_examples_to_features_adv as convert_code_to_features
    elif args.model in ['gpt2', 'gpt2-medium', 'codegpt']:
        from code_datasets.utils_codegpt import convert_examples_to_features_codegpt_adv as convert_code_to_features
    elif args.model in ['qwen0.5b', 'qwen1.5b']:
        from code_datasets.utils_qwen import convert_examples_to_features_qwen_adv as convert_code_to_features
    elif args.model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:
        from code_datasets.utils_codebert import convert_examples_to_features_codebert_adv as convert_code_to_features
    elif args.model in ['graphcodebert']:
        from code_datasets.utils_graphcodebert import convert_examples_to_features_adv as convert_code_to_features
    else:
        raise ValueError

    positions = get_identifier_posistions_from_code(words_list, variable_names)

    if len(positions) == 0:
        return None, None, None

    new_example = []
    masked_token_list, replace_token_positions = get_masked_code_by_position(words_list, positions)

    for index, tokens in enumerate([words_list] + masked_token_list):
        new_code = ' '.join(tokens)
        if args.model in ['qwen0.5b', 'qwen1.5b']:
            new_feature = convert_code_to_features(new_code, tokenizer, example[2].item(), args)
        else:
            new_feature = convert_code_to_features(new_code, tokenizer, example[1].item(), args)
        new_example.append(new_feature)

    if args.model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
        new_dataset = CodeDataset_CodeT5(new_example)
    elif args.model in ['gpt2', 'gpt2-medium', 'codegpt', 'codebert', 'roberta-large', 'roberta-base',
                        'codebert-insecure']:
        new_dataset = CodeDataset(new_example)
    elif args.model in ['qwen0.5b', 'qwen1.5b']:
        new_dataset = CodeDataset_Qwen(new_example)
    elif args.model in ['graphcodebert']:
        new_dataset = GraphCodeDataset(new_example, args)
    else:
        raise ValueError

    logits, preds = tgt_model.get_results(new_dataset, args.eval_batch_size)
    orig_probs = logits[0]
    orig_label = preds[0]
    orig_prob = max(orig_probs)

    importance_score = []
    for prob in logits[1:]:
        importance_score.append(orig_prob - prob[orig_label])

    return importance_score, replace_token_positions, positions


class Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, model_mlm, tokenizer_mlm, use_bpe, threshold_pred_score) -> None:
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.model_mlm = model_mlm
        self.tokenizer_mlm = tokenizer_mlm
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score

    def ga_attack(self, example, code, substituions, initial_replace=None):
        '''
        return
            original program: code
            program length: prog_length
            adversar program: adv_program
            true label: true_label
            original prediction: orig_label
            adversarial prediction: temp_label
            is_attack_success: is_success
            extracted variables: variable_names
            importance score of variables: names_to_importance_score
            number of changed variables: nb_changed_var
            number of changed positions: nb_changed_pos
            substitues for variables: replaced_words
        '''

        if self.args.model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
            from code_datasets.utils_codet5 import convert_defect_examples_to_features_adv as convert_code_to_features
        elif self.args.model in ['gpt2', 'gpt2-medium', 'codegpt']:
            from code_datasets.utils_codegpt import convert_examples_to_features_codegpt_adv as convert_code_to_features
        elif self.args.model in ['qwen0.5b', 'qwen1.5b']:
            from code_datasets.utils_qwen import convert_examples_to_features_qwen_adv as convert_code_to_features
        elif self.args.model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:
            from code_datasets.utils_codebert import \
                convert_examples_to_features_codebert_adv as convert_code_to_features
        elif self.args.model in ['graphcodebert']:
            from code_datasets.utils_graphcodebert import convert_examples_to_features_adv as convert_code_to_features
        else:
            raise ValueError

        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        true_label = example[1].item()

        # adv_code = ''
        adv_code = copy.deepcopy(code)
        temp_label = None

        identifiers, code_tokens = get_identifiers(code, 'c')
        prog_length = len(code_tokens)

        processed_code = " ".join(code_tokens)

        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)

        variable_names = list(substituions.keys())

        if not orig_label == true_label:
            is_success = -4
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        if len(variable_names) == 0:
            is_success = -3
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        names_positions_dict = get_identifier_posistions_from_code(words, variable_names)

        nb_changed_var = 0
        nb_changed_pos = 0
        is_success = -1

        variable_substitue_dict = {}

        for tgt_word in names_positions_dict.keys():
            variable_substitue_dict[tgt_word] = substituions[tgt_word]

        if len(variable_substitue_dict) == 0:
            is_success = -3
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        fitness_values = []

        base_chromesome = {word: word for word in variable_substitue_dict.keys()}

        population = [base_chromesome]
        for tgt_word in variable_substitue_dict.keys():

            if initial_replace is None:
                replace_examples = []
                substitute_list = []
                temp_replace = copy.deepcopy(words)
                current_prob = max(orig_prob)
                most_gap = 0.0
                initial_candidate = tgt_word
                tgt_positions = names_positions_dict[tgt_word]

                for a_substitue in variable_substitue_dict[tgt_word]:
                    substitute_list.append(a_substitue)
                    temp_code = get_example(code, tgt_word, a_substitue, "c")

                    new_feature = convert_code_to_features(temp_code, self.tokenizer_tgt, example[1].item(), self.args)

                    replace_examples.append(new_feature)

                if len(replace_examples) == 0:
                    continue

                if self.args.model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
                    new_dataset = CodeDataset_CodeT5(replace_examples)
                elif self.args.model in ['gpt2', 'gpt2-medium', 'codegpt', 'codebert', 'roberta-large', 'roberta-base',
                                         'codebert-insecure']:
                    new_dataset = CodeDataset(replace_examples)
                elif self.args.model in ['qwen0.5b', 'qwen1.5b']:
                    new_dataset = CodeDataset_Qwen(replace_examples)
                elif self.args.model in ['graphcodebert']:
                    new_dataset = GraphCodeDataset(replace_examples, self.args)
                else:
                    raise ValueError

                logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)

                _the_best_candidate = -1
                for index, temp_prob in enumerate(logits):
                    temp_label = preds[index]
                    gap = current_prob - temp_prob[temp_label]
                    if gap > most_gap:
                        most_gap = gap
                        _the_best_candidate = index
                if _the_best_candidate == -1:
                    initial_candidate = tgt_word
                else:
                    initial_candidate = substitute_list[_the_best_candidate]
            else:
                initial_candidate = initial_replace[tgt_word]

            temp_chromesome = copy.deepcopy(base_chromesome)
            temp_chromesome[tgt_word] = initial_candidate
            population.append(temp_chromesome)
            temp_fitness, temp_label = compute_fitness(temp_chromesome, self.model_tgt, self.tokenizer_tgt,
                                                       max(orig_prob), orig_label, true_label, code,
                                                       names_positions_dict, self.args)
            fitness_values.append(temp_fitness)

        cross_probability = 0.7

        max_iter = max(5 * len(population), 10)

        for i in range(max_iter):
            _temp_mutants = []
            for j in range(64):
                p = random.random()
                chromesome_1, index_1, chromesome_2, index_2 = select_parents(population)
                if p < cross_probability:
                    if chromesome_1 == chromesome_2:
                        child_1 = mutate(chromesome_1, variable_substitue_dict)
                        continue
                    child_1, child_2 = crossover(chromesome_1, chromesome_2)
                    if child_1 == chromesome_1 or child_1 == chromesome_2:
                        child_1 = mutate(chromesome_1, variable_substitue_dict)
                else:
                    child_1 = mutate(chromesome_1, variable_substitue_dict)
                _temp_mutants.append(child_1)

            # compute fitness in batch
            feature_list = []
            for mutant in _temp_mutants:
                _temp_code = map_chromesome(mutant, code, "c")

                _tmp_feature = convert_code_to_features(_temp_code, self.tokenizer_tgt, true_label, self.args)

                feature_list.append(_tmp_feature)

            if len(feature_list) == 0:
                continue

            if self.args.model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
                new_dataset = CodeDataset_CodeT5(feature_list)
            elif self.args.model in ['gpt2', 'gpt2-medium', 'codegpt', 'codebert', 'roberta-large', 'roberta-base',
                                     'codebert-insecure']:
                new_dataset = CodeDataset(feature_list)
            elif self.args.model in ['qwen0.5b', 'qwen1.5b']:
                new_dataset = CodeDataset_Qwen(feature_list)
            elif self.args.model in ['graphcodebert']:
                new_dataset = GraphCodeDataset(feature_list, self.args)
            else:
                raise ValueError

            mutate_logits, mutate_preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)

            mutate_fitness_values = []
            for index, logits in enumerate(mutate_logits):

                if mutate_preds[index] != orig_label:
                    adv_code = map_chromesome(_temp_mutants[index], code, "c")
                    for old_word in _temp_mutants[index].keys():
                        if old_word == _temp_mutants[index][old_word]:
                            nb_changed_var += 1
                            nb_changed_pos += len(names_positions_dict[old_word])

                    is_success = 1
                    code_blue = get_codebleu_512(code, adv_code, self.tokenizer_tgt, self.args.model)
                    if code_blue < self.args.codebleu_budget:
                        is_success = -1
                        adv_code = code

                    return code, prog_length, adv_code, true_label, orig_label, mutate_preds[
                        index], is_success, variable_names, None, nb_changed_var, nb_changed_pos, _temp_mutants[index]

                _tmp_fitness = max(orig_prob) - logits[orig_label]
                mutate_fitness_values.append(_tmp_fitness)

            for index, fitness_value in enumerate(mutate_fitness_values):
                min_value = min(fitness_values)
                if fitness_value > min_value:
                    min_index = fitness_values.index(min_value)
                    population[min_index] = _temp_mutants[index]
                    fitness_values[min_index] = fitness_value

        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, nb_changed_var, nb_changed_pos, None

    def greedy_attack(self, example, code, substituions):
        '''
        return
            original program: code
            program length: prog_length
            adversar program: adv_program
            true label: true_label
            original prediction: orig_label
            adversarial prediction: temp_label
            is_attack_success: is_success
            extracted variables: variable_names
            importance score of variables: names_to_importance_score
            number of changed variables: nb_changed_var
            number of changed positions: nb_changed_pos
            substitues for variables: replaced_words
        '''

        if self.args.model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
            from code_datasets.utils_codet5 import convert_defect_examples_to_features_adv as convert_code_to_features
        elif self.args.model in ['gpt2', 'gpt2-medium', 'codegpt']:
            from code_datasets.utils_codegpt import convert_examples_to_features_codegpt_adv as convert_code_to_features
        elif self.args.model in ['qwen0.5b', 'qwen1.5b']:
            from code_datasets.utils_qwen import convert_examples_to_features_qwen_adv as convert_code_to_features
        elif self.args.model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:
            from code_datasets.utils_codebert import \
                convert_examples_to_features_codebert_adv as convert_code_to_features
        elif self.args.model in ['graphcodebert']:
            from code_datasets.utils_graphcodebert import convert_examples_to_features_adv as convert_code_to_features
        else:
            raise ValueError

        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        if self.args.model in ['qwen0.5b', 'qwen1.5b']:
            true_label = example[2].item()
        else:
            true_label = example[1].item()

        adv_code = copy.deepcopy(code)
        temp_label = None

        identifiers, code_tokens = get_identifiers(code, 'c')
        prog_length = len(code_tokens)

        processed_code = " ".join(code_tokens)

        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)

        variable_names = list(substituions.keys())

        if not orig_label == true_label:
            is_success = -4
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        if len(variable_names) == 0:
            is_success = -3
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None

        sub_words = [self.tokenizer_tgt.cls_token] + sub_words[:self.args.code_length - 2] + [
            self.tokenizer_tgt.sep_token]

        importance_score, replace_token_positions, names_positions_dict = get_importance_score(self.args, example,
                                                                                               processed_code,
                                                                                               words,
                                                                                               sub_words,
                                                                                               variable_names,
                                                                                               self.model_tgt,
                                                                                               self.tokenizer_tgt,
                                                                                               [0, 1],
                                                                                               batch_size=self.args.eval_batch_size,
                                                                                               max_length=self.args.code_length,
                                                                                               model_type='classification')

        if importance_score is None:
            return code, prog_length, adv_code, true_label, orig_label, temp_label, -3, variable_names, None, None, None, None

        token_pos_to_score_pos = {}

        for i, token_pos in enumerate(replace_token_positions):
            token_pos_to_score_pos[token_pos] = i
        names_to_importance_score = {}

        for name in names_positions_dict.keys():
            total_score = 0.0
            positions = names_positions_dict[name]
            for token_pos in positions:
                total_score += importance_score[token_pos_to_score_pos[token_pos]]

            names_to_importance_score[name] = total_score

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)

        final_words = copy.deepcopy(words)
        final_code = copy.deepcopy(code)
        nb_changed_var = 0
        nb_changed_pos = 0
        is_success = -1
        replaced_words = {}
        for name_and_score in sorted_list_of_names:
            tgt_word = name_and_score[0]
            tgt_positions = names_positions_dict[tgt_word]
            all_substitues = substituions[tgt_word]

            most_gap = 0.0
            candidate = None
            replace_examples = []

            substitute_list = []

            for substitute in all_substitues:

                substitute_list.append(substitute)

                temp_code = get_example(final_code, tgt_word, substitute, "c")

                if self.args.model in ['qwen0.5b', 'qwen1.5b']:
                    new_feature = convert_code_to_features(temp_code, self.tokenizer_tgt, example[2].item(), self.args)
                else:
                    new_feature = convert_code_to_features(temp_code, self.tokenizer_tgt, example[1].item(), self.args)
                replace_examples.append(new_feature)

            if len(replace_examples) == 0:
                continue

            if self.args.model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
                new_dataset = CodeDataset_CodeT5(replace_examples)
            elif self.args.model in ['gpt2', 'gpt2-medium', 'codegpt', 'codebert', 'roberta-large', 'roberta-base',
                                     'codebert-insecure']:
                new_dataset = CodeDataset(replace_examples)
            elif self.args.model in ['qwen0.5b', 'qwen1.5b']:
                new_dataset = CodeDataset_Qwen(replace_examples)
            elif self.args.model in ['graphcodebert']:
                new_dataset = GraphCodeDataset(replace_examples, self.args)
            else:
                raise ValueError

            logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)

            assert (len(logits) == len(substitute_list))

            for index, temp_prob in enumerate(logits):
                temp_label = preds[index]

                if temp_label != orig_label:
                    is_success = 1
                    nb_changed_var += 1
                    nb_changed_pos += len(names_positions_dict[tgt_word])
                    candidate = substitute_list[index]
                    replaced_words[tgt_word] = candidate
                    adv_code = get_example(final_code, tgt_word, candidate, "c")

                    code_blue = get_codebleu_512(code, adv_code, self.tokenizer_tgt, self.args.model)
                    if code_blue < self.args.codebleu_budget:
                        is_success = -2
                        adv_code = final_code
                        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words

                    print("%s SUC! %s => %s (%.5f => %.5f)" % \
                          ('>>', tgt_word, candidate,
                           current_prob,
                           temp_prob[orig_label]), flush=True)
                    return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words

                else:
                    gap = current_prob - temp_prob[temp_label]
                    if gap > most_gap:
                        most_gap = gap
                        candidate = substitute_list[index]

            the_code = copy.deepcopy(final_code)

            if most_gap > 0:

                nb_changed_var += 1
                nb_changed_pos += len(names_positions_dict[tgt_word])
                current_prob = current_prob - most_gap

                final_code = get_example(final_code, tgt_word, candidate, "c")

                replaced_words[tgt_word] = candidate
                print("%s ACC! %s => %s (%.5f => %.5f)" % \
                      ('>>', tgt_word, candidate,
                       current_prob + most_gap,
                       current_prob), flush=True)
            else:
                replaced_words[tgt_word] = tgt_word

            code_blue = get_codebleu_512(code, final_code, self.tokenizer_tgt, self.args.model)
            if code_blue < self.args.codebleu_budget:
                is_success = -2
                adv_code = the_code
                return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words

            else:
                adv_code = final_code

        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words
