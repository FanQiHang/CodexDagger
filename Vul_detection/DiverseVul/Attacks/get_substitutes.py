import json
import sys
import copy
import torch
import argparse
from tqdm import tqdm
from load_predicts import load_predict
# sys.path.append('../../../')
# sys.path.append('../../../python_parser')

# from attacker import 
from run_parser import get_identifiers, remove_comments_and_docstrings
from utils import is_valid_variable_name, _tokenize, get_identifier_posistions_from_code, get_masked_code_by_position, \
    get_substitues, is_valid_substitue
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from process_AST import *


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--base_model", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--store_path", default=None, type=str,
                        help="results")

    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--index", nargs='+',
                        help="Optional input sequence length after tokenization.")

    args = parser.parse_args()

    eval_data = []

    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)

    codebert_mlm.to('cuda')

    with open(args.eval_data_file) as rf:
        for i, line in enumerate(rf):
            # if i < int(args.index[0]) or i >= int(args.index[1]):
            #     continue
            item = json.loads(line.strip())
            eval_data.append(item)
    # print(len(eval_data))
    with open(args.store_path, "w") as wf:
        for item in tqdm(eval_data):

            identifiers, code_tokens = get_identifiers(remove_comments_and_docstrings(item["func"], "c"), "c")
            processed_code = " ".join(code_tokens)

            print(item["func"])
            # print(processed_code)

            # identifiers = get_AST_set(remove_comments_and_docstrings(item["func"], "c"))

            words, sub_words, keys = _tokenize(processed_code, tokenizer_mlm)

            # print(words, sub_words, keys)

            variable_names = []

            for name in identifiers:
                if ' ' in name[0].strip():
                    continue
                variable_names.append(name[0])

            # print(variable_names)

            sub_words = [tokenizer_mlm.cls_token] + sub_words[:512 - 2] + [tokenizer_mlm.sep_token]

            input_ids_ = torch.tensor([tokenizer_mlm.convert_tokens_to_ids(sub_words)])

            word_predictions = codebert_mlm(input_ids_.to('cuda'))[0].squeeze()  # seq-len(sub) vocab
            word_pred_scores_all, word_predictions = torch.topk(word_predictions, 60, -1)  # seq-len k
            # print(word_predictions)

            word_predictions = word_predictions[1:len(sub_words) + 1, :]
            word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]

            names_positions_dict = get_identifier_posistions_from_code(words, variable_names)

            # print('names_positions_dict', names_positions_dict)

            variable_substitue_dict = {}
            # with torch.no_grad():
            #     orig_embeddings = codebert_mlm.roberta(input_ids_.to('cuda'))[0]

            # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            for tgt_word in names_positions_dict.keys():
                tgt_positions = names_positions_dict[tgt_word]  # the positions of tgt_word in code
                if not is_valid_variable_name(tgt_word, lang='c'):
                    # if the extracted name is not valid
                    continue
                all_substitues = []
                for one_pos in tgt_positions:
                    if keys[one_pos][0] >= word_predictions.size()[0]:
                        continue
                    substitutes = word_predictions[keys[one_pos][0]:keys[one_pos][1]]  # L, k
                    # print('substitutes', substitutes)
                    word_pred_scores = word_pred_scores_all[keys[one_pos][0]:keys[one_pos][1]]

                    substitutes = get_substitues(substitutes,
                                                 tokenizer_mlm,
                                                 codebert_mlm,
                                                 1,
                                                 word_pred_scores,
                                                 0)
                    all_substitues += substitutes
                all_substitues = set(all_substitues)

                for tmp_substitue in all_substitues:
                    if tmp_substitue.strip() in variable_names:
                        continue
                    if not is_valid_substitue(tmp_substitue.strip(), tgt_word, 'c'):
                        continue
                    try:
                        variable_substitue_dict[tgt_word].append(tmp_substitue)
                    except:
                        variable_substitue_dict[tgt_word] = [tmp_substitue]

            item["substitutes"] = variable_substitue_dict
            print(item["substitutes"])
            wf.write(json.dumps(item) + '\n')

            print('successful!')


if __name__ == "__main__":
    main()
