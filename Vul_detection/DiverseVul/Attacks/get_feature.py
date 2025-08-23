from parser.DFG import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript, DFG_c
from parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from tree_sitter import Language, Parser
import numpy as np
import torch
import tree_sitter_cpp as tscpp

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript,
    'c': DFG_c
}

# load parsers
parsers = {}
for lang in dfg_function:
    # LANGUAGE = Language('parser/my-languages.so', lang)
    LANGUAGE = Language(tscpp.language())
    # parser = Parser()
    # parser.set_language(LANGUAGE)
    parser = Parser(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
        # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node

        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')

        code_tokens = [index_to_code_token(x, code) for x in tokens_index]

        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)  # index: (root_node.start_point,root_node.end_point)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []

        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG

    except:
        dfg = []

    return code_tokens, dfg


def convert_examples_to_features_adv(code, tokenizer, args):
    # source
    parser = parsers['c']
    func = code

    # extract data flow
    code_tokens, dfg = extract_dataflow(func, parser, 'c')
    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                   enumerate(code_tokens)]
    # print('code_tokens', code_tokens)
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))

    code_tokens = [y for x in code_tokens for y in x]
    # print('code_tokens', code_tokens)

    # truncating args.code_length=512，args.data_flow_length=128
    code_tokens = code_tokens[:args.code_length + args.data_flow_length - 3 - min(len(dfg), args.data_flow_length)][
                  :512 - 3]

    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)

    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]

    dfg = dfg[:args.code_length + args.data_flow_length - len(source_tokens)]

    source_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    source_ids += [tokenizer.unk_token_id for x in dfg]
    padding_length = args.code_length + args.data_flow_length - len(source_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    source_ids += [tokenizer.pad_token_id] * padding_length

    # reindex
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]

    # calculate graph-guided masked function
    attn_mask = np.zeros((args.code_length + args.data_flow_length,
                          args.code_length + args.data_flow_length), dtype=np.bool_)
    # calculate begin index of node and max length of input
    node_index = sum([i > 1 for i in position_idx])
    max_length = sum([i != 1 for i in position_idx])
    # sequence can attend to sequence
    attn_mask[:node_index, :node_index] = True
    # special tokens attend to all tokens
    for idx, i in enumerate(source_ids):
        if i in [0, 2]:
            attn_mask[idx, :max_length] = True
    # nodes attend to code tokens that are identified from
    for idx, (a, b) in enumerate(dfg_to_code):
        if a < node_index and b < node_index:
            attn_mask[idx + node_index, a:b] = True
            attn_mask[a:b, idx + node_index] = True
    # nodes attend to adjacent nodes
    for idx, nodes in enumerate(dfg_to_dfg):
        for a in nodes:
            if a + node_index < len(position_idx):
                attn_mask[idx + node_index, a + node_index] = True

    return (source_ids,
            torch.tensor([position_idx]).to(args.device),
            torch.tensor([attn_mask]).to(args.device))


def extract_dataflow_attack(code, parser, lang):
    # remove comments
    # try:
    #     code = remove_comments_and_docstrings(code, lang)
    # except:
    #     pass
    # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node

        tokens_index = tree_to_token_index(root_node)

        # print('tokens_index', tokens_index)

        code = code.split('\n')
        print('code', code)

        code_tokens = [index_to_code_token(x, code) for x in tokens_index]

        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)  # index: (root_node.start_point,root_node.end_point)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        print('error')
        dfg = []
    return code_tokens, dfg


def convert_examples_to_features_adv_attack(code, tokenizer, args):
    # source
    parser = parsers['c']
    func = code

    # extract data flow
    code_tokens, dfg = extract_dataflow_attack(func, parser, 'c')
    # print(code_tokens)

    combine_code_tokens = ' '.join(code_tokens)

    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                   enumerate(code_tokens)]

    # code_tokens = [tokenizer.tokenize(x) for idx, x in
    #                enumerate(code_tokens)]

    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))

    code_tokens = [y for x in code_tokens for y in x]

    # truncating args.code_length=512，args.data_flow_length=128
    code_tokens = code_tokens[:args.code_length + args.data_flow_length - 3 - min(len(dfg), args.data_flow_length)][
                  :512 - 3]

    # print(code_tokens, len(code_tokens))

    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    len_source_tokens = len(source_tokens)

    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)

    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
    dfg = dfg[:args.code_length + args.data_flow_length - len(source_tokens)]
    source_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    source_ids += [tokenizer.unk_token_id for x in dfg]

    padding_length = args.code_length + args.data_flow_length - len(source_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    source_ids += [tokenizer.pad_token_id] * padding_length

    # reindex
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]

    # calculate graph-guided masked function
    attn_mask = np.zeros((args.code_length + args.data_flow_length,
                          args.code_length + args.data_flow_length), dtype=np.bool_)
    # calculate begin index of node and max length of input
    node_index = sum([i > 1 for i in position_idx])
    max_length = sum([i != 1 for i in position_idx])
    # sequence can attend to sequence
    attn_mask[:node_index, :node_index] = True
    # special tokens attend to all tokens
    for idx, i in enumerate(source_ids):
        if i in [0, 2]:
            attn_mask[idx, :max_length] = True
    # nodes attend to code tokens that are identified from
    for idx, (a, b) in enumerate(dfg_to_code):
        if a < node_index and b < node_index:
            attn_mask[idx + node_index, a:b] = True
            attn_mask[a:b, idx + node_index] = True
    # nodes attend to adjacent nodes
    for idx, nodes in enumerate(dfg_to_dfg):
        for a in nodes:
            if a + node_index < len(position_idx):
                attn_mask[idx + node_index, a + node_index] = True

    return (source_ids,
            torch.tensor([position_idx]).to(args.device),
            torch.tensor([attn_mask]).to(args.device), combine_code_tokens, len(source_tokens))
