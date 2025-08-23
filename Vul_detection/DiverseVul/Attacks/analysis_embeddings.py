import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from load_models import load_model, load_distill_model
import argparse
from load_predicts import load_predict_distill
from get_feature import convert_examples_to_features_adv
import json

first_layer_output = None


def hook_fn(module, input, output):
    global first_layer_output
    first_layer_output = output.detach()


def get_embeddings_batch_codet5_small(model, embedding_layer, tokenizer, texts, labels, device, args, batch_size=1):
    # if args.model_size == 'small':
    #     hook = model.classifier.register_forward_hook(hook_fn)
    # else:
    #     hook = model.classifier[0].register_forward_hook(hook_fn)  # 替换为你的第一层

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        source_ids = tokenizer.encode(batch_texts[0], max_length=512, padding='max_length',
                                      truncation=True)

        source_ids_ori = torch.tensor([source_ids], dtype=torch.long).to(device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

        with torch.no_grad():
            # input_embeds = embedding_layer(source_ids_ori)

            # model(source_ids_ori, batch_labels)

            batch_embeds = model.get_t5_vec(source_ids_ori).cpu().numpy()

            # print('first_layer_output', first_layer_output.shape)
            #
            # batch_embeds = first_layer_output.cpu().numpy()

        embeddings.append(batch_embeds)

    # hook.remove()

    return np.concatenate(embeddings, axis=0)


def get_embeddings_batch_codet5(model, embedding_layer, tokenizer, texts, labels, device, args, batch_size=1):
    hook = model.classifier[0].register_forward_hook(hook_fn)

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        source_ids = tokenizer.encode(batch_texts[0], max_length=512, padding='max_length',
                                      truncation=True)

        source_ids_ori = torch.tensor([source_ids], dtype=torch.long).to(device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

        with torch.no_grad():
            # input_embeds = embedding_layer(source_ids_ori)

            model(source_ids_ori, batch_labels)

            print('first_layer_output', first_layer_output.shape)

            batch_embeds = first_layer_output.cpu().numpy()

        embeddings.append(batch_embeds)

    hook.remove()

    return np.concatenate(embeddings, axis=0)


def get_embeddings_batch_qwen(model, embedding_layer, tokenizer, texts, labels, device, batch_size=1):
    embeddings = []

    hook = model.classifier[0].register_forward_hook(hook_fn)  # 替换为你的第一层

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        result = tokenizer(
            batch_texts[0],
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        source_ids_ori = result["input_ids"].to(device)
        attention_mask = result["attention_mask"].to(device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

        with torch.no_grad():
            # input_embeds = embedding_layer(source_ids_ori)
            # outputs = model.get_hidden_states(input_embeds, attention_mask)

            model(source_ids_ori, attention_mask, batch_labels)

            print('first_layer_output', first_layer_output.shape)

            batch_embeds = first_layer_output.cpu().numpy()

            # batch_embeds = outputs.cpu().numpy()

        embeddings.append(batch_embeds)

    hook.remove()

    return np.concatenate(embeddings, axis=0)


def get_embeddings_batch_codebert(model, embedding_layer, tokenizer, texts, labels, device, batch_size=1):
    embeddings = []

    hook = model.encoder.classifier.dense.register_forward_hook(hook_fn)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        code_tokens = tokenizer.tokenize(batch_texts[0])[:512 - 2]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = 512 - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length

        source_ids_ori = torch.tensor([source_ids], dtype=torch.long).to(device)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

        with torch.no_grad():
            # input_embeds = embedding_layer(source_ids_ori)
            # outputs = model.get_hidden_states(source_ids_ori, input_embeds)

            model(source_ids_ori, batch_labels)

            batch_embeds = first_layer_output.cpu().numpy()

            # batch_embeds = outputs.cpu().numpy()

        embeddings.append(batch_embeds)

    hook.remove()

    return np.concatenate(embeddings, axis=0)


def get_embeddings_batch_graphcodebert(model, embedding_layer, tokenizer, texts, labels, device, batch_size=1):
    embeddings = []

    hook = model.encoder.classifier.dense.register_forward_hook(hook_fn)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        source_ids, position_idx, attn_mask = convert_examples_to_features_adv(batch_texts[0], tokenizer, args)
        source_ids_ori = torch.tensor([source_ids], dtype=torch.long).to(device)

        # inputs_embeds = model.encoder.roberta.embeddings.word_embeddings(source_ids_ori)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
        with torch.no_grad():
            # input_embeds = embedding_layer(source_ids_ori)
            # outputs = model.get_hidden_states(inputs_embeds, position_idx,
            #                                   attn_mask)
            model(source_ids_ori, position_idx, attn_mask, batch_labels)

            batch_embeds = first_layer_output.cpu().numpy()

            # batch_embeds = outputs.cpu().numpy()

        embeddings.append(batch_embeds)

    hook.remove()

    return np.concatenate(embeddings, axis=0)


def get_embeddings_batch_codegpt_small(model, embedding_layer, tokenizer, texts, labels, device, args, batch_size=1):
    embeddings = []

    # if args.model_size == 'small':
    #     hook = model.encoder.score.register_forward_hook(hook_fn)
    # else:
    #     hook = model.encoder.score[0].register_forward_hook(hook_fn)  # 替换为你的第一层

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        code_tokens = tokenizer.tokenize(batch_texts[0])[:512 - 2]
        source_tokens = ["<|endoftext|>"] + code_tokens + ["<|endoftext|>"]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = 512 - len(source_ids)
        source_ids += [50255] * padding_length

        source_ids_ori = torch.tensor([source_ids], dtype=torch.long).to(device)
        sequence_lengths = torch.eq(source_ids_ori, model.config.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % source_ids_ori.shape[-1]
        sequence_lengths = sequence_lengths.to(source_ids_ori.device)

        batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

        with torch.no_grad():
            input_embeds = embedding_layer(source_ids_ori)
            outputs = model.get_hidden_states(source_ids_ori, input_embeds)

            # batch_embeds = outputs.cpu().numpy()
            # model(source_ids_ori, batch_labels)

            # print('first_layer_output', first_layer_output.shape)

            # batch_embeds = first_layer_output.cpu().numpy()[:, sequence_lengths, :]

            batch_embeds = outputs.cpu().numpy()[:, sequence_lengths, :]

            print(batch_embeds.shape)

        embeddings.append(batch_embeds)

    # hook.remove()

    return np.concatenate(embeddings, axis=0)


def get_embeddings_batch_codegpt(model, embedding_layer, tokenizer, texts, labels, device, args, batch_size=1):
    embeddings = []

    hook = model.encoder.score[0].register_forward_hook(hook_fn)  # 替换为你的第一层

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        code_tokens = tokenizer.tokenize(batch_texts[0])[:512 - 2]
        source_tokens = ["<|endoftext|>"] + code_tokens + ["<|endoftext|>"]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = 512 - len(source_ids)
        source_ids += [50255] * padding_length

        source_ids_ori = torch.tensor([source_ids], dtype=torch.long).to(device)
        sequence_lengths = torch.eq(source_ids_ori, model.config.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % source_ids_ori.shape[-1]
        sequence_lengths = sequence_lengths.to(source_ids_ori.device)

        batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

        with torch.no_grad():
            # input_embeds = embedding_layer(source_ids_ori)
            # outputs = model.get_hidden_states(source_ids_ori, input_embeds)

            # batch_embeds = outputs.cpu().numpy()
            model(source_ids_ori, batch_labels)

            print('first_layer_output', first_layer_output.shape)

            batch_embeds = first_layer_output.cpu().numpy()[:, sequence_lengths, :]

            print(batch_embeds.shape)

        embeddings.append(batch_embeds)

    hook.remove()

    return np.concatenate(embeddings, axis=0)


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task


def read_defect_examples_query(filename, data_num, args, idxx):
    """Read examples from filename."""
    examples = []

    final_data = json.load(open(filename, 'r'))

    final_data_idx = np.load(filename[:-4] + 'npz')['arr_0']

    ii = 0
    for key in final_data:

        # if ii == 100:
        #     break

        if int(key) not in final_data_idx:
            continue

        js = final_data[key]

        # if args.sample_num != 100:
        #
        #     if int(js['idx']) not in idxx:
        #         continue

        if int(js['idx']) not in idxx:
            continue

        code = ' '.join(js['func'].split())

        examples.append(
            Example(
                idx=int(key),
                source=code,
                target=js['target']
            )
        )

        ii += 1

    return examples


def read_defect_examples_query_graphcodebert(filename, data_num, args, idxx):
    """Read examples from filename."""
    examples = []

    final_data = json.load(open(filename, 'r'))

    final_data_idx = np.load(filename[:-4] + 'npz')['arr_0']

    for key in final_data:

        if int(key) not in final_data_idx:
            continue

        js = final_data[key]

        # if args.sample_num != 100:
        #
        #     if int(js['idx']) not in idxx:
        #         continue

        if int(js['idx']) not in idxx:
            continue

        # code = ' '.join(js['func'].split())

        examples.append(
            Example(
                idx=int(key),
                source=js['func'],
                target=js['target']
            )
        )

    return examples


def read_defect_examples_query_test(filename, data_num, args, idxx):
    """Read examples from filename."""
    examples = []
    num = 0
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):

            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())

            if int(js['idx']) not in idxx:
                continue

            num += 1
            examples.append(
                Example(
                    idx=js['idx'],
                    source=code,
                    target=js['target']
                )
            )

            if num == data_num:
                break

    return examples


def read_defect_examples_query_graphcodebert_test(filename, data_num, args, idxx):
    """Read examples from filename."""
    examples = []
    num = 0
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):

            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())

            if int(js['idx']) not in idxx:
                continue

            num += 1
            examples.append(
                Example(
                    idx=js['idx'],
                    source=code,
                    target=js['target']
                )
            )

            if num == data_num:
                break

    return examples


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description="Code_attack")
    parser.add_argument("--model", type=str, default="codet5-base", help="[]", )
    parser.add_argument("--victim_model", type=str, default="codegpt", help="[]", )

    parser.add_argument("--max_source_length", default=512, type=int, help="")

    parser.add_argument("--temperature", default=4, type=float, help="Teacher Model")
    parser.add_argument("--alpha", default=0.0, type=float, help="Teacher Model")
    parser.add_argument("--loss_name", default='', type=str, help="")

    parser.add_argument("--data_flow_length", default=114, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--code_length", default=400, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--is_sample_20", default='yes', type=str, help="")

    parser.add_argument("--sample_codebleu_budget", default=0.4, type=float, help="")

    parser.add_argument("--sample_num", type=int, default=20, help="", )

    parser.add_argument("--model_size", type=str, default='large', help="", )

    parser.add_argument("--distance", default='l1', type=str,
                        help="Optional Code input sequence length after tokenization.")

    args = parser.parse_args()

    args.data_dir = '../dataset/'
    args.device = torch.device("cuda")

    args.test_filename = '../dataset/test.jsonl'

    idx_test = load_predict_distill(args.victim_model, args)

    if args.victim_model == 'graphcodebert':
        test_examples = read_defect_examples_query_graphcodebert_test(args.test_filename, None, None, idx_test)
    else:
        test_examples = read_defect_examples_query_test(args.test_filename, None, None, idx_test)

    args.train_filename = '../dataset/test_augment_' + args.victim_model + '_0.6_20.json'

    idx = load_predict_distill(args.victim_model, args)

    if args.victim_model == 'graphcodebert':
        train_examples = read_defect_examples_query_graphcodebert(args.train_filename, None, None, idx)
    else:
        train_examples = read_defect_examples_query(args.train_filename, None, None, idx)

    all_codes = []
    all_labels = []

    iiii = 0
    for test_example in test_examples:
        all_codes.append(test_example.source)
        all_labels.append(test_example.target)
        iiii += 1

    # iii = 0
    for train_example in train_examples:
        all_codes.append(train_example.source)
        all_labels.append(train_example.target)
        # if iii == 10:
        #     break
        # iii += 1

    NUM_QUERIES = iiii
    NUM_NEIGHBORS = 10
    RANDOM_SEED = 42

    np.random.seed(RANDOM_SEED)
    # query_indices = np.random.choice(len(all_codes), NUM_QUERIES, replace=False)
    # print('query_indices', query_indices)

    # query_codes = [all_codes[i] for i in query_indices]
    query_codes = all_codes[:iiii]
    candidate_codes = all_codes[iiii:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = load_distill_model(args.model, args.victim_model, args.model_size, device, args)

    for name, param in model.named_parameters():
        print(name)

    teacher_model, teacher_tokenizer = load_model(args.victim_model, args.device, args)

    embedding_layer = model.encoder.get_input_embeddings()

    teacher_embedding_layer = teacher_model.encoder.get_input_embeddings()

    print("\nGenerating embeddings...")
    all_texts = query_codes + candidate_codes

    if 'codet5' in args.model and 'codet5' in args.victim_model:

        if args.model_size == 'small':

            embeddings_small = get_embeddings_batch_codet5_small(model, embedding_layer, tokenizer, all_texts,
                                                                 all_labels,
                                                                 device, args)
            query_embeds_small = embeddings_small[:NUM_QUERIES]
            # print(query_embeds_small[:, 0])

            candidate_embeds_small = embeddings_small[NUM_QUERIES:]

            embeddings_base = get_embeddings_batch_codet5(teacher_model, teacher_embedding_layer,
                                                          teacher_tokenizer,
                                                          all_texts, all_labels, device, args)
            query_embeds_base = embeddings_base[:NUM_QUERIES]
            candidate_embeds_base = embeddings_base[NUM_QUERIES:]

        else:

            embeddings_small = get_embeddings_batch_codet5(model, embedding_layer, tokenizer, all_texts, all_labels,
                                                           device, args)
            query_embeds_small = embeddings_small[:NUM_QUERIES]
            # print(query_embeds_small[:, 0])

            candidate_embeds_small = embeddings_small[NUM_QUERIES:]

            embeddings_base = get_embeddings_batch_codet5(teacher_model, teacher_embedding_layer, teacher_tokenizer,
                                                          all_texts, all_labels, device, args)
            query_embeds_base = embeddings_base[:NUM_QUERIES]
            candidate_embeds_base = embeddings_base[NUM_QUERIES:]

    elif ('codebert' in args.model and 'codebert' in args.victim_model) or (
            'roberta' in args.model and 'roberta' in args.victim_model):

        if 'graph' in args.victim_model:

            embeddings_small = get_embeddings_batch_codebert(model, embedding_layer, tokenizer, all_texts, all_labels,
                                                             device)
            query_embeds_small = embeddings_small[:NUM_QUERIES]
            # print(query_embeds_small[:, 0])

            candidate_embeds_small = embeddings_small[NUM_QUERIES:]

            embeddings_base = get_embeddings_batch_graphcodebert(teacher_model, teacher_embedding_layer,
                                                                 teacher_tokenizer,
                                                                 all_texts, all_labels, device)

            query_embeds_base = embeddings_base[:NUM_QUERIES]
            candidate_embeds_base = embeddings_base[NUM_QUERIES:]

        elif 'graph' in args.model:

            embeddings_small = get_embeddings_batch_graphcodebert(model, embedding_layer, tokenizer, all_texts,
                                                                  all_labels,
                                                                  device)
            query_embeds_small = embeddings_small[:NUM_QUERIES]
            # print(query_embeds_small[:, 0])

            candidate_embeds_small = embeddings_small[NUM_QUERIES:]

            embeddings_base = get_embeddings_batch_codebert(teacher_model, teacher_embedding_layer,
                                                            teacher_tokenizer,
                                                            all_texts, all_labels, device)

            query_embeds_base = embeddings_base[:NUM_QUERIES]
            candidate_embeds_base = embeddings_base[NUM_QUERIES:]

        else:

            embeddings_small = get_embeddings_batch_codebert(model, embedding_layer, tokenizer, all_texts, all_labels,
                                                             device)
            query_embeds_small = embeddings_small[:NUM_QUERIES]
            # print(query_embeds_small[:, 0])

            candidate_embeds_small = embeddings_small[NUM_QUERIES:]

            embeddings_base = get_embeddings_batch_codebert(teacher_model, teacher_embedding_layer, teacher_tokenizer,
                                                            all_texts, all_labels,
                                                            device)
            query_embeds_base = embeddings_base[:NUM_QUERIES]
            candidate_embeds_base = embeddings_base[NUM_QUERIES:]

    elif 'gpt' in args.model and 'gpt' in args.victim_model:

        if args.model_size == 'small':

            embeddings_small = get_embeddings_batch_codegpt_small(model, embedding_layer, tokenizer, all_texts,
                                                                  all_labels,
                                                                  device, args)
            query_embeds_small = embeddings_small[:NUM_QUERIES]
            # print(query_embeds_small[:, 0])

            candidate_embeds_small = embeddings_small[NUM_QUERIES:]

            embeddings_base = get_embeddings_batch_codegpt(teacher_model, teacher_embedding_layer, teacher_tokenizer,
                                                           all_texts, all_labels, device, args)
            query_embeds_base = embeddings_base[:NUM_QUERIES]
            candidate_embeds_base = embeddings_base[NUM_QUERIES:]

        else:

            embeddings_small = get_embeddings_batch_codegpt(model, embedding_layer, tokenizer, all_texts, all_labels,
                                                            device, args)
            query_embeds_small = embeddings_small[:NUM_QUERIES]
            # print(query_embeds_small[:, 0])

            candidate_embeds_small = embeddings_small[NUM_QUERIES:]

            embeddings_base = get_embeddings_batch_codegpt(teacher_model, teacher_embedding_layer, teacher_tokenizer,
                                                           all_texts, all_labels, device, args)
            query_embeds_base = embeddings_base[:NUM_QUERIES]
            candidate_embeds_base = embeddings_base[NUM_QUERIES:]

    elif 'qwen' in args.model and 'qwen' in args.victim_model:

        embeddings_small = get_embeddings_batch_qwen(model, embedding_layer, tokenizer, all_texts, all_labels, device)
        query_embeds_small = embeddings_small[:NUM_QUERIES]
        # print(query_embeds_small[:, 0])

        candidate_embeds_small = embeddings_small[NUM_QUERIES:]

        embeddings_base = get_embeddings_batch_qwen(teacher_model, teacher_embedding_layer, teacher_tokenizer,
                                                    all_texts, all_labels,
                                                    device)
        query_embeds_base = embeddings_base[:NUM_QUERIES]
        candidate_embeds_base = embeddings_base[NUM_QUERIES:]

    else:
        raise ValueError

    print("\nBuilding nearest neighbors models...")
    if args.distance == 'l1':
        nn_small = NearestNeighbors(n_neighbors=NUM_NEIGHBORS, metric='manhattan')
    elif args.distance == 'l2':
        nn_small = NearestNeighbors(n_neighbors=NUM_NEIGHBORS)
    elif args.distance == 'cosine':
        nn_small = NearestNeighbors(n_neighbors=NUM_NEIGHBORS, metric='cosine')
    else:
        raise ValueError

    nn_small.fit(candidate_embeds_small)

    if args.distance == 'l1':
        nn_base = NearestNeighbors(n_neighbors=NUM_NEIGHBORS, metric='manhattan')
    elif args.distance == 'l2':
        nn_base = NearestNeighbors(n_neighbors=NUM_NEIGHBORS)
    elif args.distance == 'cosine':
        nn_base = NearestNeighbors(n_neighbors=NUM_NEIGHBORS, metric='cosine')
    else:
        raise ValueError

    nn_base.fit(candidate_embeds_base)

    overlap_ratios = []
    for i in range(NUM_QUERIES):
        print('query_embeds_small[i]', query_embeds_small[i])
        distances_small, indices_small = nn_small.kneighbors(query_embeds_small[i].reshape(1, -1))
        distances_base, indices_base = nn_base.kneighbors(query_embeds_base[i].reshape(1, -1))

        print('indices_small ', indices_small)
        print('indices_base ', indices_base)

        overlap = len(set(indices_small[0]) & set(indices_base[0]))
        ratio = overlap / NUM_NEIGHBORS
        overlap_ratios.append(ratio)

        print(f"Query {i + 1}/{NUM_QUERIES}: Overlap ratio = {ratio:.2%}")

    mean_ratio = np.mean(overlap_ratios)
    std_ratio = np.std(overlap_ratios)

    if 'large' in args.model_size:

        with open('./saved_results/complete_embeddings/large_' + args.model + '_' + args.victim_model + '.txt', 'w',
                  encoding='utf-8') as f:
            f.write("统计结果\n")
            f.write("=" * 30 + "\n")
            f.write("{}\n".format(mean_ratio))
            f.write("{}\n".format(std_ratio))
    else:

        with open('./saved_results/complete_embeddings/' + args.model + '_' + args.victim_model + '.txt', 'w',
                  encoding='utf-8') as f:
            f.write("统计结果\n")
            f.write("=" * 30 + "\n")
            f.write("{}\n".format(mean_ratio))
            f.write("{}\n".format(std_ratio))

    print(mean_ratio)
    print(std_ratio)
