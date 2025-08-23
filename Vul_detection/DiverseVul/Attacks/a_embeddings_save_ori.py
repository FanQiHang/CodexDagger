# 扩展版：多样本统计分析
import torch
from load_models import load_model
import argparse
from load_predicts import load_predict_distill
from get_feature import convert_examples_to_features_adv
import json


def get_embeddings_batch_codet5(model, embedding_layer, tokenizer, texts, device, batch_size=1):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        source_ids = tokenizer.encode(batch_texts[0], max_length=512, padding='max_length',
                                      truncation=True)
        source_ids_ori = torch.tensor([source_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            input_embeds = embedding_layer(source_ids_ori)
            outputs = model.get_hidden_states(source_ids_ori, input_embeds)

            # outputs_ = model.get_embeddings(source_ids_ori, input_embeds, labels=None)
            #
            # print(outputs, outputs.size())
            #
            # exit()

            batch_embeds = torch.mean(outputs, dim=1).cpu()
            # batch_embeds = outputs.cpu()

        embeddings.append(batch_embeds)

    all_batch_embeds = torch.cat(embeddings, dim=0)

    mean_embed = torch.mean(all_batch_embeds, dim=0)

    return all_batch_embeds, mean_embed


def get_embeddings_batch_qwen(model, embedding_layer, tokenizer, texts, device, batch_size=1):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        #
        result = tokenizer(
            batch_texts[0],
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        source_ids_ori = result["input_ids"].to(device)
        attention_mask = result["attention_mask"].to(device)

        with torch.no_grad():
            input_embeds = embedding_layer(source_ids_ori)
            outputs = model.get_hidden_states(input_embeds, attention_mask)

            batch_embeds = torch.mean(outputs, dim=1).cpu()

        embeddings.append(batch_embeds)

    all_batch_embeds = torch.cat(embeddings, dim=0)

    print(all_batch_embeds.size())

    mean_embed = torch.mean(all_batch_embeds, dim=0)

    print('mean_embed', mean_embed)

    return all_batch_embeds, mean_embed


def get_embeddings_batch_codebert(model, embedding_layer, tokenizer, texts, device, batch_size=1):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        code_tokens = tokenizer.tokenize(batch_texts[0])[:512 - 2]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = 512 - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_ids_ori = torch.tensor([source_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            input_embeds = embedding_layer(source_ids_ori)
            outputs = model.get_hidden_states(source_ids_ori, input_embeds)
            # print(outputs, outputs.size())
            # outputs = model(**inputs).last_hidden_state
            # batch_embeds = torch.mean(outputs, dim=1).cpu().numpy()
            #
            # print(len(batch_embeds[0]))
            batch_embeds = torch.mean(outputs, dim=1).cpu()
            # batch_embeds = outputs.cpu()

        embeddings.append(batch_embeds)

    all_batch_embeds = torch.cat(embeddings, dim=0)

    mean_embed = torch.mean(all_batch_embeds, dim=0)

    return all_batch_embeds, mean_embed


def get_embeddings_batch_graphcodebert(model, embedding_layer, tokenizer, texts, device, batch_size=1):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        source_ids, position_idx, attn_mask = convert_examples_to_features_adv(batch_texts[0], tokenizer, args)
        source_ids_ori = torch.tensor([source_ids], dtype=torch.long).to(device)

        inputs_embeds = model.encoder.roberta.embeddings.word_embeddings(source_ids_ori)

        with torch.no_grad():
            # input_embeds = embedding_layer(source_ids_ori)
            outputs = model.get_hidden_states(inputs_embeds, position_idx,
                                              attn_mask)
            # print(outputs, outputs.size())
            # outputs = model(**inputs).last_hidden_state
            # batch_embeds = torch.mean(outputs, dim=1).cpu().numpy()
            #
            batch_embeds = torch.mean(outputs, dim=1).cpu()
            # batch_embeds = outputs.cpu()
            # print(len(batch_embeds[0]))

        embeddings.append(batch_embeds)

    all_batch_embeds = torch.cat(embeddings, dim=0)

    mean_embed = torch.mean(all_batch_embeds, dim=0)

    return all_batch_embeds, mean_embed


def get_embeddings_batch_codegpt(model, embedding_layer, tokenizer, texts, device, batch_size=1):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        code_tokens = tokenizer.tokenize(batch_texts[0])[:512 - 2]
        source_tokens = ["<|endoftext|>"] + code_tokens + ["<|endoftext|>"]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = 512 - len(source_ids)
        source_ids += [50255] * padding_length

        source_ids_ori = torch.tensor([source_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            input_embeds = embedding_layer(source_ids_ori)
            outputs = model.get_hidden_states(source_ids_ori, input_embeds)
            # print(outputs, outputs.size())
            # outputs = model(**inputs).last_hidden_state
            # batch_embeds = torch.mean(outputs, dim=1).cpu().numpy()
            #
            batch_embeds = torch.mean(outputs, dim=1).cpu()
            # batch_embeds = outputs.cpu()
            # print(len(batch_embeds[0]))

        embeddings.append(batch_embeds)

    all_batch_embeds = torch.cat(embeddings, dim=0)

    mean_embed = torch.mean(all_batch_embeds, dim=0)

    return all_batch_embeds, mean_embed


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


def read_defect_examples_query_graphcodebert(filename, data_num, args, idxx):
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

    parser.add_argument("--attack_model", type=str, default="codet5-base", help="[]", )
    parser.add_argument("--data_flow_length", default=114, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--code_length", default=400, type=int,
                        help="Optional Code input sequence length after tokenization.")

    args = parser.parse_args()

    args.data_dir = '../dataset/'
    args.device = torch.device("cuda")

    args.train_filename = '../dataset/test.jsonl'

    idx = load_predict_distill(args.attack_model, args)

    if args.attack_model == 'graphcodebert':
        train_examples = read_defect_examples_query_graphcodebert(args.train_filename, None, None, idx)
    else:
        train_examples = read_defect_examples_query(args.train_filename, None, None, idx)

    positive_codes = []
    negative_codes = []

    # iii = 0
    for train_example in train_examples:

        if int(train_example.target) == 1:
            positive_codes.append(train_example.source)
        elif int(train_example.target) == 0:
            negative_codes.append(train_example.source)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer = load_model(args.attack_model, args.device, args)
    embedding_layer = model.encoder.get_input_embeddings()

    if args.attack_model in ['codet5-base', 'codet5-small', 'codet5-base-multi']:

        positive_batch_embeds, positive_mean_embed = get_embeddings_batch_codet5(model, embedding_layer, tokenizer,
                                                                                 positive_codes, device)
        negative_batch_embeds, negative_mean_embed = get_embeddings_batch_codet5(model, embedding_layer, tokenizer,
                                                                                 negative_codes, device)

    elif args.attack_model in ['gpt2', 'gpt2-medium', 'codegpt']:

        positive_batch_embeds, positive_mean_embed = get_embeddings_batch_codegpt(model, embedding_layer, tokenizer,
                                                                                  positive_codes,
                                                                                  device)
        negative_batch_embeds, negative_mean_embed = get_embeddings_batch_codegpt(model, embedding_layer, tokenizer,
                                                                                  negative_codes,
                                                                                  device)

    elif args.attack_model in ['qwen0.5b', 'qwen1.5b']:

        positive_batch_embeds, positive_mean_embed = get_embeddings_batch_qwen(model, embedding_layer, tokenizer,
                                                                               positive_codes,
                                                                               device)
        negative_batch_embeds, negative_mean_embed = get_embeddings_batch_qwen(model, embedding_layer, tokenizer,
                                                                               negative_codes,
                                                                               device)

    elif args.attack_model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:

        positive_batch_embeds, positive_mean_embed = get_embeddings_batch_codebert(model, embedding_layer, tokenizer,
                                                                                   positive_codes,
                                                                                   device)
        negative_batch_embeds, negative_mean_embed = get_embeddings_batch_codebert(model, embedding_layer, tokenizer,
                                                                                   negative_codes,
                                                                                   device)

    elif args.attack_model in ['graphcodebert']:
        positive_batch_embeds, positive_mean_embed = get_embeddings_batch_graphcodebert(model, embedding_layer,
                                                                                        tokenizer, positive_codes,
                                                                                        device)
        negative_batch_embeds, negative_mean_embed = get_embeddings_batch_graphcodebert(model, embedding_layer,
                                                                                        tokenizer, negative_codes,
                                                                                        device)
    else:
        raise ValueError

    torch.save({
        'positive_batch_embeds': positive_batch_embeds,
        'positive_mean_embed': positive_mean_embed,
        'negative_batch_embeds': negative_batch_embeds,
        'negative_mean_embed': negative_mean_embed,
    }, '../dataset/' + args.attack_model + '_embeddings_and_mean_ori.pt')
