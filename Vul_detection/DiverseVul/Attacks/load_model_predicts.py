import torch
from get_feature import convert_examples_to_features_adv


def load_model_predict(args, model, tokenizer, device, source, ground_truth):
    if args.model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
        example_sample = ' '.join(source.split())
        example_sample_source_ids = tokenizer.encode(example_sample, max_length=args.max_source_length,
                                                     padding='max_length', truncation=True)
        with torch.no_grad():
            inputs = torch.tensor([example_sample_source_ids]).to(device)
            label = torch.tensor([ground_truth]).to(device)
            lm_loss, logits = model(inputs, label)

        model.zero_grad()
        preds = logits[:, 1] > 0.5

    elif args.model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:

        example_sample = ' '.join(source.split())
        code_tokens = tokenizer.tokenize(example_sample)[:512 - 2]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        example_sample_source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = 512 - len(example_sample_source_ids)
        example_sample_source_ids += [tokenizer.pad_token_id] * padding_length

        with torch.no_grad():
            inputs = torch.tensor([example_sample_source_ids]).to(device)
            label = torch.tensor([ground_truth]).to(device)
            lm_loss, logits = model(inputs, label)

        model.zero_grad()
        complement = 1.0 - logits
        logits = torch.cat((complement, logits), dim=1)
        preds = logits[:, 1] > 0.5

    elif args.model in ['gpt2', 'gpt2-medium', 'codegpt']:
        example_sample = ' '.join(source.split())
        code_tokens = tokenizer.tokenize(example_sample)[:512 - 2]
        source_tokens = ["<|endoftext|>"] + code_tokens + ["<|endoftext|>"]
        example_sample_source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = 512 - len(example_sample_source_ids)
        example_sample_source_ids += [50255] * padding_length
        with torch.no_grad():
            inputs = torch.tensor([example_sample_source_ids]).to(device)
            label = torch.tensor([ground_truth]).to(device)
            lm_loss, logits = model(inputs, label)

        model.zero_grad()
        complement = 1.0 - logits
        logits = torch.cat((complement, logits), dim=1)
        preds = logits[:, 1] > 0.5

    elif args.model in ['qwen0.5b', 'qwen1.5b']:
        example_sample = ' '.join(source.split())

        result = tokenizer(
            example_sample,
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        source_ids = result["input_ids"].to(device)
        attention_mask = result["attention_mask"].to(device)

        with torch.no_grad():
            label = torch.tensor([ground_truth]).to(device)
            lm_loss, logits = model(source_ids, attention_mask, label)

        model.zero_grad()
        complement = 1.0 - logits
        logits = torch.cat((complement, logits), dim=1)
        preds = logits[:, 1] > 0.5

    elif args.model in ['graphcodebert']:

        '''这里需要注意，不需要删除\n'''
        example_sample = source

        example_sample_source_ids, position_idx, attn_mask = convert_examples_to_features_adv(example_sample,
                                                                                              tokenizer,
                                                                                              args)

        with torch.no_grad():
            inputs = torch.tensor([example_sample_source_ids]).to(device)
            label = torch.tensor([ground_truth]).to(device)
            lm_loss, logits = model(inputs, position_idx, attn_mask, label)

        model.zero_grad()
        complement = 1.0 - logits
        logits = torch.cat((complement, logits), dim=1)
        preds = logits[:, 1] > 0.5

    return preds, lm_loss, logits


def load_embed(args, ori_sample, tokenizer, embedding_layer, model, device):
    if args.model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
        ori_sample = ' '.join(ori_sample.split())
        source_ids = tokenizer.encode(ori_sample, max_length=args.max_source_length, padding='max_length',
                                      truncation=True)

        source_ids_ori = torch.tensor([source_ids], dtype=torch.long).to(device)
        input_embeds = embedding_layer(source_ids_ori)
        embeddings_sample1 = model.get_hidden_states(source_ids_ori, input_embeds).detach()
        print(embeddings_sample1.size())
        embedding1_mean = embeddings_sample1.mean(dim=1).detach()
        # embedding1_mean = embeddings_sample1

    elif args.model in ['qwen0.5b', 'qwen1.5b']:
        ori_sample = ' '.join(ori_sample.split())

        result = tokenizer(
            ori_sample,
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        source_ids = result["input_ids"].to(device)
        attention_mask = result["attention_mask"].to(device)

        input_embeds = embedding_layer(source_ids)
        embeddings_sample1 = model.get_hidden_states(input_embeds, attention_mask).detach()
        print(embeddings_sample1.size())
        embedding1_mean = embeddings_sample1.mean(dim=1).detach()

    elif args.model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:
        ori_sample = ' '.join(ori_sample.split())
        code_tokens = tokenizer.tokenize(ori_sample)[:512 - 2]
        source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = 512 - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length

        source_ids_ori = torch.tensor([source_ids], dtype=torch.long).to(device)
        input_embeds = embedding_layer(source_ids_ori)
        embeddings_sample1 = model.get_hidden_states(source_ids_ori, input_embeds).detach()
        embedding1_mean = embeddings_sample1.mean(dim=1).detach()
        # embedding1_mean = embeddings_sample1

    elif args.model in ['gpt2', 'gpt2-medium', 'codegpt']:
        ori_sample = ' '.join(ori_sample.split())
        code_tokens = tokenizer.tokenize(ori_sample)[:512 - 2]
        source_tokens = ["<|endoftext|>"] + code_tokens + ["<|endoftext|>"]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        padding_length = 512 - len(source_ids)
        source_ids += [50255] * padding_length

        source_ids_ori = torch.tensor([source_ids], dtype=torch.long).to(device)
        input_embeds = embedding_layer(source_ids_ori)
        embeddings_sample1 = model.get_hidden_states(source_ids_ori, input_embeds).detach()
        embedding1_mean = embeddings_sample1.mean(dim=1).detach()
        # embedding1_mean = embeddings_sample1

    elif args.model in ['graphcodebert']:
        # ori_sample = ' '.join(ori_sample.split())
        source_ids, position_idx, attn_mask = convert_examples_to_features_adv(ori_sample, tokenizer, args)
        source_ids_ori = torch.tensor([source_ids], dtype=torch.long).to(device)

        inputs_embeds = model.encoder.roberta.embeddings.word_embeddings(source_ids_ori)
        embeddings_sample1 = model.get_hidden_states(inputs_embeds, position_idx,
                                                     attn_mask).detach()
        embedding1_mean = embeddings_sample1.mean(dim=1).detach()
        # embedding1_mean = embeddings_sample1

    return embeddings_sample1, embedding1_mean
