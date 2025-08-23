from code_datasets.utils_codet5 import load_and_cache_defect_data_codet5
from code_datasets.utils_codegpt import TextDataset_codegpt
from code_datasets.utils_codebert import TextDataset_codebert
from code_datasets.utils_graphcodebert import TextDataset_graphcodebert, TextDataset_graphcodebert_bert
from code_datasets.utils_qwen import TextDataset_qwen


def process_dataset(victim_model, tokenizer, args, file_path):
    if victim_model in ['gpt2', 'gpt2-medium', 'codegpt']:
        data = TextDataset_codegpt(tokenizer, args, file_path)

    elif victim_model in ['qwen0.5b', 'qwen1.5b']:

        data = TextDataset_qwen(tokenizer, args, file_path)

    elif victim_model in ['graphcodebert']:
        data = TextDataset_graphcodebert(tokenizer, args, file_path)

    elif victim_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
        _, data = load_and_cache_defect_data_codet5(tokenizer, args, file_path)

    elif victim_model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:

        data = TextDataset_codebert(tokenizer, args, file_path)

    else:
        raise ValueError

    return data


def process_dataset_bert(victim_model, tokenizer, args, idxx, file_path):
    if victim_model in ['graphcodebert']:
        data = TextDataset_graphcodebert_bert(tokenizer, args, idxx, file_path)

    return data
