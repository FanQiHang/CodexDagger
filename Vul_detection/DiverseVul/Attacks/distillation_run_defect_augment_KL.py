from __future__ import absolute_import
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer, GPTNeoXConfig, GPTNeoXTokenizerFast,
                          GPTNeoXForCausalLM,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer, T5Config,
                          T5ForConditionalGeneration, T5Tokenizer, Qwen2Config, AutoModel, AutoTokenizer)

MODEL_CLASSES = {
    'polycoder': (GPTNeoXConfig, GPTNeoXForCausalLM, GPT2Tokenizer),
    'gpt2': (GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
    'qwen': (Qwen2Config, AutoModel, AutoTokenizer)
}

import multiprocessing
import time
from distillation_configs import add_args, set_seed
from code_distill_datasets.distillation_utils_codet5 import load_and_cache_defect_data
from code_distill_datasets.distillation_utils_codet5 import load_and_cache_defect_data_query, \
    load_and_cache_defect_data_dist
from code_distill_datasets.distillation_utils_codegpt import TextDataset, TextDataset_query, TextDataset_dist
# Qwen2.5
from code_distill_datasets.distillation_utils_qwen import TextDataset_qwen, TextDataset_query_qwen, \
    TextDataset_dist_qwen
# from code_distill_datasets.distillation_utils_polycoder import TextDataset_polycoder, TextDataset_polycoder_query, \
#     TextDataset_polycoder_dist
from code_distill_datasets.distillation_utils_codebert import TextDataset_bert, TextDataset_bert_query, \
    TextDataset_bert_dist
from code_distill_datasets.distillation_utils_graphcodebert import TextDataset_graphcodebert, \
    TextDataset_graphcodebert_query, TextDataset_graphcodebert_dist
from code_datasets.filenames import get_filenames
from load_predicts import load_predict_distill
from load_models import load_model_distillation, load_model, load_distill_model_nonsoftmax
from process_datasets import process_dataset

cpu_cont = multiprocessing.cpu_count()
import torch.nn.functional as F
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def evaluate(args, model, teacher_model, eval_data, teacher_eval_data, write_to_pred=False):
    '''Teacher'''
    teacher_eval_sampler = SequentialSampler(teacher_eval_data)
    teacher_eval_dataloader = DataLoader(teacher_eval_data, sampler=teacher_eval_sampler,
                                         batch_size=args.eval_batch_size)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    eval_loss = 0.0
    kl_eval_loss = 0.0
    all_loss = 0.0
    teacher_eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    teacher_model.eval()
    logits = []
    labels_ls = []
    logits_teacher = []

    T = args.temperature
    device = args.device

    for batch, teacher_batch in zip(eval_dataloader, teacher_eval_dataloader):

        if args.victim_model == 'graphcodebert':
            (teacher_source_ids, position_idx, attn_mask,
             teacher_labels) = tuple(t.to(args.device) for t in teacher_batch)
            with torch.no_grad():
                teacher_loss, teacher_logits, teacher_no_sigmod_logits = teacher_model(teacher_source_ids,
                                                                                       position_idx,
                                                                                       attn_mask,
                                                                                       teacher_labels)
        elif args.victim_model in ['qwen0.5b', 'qwen1.5b']:

            (teacher_source_ids, attn_mask,
             teacher_labels) = tuple(t.to(args.device) for t in teacher_batch)
            with torch.no_grad():
                teacher_loss, teacher_logits, teacher_no_sigmod_logits = teacher_model(teacher_source_ids, attn_mask,
                                                                                       teacher_labels)

        else:
            teacher_batch = tuple(t.to(device) for t in teacher_batch)
            teacher_source_ids, teacher_labels = teacher_batch

            with torch.no_grad():
                teacher_loss, teacher_logits, teacher_no_sigmod_logits = teacher_model(teacher_source_ids,
                                                                                       teacher_labels)

        predict_labels = torch.argmax(teacher_logits, dim=1)

        if args.attack_model == 'graphcodebert':
            (source_ids, position_idx, attn_mask,
             labels) = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                hard_loss, student_logits, student_no_sigmod_logits = model(source_ids, position_idx, attn_mask,
                                                                            predict_labels)
        elif args.attack_model in ['qwen0.5b', 'qwen1.5b']:

            (source_ids, attn_mask,
             labels) = tuple(t.to(args.device) for t in teacher_batch)
            with torch.no_grad():
                hard_loss, student_logits, student_no_sigmod_logits = teacher_model(source_ids, attn_mask, labels)
        else:
            batch = tuple(t.to(device) for t in batch)
            source_ids, labels = batch
            with torch.no_grad():
                hard_loss, student_logits, student_no_sigmod_logits = model(source_ids, predict_labels)

        if args.attack_model in ['gpt2', 'gpt2-medium', 'codegpt', 'codebert', 'roberta-large',
                                 'roberta-base', 'graphcodebert', 'codebert-insecure', 'qwen0.5b', 'qwen1.5b']:
            complement = 1.0 - student_logits
            student_logits_output = torch.cat((complement, student_logits), dim=1)

        elif args.attack_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
            student_logits_output = F.softmax(student_logits)

        else:
            raise ValueError

        if args.victim_model in ['gpt2', 'gpt2-medium', 'codegpt', 'codebert', 'roberta-large',
                                 'roberta-base', 'graphcodebert', 'codebert-insecure', 'qwen0.5b', 'qwen1.5b']:
            complement = 1.0 - teacher_logits
            teacher_logits_output = torch.cat((complement, teacher_logits), dim=1)

        elif args.victim_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
            teacher_logits_output = F.softmax(teacher_logits)

        else:
            raise ValueError

        if args.loss_name == 'kl':

            if args.victim_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:

                if args.attack_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:

                    prob_teacher = F.softmax(teacher_logits / T, dim=1)
                    log_prob_student = F.log_softmax(student_logits / T, dim=1)
                    soft_loss = F.kl_div(
                        log_prob_student,
                        prob_teacher,
                        reduction="batchmean",
                    ) * (T ** 2)

                else:

                    prob_teacher = F.softmax(teacher_logits / T, dim=1)

                    student_logits_scaled = student_no_sigmod_logits / T

                    student_logits = torch.sigmoid(student_logits_scaled.squeeze(1))

                    student_probs = torch.stack([1 - student_logits, student_logits], dim=1)

                    student_log_probs = torch.log(student_probs + 1e-12)

                    soft_loss = F.kl_div(
                        student_log_probs,
                        prob_teacher,
                        reduction="batchmean",
                    ) * (T ** 2)

            elif args.victim_model in ['gpt2', 'gpt2-medium', 'codegpt', 'codebert', 'roberta-large',
                                       'roberta-base', 'graphcodebert', 'codebert-insecure', 'qwen0.5b', 'qwen1.5b']:

                if args.attack_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:

                    log_prob_student = F.log_softmax(student_logits / T, dim=1)

                    teacher_logits_scaled = teacher_no_sigmod_logits / T

                    teacher_logits = torch.sigmoid(teacher_logits_scaled.squeeze(1))

                    teacher_probs = torch.stack([1 - teacher_logits, teacher_logits], dim=1)

                    soft_loss = F.kl_div(
                        log_prob_student,
                        teacher_probs,
                        reduction="batchmean",
                    ) * (T ** 2)

                else:

                    epsilon = 1e-12

                    teacher_probs = torch.sigmoid(teacher_no_sigmod_logits / T)
                    teacher_probs = torch.clamp(teacher_probs, epsilon, 1 - epsilon)

                    student_probs = torch.sigmoid(student_no_sigmod_logits / T)
                    student_probs = torch.clamp(student_probs, epsilon, 1 - epsilon)

                    teacher_probs_2d = torch.stack([teacher_probs, 1 - teacher_probs], dim=1)
                    student_log_probs_2d = torch.log(
                        torch.stack([student_probs, 1 - student_probs], dim=1)
                    )

                    soft_loss = F.kl_div(
                        input=student_log_probs_2d,
                        target=teacher_probs_2d,
                        reduction='batchmean',
                        log_target=False
                    ) * (T ** 2)

        elif args.loss_name == 'mae':

            if args.victim_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:

                if args.attack_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:

                    student_softmax_logits = F.softmax(student_logits / T, dim=1)
                    teacher_softmax_logits = F.softmax(teacher_logits / T, dim=1)

                    soft_loss = F.l1_loss(student_softmax_logits, teacher_softmax_logits) * T

                else:

                    teacher_probs = F.softmax(teacher_logits / T, dim=1)
                    teacher_positive_probs = teacher_probs[:, 1]

                    student_probs = torch.sigmoid(student_no_sigmod_logits / T)

                    soft_loss = F.l1_loss(student_probs, teacher_positive_probs) * T


            elif args.victim_model in ['gpt2', 'gpt2-medium', 'codegpt', 'codebert', 'roberta-large',
                                       'roberta-base', 'graphcodebert', 'codebert-insecure', 'qwen0.5b', 'qwen1.5b']:

                if args.attack_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:

                    student_probs = F.softmax(student_logits / T, dim=1)
                    student_positive_probs = student_probs[:, 1]

                    teacher_probs = torch.sigmoid(teacher_no_sigmod_logits / T)

                    soft_loss = F.l1_loss(student_positive_probs, teacher_probs) * T

                else:

                    student_probs = torch.sigmoid(student_no_sigmod_logits / T)

                    teacher_probs = torch.sigmoid(teacher_no_sigmod_logits / T)

                    soft_loss = F.l1_loss(student_probs, teacher_probs) * T

                    # soft_loss = F.l1_loss(student_no_sigmod_logits / T, teacher_no_sigmod_logits / T) * T

        else:
            raise ValueError

        loss = (1 - args.alpha) * hard_loss + args.alpha * soft_loss

        logits.append(student_logits_output.cpu().numpy())
        logits_teacher.append(teacher_logits_output.cpu().numpy())
        labels_ls.append(labels.cpu().numpy())

        eval_loss += hard_loss.mean().item()
        kl_eval_loss += soft_loss.mean().item()
        all_loss += loss.mean().item()
        teacher_eval_loss += teacher_loss.mean().item()

        nb_eval_steps += 1

    logits = np.concatenate(logits, 0)
    labels_ls = np.concatenate(labels_ls, 0)
    logits_teacher = np.concatenate(logits_teacher, 0)

    preds = logits[:, 1] > 0.5
    preds_teacher = logits_teacher[:, 1] > 0.5

    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(preds, preds_teacher)

    eval_acc_teacher = np.mean(labels_ls == preds_teacher)
    eval_acc = np.mean(labels_ls == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    from sklearn.metrics import confusion_matrix, f1_score
    tn, fp, fn, tp = confusion_matrix(labels_ls, preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = f1_score(labels_ls, preds)
    from sklearn.metrics import roc_curve, auc
    fpr_, tpr_, thresholds = roc_curve(labels_ls, logits[:, 1])
    fnr_ls = 1 - tpr_
    index = np.argmin(np.abs(fpr_ - 0.1))
    fnr_at_fpr_0_1 = fnr_ls[index]
    index = np.argmin(np.abs(fpr_ - 0.01))
    fnr_at_fpr_0_01 = fnr_ls[index]
    roc_auc = auc(fpr_, tpr_)
    file = os.path.join(args.output_dir, str(args.temperature) + '_' + str(
        args.alpha) + '_' + args.is_sample_20 + '_' + args.loss_name + '_' + str(
        args.sample_codebleu_budget) + '_' + str(args.sample_num) + '_' + str(
        args.model_size) + '_roc_data.npz')
    np.savez(file, fpr_=fpr_, tpr_=tpr_)

    tn, fp, fn, tp = confusion_matrix(labels_ls, preds, labels=[1, 0]).ravel()
    fpr_0 = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr_0 = fn / (fn + tp) if (fn + tp) > 0 else 0
    recall_0 = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision_0 = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_0 = f1_score(labels_ls, preds, pos_label=0)
    fpr_, tpr_, thresholds = roc_curve(labels_ls, 1 - logits[:, 1], pos_label=0)
    fnr_ls = 1 - tpr_
    index = np.argmin(np.abs(fpr_ - 0.1))
    fnr_at_fpr_0_1_ = fnr_ls[index]
    index = np.argmin(np.abs(fpr_ - 0.01))
    fnr_at_fpr_0_01_ = fnr_ls[index]

    roc_auc_0 = auc(fpr_, tpr_)
    file = os.path.join(args.output_dir, str(args.temperature) + '_' + str(
        args.alpha) + '_' + args.is_sample_20 + '_' + args.loss_name + '_' + str(
        args.sample_codebleu_budget) + '_' + str(args.sample_num) + '_' + str(
        args.model_size) + '_roc_data_0.npz')
    np.savez(file, fpr_=fpr_, tpr_=tpr_)

    result = {
        "kappa": round(kappa, 4),
        args.loss_name + "_eval_loss": float(torch.tensor(kl_eval_loss) / nb_eval_steps),
        "eval_acc": round(eval_acc, 4),
        "eval_acc_teacher": round(eval_acc_teacher, 4),
        "eval_loss": float(perplexity),
        "eval_fpr": round(fpr, 4),
        "eval_fnr": round(fnr, 4),
        "eval_recall": round(recall, 4),
        "eval_precision": round(precision, 4),
        "eval_auc": round(roc_auc, 4),
        "f1-score": round(f1, 4),
        "eval_fpr_0": round(fpr_0, 4),
        "eval_fnr_0": round(fnr_0, 4),
        "eval_recall_0": round(recall_0, 4),
        "eval_precision_0": round(precision_0, 4),
        "f1-score_0": round(f1_0, 4),
        "eval_auc_0": round(roc_auc_0, 4),
        "fnr_at_fpr_0_1": round(fnr_at_fpr_0_1, 4),
        "fnr_at_fpr_0_01": round(fnr_at_fpr_0_01, 4),
        "fnr_at_fpr_0_1_0": round(fnr_at_fpr_0_1_, 4),
        "fnr_at_fpr_0_01_0": round(fnr_at_fpr_0_01_, 4),
        "teacher_eval_loss": float(torch.tensor(teacher_eval_loss) / nb_eval_steps),
        "all_loss": float(torch.tensor(all_loss) / nb_eval_steps),
    }
    print('Evaluating...result', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    t0 = time.time()
    args = add_args(parser)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 3
        # if args.is_sample_20 == 'no':
        #     args.n_gpu = 2
        # args.n_gpu = 2
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    # device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    # args.n_gpu = 1

    args.device = device
    set_seed(args)

    '''Student Model'''
    model, tokenizer = load_model_distillation(args.attack_model, args.model_size, device, args)

    pool = multiprocessing.Pool(cpu_cont)

    args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.victim_model,
                                                                               args=args)

    '''Teacher Model'''
    teacher_model, teacher_tokenizer = load_distill_model_nonsoftmax(args.victim_model, device, args)
    idx = load_predict_distill(args.victim_model, args)

    if args.do_train:

        if args.n_gpu > 1:
            # multi-gpu training
            model = torch.nn.DataParallel(model)
            teacher_model = torch.nn.DataParallel(teacher_model)

        # if args.local_rank in [-1, 0] and args.data_num == -1:
        if args.local_rank in [-1, 0]:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            # tb_writer = SummaryWriter(summary_fn)

        if args.is_sample_20 == 'yes':

            if args.attack_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
                train_examples, train_data = load_and_cache_defect_data_query(args, args.train_filename,
                                                                              pool, tokenizer,
                                                                              'none', idx,
                                                                              is_sample=False)

            elif args.attack_model in ['gpt2', 'gpt2-medium', 'codegpt']:
                train_data = TextDataset_query(tokenizer, args, idx, args.train_filename)

            elif args.attack_model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:
                train_data = TextDataset_bert_query(tokenizer, args, idx, args.train_filename)

            elif args.attack_model in ['graphcodebert']:
                train_data = TextDataset_graphcodebert_query(tokenizer, args, idx, args.train_filename)

            elif args.attack_model in ['qwen0.5b', 'qwen1.5b']:

                train_data = TextDataset_query_qwen(tokenizer, args, idx, args.train_filename)
            else:
                raise ValueError

            if args.victim_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
                teacher_train_examples, teacher_train_data = load_and_cache_defect_data_query(args, args.train_filename,
                                                                                              pool, teacher_tokenizer,
                                                                                              'none', idx,
                                                                                              is_sample=False)
            elif args.victim_model in ['gpt2', 'gpt2-medium', 'codegpt']:
                teacher_train_data = TextDataset_query(teacher_tokenizer, args, idx, args.train_filename)

            elif args.victim_model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:
                teacher_train_data = TextDataset_bert_query(teacher_tokenizer, args, idx, args.train_filename)

            elif args.victim_model == 'graphcodebert':
                teacher_train_data = TextDataset_graphcodebert_query(teacher_tokenizer, args, idx, args.train_filename)

            elif args.victim_model in ['qwen0.5b', 'qwen1.5b']:
                teacher_train_data = TextDataset_query_qwen(teacher_tokenizer, args, idx, args.train_filename)
            else:
                raise ValueError

        elif args.is_sample_20 == 'no':

            if args.attack_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
                _, train_data = load_and_cache_defect_data_dist(args, args.test_filename,
                                                                pool,
                                                                tokenizer,
                                                                'none',
                                                                idx,
                                                                is_sample=False)

            elif args.attack_model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:

                train_data = TextDataset_bert_dist(tokenizer, args, idx, args.test_filename)

            elif args.attack_model in ['graphcodebert']:

                train_data = TextDataset_graphcodebert_dist(tokenizer, args, idx, args.test_filename)

            elif args.attack_model in ['gpt2', 'gpt2-medium', 'codegpt']:

                train_data = TextDataset_dist(tokenizer, args, idx, args.test_filename)

            elif args.attack_model in ['qwen0.5b', 'qwen1.5b']:
                train_data = TextDataset_dist_qwen(tokenizer, args, idx, args.test_filename)

            else:
                raise ValueError

            if args.victim_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
                teacher_train_examples, teacher_train_data = load_and_cache_defect_data_dist(args, args.test_filename,
                                                                                             pool,
                                                                                             teacher_tokenizer,
                                                                                             'none',
                                                                                             idx,
                                                                                             is_sample=False)

            elif args.victim_model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:

                teacher_train_data = TextDataset_bert_dist(teacher_tokenizer, args, idx, args.test_filename)

            elif args.victim_model in ['graphcodebert']:

                teacher_train_data = TextDataset_graphcodebert_dist(teacher_tokenizer, args, idx, args.test_filename)

            elif args.victim_model in ['gpt2', 'gpt2-medium', 'codegpt']:

                teacher_train_data = TextDataset_dist(teacher_tokenizer, args, idx, args.test_filename)

            elif args.victim_model in ['qwen0.5b', 'qwen1.5b']:

                teacher_train_data = TextDataset_dist_qwen(teacher_tokenizer, args, idx, args.test_filename)

            else:
                raise ValueError

        if args.local_rank == -1:
            # train_sampler = RandomSampler(train_data)
            train_sampler = SequentialSampler(train_data)
            teacher_train_sampler = SequentialSampler(teacher_train_data)
        else:
            train_sampler = DistributedSampler(train_data, shuffle=False)
            teacher_train_sampler = DistributedSampler(teacher_train_data, shuffle=False)

        print(len(train_data), len(teacher_train_data))

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        '''Teacher Model'''
        teacher_train_dataloader = DataLoader(teacher_train_data, sampler=teacher_train_sampler,
                                              batch_size=args.train_batch_size)

        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        save_steps = max(len(train_dataloader), 1)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        if args.warmup_steps < 1:
            warmup_steps = num_train_optimization_steps * args.warmup_steps
        else:
            warmup_steps = int(args.warmup_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        global_step, best_acc = 0, np.inf
        not_acc_inc_cnt = 0
        is_early_stop = False
        T = args.temperature
        alpha = args.alpha
        args.data_num = -1

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = zip(train_dataloader, teacher_train_dataloader)
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            print('cur_epoch...', cur_epoch)
            model.train()

            # temp_scheduler.update(cur_epoch, int(args.num_train_epochs))

            for step, (batch, teacher_batch) in enumerate(bar):

                # print('cur_step...', step)

                if args.victim_model == 'graphcodebert':
                    (teacher_source_ids, position_idx, attn_mask,
                     teacher_labels) = tuple(t.to(args.device) for t in teacher_batch)
                    with torch.no_grad():
                        teacher_loss, teacher_logits, teacher_no_sigmod_logits = teacher_model(teacher_source_ids,
                                                                                               position_idx,
                                                                                               attn_mask,
                                                                                               teacher_labels)
                elif args.victim_model in ['qwen0.5b', 'qwen1.5b']:

                    (teacher_source_ids, attn_mask,
                     teacher_labels) = tuple(t.to(args.device) for t in teacher_batch)
                    with torch.no_grad():
                        teacher_loss, teacher_logits, teacher_no_sigmod_logits = teacher_model(teacher_source_ids,
                                                                                               attn_mask,
                                                                                               teacher_labels)
                else:
                    teacher_batch = tuple(t.to(device) for t in teacher_batch)
                    teacher_source_ids, teacher_labels = teacher_batch

                    with torch.no_grad():
                        teacher_loss, teacher_logits, teacher_no_sigmod_logits = teacher_model(teacher_source_ids,
                                                                                               teacher_labels)

                predict_labels = torch.argmax(teacher_logits, dim=1)

                if args.attack_model == 'graphcodebert':
                    (source_ids, position_idx, attn_mask,
                     labels) = tuple(t.to(args.device) for t in batch)
                    hard_loss, student_logits, student_no_sigmod_logits = model(source_ids, position_idx, attn_mask,
                                                                                predict_labels)

                elif args.attack_model in ['qwen0.5b', 'qwen1.5b']:

                    (source_ids, attn_mask,
                     labels) = tuple(t.to(args.device) for t in batch)
                    hard_loss, student_logits, student_no_sigmod_logits = model(source_ids, attn_mask,
                                                                                predict_labels)
                else:
                    batch = tuple(t.to(device) for t in batch)
                    source_ids, labels = batch
                    hard_loss, student_logits, student_no_sigmod_logits = model(source_ids, predict_labels)

                if args.loss_name == 'kl':

                    if args.victim_model in ['codet5-small', 'codet5-base', 'codet5-base-multi', 'flan-t5-small']:

                        if args.attack_model in ['codet5-small', 'codet5-base', 'codet5-base-multi', 'flan-t5-small']:

                            prob_teacher = F.softmax(teacher_logits / T, dim=1)
                            log_prob_student = F.log_softmax(student_logits / T, dim=1)
                            soft_loss = F.kl_div(
                                log_prob_student,
                                prob_teacher,
                                reduction="batchmean",
                            ) * (T ** 2)

                        else:

                            prob_teacher = F.softmax(teacher_logits / T, dim=1)

                            student_logits_scaled = student_no_sigmod_logits / T

                            student_logits = torch.sigmoid(student_logits_scaled.squeeze(1))

                            student_probs = torch.stack([1 - student_logits, student_logits], dim=1)

                            student_log_probs = torch.log(student_probs + 1e-12)

                            soft_loss = F.kl_div(
                                student_log_probs,
                                prob_teacher,
                                reduction="batchmean",
                            ) * (T ** 2)

                    elif args.victim_model in ['gpt2', 'gpt2-medium', 'codegpt', 'codebert', 'roberta-large',
                                               'roberta-base', 'graphcodebert', 'codebert-insecure', 'qwen0.5b',
                                               'qwen1.5b']:

                        if args.attack_model in ['codet5-small', 'codet5-base', 'codet5-base-multi', 'flan-t5-small']:

                            log_prob_student = F.log_softmax(student_logits / T, dim=1)

                            teacher_logits_scaled = teacher_no_sigmod_logits / T

                            teacher_logits = torch.sigmoid(teacher_logits_scaled.squeeze(1))

                            teacher_probs = torch.stack([1 - teacher_logits, teacher_logits], dim=1)

                            soft_loss = F.kl_div(
                                log_prob_student,
                                teacher_probs,
                                reduction="batchmean",
                            ) * (T ** 2)

                        else:

                            epsilon = 1e-12

                            teacher_probs = torch.sigmoid(teacher_no_sigmod_logits / T)
                            teacher_probs = torch.clamp(teacher_probs, epsilon, 1 - epsilon)

                            student_probs = torch.sigmoid(student_no_sigmod_logits / T)
                            student_probs = torch.clamp(student_probs, epsilon, 1 - epsilon)

                            teacher_probs_2d = torch.stack([teacher_probs, 1 - teacher_probs], dim=1)
                            student_log_probs_2d = torch.log(
                                torch.stack([student_probs, 1 - student_probs], dim=1)
                            )
                            soft_loss = F.kl_div(
                                input=student_log_probs_2d,
                                target=teacher_probs_2d,
                                reduction='batchmean',
                                log_target=False
                            ) * (T ** 2)

                elif args.loss_name == 'mae':

                    if args.victim_model in ['codet5-small', 'codet5-base', 'codet5-base-multi', 'flan-t5-small']:

                        if args.attack_model in ['codet5-small', 'codet5-base', 'codet5-base-multi', 'flan-t5-small']:

                            student_softmax_logits = F.softmax(student_logits / T, dim=1)
                            teacher_softmax_logits = F.softmax(teacher_logits / T, dim=1)

                            soft_loss = F.l1_loss(student_softmax_logits, teacher_softmax_logits) * T

                        else:  # codegpt

                            teacher_probs = F.softmax(teacher_logits / T, dim=1)  # (batch_size, 2)
                            teacher_positive_probs = teacher_probs[:, 1]

                            student_probs = torch.sigmoid(student_no_sigmod_logits / T)

                            soft_loss = F.l1_loss(student_probs, teacher_positive_probs) * T


                    elif args.victim_model in ['gpt2', 'gpt2-medium', 'codegpt', 'codebert', 'roberta-large',
                                               'roberta-base', 'graphcodebert', 'codebert-insecure', 'qwen0.5b',
                                               'qwen1.5b']:

                        if args.attack_model in ['codet5-small', 'codet5-base', 'codet5-base-multi', 'flan-t5-small']:

                            student_probs = F.softmax(student_logits / T, dim=1)  # (batch_size, 2)
                            student_positive_probs = student_probs[:, 1]

                            teacher_probs = torch.sigmoid(teacher_no_sigmod_logits / T)

                            soft_loss = F.l1_loss(student_positive_probs, teacher_probs) * T

                        else:

                            student_probs = torch.sigmoid(student_no_sigmod_logits / T)

                            teacher_probs = torch.sigmoid(teacher_no_sigmod_logits / T)

                            soft_loss = F.l1_loss(student_probs, teacher_probs) * T

                            # soft_loss = F.l1_loss(student_no_sigmod_logits / T, teacher_no_sigmod_logits / T) * T

                else:
                    raise ValueError

                loss = (1 - alpha) * hard_loss + alpha * soft_loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                    hard_loss = hard_loss.mean()
                    soft_loss = soft_loss.mean()
                    teacher_loss = teacher_loss.mean()

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                if (step + 1) % 100 == 0:
                    print('Training...', 'loss', loss.item(), 'hard_loss', hard_loss.item(), 'teacher_loss',
                          teacher_loss.item(), 'soft_loss', soft_loss.item())

                # print('Training...', 'loss', loss.item(), 'hard_loss', hard_loss.item(), 'teacher_loss',
                #       teacher_loss.item(), 'soft_loss', soft_loss.item())

                if (step + 1) % save_steps == 0 and args.do_eval:
                    # if args.do_eval:

                    print('Training...', 'loss', loss.item(), 'hard_loss', hard_loss.item(), 'teacher_loss',
                          teacher_loss.item(), 'soft_loss', soft_loss.item())

                    torch.cuda.empty_cache()

                    if args.attack_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
                        _, eval_data = load_and_cache_defect_data_dist(args, args.test_filename,
                                                                       pool,
                                                                       tokenizer,
                                                                       'none',
                                                                       idx,
                                                                       is_sample=False)

                    elif args.attack_model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:

                        eval_data = TextDataset_bert_dist(tokenizer, args, idx, args.test_filename)

                    elif args.attack_model in ['graphcodebert']:

                        eval_data = TextDataset_graphcodebert_dist(tokenizer, args, idx, args.test_filename)

                    elif args.attack_model in ['gpt2', 'gpt2-medium', 'codegpt']:

                        eval_data = TextDataset_dist(tokenizer, args, idx, args.test_filename)

                    elif args.attack_model in ['qwen0.5b', 'qwen1.5b']:

                        eval_data = TextDataset_dist_qwen(tokenizer, args, idx, args.test_filename)

                    if args.victim_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
                        teacher_eval_examples, teacher_eval_data = load_and_cache_defect_data_dist(args,
                                                                                                   args.test_filename,
                                                                                                   pool,
                                                                                                   teacher_tokenizer,
                                                                                                   'none',
                                                                                                   idx,
                                                                                                   is_sample=False)

                    elif args.victim_model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:
                        teacher_eval_data = TextDataset_bert_dist(teacher_tokenizer, args, idx, args.test_filename)

                    elif args.victim_model in ['graphcodebert']:
                        teacher_eval_data = TextDataset_graphcodebert_dist(teacher_tokenizer, args, idx,
                                                                           args.test_filename)

                    elif args.victim_model in ['gpt2', 'gpt2-medium', 'codegpt']:
                        teacher_eval_data = TextDataset_dist(teacher_tokenizer, args, idx, args.test_filename)

                    elif args.victim_model in ['qwen0.5b', 'qwen1.5b']:

                        teacher_eval_data = TextDataset_dist_qwen(teacher_tokenizer, args, idx, args.test_filename)

                    else:
                        raise ValueError

                    print('len(eval_data)', len(eval_data))

                    result = evaluate(args, model, teacher_model, eval_data, teacher_eval_data,
                                      write_to_pred=True)

                    eval_acc = result[args.loss_name + '_eval_loss']
                    print(args.loss_name + '_loss', eval_acc)
                    if eval_acc < best_acc:
                        not_acc_inc_cnt = 0

                        best_acc = eval_acc

                        print('best_acc', best_acc, 'cur_epoch', cur_epoch)

                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-acc')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or True:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            if args.valid_or_test == 'test':
                                output_model_file = os.path.join(output_dir, str(args.temperature) + '_' + str(
                                    args.alpha) + '_' + args.is_sample_20 + '_' + args.loss_name + '_' + str(
                                    args.sample_codebleu_budget) + '_' + str(args.sample_num) + '_' + str(
                                    args.model_size) + '_pytorch_model.bin')
                            torch.save(model_to_save.state_dict(), output_model_file)

                    else:
                        not_acc_inc_cnt += 1
                        if not_acc_inc_cnt > args.patience:
                            is_early_stop = True
                            break

                model.train()

            if is_early_stop:
                break

            torch.cuda.empty_cache()

    if args.do_test:
        print('Testing...')

        for criteria in ['best-acc']:

            if args.valid_or_test == 'test':
                file = os.path.join(args.output_dir, 'checkpoint-best-acc/' + str(args.temperature) + '_' + str(
                    args.alpha) + '_' + args.is_sample_20 + '_' + args.loss_name + '_' + str(
                    args.sample_codebleu_budget) + '_' + str(args.sample_num) + '_' + str(
                    args.model_size) + '_pytorch_model.bin')

            print("Reload model from {}".format(file))

            if hasattr(model, 'module'):
                model.module.load_state_dict(torch.load(file))
            else:
                model.load_state_dict(torch.load(file))

            if args.n_gpu > 1:
                # multi-gpu training
                model = torch.nn.DataParallel(model)
                teacher_model = torch.nn.DataParallel(teacher_model)

            model.to(device)
            teacher_model.to(device)

            if args.attack_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
                _, test_data = load_and_cache_defect_data_dist(args, args.test_filename,
                                                               pool,
                                                               tokenizer,
                                                               'none',
                                                               idx,
                                                               is_sample=False)

            elif args.attack_model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:

                test_data = TextDataset_bert_dist(tokenizer, args, idx, args.test_filename)

            elif args.attack_model in ['graphcodebert']:

                test_data = TextDataset_graphcodebert_dist(tokenizer, args, idx, args.test_filename)

            elif args.attack_model in ['gpt2', 'gpt2-medium', 'codegpt']:

                test_data = TextDataset_dist(tokenizer, args, idx, args.test_filename)

            elif args.attack_model in ['qwen0.5b', 'qwen1.5b']:
                test_data = TextDataset_dist_qwen(tokenizer, args, idx, args.test_filename)

            else:
                raise ValueError

            if args.victim_model in ['codet5-base', 'codet5-small', 'codet5-base-multi', 'flan-t5-small']:
                teacher_test_examples, teacher_test_data = load_and_cache_defect_data_dist(args, args.test_filename,
                                                                                           pool,
                                                                                           teacher_tokenizer,
                                                                                           'none',
                                                                                           idx,
                                                                                           is_sample=False)

            elif args.victim_model in ['codebert', 'roberta-large', 'roberta-base', 'codebert-insecure']:

                teacher_test_data = TextDataset_bert_dist(teacher_tokenizer, args, idx, args.test_filename)

            elif args.victim_model in ['graphcodebert']:

                teacher_test_data = TextDataset_graphcodebert_dist(teacher_tokenizer, args, idx, args.test_filename)

            elif args.victim_model in ['gpt2', 'gpt2-medium', 'codegpt']:

                teacher_test_data = TextDataset_dist(teacher_tokenizer, args, idx, args.test_filename)

            elif args.victim_model in ['qwen0.5b', 'qwen1.5b']:
                teacher_test_data = TextDataset_dist_qwen(teacher_tokenizer, args, idx, args.test_filename)

            else:
                raise ValueError

            result = evaluate(args, model, teacher_model, test_data, teacher_test_data,
                              write_to_pred=True)


if __name__ == "__main__":
    main()
