# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.query = 0

    def get_hidden_states(self, input_embeds, attention_mask):
        outputs = self.encoder(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        # seq_lengths = attention_mask.sum(1) - 1
        # seq_lengths = seq_lengths.to(outputs.last_hidden_state.device)
        # batch_indices = torch.arange(input_embeds.size(0), device=outputs.last_hidden_state.device)
        # hidden_states = outputs.last_hidden_state[batch_indices, seq_lengths]
        return hidden_states

    def get_embeddings(self, input_embeds, attention_mask, labels):
        outputs = self.encoder(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True)
        # pooled_output = outputs.last_hidden_state[:, -1, :]
        seq_lengths = attention_mask.sum(1) - 1
        seq_lengths = seq_lengths.to(outputs.last_hidden_state.device)
        batch_indices = torch.arange(input_embeds.size(0), device=outputs.last_hidden_state.device)
        pooled_output = outputs.last_hidden_state[batch_indices, seq_lengths]

        logits = self.classifier(pooled_output)
        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob

    def forward(self, input_ids, att_mask, labels):
        outputs = self.encoder(input_ids, attention_mask=att_mask)

        seq_lengths = att_mask.sum(1) - 1
        seq_lengths = seq_lengths.to(outputs.last_hidden_state.device)
        batch_indices = torch.arange(input_ids.size(0), device=outputs.last_hidden_state.device)
        pooled_output = outputs.last_hidden_state[batch_indices, seq_lengths]

        logits = self.classifier(pooled_output)
        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob

    def forward_classifer(self, input_ids, att_mask):
        outputs = self.encoder(input_ids, attention_mask=att_mask)
        sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(outputs.last_hidden_state.device)

        batch_indices = torch.arange(input_ids.size(0), device=outputs.last_hidden_state.device)
        pooled_output = outputs.last_hidden_state[batch_indices, sequence_lengths]
        logits = self.classifier(pooled_output)
        prob = F.sigmoid(logits)
        return prob

    def get_results(self, dataset, batch_size):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=4,
                                     pin_memory=False)

        self.eval()
        logits = []
        labels = []
        for batch in eval_dataloader:
            inputs_ids = batch[0].to("cuda")
            att_mask = batch[1].to("cuda")
            label = batch[2].to("cuda")
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs_ids, att_mask, label)
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())

        logits = np.concatenate(logits, 0)
        labels = np.concatenate(labels, 0)

        probs = [[1 - prob[0], prob[0]] for prob in logits]
        pred_labels = [1 if label else 0 for label in logits[:, 0] > 0.5]

        return probs, pred_labels

    def compute(self, dataset, batch_size):
        '''Given a dataset, return probabilities and labels.'''
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=4,
                                     pin_memory=False)

        self.eval()
        logits = []
        labels = []
        for batch in eval_dataloader:
            inputs_ids = batch[0].to("cuda")
            att_mask = batch[1].to("cuda")
            label = batch[2].to("cuda")
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs_ids, att_mask, label)
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())

        logits = np.concatenate(logits, 0)
        labels = np.concatenate(labels, 0)

        probs = [[1 - prob[0], prob[0]] for prob in logits]
        pred_labels = [1 if label else 0 for label in logits[:, 0] > 0.5]

        return probs, pred_labels, labels
