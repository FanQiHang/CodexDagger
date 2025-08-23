# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


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

    def get_qwen_hidden_states(self, input_embeds, attention_mask):
        outputs = self.encoder(inputs_embeds=input_embeds, attention_mask=attention_mask)
        hidden_states = outputs.hidden_state[-1]
        # seq_lengths = attention_mask.sum(1) - 1  # 得到每个序列最后一个有效token的位置
        # seq_lengths = seq_lengths.to(outputs.last_hidden_state.device)  # 确保设备一致
        # batch_indices = torch.arange(input_embeds.size(0), device=outputs.last_hidden_state.device)
        # hidden_states = outputs.last_hidden_state[batch_indices, seq_lengths]
        return hidden_states

    def get_qwen_embeddings(self, input_embeds, attention_mask, labels):
        outputs = self.encoder(inputs_embeds=input_embeds, attention_mask=attention_mask)
        # pooled_output = outputs.last_hidden_state[:, -1, :]
        seq_lengths = attention_mask.sum(1) - 1  # 得到每个序列最后一个有效token的位置
        seq_lengths = seq_lengths.to(outputs.last_hidden_state.device)  # 确保设备一致
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

        seq_lengths = att_mask.sum(1) - 1  # 得到每个序列最后一个有效token的位置
        seq_lengths = seq_lengths.to(outputs.last_hidden_state.device)  # 确保设备一致
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
