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
        self.encoder.score = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

    def get_gpt_hidden_states(self, input_ids=None, input_embeds=None):
        outputs = self.encoder(inputs_embeds=input_embeds, attention_mask=input_ids.ne(50255),
                               output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        return hidden_states

    def get_gpt_embeddings(self, input_ids=None, input_embeds=None, labels=None):
        outputs = self.encoder(inputs_embeds=input_embeds, attention_mask=input_ids.ne(50255),
                               output_hidden_states=True)[0]
        logits = outputs  # 4*1
        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob

    def forward(self, input_ids=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(50255))[0]
        logits = outputs  # 4*1
        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob
