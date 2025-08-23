# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


# class RobertaClassificationHead(nn.Module):
#     """Head for sentence-level classification tasks."""
#
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         # self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
#         self.out_proj = nn.Sequential(
#             nn.Linear(config.hidden_size, 256),
#             nn.ReLU(),
#             nn.Linear(256, config.num_labels),
#         )
#
#     def forward(self, features, **kwargs):
#         x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = torch.tanh(x)
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         return x


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder

        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        # self.encoder.classifier = RobertaClassificationHead(self.config)

        self.encoder.classifier.out_proj = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, config.num_labels),
        )

    def get_bert_embeddings(self, input_ids=None, input_embeds=None, labels=None):
        outputs = self.encoder(inputs_embeds=input_embeds, attention_mask=input_ids.ne(1),
                               output_hidden_states=True)[0]
        logits = outputs  # 4*1
        prob = F.sigmoid(logits)
        print('prob', prob)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob

    def get_bert_hidden_states(self, input_ids=None, input_embeds=None):
        outputs = self.encoder(inputs_embeds=input_embeds, attention_mask=input_ids.ne(1),
                               output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        return hidden_states

    def forward(self, input_ids=None, labels=None):
        # labels : B
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        logits = outputs  # 4*1
        prob = F.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob
