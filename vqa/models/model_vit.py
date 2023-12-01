#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:55:48 2018

@author: sylvain
"""

from torchvision import models as torchmodels
import torch.nn as nn
import models.seq2vec
import torch.nn.functional as F
import torch

from models.mcb import CompactBilinearPooling

from block import fusions

VISUAL_OUT = 2048
VISUAL_OUT_VIT = 768
QUESTION_OUT = 2400
FUSION_IN = 1200
FUSION_HIDDEN = 256
DROPOUT_V = 0.5
DROPOUT_Q = 0.5
DROPOUT_F = 0.5

class VQAModel(nn.Module):
    def __init__(self):
        
        super(VQAModel, self).__init__()
        self.num_classes = 95
        
        self.dropoutV = torch.nn.Dropout(DROPOUT_V)
        self.dropoutQ = torch.nn.Dropout(DROPOUT_Q)
        self.dropoutF = torch.nn.Dropout(DROPOUT_F)
        
        self.linear_q = nn.Linear(QUESTION_OUT, FUSION_IN)
        self.linear_v = nn.Linear(VISUAL_OUT_VIT, FUSION_IN)

        # Initialize CompactBilinearPooling layer
        # Adjust the input1_size, input2_size, and output_size as needed
        self.multihead_attn = nn.MultiheadAttention(embed_dim=FUSION_IN, num_heads=1, dropout=0.5, batch_first=True)
        self.fusion = fusions.Block([FUSION_IN, FUSION_IN], FUSION_IN)
        
        self.linear_classif1 = nn.Linear(FUSION_IN, FUSION_HIDDEN)
        self.linear_classif2 = nn.Linear(FUSION_HIDDEN, self.num_classes)
        
    def forward(self, input_v, input_q):
        input_v = self.dropoutV(input_v)
        x_v = self.linear_v(input_v)
        x_v = nn.Tanh()(x_v)

        x_q = self.dropoutV(input_q)
        x_q = self.linear_q(x_q)
        x_q = nn.Tanh()(x_q)
        
        #x = torch.mul(x_v, x_q)
        #x = torch.squeeze(x, 1)
        # x = torch.cat((x_v, x_q), 1)
        # x = nn.Tanh()(x)
        x_v_new, _ = self.multihead_attn(query=x_q, key=x_v, value=x_v)
        x = self.fusion([x_v_new, x_q])
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)

        return x
        