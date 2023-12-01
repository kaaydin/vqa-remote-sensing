from torchvision import models as torchmodels
import torch.nn as nn
import torch.nn.functional as F
import torch
from attention import SelfAttentionQuestion, CrossAttention, SelfAttentionImage

VISUAL_OUT = 768
QUESTION_OUT = 768
HIDDEN_DIMENSION_ATTENTION = 512
HIDDEN_DIMENSION_CROSS = 5000
FUSION_IN = 1200
FUSION_HIDDEN = 256
DROPOUT_V = 0.5
DROPOUT_Q = 0.5
DROPOUT_F = 0.5

class VQAModel(nn.Module):
    def __init__(self):
        
        super(VQAModel, self).__init__()
        
        self.num_classes = 95
        
        ## Dropouts
        self.dropoutV = torch.nn.Dropout(DROPOUT_V)
        self.dropoutQ = torch.nn.Dropout(DROPOUT_Q)
        self.dropoutF = torch.nn.Dropout(DROPOUT_F)

        ## Attention Modules
        self.selfattention_q = SelfAttentionQuestion(QUESTION_OUT, HIDDEN_DIMENSION_ATTENTION, 1)
        self.crossattention = CrossAttention(HIDDEN_DIMENSION_CROSS, QUESTION_OUT, VISUAL_OUT)
        self.selfattention_v = SelfAttentionImage(HIDDEN_DIMENSION_CROSS, HIDDEN_DIMENSION_ATTENTION, 1)

        ## Prepare Fusion
        self.linear_q = nn.Linear(QUESTION_OUT, FUSION_IN)
        self.linear_v = nn.Linear(VISUAL_OUT, FUSION_IN)
        
        ## Classification layers
        self.linear_classif1 = nn.Linear(FUSION_IN, FUSION_HIDDEN)
        self.linear_classif2 = nn.Linear(FUSION_HIDDEN, self.num_classes)

        
    def forward(self, input_v, input_q):

        ## Dropouts
        input_q = self.dropoutQ(input_q)
        input_v = self.dropoutV(input_v)
        
        ## Self-Attention for Question
        q = self.selfattention_q(input_q)
        c = self.crossattention(q, input_v)
        v = self.selfattention_v(c, input_v)
    
        ## Prepare fusion
        q = self.linear_q(q)
        q = nn.Tanh()(q)
        v = self.linear_v(v)
        v = nn.Tanh()(v)

        ## Fusion & Classification         
        x = torch.mul(v, q)
        x = torch.squeeze(x, 1)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)
        
        return x