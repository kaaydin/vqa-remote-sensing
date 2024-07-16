from torchvision import models as torchmodels
import torch.nn as nn
import torch.nn.functional as F
import torch
from models.attention import SelfAttentionQuestion, CrossAttention, SelfAttentionImage
from block import fusions

class VQAModel(nn.Module):
    def __init__(self, 
                 visual_out=768, 
                 question_out=768, 
                 fusion_in=1200, 
                 fusion_hidden=256, 
                 num_classes=95, 
                 dropout_v=0.5, 
                 dropout_q=0.5, 
                 dropout_f=0.5, 
                 use_attention=True,
                 hidden_dimension_attention=512,
                 hidden_dimension_cross=5000):
        
        super(VQAModel, self).__init__()
                
        ## Dropouts
        self.dropoutV = torch.nn.Dropout(dropout_v)
        self.dropoutQ = torch.nn.Dropout(dropout_q)
        self.dropoutF = torch.nn.Dropout(dropout_f)

        ## Attention Modules
        if use_attention:
            self.selfattention_q = SelfAttentionQuestion(question_out, hidden_dimension_attention, 1)
            self.crossattention = CrossAttention(hidden_dimension_cross, question_out, visual_out)
            self.selfattention_v = SelfAttentionImage(hidden_dimension_cross, hidden_dimension_attention, 1)

        ## Prepare Fusion
        self.linear_q = nn.Linear(question_out, fusion_in)
        self.linear_v = nn.Linear(visual_out, fusion_in)

        ## Fusion Layer 
        self.fusion = fusions.Mutan([fusion_in, fusion_in], fusion_hidden)
        
        ## Classification layers
        self.linear_classif1 = nn.Linear(fusion_in, fusion_hidden) # for concat, FUSION_IN*2
        self.linear_classif2 = nn.Linear(fusion_hidden, num_classes)

        
    def forward(self, input_v, input_q):

        ## Dropouts
        input_q = self.dropoutQ(input_q)
        input_v = self.dropoutV(input_v)
        
        ## Self-Attention for Question
        if hasattr(self, 'selfattention_q'):
            q = self.selfattention_q(input_q)
            c = self.crossattention(q, input_v)
            v = self.selfattention_v(c, input_v)
    
        ## Prepare fusion
        q = self.linear_q(q)
        q = nn.Tanh()(q)
        v = self.linear_v(v)
        v = nn.Tanh()(v)

        ## Fusion & Classification

        # multiplication         
        #x = torch.mul(v, q)

        # concatenation
        #x = torch.cat((v, q), 1)
        #x = torch.squeeze(x, 1)
        
        # mutan
        x = self.fusion([v, q])
        
        # use tanh for multiplication, optional for concatenation and mutan
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear_classif2(x)
        
        return x