from torchvision import models as torchmodels
import torch.nn as nn
import torch.nn.functional as F
import torch

class SelfAttentionQuestion(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SelfAttentionQuestion, self).__init__()
        self.inner_matrix = nn.Linear(input_dim, hidden_dim, bias=False)
        self.outer_matrix = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, input_q):
        # print("Shape of input_q:", input_q.shape)
        
        q = self.inner_matrix(input_q)
        q = nn.ReLU()(q)
        #print("Shape of q after inner matrix:", q.shape)

        q = self.outer_matrix(q)
        q = nn.Softmax(dim=1)(q)
        #print("Shape of q after outer matrix and before squeeze:", q.shape) 
        
        q = torch.squeeze(q, -1)  # Removing the last dimension, now shape is batch_size x 35
        #print("Shape of q after squeeze:", q.shape)            

        q = torch.unsqueeze(q, 1)  # Adding a dimension at index 1, now shape is batch_size x 1 x 35  
        #print("Shape of q after unsqueeze:", q.shape)

        output_q = torch.matmul(q, input_q).squeeze(1)
        #print("Shape of q after matmul:", output_q.shape)

        #print("Self-Attention Question completed")

        return output_q
    
class CrossAttention(nn.Module):
    def __init__(self, output_dim, input_dim_q, input_dim_v):
        super(CrossAttention, self).__init__()

        self.cross_q = nn.Linear(input_dim_q, output_dim, bias=False)
        self.cross_v = nn.Linear(input_dim_v, output_dim, bias=False)

    def forward(self, input_q, input_v):
        # print("Shape of input q:", input_q.shape)
        # print("Shape of input v:", input_v.shape)
        
        x_q = self.cross_q(input_q)
        x_v = self.cross_v(input_v)
        # print("Shape of x_q:", x_q.shape)
        # print("Shape of x_v:", x_v.shape)

        x_q = x_q.unsqueeze(1)
        # print("Shape of x_q after unsqueeze:", x_q.shape)
        
        x_v_attn = torch.mul(x_v, x_q)
        # print("Shape of x_v_attn:", x_v_attn.shape)
        
        x_v_attn = nn.Dropout(p=0.15)(x_v_attn)
        x_v_attn = F.normalize(x_v_attn, p=2, dim=2)
        # print("Shape of x_v_attn after dropout and norm:", x_v_attn.shape)

        # print("Cross-Attention completed")
        
        return x_v_attn
    
class SelfAttentionImage(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SelfAttentionImage, self).__init__()
        self.inner_matrix = nn.Linear(input_dim, hidden_dim, bias=False)
        self.outer_matrix = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, v_proc, input_v):
        # print("Shape of input_v:", input_v.shape)
        # print("Shape of v_proc:", v_proc.shape)
        
        v = self.inner_matrix(v_proc)
        v = nn.ReLU()(v)
        # print("Shape of v after inner matrix:", v.shape)

        v = self.outer_matrix(v)
        v = nn.Softmax(dim=1)(v)
        # print("Shape of v after outer matrix and before squeeze:", v.shape) 
        
        v = torch.squeeze(v, -1)  # Removing the last dimension, now shape is batch_size x 35
        # print("Shape of v after squeeze:", v.shape)            

        v = torch.unsqueeze(v, 1)  # Adding a dimension at index 1, now shape is batch_size x 1 x 35  
        # print("Shape of v after unsqueeze:", v.shape)

        output_v = torch.matmul(v, input_v).squeeze(1)
        # print(output_v.shape)

        # print("Cross-Attention completed")

        return output_v