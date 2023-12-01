import torch
import torch.nn as nn
from models import attention

VISUAL_OUT_VIT = 768
QUESTION_OUT = 2400
FUSION_IN = 1200
FUSION_HIDDEN = 256
DROPOUT_V = 0.5
DROPOUT_Q = 0.5
DROPOUT_F = 0.5

from block import fusions

# used in dataset, only for reference
# question_type_to_idx = {
#     "presence": 0,
#     "comp": 0,
#     "area": 1,
#     "count": 2,
# }


# question_type_to_indices = {
#             "presence": [0, 1],
#             "area": list(range(2, 6)),
#             "count": list(range(6, 95))
#         }

class CustomFusionModule(nn.Module):
    def __init__(self, fusion_in, fusion_hidden, num_answers):
        super(CustomFusionModule, self).__init__()

        self.dropout = nn.Dropout(DROPOUT_F)
        
        self.linear1 = nn.Linear(fusion_in, fusion_hidden)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(fusion_hidden, num_answers)

    def forward(self, fused):
        x = self.dropout(fused)
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiTaskVQAModel(nn.Module):
    def __init__(self):
        super(MultiTaskVQAModel, self).__init__()

        # Mapping question types to number of unique answers
        question_type_to_num_answers = {
            0: 2,
            1: 2,
            2: 5,
            3: 88
        }

        self.dropoutV = nn.Dropout(DROPOUT_V)
        self.dropoutQ = nn.Dropout(DROPOUT_Q)
        self.dropoutF = nn.Dropout(DROPOUT_F)
        
        self.linear_q = nn.Linear(QUESTION_OUT, FUSION_IN)
        self.linear_v = nn.Linear(VISUAL_OUT_VIT, FUSION_IN)

        self.fusion = fusions.Mutan([FUSION_IN, FUSION_IN], FUSION_IN)

        self.question_type_to_num_answers = question_type_to_num_answers
        self.total_num_classes = 95

        # self.selfattention = attention.SelfAttention(FUSION_IN)
        # self.crossattention = attention.CrossAttention(FUSION_IN)

        self.classifiers = nn.ModuleList([
            CustomFusionModule(FUSION_IN, FUSION_HIDDEN, num_answers) 
            for num_answers in question_type_to_num_answers.values()
        ])

    def shared_parameters(self):
        # Return all parameters that are not part of the classifier heads
        for name, param in self.named_parameters():
            if not any(name.startswith(f'classifiers.{i}') for i in range(len(self.classifiers))):
                yield param

    def forward(self, input_v, input_q, question_type):
        
        x_v = self.dropoutV(input_v)
        x_v = self.linear_v(x_v)
        x_v = torch.tanh(x_v)

        x_q = self.dropoutQ(input_q)
        x_q = self.linear_q(x_q)
        x_q = torch.tanh(x_q)
        x = self.fusion([x_v, x_q])
        
        # x_q, _ = self.selfattention(x_q)
        # x_v, _ = self.crossattention(x_q, x_v)

        #x = torch.mul(x_v, x_q)
        # x = torch.squeeze(x, 1)
        # x = nn.Tanh()(x)

        # Initialize a tensor to hold the final predictions for the entire batch
        batch_size = x_q.size(0)
        final_output = torch.zeros(batch_size, self.total_num_classes, device=x_q.device)

        for qt, classifier in zip(self.question_type_to_num_answers.keys(), self.classifiers):
            mask = (question_type == qt)
            x_masked = x[mask]
            
            if x_masked.size(0) > 0:  # Check if there are any items of this question type
                classifier_output = classifier(x_masked)
                output_masked = self.get_final_prediction(pred=classifier_output, question_type=qt, num_classes=self.total_num_classes)
                
                # Place the result back in the correct positions of the final_output tensor
                final_output[mask] = output_masked

        return final_output
    
    def get_final_prediction(self, pred, question_type, num_classes):
        question_type_to_indices = {
            0: [0, 1],
            1: [0, 1],
            2: list(range(2, 7)),
            3: list(range(7, 95))
        }
        final_pred = torch.zeros((pred.shape[0], num_classes), device=pred.device)
        indices = question_type_to_indices[question_type]
        
        # Assign the predictions to the relevant indices.
        final_pred[:, indices] = pred
        return final_pred
