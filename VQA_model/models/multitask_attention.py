import torch
import torch.nn as nn
from models.attention import SelfAttentionQuestion, CrossAttention, SelfAttentionImage

from block import fusions

# used in dataset, only for reference
# question_type_to_idx = {
#     "presence": 0,
#     "comp": 1,
#     "area": 2,
#     "count": 3,
# }


# question_type_to_indices = {
#             "presence": [0, 1],
#             "area": list(range(2, 6)),
#             "count": list(range(6, 95))
#         }

class CustomFusionModule(nn.Module):
    def __init__(self, 
                visual_out=768, 
                question_out=768, 
                fusion_in=1200, 
                fusion_hidden=256, 
                num_answers=2, 
                dropout_v=0.5, 
                dropout_q=0.5, 
                dropout_f=0.5, 
                use_attention=True,
                hidden_dimension_attention=512,
                hidden_dimension_cross=5000):
        super(CustomFusionModule, self).__init__()

        self.dropoutV = nn.Dropout(dropout_v)
        self.dropoutQ = nn.Dropout(dropout_q)
        self.dropoutF = nn.Dropout(dropout_f)

        ## Attention Modules
        if use_attention:
            self.selfattention_q = SelfAttentionQuestion(question_out, hidden_dimension_attention, 1)
            self.crossattention = CrossAttention(hidden_dimension_cross, question_out, visual_out)
            self.selfattention_v = SelfAttentionImage(hidden_dimension_cross, hidden_dimension_attention, 1)
        
        self.linear_q = nn.Linear(question_out, fusion_in)
        self.linear_v = nn.Linear(visual_out, fusion_in)

        self.dropout = nn.Dropout(dropout_f)

        self.fusion = fusions.Mutan([fusion_in, fusion_in], fusion_in)
        
        self.linear1 = nn.Linear(fusion_in, fusion_hidden) # for concat, FUSION_IN*2
        self.linear2 = nn.Linear(fusion_hidden, num_answers)

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
        # when using concat        
        #x = torch.cat((q, v), dim=1)
        #x = torch.squeeze(x, 1)

        # when using element-wise multiplication
        #x = torch.mul(q, v)
        #x = nn.Tanh()(x)

        # when using mutan
        x = self.fusion([v, q])
        
        
        #x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear1(x)
        x = nn.Tanh()(x)
        x = self.dropoutF(x)
        x = self.linear2(x)
        return x


class MultiTaskVQAModel(nn.Module):
    def __init__(self,
                visual_out=768, 
                question_out=768, 
                fusion_in=1200, 
                fusion_hidden=256, 
                num_answers=95, 
                dropout_v=0.5, 
                dropout_q=0.5, 
                dropout_f=0.5, 
                use_attention=True,
                hidden_dimension_attention=512,
                hidden_dimension_cross=5000):
        super(MultiTaskVQAModel, self).__init__()

        # Mapping question types to number of unique answers
        question_type_to_num_answers = {
            0: 2,
            1: 2,
            2: 5,
            3: 88
        }

        self.question_type_to_num_answers = question_type_to_num_answers
        self.total_num_classes = num_answers

        self.classifiers = nn.ModuleList([
            CustomFusionModule(visual_out, question_out, fusion_in, fusion_hidden, num_answers, dropout_v, dropout_q, dropout_f, use_attention, hidden_dimension_attention, hidden_dimension_cross) 
            for num_answers in question_type_to_num_answers.values()
        ])

    def shared_parameters(self):
        # Return all parameters that are not part of the classifier heads
        for name, param in self.named_parameters():
            if not any(name.startswith(f'classifiers.{i}') for i in range(len(self.classifiers))):
                yield param

    def forward(self, input_v, input_q, question_type):

        # Initialize a tensor to hold the final predictions for the entire batch
        batch_size = input_v.size(0)
        final_output = torch.zeros(batch_size, self.total_num_classes, device=input_q.device)

        for qt, classifier in zip(self.question_type_to_num_answers.keys(), self.classifiers):
            mask = (question_type == qt)
            v_masked = input_v[mask]
            q_masked = input_q[mask]
            
            if v_masked.size(0) > 0:  # Check if there are any items of this question type
                classifier_output = classifier(v_masked, q_masked)
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
