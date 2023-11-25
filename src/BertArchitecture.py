import transformers
from transformers import BertForTokenClassification
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F

class BertNER(nn.Module):
    def __init__(self, tokens_dim):
        super(BertNER,self).__init__()
        self.pretrained = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels = tokens_dim)

    def forward(self, input_ids, attention_mask, labels = None): #labels for loss calculation
        if labels == None: #no labels at inference time
            out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask )
            return out
        out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask , labels = labels)
        return out
