import transformers
from transformers import BertForTokenClassification
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F

class BertNER(nn.Module):
    """
    Architecture using bert-base-uncased.
    """
    def __init__(self, tokens_dim):
        super(BertNER,self).__init__()
        self.pretrained = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels = tokens_dim)

    def forward(self, input_ids, attention_mask, labels = None): #labels for loss calculation
        if labels == None:
            out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask )
            return out
        out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask , labels = labels)
        return out


class BioBertNER(nn.Module):
    """
    Architecture using the BioBERT diseases NER for transfer learning.
    """
    def __init__(self, tokens_dim):
        super(BioBertNER,self).__init__()
        self.pretrained = BertForTokenClassification.from_pretrained("alvaroalon2/biobert_diseases_ner", num_labels = tokens_dim)

    def forward(self, input_ids, attention_mask, labels = None): #labels for loss calculation
        if labels == None:
            out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask )
            return out
        out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask , labels = labels)
        return out
