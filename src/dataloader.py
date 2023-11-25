import numpy as np
import pandas as pd
import torch
import re
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from transformers import BertModel
#bert = BertModel.from_pretrained('bert-base-uncased')

from transformers import BertTokenizer,BertForTokenClassification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(['B-MEDCOND', 'I-MEDCOND'])


label_to_ids = {
 'B-MEDCOND': 1,
 'I-MEDCOND': 2,
 'O': 0}

ids_to_label = {
 1:'B-MEDCOND',
 2:'I-MEDCOND',
 0:'O'}

max_tokens = 128

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        if(len(tokenized_sentence)>=max_tokens): #truncate
            return tokenized_sentence, labels

        tokenized_sentence.extend(tokenized_word)

        if label.startswith("B-"):
            labels.extend([label])
            labels.extend([ids_to_label.get(label_to_ids.get(label)+1)]*(n_subwords-1))
        else:
            labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

class Custom_Dataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = re.findall(r"\w+|\w+(?='s)|'s|['\".,!?;]", self.data['text'][idx].strip(), re.UNICODE)
        word_labels = self.data['entity'][idx].split(" ")
        t_sen, t_labl = tokenize_and_preserve_labels(sentence, word_labels, tokenizer)

        sen_code = tokenizer.encode_plus(t_sen,
            add_special_tokens=True, # adds [CLS] and [SEP]
            max_length = max_tokens, # maximum tokens of a sentence
            pad_to_max_length=True, # adds [PAD]s
            return_attention_mask=True, # generates the attention mask
            truncation = True
#             return_tensors = 'pt'
            )


        #shift labels (due to [CLS] and [SEP])
        labels = [-1]*max_tokens
        for i, tok in enumerate(t_labl):
            if tok != None and i < max_tokens-1:
                labels[i+1]=label_to_ids.get(tok)

        item = {key: torch.as_tensor(val) for key, val in sen_code.items()}
        item['entity'] = torch.as_tensor(labels)

        return item

def load_dataset():
    data = pd.read_csv('../datasets/labelled_data/all.csv', names=['text', 'entity'], header=None, sep="|")

    train_data = Custom_Dataset(data)

    return train_data

    #https://www.kaggle.com/code/mdmustafijurrahman/bert-named-entity-recognition-ner-data

    #train_data[2]
    #print(train_data[2]['input_ids'].detach().numpy())
    #print(tokenizer.convert_ids_to_tokens(train_data[2]['input_ids'].detach().numpy()))

    #print('#####')
    #for i in train_data[2]['entity'].detach().numpy():
    #    print(ids_to_label.get(i))