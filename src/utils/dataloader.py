import numpy as np
import pandas as pd
import torch
import re
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer,BertForTokenClassification

class Dataloader():

    def __init__(self, label_to_ids, ids_to_label, transfer_learning=False, max_tokens=128):
        self.label_to_ids = label_to_ids
        self.ids_to_label = ids_to_label
        self.max_tokens = max_tokens
        self.transfer_learning = transfer_learning

    def load_dataset(self, full = False):
        data = pd.read_csv('../datasets/labelled_data/all.csv', names=['text', 'entity'], header=None, sep="|").head(20)

        if self.transfer_learning:
            tokenizer = BertTokenizer.from_pretrained('alvaroalon2/biobert_diseases_ner')
            data['entity'] = data['entity'].apply(lambda x: x.replace('B-MEDCOND', 'B-DISEASE'))
            data['entity'] = data['entity'].apply(lambda x: x.replace('I-MEDCOND', 'I-DISEASE'))
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            tokenizer.add_tokens(['B-MEDCOND', 'I-MEDCOND'])

        if not full:
            train_data = data.sample((int) (len(data)*0.8), random_state=7).reset_index(drop=True)
            test_data = data.drop(train_data.index).reset_index(drop=True)

            train_dataset = Custom_Dataset(train_data, tokenizer, self.label_to_ids, self.ids_to_label)
            test_dataset = Custom_Dataset(test_data, tokenizer, self.label_to_ids, self.ids_to_label)

            return train_dataset, test_dataset
        else:
            dataset = Custom_Dataset(data, tokenizer, self.label_to_ids, self.ids_to_label)
            return dataset

    def load_custom(self, data):
        if self.transfer_learning:
            tokenizer = BertTokenizer.from_pretrained('alvaroalon2/biobert_diseases_ner')
            data['entity'] = data['entity'].apply(lambda x: x.replace('B-MEDCOND', 'B-DISEASE'))
            data['entity'] = data['entity'].apply(lambda x: x.replace('I-MEDCOND', 'I-DISEASE'))
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            tokenizer.add_tokens(['B-MEDCOND', 'I-MEDCOND'])
        dataset = Custom_Dataset(data, tokenizer, self.label_to_ids, self.ids_to_label)
        return dataset

        #https://www.kaggle.com/code/mdmustafijurrahman/bert-named-entity-recognition-ner-data

    def convert_id_to_label(self, ids):
        return [self.ids_to_label.get(x) for x in ids.numpy()[0]]

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer, label_to_ids, ids_to_label, max_tokens):
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

    def __init__(self, data, tokenizer, label_to_ids, ids_to_label, max_tokens=128):
        self.data = data
        self.tokenizer = tokenizer
        self.label_to_ids = label_to_ids
        self.ids_to_label = ids_to_label
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = re.findall(r"\w+|\w+(?='s)|'s|['\".,!?;]", self.data['text'][idx].strip(), re.UNICODE)
        word_labels = self.data['entity'][idx].split(" ")
        t_sen, t_labl = tokenize_and_preserve_labels(sentence, word_labels, self.tokenizer, self.label_to_ids, self.ids_to_label, self.max_tokens)

        sen_code = self.tokenizer.encode_plus(t_sen,
            add_special_tokens=True, # adds [CLS] and [SEP]
            max_length = self.max_tokens, # maximum tokens of a sentence
            padding='max_length',
            return_attention_mask=True, # generates the attention mask
            truncation = True
            )


        #shift labels (due to [CLS] and [SEP])
        labels = [-100]*self.max_tokens
        for i, tok in enumerate(t_labl):
            if tok != None and i < self.max_tokens-1:
                labels[i+1]=self.label_to_ids.get(tok)

        item = {key: torch.as_tensor(val) for key, val in sen_code.items()}
        item['entity'] = torch.as_tensor(labels)

        return item
