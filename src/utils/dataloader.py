import numpy as np
import pandas as pd
import torch
import re
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer,BertForTokenClassification

class Dataloader():
    """
    Dataloader used for loading the dataset used in this project. Also provides a framework for automatic
    tokenization of the data.
    """

    def __init__(self, label_to_ids, ids_to_label, transfer_learning, max_tokens):
        self.label_to_ids = label_to_ids
        self.ids_to_label = ids_to_label
        self.max_tokens = max_tokens
        self.transfer_learning = transfer_learning

    def load_dataset(self, full = False):
        """
        Loads the dataset and automatically initialized a tokenizer for the Custom_Dataset initialization.

        Parameters:
        full (bool): Whether the function should return the whole dataset or not - will return a train-test split
                     according to the Pareto principle (80:20).

        Returns:
        if full:
            dataset (Custom_Dataset): the full dataset in one.
        else:
            tuple:
                - train_dataset (Custom_Dataset): Dataset used for training.
                - val_dataset (Custom_Dataset): Dataset used for validation.
                - test_dataset (Custom_Dataset): Dataset sued for testing.
        """
        data = pd.read_csv('../datasets/labelled_data/all.csv', names=['text', 'entity'], header=None, sep="|")

        if self.transfer_learning:
            tokenizer = BertTokenizer.from_pretrained('alvaroalon2/biobert_diseases_ner')
            data['entity'] = data['entity'].apply(lambda x: x.replace('B-MEDCOND', 'B-DISEASE'))
            data['entity'] = data['entity'].apply(lambda x: x.replace('I-MEDCOND', 'I-DISEASE'))
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            tokenizer.add_tokens(['B-MEDCOND', 'I-MEDCOND'])

        if not full:
            #train_data = data.sample((int) (len(data)*0.8), random_state=7).reset_index(drop=True)
            #test_data = data.drop(train_data.index).reset_index(drop=True)

            train_data = data.sample(frac=0.7, random_state=7).reset_index(drop=True)

            remaining_data = data.drop(train_data.index).reset_index(drop=True)
            val_data = remaining_data.sample(frac=0.2857, random_state=7).reset_index(drop=True)

            test_data = remaining_data.drop(val_data.index).reset_index(drop=True)

            train_dataset = Custom_Dataset(train_data, tokenizer, self.label_to_ids, self.ids_to_label, self.max_tokens)
            val_dataset = Custom_Dataset(val_data, tokenizer, self.label_to_ids, self.ids_to_label, self.max_tokens)
            test_dataset = Custom_Dataset(test_data, tokenizer, self.label_to_ids, self.ids_to_label, self.max_tokens)

            return train_dataset, val_dataset, test_dataset
        else:
            dataset = Custom_Dataset(data, tokenizer, self.label_to_ids, self.ids_to_label, self.max_tokens)
            return dataset

    def load_custom(self, data):
        """
        Loads the dataset, but with entities swapped from MEDCOND to DISEASE (if transfer learning
        is enabled).

        Parameters:
        data (dataframe): Data extracted from csv file.

        Returns:
        dataset (Custom_Dataset): Dataset changed accordingly.
        """
        if self.transfer_learning:
            tokenizer = BertTokenizer.from_pretrained('alvaroalon2/biobert_diseases_ner')
            data['entity'] = data['entity'].apply(lambda x: x.replace('B-MEDCOND', 'B-DISEASE'))
            data['entity'] = data['entity'].apply(lambda x: x.replace('I-MEDCOND', 'I-DISEASE'))
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            tokenizer.add_tokens(['B-MEDCOND', 'I-MEDCOND'])
        dataset = Custom_Dataset(data, tokenizer, self.label_to_ids, self.ids_to_label, self.max_tokens)
        return dataset

    def convert_id_to_label(self, ids):
        return [self.ids_to_label.get(x) for x in ids.numpy()[0]]

def tokenize_and_preserve_labels(sentence, text_labels, tokenizer, label_to_ids, ids_to_label, max_tokens):
    """
    Tokenizes each word separately. This may take longer, but increases accuracy. Preserves the labels
    of each word, adhereing to B and I prefixes.

    Parameters:
    sentence (string): Sentence to be tokenized.
    text_labels (numpy.array): Contains the labels of the sentence.
    tokenizer (BertTokenizer): Tokenizer used for tokenizing sentences.
    label_to_ids (dict): Dictionary containing label-id mappings.
    ids_to_label (dict): Dictionary containing id-label mappings.
    max_tokens (int): The maximum tokens allowed (input size of BERT model).

    Returns:
        tuple:
            - tokenized_sentence (numpy.array): Array containing all tokens of the give sentence.
            - labels (numpy.array): Array containing the corresponding labels of the tokens.
    """
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
    """
    Dataset used for loading and tokenizing sentences on-the-fly.
    """

    def __init__(self, data, tokenizer, label_to_ids, ids_to_label, max_tokens):
        self.data = data
        self.tokenizer = tokenizer
        self.label_to_ids = label_to_ids
        self.ids_to_label = ids_to_label
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Takes the current sentence with its labels and tokenizes it on-the-fly.

        Returns:
        item (torch.tensor): Tensor which can be fed into model.
        """
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
        labels = [-100]*self.max_tokens #-100 is ignore token
        for i, tok in enumerate(t_labl):
            if tok != None and i < self.max_tokens-1:
                labels[i+1]=self.label_to_ids.get(tok)

        item = {key: torch.as_tensor(val) for key, val in sen_code.items()}
        item['entity'] = torch.as_tensor(labels)

        return item
