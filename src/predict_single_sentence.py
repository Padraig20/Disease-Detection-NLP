from utils.dataloader import Dataloader
from utils.BertArchitecture import BertNER
from utils.metric_tracking import MetricsTracking

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from tqdm import tqdm
from transformers import BertTokenizer,BertForTokenClassification

import argparse
parser = argparse.ArgumentParser(description='Enter a sentence for the model to work on.')

parser.add_argument('-l', '--length', type=bool, default=128,
                    help='Choose the maximum length of the model\'s input layer.')
parser.add_argument('-m', '--model', type=str, default='../models/bert_medcond.pth',
                    help='Choose the directory of the model to be used for prediction.')
parser.add_argument('sentence', type=str, help='Write your input sentence, preferrably an admission note!')

args = parser.parse_args()
max_length = args.length
model_path = args.model
sentence = args.sentence

ids_to_label = {
    1:'B-MEDCOND',
    2:'I-MEDCOND',
    0:'O'
    }

model = BertNER(3)
model.load_state_dict(torch.load(model_path))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(['B-MEDCOND', 'I-MEDCOND'])

t_sen = tokenizer.tokenize(sentence)

sen_code = tokenizer.encode_plus(sentence,
    return_tensors='pt',
    add_special_tokens=True, # adds [CLS] and [SEP]
    max_length = max_length, # maximum tokens of a sentence
    padding='max_length',
    return_attention_mask=True, # generates the attention mask
    truncation = True
    )
inputs = {key: torch.as_tensor(val) for key, val in sen_code.items()}

attention_mask = inputs['attention_mask'].squeeze(1)
input_ids = inputs['input_ids'].squeeze(1)

outputs = model(input_ids, attention_mask)

predictions = outputs.logits.argmax(dim=-1)
predictions = [ids_to_label.get(x) for x in predictions.numpy()[0]]

#beware special tokens
cutoff = min(len(predictions)-1, len(t_sen))
predictions = predictions[1:cutoff+1]
t_sen = t_sen[:cutoff]

df = pd.DataFrame({
    'Sentence': t_sen,
    'Prediction': predictions
})
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
print(df)
