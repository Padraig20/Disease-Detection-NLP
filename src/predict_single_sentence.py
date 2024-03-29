import argparse
parser = argparse.ArgumentParser(description='Enter a sentence for the model to work on.')

parser.add_argument('-l', '--length', type=int, default=128,
                    help='Choose the maximum length of the model\'s input layer.')
parser.add_argument('-m', '--model', type=str, default='../models/medcondbert.pth',
                    help='Choose the directory of the model to be used for prediction.')
parser.add_argument('-tr', '--transfer_learning', type=bool, default=False,
                    help='Choose whether the given model has been trained on BioBERT or not. \
                    Careful: It will not work if wrongly specified!')
parser.add_argument('sentence', type=str,
                    help='Write your input sentence, preferrably an admission note!')
parser.add_argument('-t', '--type', type=str, required=True,
                    help='Specify the type of annotation to process. Type of annotation needs to be one of the following: Medical Condition, Symptom, Medication, Vital Statistic, Measurement Value, Negation Cue, Medical Procedure')

args = parser.parse_args()
max_length = args.length
model_path = args.model
sentence = args.sentence

from utils.dataloader import Dataloader
from utils.BertArchitecture import BertNER, BioBertNER
from utils.metric_tracking import MetricsTracking

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from tqdm import tqdm
from transformers import BertTokenizer,BertForTokenClassification

if not args.transfer_learning:
    print("Training base BERT model...")
    model = BertNER(3) #O, B-, I- -> 3 entities

    if args.type == 'Medical Condition':
        type = 'MEDCOND'
    elif args.type == 'Symptom':
        type = 'SYMPTOM'
    elif args.type == 'Medication':
        type = 'MEDICATION'
    elif args.type == 'Vital Statistic':
        type = 'VITALSTAT'
    elif args.type == 'Measurement Value':
        type = 'MEASVAL'
    elif args.type == 'Negation Cue':
        type = 'NEGATION'
    elif args.type == 'Medical Procedure':
        type = 'PROCEDURE'
    else:    
        raise ValueError('Type of annotation needs to be one of the following: Medical Condition, Symptom, Medication, Vital Statistic, Measurement Value, Negation Cue, Medical Procedure')
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_tokens(['B-' + args.type, 'I-' + args.type])
else:
    print("Training BERT model based on BioBERT diseases...")

    if not args.type == 'Medical Condition':
        raise ValueError('Type of annotation needs to be Medical Condition when using BioBERT as baseline.')

    model = BioBertNER(3) #O, B-, I- -> 3 entities
    tokenizer = BertTokenizer.from_pretrained('alvaroalon2/biobert_diseases_ner')
    type = 'DISEASE'

label_to_ids = {
    'B-' + type: 0,
    'I-' + type: 1,
    'O': 2
    }

ids_to_label = {
    0:'B-' + type,
    1:'I-' + type,
    2:'O'
    }

model.load_state_dict(torch.load(model_path))
model.eval()


t_sen = tokenizer.tokenize(sentence)

sen_code = tokenizer.encode_plus(sentence,
    return_tensors='pt',
    add_special_tokens=True,
    max_length = max_length,
    padding='max_length',
    return_attention_mask=True,
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
