import argparse

parser = argparse.ArgumentParser(
        description='This class is used to train a transformer-based model on admission notes, labelled with <3 by Patrick.')

parser.add_argument('-o', '--output', type=str, default="",
                    help='Choose where to save the model after training. Saving is optional.')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2,
                    help='Choose the learning rate of the model.')
parser.add_argument('-b', '--batch_size', type=int, default=16,
                    help='Choose the batch size of the model.')
parser.add_argument('-e', '--epochs', type=int, default=5,
                    help='Choose the epochs of the model.')
parser.add_argument('-opt', '--optimizer', type=str, default='SGD',
                    help='Choose the optimizer to be used for the model: SDG | Adam')
parser.add_argument('-tr', '--transfer_learning', type=bool, default=False,
                    help='Choose whether the BioBERT model should be used as baseline or not.')
parser.add_argument('-v', '--verbose', type=bool, default=False,
                    help='Choose whether the model should be evaluated after each epoch or only after the training.')
parser.add_argument('-l', '--input_length', type=int, default=128,
                    help='Choose the maximum length of the model\'s input layer.')
parser.add_argument('-ag', '--data_augmentation', type=bool, default=False,
                    help='Choose whether data-augmentation should be used.')
parser.add_argument('-t', '--type', type=str, required=True,
                    help='Specify the type of annotation to process. Type of annotation needs to be one of the following: Medical Condition, Symptom, Medication, Vital Statistic, Measurement Value, Negation Cue, Medical Procedure')

args = parser.parse_args()

if args.type not in ['Medical Condition', 'Symptom', 'Medication', 'Vital Statistic', 'Measurement Value', 'Negation Cue', 'Medical Procedure']:
    raise ValueError('Type of annotation needs to be one of the following: Medical Condition, Symptom, Medication, Vital Statistic, Measurement Value, Negation Cue, Medical Procedure')

from utils.dataloader import Dataloader
from utils.BertArchitecture import BertNER
from utils.BertArchitecture import BioBertNER
from utils.metric_tracking import MetricsTracking
from utils.training import train_loop, testing

import torch
from torch.optim import SGD
from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from tqdm import tqdm

#-------MAIN-------#

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

else:
    print("Training BERT model based on BioBERT diseases...")

    if not args.type == 'Medical Condition':
        raise ValueError('Type of annotation needs to be Medical Condition when using BioBERT as baseline.')

    model = BioBertNER(3) #O, B-, I- -> 3 entities
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

dataloader = Dataloader(label_to_ids, ids_to_label, args.transfer_learning, args.input_length, type)

train, val, test = dataloader.load_dataset(augment = args.data_augmentation)

if args.optimizer == 'SGD':
    print("Using SGD optimizer...")
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum = 0.9)
else:
    print("Using Adam optimizer...")
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

parameters = {
    "model": model,
    "train_dataset": train,
    "eval_dataset" : val,
    "optimizer" : optimizer,
    "batch_size" : args.batch_size,
    "epochs" : args.epochs,
    "type" : type
}

train_loop(**parameters, verbose=args.verbose)

testing(model, test, args.batch_size, type)

#save model if wanted
if args.output:
    torch.save(model.state_dict(), args.output)
    print(f"Model has successfully been saved at {args.output}!")
