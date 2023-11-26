from utils.dataloader import Dataloader
from utils.BertArchitecture import BertNER
from utils.metric_tracking import MetricsTracking

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from tqdm import tqdm

def train_loop(model, train_dataset, eval_dataset, optimizer, batch_size, epochs):

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
    eval_dataloader = DataLoader(eval_dataset, batch_size = batch_size, shuffle = False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #training
    for epoch in range(epochs) :

        train_metrics = MetricsTracking()

        model.train() #train mode

        for train_data in tqdm(train_dataloader):

            train_label = train_data['entity'].to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()

            output = model(input_id, mask, train_label)
            loss, logits = output.loss, output.logits
            predictions = logits.argmax(dim=-1)

            #compute metrics
            train_metrics.update(predictions, train_label, loss.item())

            loss.backward()
            optimizer.step()


        model.eval() #evaluation mode

        eval_metrics = MetricsTracking()

        with torch.no_grad():

            for eval_data in eval_dataloader:

                eval_label = eval_data['entity'].to(device)
                mask = eval_data['attention_mask'].squeeze(1).to(device)
                input_id = eval_data['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask, eval_label)
                loss, logits = output.loss, output.logits

                predictions = logits.argmax(dim=-1)

                eval_metrics.update(predictions, eval_label, loss.item())

        train_results = train_metrics.return_avg_metrics(len(train_dataloader))
        eval_results = eval_metrics.return_avg_metrics(len(eval_dataloader))

        print(f"Epoch {epoch} of {epochs} finished!")
        print(f"TRAIN\nMetrics {train_results}\n")
        print(f"VALIDATION\nMetrics {eval_results}\n")



#-------MAIN-------#

import argparse

parser = argparse.ArgumentParser(
        description='This class is used to train a transformer-based model on admission notes, labelled with <3 by Patrick.')

parser.add_argument('-s', '--save', type=str, default="",
                    help='Choose where to save the model after training. Saving is optional.')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2,
                    help='Choose the learning rate of the model.')
parser.add_argument('-b', '--batch_size', type=int, default=16,
                    help='Choose the batch size of the model.')
parser.add_argument('-e', '--epochs', type=int, default=5,
                    help='Choose the epochs of the model.')

args = parser.parse_args()

model = BertNER(3) #O, B-MEDCOND, I-MEDCOND -> 3 entities

label_to_ids = {
    'B-MEDCOND': 1,
    'I-MEDCOND': 2,
    'O': 0
    }

ids_to_label = {
    1:'B-MEDCOND',
    2:'I-MEDCOND',
    0:'O'
    }

dataloader = Dataloader(label_to_ids, ids_to_label)

train, test = dataloader.load_dataset()

optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum = 0.9)

parameters = {
    "model": model,
    "train_dataset": train,
    "eval_dataset" : test,
    "optimizer" : optimizer,
    "batch_size" : args.batch_size,
    "epochs" : args.epochs
}

train_loop(**parameters)

#save model if wanted
if args.save:
    torch.save(model.state_dict(), args.save)
    print(f"Model has successfully been saved at {args.save}!")
