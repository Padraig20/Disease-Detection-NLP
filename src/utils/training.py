from utils.dataloader import Dataloader
from utils.BertArchitecture import BertNER
from utils.BertArchitecture import BioBertNER
from utils.metric_tracking import MetricsTracking

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from tqdm import tqdm

def train_loop(model, train_dataset, eval_dataset, optimizer, batch_size, epochs, train_sampler=None, eval_sampler=None, verbose=True):
    """
    Usual training loop, including training and evaluation.

    Parameters:
    model (BertNER | BioBertNER): Model to be trained.
    train_dataset (Custom_Dataset): Dataset used for training.
    eval_dataset (Custom_Dataset): Dataset used for testing.
    optimizer (torch.optim): Optimizer used, usually SGD or Adam.
    batch_size (int): Batch size used during training.
    epochs (int): Number of epochs used for training.
    train_sampler (SubsetRandomSampler): Sampler used during hyperparameter-tuning.
    val_subsampler (SubsetRandomSampler): Sampler used during hyperparameter-tuning.
    verbose (bool): Whether the model should be evaluated after each epoch or not.

    Returns:
    tuple:
        - train_res (dict): A dictionary containing the results obtained during training.
        - test_res (dict): A dictionary containing the results obtained during testing.
    """

    if train_sampler == None or eval_sampler == None:
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False, sampler=train_sampler)
        eval_dataloader = DataLoader(eval_dataset, batch_size = batch_size, shuffle = False, sampler=eval_sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
        eval_dataloader = DataLoader(eval_dataset, batch_size = batch_size, shuffle = False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #training
    for epoch in range(epochs):

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

        if verbose:
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

            print(f"Epoch {epoch+1} of {epochs} finished!")
            print(f"TRAIN\nMetrics {train_results}\n")
            print(f"VALIDATION\nMetrics {eval_results}\n")

    if not verbose:
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

        print(f"Epoch {epoch+1} of {epochs} finished!")
        print(f"TRAIN\nMetrics {train_results}\n")
        print(f"VALIDATION\nMetrics {eval_results}\n")

    return train_results, eval_results
