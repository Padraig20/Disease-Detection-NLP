from dataloader import load_dataset
from BertArchitecture import BertNER
from metric_tracking import MetricsTracking

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from tqdm import tqdm

def train_loop(model, train_dataset, eval_dataset, optimizer, batch_size, epochs):

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    eval_dataloader = DataLoader(eval_dataset, batch_size = batch_size, shuffle = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #training
    for epoch in range(epochs) :

        train_metrics = MetricsTracking()
        total_loss_train = 0

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
        train_metrics.update(predictions, train_label)
        total_loss_train += loss.item()

        loss.backward()
        optimizer.step()


    #evaluation
    model.eval()

    eval_metrics = MetricsTracking()
    total_loss_eval = 0

    with torch.no_grad():

        for eval_data in eval_dataloader:

            eval_label = eval_data['entity'].to(device)
            mask = eval_data['attention_mask'].squeeze(1).to(device)
            input_id = eval_data['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask, eval_label)
            loss, logits = output.loss, output.logits

            predictions = logits.argmax(dim=-1)

            eval_metrics.update(predictions, eval_label)
            total_loss_eval += loss.item()

    train_results = train_metrics.return_avg_metrics(len(train_dataloader))
    eval_results = eval_metrics.return_avg_metrics(len(eval_dataloader))

    print(f"TRAIN\nLoss: {total_loss_train / len(train_dataset)}\nMetrics {train_results}\n" )
    print(f"VALIDATION\nLoss {total_loss_eval / len(eval_dataset)}\nMetrics {eval_results}\n" )



#-------MAIN-------#

model = BertNER(3) #O, B-MEDCOND, I-MEDCOND -> 3 entities

train, test = load_dataset()

lr = 1e-2
optimizer = SGD(model.parameters(), lr=lr, momentum = 0.9)

parameters = {
    "model": model,
    "train_dataset": train,
    "eval_dataset" : test,
    "optimizer" : optimizer,
    "batch_size" : 8,
    "epochs" : 8
}

train_loop(**parameters)
