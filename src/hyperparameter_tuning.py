from transformers import BertTokenizer, BertForSequenceClassification
from utils.BertArchitecture import BertNER, BioBertNER
from utils.training import train_loop
from utils.dataloader import Dataloader
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.optim import SGD
from torch.optim import Adam

def get_label_descriptions(transfer_learning, type):
    """
    This function returns the correct label descriptions according to the
    current Model Architecture in use.

    Parameters:
    transfer_learning (bool): Whether we train BioBERT for transfer learning or not.

    Returns:
    tuple:
        - label_to_ids (dict): A dictionary mapping labels to their respective IDs.
        - ids_to_label (dict): A dictionary mapping IDs back to their respective labels.
    """
    if not transfer_learning:

        if type == 'Medical Condition':
            type = 'MEDCOND'
        elif type == 'Symptom':
            type = 'SYMPTOM'
        elif type == 'Medication':
            type = 'MEDICATION'
        elif type == 'Vital Statistic':
            type = 'VITALSTAT'
        elif type == 'Measurement Value':
            type = 'MEASVAL'
        elif type == 'Negation Cue':
            type = 'NEGATION'
        elif type == 'Medical Procedure':
            type = 'PROCEDURE'
        else:    
            raise ValueError('Type of annotation needs to be one of the following: Medical Condition, Symptom, Medication, Vital Statistic, Measurement Value, Negation Cue, Medical Procedure')
    else:
        if not type == 'Medical Condition':
            raise ValueError('Type of annotation needs to be Medical Condition when using BioBERT as baseline.')
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
    return label_to_ids, ids_to_label, type

def initialize_model(transfer_learning):
    """
    Initializes the model architecture according to whether we would like to use BioBERT
    or not.

    Parameters:
    transfer_learning (bool): Whether we train BioBERT for transfer learning or not.

    Returns:
    model (BertNER | BioBertNER)
    """
    if not transfer_learning:
        model = BertNER(3) #O, B-, I- -> 3 entities
    else:
        model = BioBertNER(3)
    return model

def train_fold(transfer_learning, train_idx, val_idx, batch_size, learning_rate, optimizer_name, epoch, type):
    """
    Trains a whole fold during hyperparameter tuning.

    Parameters:
    transfer_learning (bool): Whether we train BioBERT for transfer learning or not.
    train_idx (int): Index of the current split in training data.
    val_idx (int): Index of the current split in testing data.
    batch_size (int): Batch size used for training.
    learning_rate (float): Learning rate used for optimizer.
    optimizer_name (string): Name of the optimizer to be used (SGD or Adam).
    epoch (int): Number of epochs used for training.

    Returns:
    tuple:
        - train_res (dict): A dictionary containing the results obtained during training.
        - test_res (dict): A dictionary containing the results obtained during testing.
    """
    model = initialize_model(transfer_learning)

    if optimizer_name == 'SGD':
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum = 0.9)
    else:
        optimizer = Adam(model.parameters(), lr=learning_rate)

    train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

    parameters = {
        "model": model,
        "train_dataset": data,
        "eval_dataset" : data,
        "optimizer" : optimizer,
        "batch_size" : batch_size,
        "epochs" : epoch,
        "train_sampler": train_subsampler,
        "eval_sampler": val_subsampler,
        "type" : type
    }

    train_res, test_res = train_loop(**parameters, verbose=False)
    return train_res, test_res

import argparse

parser = argparse.ArgumentParser(
        description='This class is used to optimize the hyperparameters of either the pretrained BioBERT or the base BERT.')
parser.add_argument('-tr', '--transfer_learning', type=bool, default=False,
                    help='Choose whether the BioBERT model should be used as baseline or not.')
parser.add_argument('-t', '--type', type=str, required=True,
                    help='Specify the type of annotation to process. Type of annotation needs to be one of the following: Medical Condition, Symptom, Medication, Vital Statistic, Measurement Value, Negation Cue, Medical Procedure')

args = parser.parse_args()

if args.type not in ['Medical Condition', 'Symptom', 'Medication', 'Vital Statistic', 'Measurement Value', 'Negation Cue', 'Medical Procedure']:
    raise ValueError('Type of annotation needs to be one of the following: Medical Condition, Symptom, Medication, Vital Statistic, Measurement Value, Negation Cue, Medical Procedure')


#-----hyperparameter grids-----#

batch_sizes = [8, 16]  #[8,16,32]
learning_rates = [0.1] #[0.1, 0.01, 0.001, 0.0001]
optimizers = ['SGD']  #['SGD', 'Adam']
epochs = [1] #[5, 10]
max_tokens = 128

label_to_ids, ids_to_label, type = get_label_descriptions(args.transfer_learning, args.type)
dataloader = Dataloader(label_to_ids, ids_to_label, args.transfer_learning, max_tokens, type)
data = dataloader.load_dataset(full=True)

best_f1_score = 0
best_param_grid = {
    "batch_size": 0,
    "learning_rate": 0,
    "epochs" : 0,
    "optimizer" : "",
    "max_tokens" : 0 
}

for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        for optimizer_name in optimizers:
            for epoch in epochs:
                kf = KFold(n_splits=4, shuffle=True, random_state=7)
                test_f1_scores = []
                for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
                    train_res, test_res = train_fold(args.transfer_learning, train_idx, val_idx, batch_size, learning_rate, optimizer_name, epoch, type)
                    test_f1_scores.append(test_res['avg_f1_score'])
                    print(f"Finished fold {fold+1} of 4!")
                local_best_f1 = sum(test_f1_scores) / len(test_f1_scores)
                if local_best_f1 > best_f1_score:
                    best_f1_score = local_best_f1
                    best_param_grid = {
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "epochs" : epoch,
                        "optimizer" : optimizer_name,
                        "max_tokens" : max_tokens
                    }
                    print(f"Found new best f1 score: {best_f1_score}")
                    print(best_param_grid)

print("-------FINAL RESULTS-------")
print(f"Best f1 score: {best_f1_score}")
print(best_param_grid)
