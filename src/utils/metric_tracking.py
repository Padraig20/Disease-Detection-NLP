from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score
import torch

class MetricsTracking():
    def __init__(self):
        self.total_acc = 0
        self.total_f1 = 0
        self.total_precision = 0
        self.total_recall = 0
        self.total_loss = 0

    def update(self, predictions, labels, loss, ignore_token=-100):
        predictions = predictions.flatten()
        labels = labels.flatten()

        predictions = predictions[labels != ignore_token]
        labels = labels[labels != ignore_token]

        predictions = predictions.to("cpu")
        labels = labels.to("cpu")

        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, zero_division=0, average = "macro")
        precision = precision_score(labels, predictions, zero_division=0, average = "macro")
        recall = recall_score(labels, predictions, zero_division=0, average = "macro")

        self.total_acc  += acc
        self.total_f1 += f1
        self.total_precision += precision
        self.total_recall  += recall
        self.total_loss += loss

    def return_avg_metrics(self, data_loader_size):
        n = data_loader_size
        metrics = {
            "acc": round(self.total_acc / n ,3),
            "f1": round(self.total_f1 / n, 3),
            "precision" : round(self.total_precision / n, 3),
            "recall": round(self.total_recall / n, 3),
            "loss": round(self.total_loss / n, 3)
            }
        return metrics
