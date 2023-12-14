from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score
import torch

class MetricsTracking():
    """
    Class used for tracking the most prominent metrics, including accuracy, f1, precision, recall and loss.
    """
    def __init__(self):
        self.total_acc = 0
        self.total_f1 = 0
        self.total_precision = 0
        self.total_recall = 0
        self.total_loss = 0

    def update(self, predictions, labels, loss, ignore_token=-100):
        """
        Updates the current metrics. Takes into account which tokens to ignore during evaluation (e.g. [¢LS]).

        Parameters:
        predictions (numpy.array): Array containing the predictions of the model.
        labels (numpy.array): Array containing the ground truth.
        loss (float): Loss of the model.
        ignore_token (int): Specifies which token to ignore - -100 in this project's architecture.
        """
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
        """
        Returns the metrics stored, but averages it to the size of the data.

        Parameters:
        data_loader_size (int): length of the data.

        Returns:
        metrics (dict): All the metrics calculated until now.
        """
        n = data_loader_size
        metrics = {
            "acc": round(self.total_acc / n ,3),
            "f1": round(self.total_f1 / n, 3),
            "precision" : round(self.total_precision / n, 3),
            "recall": round(self.total_recall / n, 3),
            "loss": round(self.total_loss / n, 3)
            }
        return metrics
