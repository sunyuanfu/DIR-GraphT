from utils import init_path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

import torch


def get_gnn_trainer(model):
    if model in ['GCN', 'RevGAT', 'SAGE']:
        from core.GNNs.gnn_trainer import GNNTrainer
    else:
        raise ValueError(f'GNN-Trainer for model {model} is not defined')
    return GNNTrainer


class Evaluator:
    def __init__(self, name):
        self.name = name

    def eval(self, input_dict):
        y_true, y_pred = input_dict["y_true"], input_dict["y_pred"]
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        acc_list = []
        roc_auc_list = []
        ap_list = []

        if self.name == "chemhiv" or self.name == "ogbg-hiv":
            for i in range(y_true.shape[1]):
                is_labeled = y_true[:, i] == y_true[:, i]
                # Ensure y_true and y_pred for ROC-AUC calculation are valid
                if len(np.unique(y_true[is_labeled, i])) > 1:  # At least one positive and one negative sample
                    roc_auc = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
                    roc_auc_list.append(roc_auc)
                else:
                    roc_auc_list.append(float('nan'))
            return {'rocauc': np.nanmean(roc_auc_list)}
        
        elif self.name == "chempcba" or self.name == "ogbg-pcba":
            for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
                if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                    # ignore nan values
                    is_labeled = y_true[:,i] == y_true[:,i]
                    ap = average_precision_score(y_true[is_labeled,i], y_pred[is_labeled,i])
                    ap_list.append(ap)
            if len(ap_list) == 0:
                raise RuntimeError('No positively labeled data available. Cannot compute Average Precision.')
            return {'ap': sum(ap_list)/len(ap_list)}

        else:
            for i in range(y_true.shape[1]):
                is_labeled = y_true[:, i] == y_true[:, i]
                correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
                acc_list.append(float(np.sum(correct))/len(correct))

            return {'acc': sum(acc_list)/len(acc_list)}


"""
Early stop modified from DGL implementation
"""


class EarlyStopping:
    def __init__(self, patience=10, path='es_checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False
        if isinstance(path, list):
            self.path = [init_path(p) for p in path]
        else:
            self.path = init_path(path)

    def step(self, acc, model, epoch):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
            self.counter = 0
        es_str = f'{self.counter:02d}/{self.patience:02d} | BestVal={self.best_score:.4f}@E{self.best_epoch}'
        return self.early_stop, es_str

    def save_checkpoint(self, model):
        """Saves model when validation loss decrease."""
        if isinstance(model, list):
            for i, m in enumerate(model):
                torch.save(m.state_dict(), self.path[i])
        else:
            torch.save(model.state_dict(), self.path)
