# trainer.py
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


class Trainer:
    def __init__(self, model, train_loader, test_loader, optimizer, criterion, adj, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.adj = adj
        self.device = device

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X_batch, self.adj)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch, self.adj)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        return acc
