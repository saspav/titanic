import os
from pathlib import Path
import numpy as np
import pandas as pd
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from tqdm import tqdm

from data_process import (RANDOM_SEED, WORK_PATH, get_max_num, DataTransform,
                          make_predict, add_info_to_log, merge_submits)

from set_all_seeds import set_all_seeds

__import__("warnings").filterwarnings('ignore')

set_all_seeds(seed=RANDOM_SEED)


class ConfigNN:
    def __init__(self, hidden_size=128, dropout=0.1, lr=1e-3, batch_size=128, max_epochs=20,
                 num_features=None, num_tar_class=2, **kwargs):
        self.hidden_size = hidden_size
        self.dropout1 = kwargs.get('dropout1', dropout)
        self.dropout2 = kwargs.get('dropout2', dropout)
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = kwargs.get('best_epoch', max_epochs)
        self.num_features = num_features  # количество признаков для подачи на вход
        self.num_tar_class = num_tar_class  # количество классов


class TabularNN(nn.Module):
    def __init__(self, params):
        super().__init__()
        half_hidden_size = params['hidden_size'] // 2
        self.mlp = nn.Sequential(
            nn.Linear(params['num_features'], params['hidden_size']),
            nn.Dropout(params['dropout1']),
            nn.ReLU(),
            nn.Linear(params['hidden_size'], half_hidden_size),
            nn.Dropout(params['dropout2']),
            nn.ReLU(),
            nn.Linear(half_hidden_size, half_hidden_size // 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(half_hidden_size // 2, params['num_tar_class'])
        )

    def forward(self, data):
        x = self.mlp(data)
        tar_class = self.classifier(x)
        return tar_class


class TabularClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, device=None, hidden_size=64, dropout1=0.1, dropout2=0.1, lr=1e-3,
                 batch_size=128, max_epochs=100, patience=10, show_progress=False, **kwargs):
        self.device = 'cpu' if device is None else device
        self.hidden_size = hidden_size
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.show_progress = show_progress
        self.random_state = kwargs.get('random_state', 17)
        self.params = {
            'device': self.device,
            'hidden_size': self.hidden_size,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'patience': self.patience,
            'show_progress': self.show_progress,
            'random_state': self.random_state,
        }
        set_all_seeds(seed=self.random_state)

    @property
    def n_estimators(self):
        return self.params.get('best_epoch', self.params['max_epochs'])

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.params:
                setattr(self, key, value)
                self.params[key] = value
        return self

    def fit(self, X_train, y_train, eval_set=None, verbose=1, **kwargs):
        eval_metric = kwargs.get('eval_metric', ['accuracy'])
        max_epochs = kwargs.get('max_epochs')
        patience = kwargs.get('patience')
        if max_epochs is not None:
            self.params['max_epochs'] = max_epochs
        if patience is not None:
            self.params['patience'] = patience
        self.verbose = verbose
        # Обработка целевых меток
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)

        self.params['num_features'] = X_train.shape[1]
        # Получение количества классов и списка классов
        self.classes_ = self.label_encoder.classes_
        self.params['num_tar_class'] = len(self.classes_)

        # Преобразование входных данных в тензоры
        X_train_tensor = torch.tensor(X_train).float()
        y_train_tensor = torch.tensor(y_train_encoded).long()

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.params['batch_size'],
                                  shuffle=True)

        # Evaluation Set
        if eval_set is not None:
            X_valid, y_valid = eval_set
            y_valid_encoded = self.label_encoder.transform(y_valid)
            X_valid_tensor = torch.tensor(X_valid).float()
            y_valid_tensor = torch.tensor(y_valid_encoded).long()
            valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
            valid_loader = DataLoader(valid_dataset, batch_size=self.params['batch_size'])

        # Инициализация модели. Передаем параметры в TabularNN как словарь
        nn_params = {
            "num_features": X_train.shape[1],
            "hidden_size": self.hidden_size,
            "dropout1": self.dropout1,
            "dropout2": self.dropout2,
            "num_tar_class": len(np.unique(y_train)),
        }
        self.model = TabularNN(nn_params).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'])
        classification_criterion = nn.CrossEntropyLoss().to(self.device)

        # Тренировка модели
        best_loss = np.inf
        early_steps = 0
        best_step = 0

        for epoch in range(self.params['max_epochs']):
            self.model.train()
            running_loss = 0.0

            if self.show_progress:
                loader = tqdm(train_loader, desc='Training')
            else:
                loader = train_loader

            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = classification_criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)

            train_loss = f'Epoch {epoch + 1}/{self.params["max_epochs"]}, ' \
                         f'Train Loss: {epoch_loss:.4f}'

            # Evaluation with valid set
            if eval_set is not None:
                valid_loss = self.evaluate(valid_loader, classification_criterion)
                train_loss += f', Valid Loss: {valid_loss:.5f}'

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_step = epoch + 1
                    early_steps = 0
                else:
                    early_steps += 1
                    if early_steps >= self.patience:
                        if self.verbose:
                            print(f'Early stopping... Best Epoch: {best_step}, '
                                  f'Valid Loss: {best_loss:.5f}')
                        self.params['patience'] = self.patience
                        self.params['best_epoch'] = best_step
                        self.params['best_loss'] = round(best_loss, 7)
                        self.params['deep'] = best_step
                        break
            if self.verbose and not (epoch + 1) % self.verbose:
                print(train_loss)

        return self

    def evaluate(self, data_loader, criterion):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(data_loader.dataset)
        return epoch_loss

    def predict(self, X):
        X_tensor = torch.tensor(X).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, preds = torch.max(outputs, 1)

        return self.label_encoder.inverse_transform(preds.cpu().numpy())

    def predict_proba(self, X):
        X_tensor = torch.tensor(X).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def get_depth(self):
        return self.params['best_epoch']

    get_n_leaves = get_depth


if __name__ == '__main__':
    device = torch.device('cpu')
    print('device:', device)

    # # Пример использования
    # clf = TabularClassifier(hidden_size=128, lr=1e-3, batch_size=128,
    #                         num_epochs=1000, patience=100)
    # clf.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    # predictions = clf.predict(X_valid)
    # probabilities = clf.predict_proba(X_valid)
