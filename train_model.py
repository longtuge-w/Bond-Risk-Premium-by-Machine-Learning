import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from sklearn.pipeline import Pipeline


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from preprocess import required_maturities

import sys
import re


# Set the device to CUDA if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Autoencoder neural network class
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(True),
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
        )

    def forward(self, x):
        # Encoder pass
        x = self.encoder(x)
        # Decoder pass
        x = self.decoder(x)
        return x


# Variational Autoencoder neural network class
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(True),
        )
        # Variance layers
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
        )

    # Reparameterization trick for VAE
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoder pass
        x = self.encoder(x)
        # Obtain mean and log variance
        mu = self.mu(x)
        log_var = self.log_var(x)
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        # Decoder pass
        x_decoded = self.decoder(z)
        
        return x_decoded, mu, log_var


# Function to train an Autoencoder or Variational Autoencoder model
def train_autoencoder_model(X_train, y_train, desc, model, model_path, epochs, lr):
    """
    Function to train an Autoencoder or Variational Autoencoder model.
    """
    # Convert data to tensors and move to device
    X_train_tensor = Variable(torch.from_numpy(X_train.values)).float().to(device)
    # Load existing model or initialize new one
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    else:
        model.to(device)
        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            # Forward pass
            if 'VarAutoencoder' in desc:
                X_decoded, mu, log_var = model(X_train_tensor)
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = nn.MSELoss()(X_decoded, X_train_tensor) + kl_loss
            else:
                X_decoded = model(X_train_tensor)
                loss = nn.MSELoss()(X_decoded, X_train_tensor)
            # Backpropagation
            loss.backward()
            optimizer.step()
        # Save trained model
        torch.save(model.state_dict(), model_path)
    return model


# Function to train sklearn models
def train_sklearn_model(X_train, y_train, model, desc, model_path, is_pipeline=False):
    """
    Function to train sklearn models.
    """
    # Load existing model or train new one
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        # Train model
        if is_pipeline:
            pipeline_model = Pipeline([(desc, model), ('LR', LinearRegression())])
            pipeline_model.fit(X_train, y_train)
            joblib.dump(pipeline_model, model_path)
            return pipeline_model
        else:
            model.fit(X_train, y_train)
            joblib.dump(model, model_path)
            return model


# Function for recursive forecasting
def recursive_forecast(X, y, model, train_size=0.7, desc=None, epochs=100, lr=1e-3):
    """
    Function for recursive forecasting.
    """
    # Initialize variables
    n_samples = X.shape[0]
    train_end = int(n_samples * train_size)
    start_month = X.index[train_end]
    monthLst = np.sort(pd.unique(X.index))
    start_idx = np.where(start_month == monthLst)[0][0]
    y_pred_list = []
    date_list = []
    model_folder = f"Model/{desc}"
    os.makedirs(model_folder, exist_ok=True)

    # Iterate over months for forecasting
    with tqdm(total=len(monthLst)-start_idx-1, desc=desc) as pbar:
        for i in range(start_idx, len(monthLst)-1):
            # Train-test split for each month
            X_train, y_train = X.loc[X.index <= monthLst[i]], y.loc[y.index <= monthLst[i]]
            X_test, y_test = X.loc[X.index == monthLst[i+1]], y.loc[y.index == monthLst[i+1]]
            start_date, end_date = monthLst[0], monthLst[i]
            model_name = f"{desc}_{start_date}_{end_date}"
            model_path = os.path.join(model_folder, f"{model_name}.m")
            
            # Train model and make predictions
            if 'Autoencoder' in desc or 'VarAutoencoder' in desc:
                model = train_autoencoder_model(X_train, y_train, desc, model, model_path, epochs, lr)
                X_train_tensor = Variable(torch.from_numpy(X_train.values)).float().to(device)
                X_test_tensor = Variable(torch.from_numpy(X_test.values)).float().to(device)
                X_test_transformed = model.encoder(X_test_tensor).detach().cpu().numpy()
                lr_model = LinearRegression()
                lr_model.fit(model.encoder(X_train_tensor).detach().cpu().numpy(), y_train)
                y_pred = lr_model.predict(X_test_transformed)
            elif 'PCA' in desc:
                if os.path.exists(model_path):
                    modelfitted = joblib.load(model_path)
                else:
                    modelfitted = train_sklearn_model(X_train, y_train, model, desc, model_path, is_pipeline=True)
                y_pred = modelfitted.predict(X_test)
            else:
                if os.path.exists(model_path):
                    modelfitted = joblib.load(model_path)
                else:
                    modelfitted = train_sklearn_model(X_train, y_train, model, desc, model_path)
                y_pred = modelfitted.predict(X_test)
            
            # Append predictions and update progress bar
            y_pred_list.append(y_pred)
            date_list.append(end_date)
            pbar.set_postfix({'Month': end_date})
            pbar.update(1)
    
    # Concatenate predictions and calculate evaluation metrics
    y_pred = np.concatenate(y_pred_list)
    y_true = y.loc[y.index > start_month]
    mse = mean_squared_error(y_true, y_pred)
    r2 = 1 - mse / np.var(y_true)
    
    return mse, r2, y_pred, date_list

# Create a regular expression pattern to match special JSON characters
pattern = re.compile(r'[\{\}\[\]:,]')

# Function to get predicted excess returns using only bond features
def get_predicted_excess_returns(data, models, train_size=0.7, epochs=100, lr=1e-3):
    predicted_returns = {}
    # Loop through each maturity except the first one
    for maturity in required_maturities[1:]:
        # Extract features and target for the current maturity
        X = data[feature_cols]
        y = data[f'{maturity}_excess_return']
        predicted_returns[maturity] = {}
        # Loop through each model
        for model_name, model in models.items():
            # Description for the model used in logging
            desc = f'{maturity} - {model_name}'
            # Perform recursive forecasting and get predictions and dates
            _, _, preds, dates = recursive_forecast(X, y, model, train_size, desc, epochs, lr)
            # Store predicted returns for the current model
            predicted_returns[maturity][model_name] = {'pred_return': preds}
    return predicted_returns, dates

# Function to get predicted excess returns using both bond and macro features
def get_predicted_excess_returns_macro(data, models, train_size=0.7, epochs=100, lr=1e-3, change_col=False):
    predicted_returns = {}
    # Loop through each maturity except the first one
    for maturity in required_maturities[1:]:
        # Extract features (bond and macro) and target for the current maturity
        X = data[feature_cols+macro_cols]
        y = data[f'{maturity}_excess_return']

        if change_col:
            # Replace any found characters with an underscore, or just remove them
            X.columns = [pattern.sub('_', col) for col in X.columns]

        predicted_returns[maturity] = {}
        # Loop through each model
        for model_name, model in models.items():
            # Description for the model used in logging
            desc = f'{maturity} - {model_name} - macro'
            # Perform recursive forecasting and get predictions and dates
            _, _, preds, dates = recursive_forecast(X, y, model, train_size, desc, epochs, lr)
            # Store predicted returns for the current model
            predicted_returns[maturity][model_name] = {'pred_return': preds}
    return predicted_returns, dates


import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRanker, Pool
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import re

# Create a regular expression pattern to match special JSON characters
pattern = re.compile(r'[\{\}\[\]:,]')

def lightgbm_ranker(X, y, model_path):

    X, y = X.copy(), y.copy()
    X.index = pd.to_datetime(X.index)
    y.index = pd.to_datetime(y.index)

    y_ranked = y.groupby(y.index.year).apply(lambda x: pd.qcut(x, q=12, labels=False, duplicates='drop'))
    y_ranked = y_ranked.reset_index(level=0, drop=True)

    # Drop NaN values from y_ranked and the corresponding data points in X
    y_ranked = y_ranked.dropna()
    valid_indices = y_ranked.index
    X = X.loc[valid_indices]

    group_sizes = X.index.year.value_counts().sort_index().tolist()
    
    if os.path.exists(model_path):
        model = lgb.Booster(model_file=model_path)
    else:
        # Prepare data for LightGBM
        train_data = lgb.Dataset(X, label=y_ranked, group=group_sizes)
        
        # Set LightGBM parameters for ranking
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'eval_at': [1, 3, 5],
            'verbosity': -1
        }
        
        # Train LightGBM model
        model = lgb.train(params, train_data)
        
        # Save trained model
        model.save_model(model_path)
    
    return model

# Function for XGBoost ranker
def xgboost_ranker(X, y, model_path):

    X, y = X.copy(), y.copy()
    X.index = pd.to_datetime(X.index)
    y.index = pd.to_datetime(y.index)

    y_ranked = y.groupby(y.index.year).apply(lambda x: pd.qcut(x, q=12, labels=False, duplicates='drop'))
    y_ranked = y_ranked.reset_index(level=0, drop=True)

    # Drop NaN values from y_ranked and the corresponding data points in X
    y_ranked = y_ranked.dropna()
    valid_indices = y_ranked.index
    X = X.loc[valid_indices]

    group_sizes = X.index.year.value_counts().sort_index().tolist()

    if os.path.exists(model_path):
        model = xgb.Booster(model_file=model_path)
    else:
        train_data = xgb.DMatrix(X, label=y_ranked)
        train_data.set_group(group_sizes)

        params = {
            'objective': 'rank:ndcg',
            'eval_metric': 'ndcg@1-',
            'verbosity': 0
        }

        model = xgb.train(params, train_data)
        model.save_model(model_path)

    return model

# Function for CatBoost ranker
def catboost_ranker(X, y, model_path):

    X, y = X.copy(), y.copy()
    X.index = pd.to_datetime(X.index)
    y.index = pd.to_datetime(y.index)

    y_ranked = y.groupby(y.index.year).apply(lambda x: pd.qcut(x, q=12, labels=False, duplicates='drop'))
    y_ranked = y_ranked.reset_index(level=0, drop=True)

    # Drop NaN values from y_ranked and the corresponding data points in X
    y_ranked = y_ranked.dropna()
    valid_indices = y_ranked.index
    X = X.loc[valid_indices]

    if os.path.exists(model_path):
        model = CatBoostRanker().load_model(model_path)
    else:
        train_pool = Pool(X, y_ranked, group_id=X.index.year.tolist())

        params = {
            'loss_function': 'YetiRank',
            'verbose': False
        }

        model = CatBoostRanker(**params)
        model.fit(train_pool)
        model.save_model(model_path)

    return model


# def lightgbm_ranker(X, y, model_path):
#     X, y = X.copy(), y.copy()
#     X.index = pd.to_datetime(X.index)
#     y.index = pd.to_datetime(y.index)

#     y = pd.qcut(y, q=10, labels=False, duplicates='drop')

#     if os.path.exists(model_path):
#         model = lgb.Booster(model_file=model_path)
#     else:
#         # Prepare data for LightGBM
#         train_data = lgb.Dataset(X, label=y, group=[X.shape[0]])

#         # Set LightGBM parameters for ranking
#         params = {
#             'objective': 'lambdarank',
#             'metric': 'ndcg',
#             'eval_at': [1, 3, 5],
#             'verbosity': -1
#         }

#         # Train LightGBM model
#         model = lgb.train(params, train_data)

#         # Save trained model
#         model.save_model(model_path)

#     return model

# # Function for XGBoost ranker
# def xgboost_ranker(X, y, model_path):
#     X, y = X.copy(), y.copy()
#     X.index = pd.to_datetime(X.index)
#     y.index = pd.to_datetime(y.index)

#     y = pd.qcut(y, q=10, labels=False, duplicates='drop')

#     if os.path.exists(model_path):
#         model = xgb.Booster(model_file=model_path)
#     else:
#         train_data = xgb.DMatrix(X, label=y)
#         train_data.set_group([X.shape[0]])

#         params = {
#             'objective': 'rank:ndcg',
#             'eval_metric': 'ndcg@1-',
#             'verbosity': 0
#         }

#         model = xgb.train(params, train_data)
#         model.save_model(model_path)

#     return model

# # Function for CatBoost ranker
# def catboost_ranker(X, y, model_path):
#     X, y = X.copy(), y.copy()
#     X.index = pd.to_datetime(X.index)
#     y.index = pd.to_datetime(y.index)

#     y = pd.qcut(y, q=10, labels=False, duplicates='drop')

#     if os.path.exists(model_path):
#         model = CatBoostRanker().load_model(model_path)
#     else:
#         train_pool = Pool(X, y, group_id=[0] * X.shape[0])

#         params = {
#             'loss_function': 'YetiRank',
#             'verbose': False
#         }

#         model = CatBoostRanker(**params)
#         model.fit(train_pool)
#         model.save_model(model_path)

#     return model


# Function for recursive forecasting with LightGBM ranker
def lightgbm_ranker_forecast(X, y, maturity, train_size=0.7):
    # Initialize variables
    n_samples = X.shape[0]
    train_end = int(n_samples * train_size)
    start_month = X.index[train_end]
    monthLst = np.sort(pd.unique(X.index))
    start_idx = np.where(start_month == monthLst)[0][0]
    y_pred_list = []
    dateLst = []
    model_folder = f"Model/{maturity} - LightGBM Rank"
    os.makedirs(model_folder, exist_ok=True)

    # Iterate over months for forecasting
    with tqdm(total=len(monthLst)-start_idx-1, desc=f'{maturity} - LightGBM Rank') as pbar:
        for i in range(start_idx, len(monthLst)-1):
            # Train-test split for each month
            X_train, y_train = X.loc[X.index <= monthLst[i]], y.loc[y.index <= monthLst[i]]
            test_X_current = X.loc[X.index == monthLst[i+1]]
            start_date, end_date = monthLst[0], monthLst[i]
            model_name = f"LightGBM Rank_{start_date}_{end_date}"
            model_path = os.path.join(model_folder, f"{model_name}.txt")
            
            # Load pretrained model if exists, else train a new model
            if os.path.exists(model_path):
                model = lgb.Booster(model_file=model_path)
            else:
                model = lightgbm_ranker(X_train, y_train, model_path)
            
            # Make predictions on test_X_current
            y_pred = model.predict(test_X_current)
            
            # Append predictions and update progress bar
            y_pred_list.append(y_pred)
            dateLst.append(end_date)
            pbar.update(1)

    return np.concatenate(y_pred_list), dateLst


# Function for recursive forecasting with XGBoost ranker
def xgboost_ranker_forecast(X, y, maturity, train_size=0.7):

    # Initialize variables
    n_samples = X.shape[0]
    train_end = int(n_samples * train_size)
    start_month = X.index[train_end]
    monthLst = np.sort(pd.unique(X.index))
    start_idx = np.where(start_month == monthLst)[0][0]
    y_pred_list = []
    dateLst = []
    model_folder = f"Model/{maturity}_XGBoost Rank"
    os.makedirs(model_folder, exist_ok=True)

    # Iterate over months for forecasting
    with tqdm(total=len(monthLst)-start_idx-1, desc=f'{maturity} - XGBoost Rank') as pbar:
        for i in range(start_idx, len(monthLst)-1):
            # Train-test split for each month
            X_train, y_train = X.loc[X.index <= monthLst[i]], y.loc[y.index <= monthLst[i]]
            test_X_current = X.loc[X.index == monthLst[i+1]]
            start_date, end_date = monthLst[0], monthLst[i]
            model_name = f"XGBoost Rank_{start_date}_{end_date}"
            model_path = os.path.join(model_folder, f"{model_name}.json")
            
            # Load pretrained model if exists, else train a new model
            if os.path.exists(model_path):
                model = xgb.Booster(model_file=model_path)
            else:
                model = xgboost_ranker(X_train, y_train, model_path)
            
            # Make predictions on test_X_current
            test_data = xgb.DMatrix(test_X_current)
            y_pred = model.predict(test_data)
            
            # Append predictions and update progress bar
            y_pred_list.append(y_pred)
            dateLst.append(end_date)
            pbar.update(1)
    
    return np.concatenate(y_pred_list), dateLst


# Function for recursive forecasting with CatBoost ranker
def catboost_ranker_forecast(X, y, maturity, train_size=0.7):
    # Initialize variables
    n_samples = X.shape[0]
    train_end = int(n_samples * train_size)
    start_month = X.index[train_end]
    monthLst = np.sort(pd.unique(X.index))
    start_idx = np.where(start_month == monthLst)[0][0]
    y_pred_list = []
    dateLst = []
    model_folder = f"Model/{maturity}_CatBoost Rank"
    os.makedirs(model_folder, exist_ok=True)

    # Iterate over months for forecasting
    with tqdm(total=len(monthLst)-start_idx-1, desc=f'{maturity} - CatBoost Rank') as pbar:
        for i in range(start_idx, len(monthLst)-1):
            # Train-test split for each month
            X_train, y_train = X.loc[X.index <= monthLst[i]], y.loc[y.index <= monthLst[i]]
            test_X_current = X.loc[X.index == monthLst[i+1]]
            start_date, end_date = monthLst[0], monthLst[i]
            model_name = f"CatBoost Rank_{start_date}_{end_date}"
            model_path = os.path.join(model_folder, f"{model_name}.cbm")
            
            # Load pretrained model if exists, else train a new model
            if os.path.exists(model_path):
                model = CatBoostRanker().load_model(model_path)
            else:
                model = catboost_ranker(X_train, y_train, model_path)
            
            # Make predictions on test_X_current
            y_pred = model.predict(test_X_current)
            
            # Append predictions and update progress bar
            y_pred_list.append(y_pred)
            dateLst.append(end_date)
            pbar.update(1)
    
    return np.concatenate(y_pred_list), dateLst


def get_predicted_excess_returns_macro_lightgbm(data, train_size=0.7, change_col=True):
    
    predicted_returns = {}
    
    # Loop through each maturity except the first one
    for maturity in required_maturities[1:]:
        # Extract features (bond and macro) and target for the current maturity
        X = data[feature_cols+macro_cols]
        y = data[f'{maturity}_excess_return']
        
        if change_col:
            # Replace any found characters with an underscore, or just remove them
            X.columns = [pattern.sub('_', col) for col in X.columns]
        
        predicted_returns[maturity] = {}
        
        # Perform recursive forecasting and get predictions and dates
        preds, dates = lightgbm_ranker_forecast(X, y, maturity, train_size)
        
        # Store predicted returns for the current model
        predicted_returns[maturity]['LightGBM Rank'] = {'pred_return': preds}
    
    return predicted_returns, dates


def get_predicted_excess_returns_macro_xgboost(data, train_size=0.7, change_col=False):

    predicted_returns = {}
    
    # Loop through each maturity except the first one
    for maturity in required_maturities[1:]:
        # Extract features (bond and macro) and target for the current maturity
        X = data[feature_cols+macro_cols]
        y = data[f'{maturity}_excess_return']
        
        if change_col:
            # Replace any found characters with an underscore, or just remove them
            X.columns = [pattern.sub('_', col) for col in X.columns]
        
        predicted_returns[maturity] = {}
        
        # Perform recursive forecasting and get predictions and dates
        preds, dates = xgboost_ranker_forecast(X, y, maturity, train_size)
        
        # Store predicted returns for the current model
        predicted_returns[maturity]['XGBoost Rank'] = {'pred_return': preds}
    
    return predicted_returns, dates


def get_predicted_excess_returns_macro_catboost(data, train_size=0.7, change_col=False):
    predicted_returns = {}
    
    # Loop through each maturity except the first one
    for maturity in required_maturities[1:]:
        # Extract features (bond and macro) and target for the current maturity
        X = data[feature_cols+macro_cols]
        y = data[f'{maturity}_excess_return']
        
        if change_col:
            # Replace any found characters with an underscore, or just remove them
            X.columns = [pattern.sub('_', col) for col in X.columns]
        
        predicted_returns[maturity] = {}
        
        # Perform recursive forecasting and get predictions and dates
        preds, dates = catboost_ranker_forecast(X, y, maturity, train_size)
        
        # Store predicted returns for the current model
        predicted_returns[maturity]['CatBoost Rank'] = {'pred_return': preds}
    
    return predicted_returns, dates