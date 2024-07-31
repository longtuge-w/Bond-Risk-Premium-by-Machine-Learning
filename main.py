# 1. Train model on price dataset
# Dictionary containing models to be evaluated
models = {
    'PCA_3': PCA(n_components=3),
    'PCA_5': PCA(n_components=5),
    'OLS': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'PLS_3': PLSRegression(n_components=3),
    'PLS_5': PLSRegression(n_components=5),
    'GradientBoosting': GradientBoostingRegressor(),
    'RandomForest': RandomForestRegressor(n_jobs=-1),
    'ExtraTrees': ExtraTreesRegressor(n_jobs=-1),
    'NN_2_layer_3_nodes': MLPRegressor(hidden_layer_sizes=(3,)),
    'NN_2_layer_5_nodes': MLPRegressor(hidden_layer_sizes=(5,)),
    'NN_3_layer_3_nodes': MLPRegressor(hidden_layer_sizes=(3, 3)),
    'NN_3_layer_5_nodes': MLPRegressor(hidden_layer_sizes=(5, 5)),
    'NN_4_layer_3_nodes': MLPRegressor(hidden_layer_sizes=(3, 3, 3)),
    'NN_4_layer_pyramid': MLPRegressor(hidden_layer_sizes=(4, 3, 2)),
    'Autoencoder_3': Autoencoder(input_dim=15, encoding_dim=3),
    'Autoencoder_5': Autoencoder(input_dim=15, encoding_dim=5),
    'VarAutoencoder_3': VAE(input_dim=15, latent_dim=3),
    'VarAutoencoder_5': VAE(input_dim=15, latent_dim=5),
}


# Dictionary to store results
results = {}

# Iterate over required maturities for bond excess returns
for maturity in required_maturities[1:]:
    # Copying target variable and feature columns
    y = preprocessed_data[f'{maturity}_excess_return'].copy()
    X = preprocessed_data[feature_cols].copy()
    
    # Initialize results dictionary for current maturity
    results[maturity] = {}
    
    # Iterate over models
    for model_name, model in models.items():
        # Perform recursive forecasting and store results
        mse, r2, _, _ = recursive_forecast(X, y, model, desc=f'{maturity} - {model_name}')
        results[maturity][model_name] = {'MSPE': mse, 'R2_oos': r2}


# 2. Train model on price and macro indicator dataset
models = {
    'PCA_3': PCA(n_components=3),
    'PCA_5': PCA(n_components=5),
    'OLS': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'PLS_3': PLSRegression(n_components=3),
    'PLS_5': PLSRegression(n_components=5),
    'GradientBoosting': GradientBoostingRegressor(),
    'RandomForest': RandomForestRegressor(n_jobs=-1),
    'ExtraTrees': ExtraTreesRegressor(n_jobs=-1),
    'NN_2_layer_3_nodes': MLPRegressor(hidden_layer_sizes=(3,)),
    'NN_2_layer_5_nodes': MLPRegressor(hidden_layer_sizes=(5,)),
    'NN_3_layer_3_nodes': MLPRegressor(hidden_layer_sizes=(3, 3)),
    'NN_3_layer_5_nodes': MLPRegressor(hidden_layer_sizes=(5, 5)),
    'NN_4_layer_3_nodes': MLPRegressor(hidden_layer_sizes=(3, 3, 3)),
    'NN_4_layer_pyramid': MLPRegressor(hidden_layer_sizes=(4, 3, 2)),
    'Autoencoder_3': Autoencoder(input_dim=142, encoding_dim=3),
    'Autoencoder_5': Autoencoder(input_dim=142, encoding_dim=5),
    'VarAutoencoder_3': VAE(input_dim=142, latent_dim=3),
    'VarAutoencoder_5': VAE(input_dim=142, latent_dim=5),
}


results = {}

for maturity in required_maturities[1:]:
    y = preprocessed_data_macro[f'{maturity}_excess_return'].copy()
    X = preprocessed_data_macro[feature_cols+macro_cols].copy()
    
    results[maturity] = {}
    
    for model_name, model in models.items():
        mse, r2, _, _ = recursive_forecast(X, y, model, desc=f'{maturity} - {model_name} - macro')
        results[maturity][model_name] = {'MSPE': mse, 'R2_oos': r2}


# 3. Train some other models
import pandas as pd
from functools import reduce

# Define your models
models = {
    'AdaBoost': AdaBoostRegressor(),
    'LightGBM': LGBMRegressor(n_jobs=-1),
    'Xgboost': XGBRegressor(n_jobs=-1),
    'CatBoost': CatBoostRegressor(verbose=False),
}

import re

# Create a regular expression pattern to match special JSON characters
pattern = re.compile(r'[\{\}\[\]:,]')

y_pred_lst = []

for maturity in required_maturities[1:]:
    y = preprocessed_data_macro[f'{maturity}_excess_return'].copy()
    X = preprocessed_data_macro[feature_cols+macro_cols].copy()

    # Replace any found characters with an underscore, or just remove them
    X.columns = [pattern.sub('_', col) for col in X.columns]
        
    for model_name, model in models.items():
        _, _, y_pred, dateLst = recursive_forecast(X, y, model, desc=f'{maturity} - {model_name} - macro')
        df_i = pd.DataFrame({
            'date': dateLst,
            f'{model_name}_{maturity}_pred_return': y_pred,
        })
        y_pred_lst.append(df_i)

# Assuming your list of DataFrames is named 'dfs'
pred_df = reduce(lambda left, right: pd.merge(left, right, on='date'), y_pred_lst)

excess_ret_df = preprocessed_data_macro[[f'{maturity}_excess_return' for maturity in required_maturities[1:]]].reset_index(drop=False)

pred_df = pd.merge(pred_df, excess_ret_df, how='left', on='date')
pred_df.head()


# 4. Learning to Rank
# Dictionary to store results
results = {}

# Iterate over required maturities for bond excess returns
for maturity in required_maturities[1:]:
    # Copying target variable and feature columns
    y = preprocessed_data_macro[f'{maturity}_excess_return'].copy()
    X = preprocessed_data_macro[feature_cols+macro_cols].copy()

    # Replace any found characters with an underscore, or just remove them
    X.columns = [pattern.sub('_', col) for col in X.columns]

    # Initialize results dictionary for current maturity
    results[maturity] = {}
    
    # Assuming 'date' column represents the query groups
    query_col = 'date'
    
    # Perform recursive forecasting for each ranker model
    # results[maturity]['LightGBM Rank'] = lightgbm_ranker_forecast(X, y, maturity)
    results[maturity]['XGBoost Rank'] = xgboost_ranker_forecast(X, y, maturity)
    # results[maturity]['CatBoost Rank'] = catboost_ranker_forecast(X, y, maturity)