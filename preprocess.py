import os
import numpy as np
import pandas as pd


required_maturities = ['1 Yr', '2 Yr', '3 Yr', '4 Yr', '5 Yr']
feature_cols = ['1 Yr', '2 Yr', '3 Yr', '4 Yr', '5 Yr']


def preprocess(df: pd.DataFrame):

    df['qdate'] = pd.to_datetime(df['qdate'], format='%Y%m%d')
    df['qdate'] = df['qdate'].dt.strftime('%Y-%m')
    df.columns = ['date', '1 Yr', '2 Yr', '3 Yr', '4 Yr', '5 Yr']

    bond_prices = pd.DataFrame()
    for col in required_maturities:
        years = int(col.split(' ')[0])
        bond_prices[col + '_price'] = np.log(1 / ((1 + df[col] / 100) ** years))
        feature_cols.append(col + '_price')

    one_period_yields = pd.DataFrame()
    for col in required_maturities:
        years = int(col.split(' ')[0])
        one_period_yields[col + '_yield'] = bond_prices[col + '_price'] / (-years)
        feature_cols.append(col + '_yield')

    bond_excess_returns = pd.DataFrame()
    for col in required_maturities[1:]:
        years = int(col.split(' ')[0])
        bond_excess_returns[col + '_excess_return'] = (bond_prices[f'{(years-1)} Yr_price'].shift(-12) - bond_prices[col + '_price'] - one_period_yields['1 Yr_yield']) * 100

    preprocessed_data = pd.concat([df, bond_prices, one_period_yields, bond_excess_returns], axis=1)
    preprocessed_data.dropna(inplace=True)
    preprocessed_data['date'] = pd.to_datetime(preprocessed_data['date'])
    preprocessed_data['date'] = preprocessed_data['date'].dt.strftime('%Y-%m')
    preprocessed_data.set_index('date', inplace=True)

    return preprocessed_data


def standardize_features(df, identifier_columns):
    """
    Standardizes the feature columns of the DataFrame using z-score.
    
    Parameters:
    df (pd.DataFrame): The DataFrame with features to standardize.
    identifier_columns (list): List of columns to exclude from standardization.
    
    Returns:
    pd.DataFrame: The DataFrame with standardized feature columns.
    """
    df = df.copy()
    feature_columns = [col for col in df.columns if col in identifier_columns]
    features_array = df[feature_columns].to_numpy()
    
    # Handle NaN and infinite values
    mean = np.nanmean(features_array, axis=1, keepdims=True)
    std = np.nanstd(features_array, axis=1, keepdims=True)
    valid_mask = np.isfinite(features_array)
    
    # Replace NaN and infinite values with 0 before standardization
    features_array[~valid_mask] = 0
    
    # Perform standardization
    standardized_features = (features_array - mean) / std
    standardized_features[~valid_mask] = 0  # Set standardized values for NaN/infinite to 0
    
    df[feature_columns] = standardized_features
    return df