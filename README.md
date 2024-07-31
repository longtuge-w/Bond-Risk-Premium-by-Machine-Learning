# Bond Risk Premia with Machine Learning

This project aims to replicate the findings of the paper "Bond Risk Premia with Machine Learning" by Daniele Bianchi, Matthias Büchner, and Andrea Tamoni (2019). The paper explores the use of various machine learning methods for predicting bond excess returns and measuring bond risk premia.

## Overview

The main objectives of this project are:

1. Implement and compare different machine learning methods for predicting bond excess returns, including:
   - Principal Component Regression (PCR)
   - Partial Least Squares (PLS)
   - Penalized Linear Regressions (Ridge, Lasso, Elastic Net)
   - Regression Trees and Random Forests
   - Neural Networks (Shallow and Deep)
   - Autoencoders and Variational Autoencoders (VAEs) for feature extraction
   - Ranking models using LightGBM, XGBoost, and CatBoost

2. Evaluate the out-of-sample predictive performance of these methods using mean squared prediction error (MSPE) and out-of-sample R-squared (R2_oos).

3. Investigate the economic significance of the predictability by calculating Sharpe ratios based on the predicted bond excess returns.

4. Analyze the relative importance of macroeconomic variables in predicting bond excess returns using neural networks and ranking models.

5. Implement a backtesting framework to simulate trading strategies based on the model's predictions and assess their profitability.

## Data

The project uses the following datasets:

1. Yield curve data: U.S. Treasury bond yields for different maturities (1-year, 2-year, 3-year, 5-year).
2. Macroeconomic variables: A large panel of macroeconomic and financial variables.

## Code Structure

The code for this project is organized as follows:

- `main.ipynb`: Jupyter notebook containing the main code for data preprocessing, model training, evaluation, and analysis.
- `utils.py`: Python module containing utility functions for plotting and analysis.
- `preprocess.py`: Python script for data preprocessing and standardization.
- `train_model.py`: Python script for training machine learning models, including neural networks, autoencoders, and ranking models.
- `backtest.py`: Python module for backtesting trading strategies based on the models' predictions.
- `data/`: Directory containing the raw and processed datasets.
- `models/`: Directory to store the trained machine learning models.
- `results/`: Directory to store the evaluation results, backtesting logs, and figures.

## Dependencies

The project requires the following dependencies:

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- PyTorch
- LightGBM
- XGBoost
- CatBoost
- Matplotlib
- Seaborn
- Plotly

## Usage

1. Install the required dependencies.
2. Place the raw datasets in the `data/` directory.
3. Preprocess the data using `preprocess.py`.
4. Train the models using `train_model.py`.
5. Run backtests using `backtest.py` to evaluate the trading strategies.
6. For a comprehensive walkthrough, refer to the `main.ipynb` notebook.

## Results

The main findings of this project are:

1. Machine learning methods, particularly neural networks and ranking models, can capture a significant amount of time-series variation in expected bond excess returns and outperform traditional benchmarks like PCR.
2. Macroeconomic information has substantial out-of-sample forecasting power for bond excess returns, especially when combined with non-linear methods like deep neural networks and feature extraction techniques.
3. The backtesting framework provides a practical assessment of the economic significance of the models' predictions, demonstrating the potential for profitable trading strategies.
4. The composition of the best predictors varies across the term structure, with financial variables being more important for short-term bonds and macroeconomic variables being more relevant for long-term bonds.

For detailed results and analysis, please refer to the `main.ipynb` notebook and the `results/` directory.

## References

Bianchi, D., Büchner, M., & Tamoni, A. (2019). Bond Risk Premia with Machine Learning. Journal of Financial Economics, forthcoming.
