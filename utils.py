import pandas as pd
import shap
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline

import torch
from torch.autograd import Variable


# Set the device to CUDA if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_results_table_plotly(results):
    # Create the table data
    table_data = []
    for main_key in results:
        for sub_key in results[main_key]:
            mspe = results[main_key][sub_key]['MSPE']
            r2_oos = results[main_key][sub_key]['R2_oos']
            table_data.append([main_key, sub_key, mspe, r2_oos])

    # Create the table trace
    table_trace = go.Table(
        header=dict(values=['Maturity', 'Method', 'MSPE', 'R2_oos'],
                    fill_color='lightgray',
                    align='center'),
        cells=dict(values=[list(x) for x in zip(*table_data)],
                   fill_color='white',
                   align='center'))

    # Create the layout
    layout = go.Layout(
        title='Out-of-Sample Results',
        font=dict(size=12),
        width=800,
        height=600
    )

    # Create the figure and display the plot
    fig = go.Figure(data=[table_trace], layout=layout)
    fig.show()


def plot_results_table_seaborn(results, desc):
    # Create the table data
    table_data = []
    for main_key in results:
        for sub_key in results[main_key]:
            mspe = results[main_key][sub_key]['MSPE']
            r2_oos = results[main_key][sub_key]['R2_oos']
            table_data.append([main_key, sub_key, mspe, r2_oos])

    # Create a DataFrame
    df = pd.DataFrame(table_data, columns=['Maturity', 'Method', 'MSPE', 'R2_oos'])

    # Set the figure size and style
    plt.figure(figsize=(20, 14))
    sns.set(style='whitegrid')

    # Create the heatmap
    sns.heatmap(df.pivot_table(index='Maturity', columns='Method', values='MSPE'),
                cmap='YlGnBu', annot=True, fmt='.2f', cbar_kws={'label': 'MSPE'})

    # Set the plot title and labels
    plt.title('Out-of-Sample Results (MSPE)', fontsize=16)
    plt.xlabel('Method')
    plt.ylabel('Maturity')

    # Adjust the plot layout and display
    plt.tight_layout()
    plt.show()

    # Set the figure size and style
    plt.figure(figsize=(20, 14))
    sns.set(style='whitegrid')

    # Create the heatmap
    sns.heatmap(df.pivot_table(index='Maturity', columns='Method', values='R2_oos'),
                cmap='YlGnBu', annot=True, fmt='.2f', cbar_kws={'label': 'R2_oos'})

    # Set the plot title and labels
    plt.title('Out-of-Sample Results (R2_oos)', fontsize=16)
    plt.xlabel('Method')
    plt.ylabel('Maturity')

    # Adjust the plot layout and display
    plt.tight_layout()
    plt.savefig(f'Graph/{desc}_table_results.png')
    plt.show()


def plot_feature_importance(X, desc, start_month, end_month):
    X = X.loc[(X.index >= start_month) & (X.index <= end_month)]
    # Assuming 'device' is defined globally
    if 'Autoencoder' in desc or 'VarAutoencoder' in desc:
        X_tensor = Variable(torch.from_numpy(X)).float().to(device)
        X_transformed = model.encoder(X_tensor).detach().cpu().numpy()
        explainer = shap.KernelExplainer(model, X_transformed)
        shap_values = explainer.shap_values(X_transformed)
    else:
        model_path = f'Model/{desc}/{desc}_{start_month}_{end_month}.m'
        model = joblib.load(model_path)
        
        if isinstance(model, Pipeline):
            # If the model is a pipeline, use the last step (estimator) for explanation
            estimator = model.steps[-1][1]
            X = model.steps[0][1].fit_transform(X)
            explainer = shap.KernelExplainer(estimator.predict, X)
        else:
            # If the model is a single estimator, use it directly for explanation
            explainer = shap.KernelExplainer(model.predict, X)
        
        shap_values = explainer.shap_values(X)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(f"Feature Importance - {desc}")
    plt.xlabel("SHAP Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f'Graph/{desc}_feature_importance.png')
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt

def plot_pred_and_true_returns(data, maturities, model_name):
    # Convert the 'date' column to datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Create a figure with subplots for each maturity
    fig, axes = plt.subplots(len(maturities), 1, figsize=(10, 4*len(maturities)), sharex=True)

    # Iterate over each maturity
    for i, maturity in enumerate(maturities):
        # Get the corresponding predicted return and true return columns
        pred_return_col = f'{model_name}_{maturity}_pred_return'
        true_return_col = f'{maturity}_excess_return'

        # Plot the predicted return and true return for the current maturity
        axes[i].plot(data['date'], data[pred_return_col], label=f'{maturity} Predicted Return')
        axes[i].plot(data['date'], data[true_return_col], label=f'{maturity} True Return')

        # Set the title and labels for the current subplot
        axes[i].set_title(f'{maturity} Returns by model {model_name}')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Return')

        # Add a legend to the current subplot
        axes[i].legend()

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.savefig(f"Graph/{model_name}_fit_curve.png")
    plt.show()