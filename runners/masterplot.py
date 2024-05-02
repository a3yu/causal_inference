import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_bias_variance(tracked_data, figsize=(10, 5), title='Bias and Variance Evolution'):
    """
    Plot bias and variance for multiple estimators.

    Parameters:
    - tracked_data: dict, keys are estimator names and values are dicts with keys 'tracked_values', 'bias', 'variance'
    - figsize: tuple, the figure size
    - title: str, the title of the plot
    """
    plt.figure(figsize=figsize)
    for estimator_name, data in tracked_data.items():
        x = data['tracked_values']
        y = data['std_dev']
        std_dev = np.sqrt(data['variance'])
        
        # Line plot for bias
        sns.lineplot(x=x, y=y, label=f'Std. dev of {estimator_name}')
        
        plt.fill_between(x, y - np.array(std_dev), y + np.array(std_dev), alpha=0.5, label=f'Std. dev of {estimator_name}', edgecolor='gray')
    
    plt.title(title)
    plt.xlabel('Tracked Parameter')
    plt.ylabel('Std. dev')
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))  # Moving the legend outside the plot
    plt.show()

# Example usage remains the same
