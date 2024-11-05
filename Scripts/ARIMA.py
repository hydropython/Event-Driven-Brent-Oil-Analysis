import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_switching import MarkovSwitching
import matplotlib.pyplot as plt

class ARIMAComparison:
    def __init__(self, data):
        self.data = data
        self.simple_model = None
        self.simple_results = None
        self.ms_model = None
        self.ms_results = None

    def fit_simple_arima(self, order=(1, 1, 1)):
        """Fit a simple ARIMA model to the data."""
        self.simple_model = ARIMA(self.data['Price'], order=order)
        self.simple_results = self.simple_model.fit()
        print("Simple ARIMA Results:")
        print(self.simple_results.summary())

    def fit_markov_switching_arima(self, k_regimes=2, order=(1, 1, 1)):
        """Fit a Markov-Switching ARIMA model to the data."""
        self.ms_model = MarkovSwitching(self.data['Price'], k_regimes=k_regimes, order=order)
        self.ms_results = self.ms_model.fit()
        print("Markov-Switching ARIMA Results:")
        print(self.ms_results.summary())

    def plot_comparison(self):
        """Plot the forecast results of both ARIMA and Markov-Switching ARIMA models."""
        plt.figure(figsize=(12, 8))

        # Plot observed data
        plt.plot(self.data['Price'], label='Observed', color='black')

        # Plot Simple ARIMA fitted values
        if self.simple_results is not None:
            plt.plot(self.simple_results.fittedvalues, label='Fitted (Simple ARIMA)', color='red')

        # Plot Markov-Switching ARIMA fitted values
        if self.ms_results is not None:
            plt.plot(self.ms_results.fittedvalues, label='Fitted (MS-ARIMA)', color='green')

        plt.title('Comparison of Simple ARIMA and Markov-Switching ARIMA Models')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.show()