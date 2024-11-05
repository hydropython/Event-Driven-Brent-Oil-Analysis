import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import pymc as pm
import ruptures as rpt  # Assuming ruptures is imported for PELT method
from joblib import Parallel, delayed
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChangePointAnalysis:
    def __init__(self, data=None):
        self.data = data if data is not None else pd.DataFrame()
        self.change_points = {}

    def load_data(self, csv_file):
        """Load data from a CSV file."""
        logging.info(f"Loading data from {csv_file}")
        self.data = pd.read_csv(csv_file, parse_dates=['Date'])
        self.data['Price'] = self.data['Price'].astype(float)
        logging.info("Data loaded successfully.")

    def detect_cusum(self):
        """Detect change points using the CUSUM method."""
        logging.info("Starting CUSUM change point detection.")
        price_data = self.data['Price'].values
        mean_price = np.mean(price_data)
        cusum = np.cumsum(price_data - mean_price)
        
        # Identify change points based on CUSUM
        threshold = 5 * np.std(cusum)  # Adjust threshold as needed
        change_points = np.where(np.abs(cusum) > threshold)[0]
        self.change_points['CUSUM'] = change_points.tolist()
        
        logging.info(f"CUSUM detected change points at indices: {change_points}")
        self._plot_cusum(cusum)

    def _plot_cusum(self, cusum):
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['Date'], cusum, label='CUSUM')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('CUSUM Value')
        plt.title('CUSUM Analysis')
        plt.legend()
        plt.show()

    def detect_bayesian(self):
        """Detect change points using Bayesian Change Point Detection."""
        logging.info("Starting Bayesian change point detection.")
        with pm.Model() as model:
            mu_1 = pm.Normal('mu_1', mu=self.data['Price'].mean(), sigma=10)
            mu_2 = pm.Normal('mu_2', mu=self.data['Price'].mean(), sigma=10)
            tau = pm.DiscreteUniform('tau', lower=0, upper=len(self.data)-1)

            likelihood = pm.Normal('y', 
                                mu=pm.math.switch(tau, mu_1, mu_2), 
                                sigma=1, 
                                observed=self.data['Price'].values)

            trace = pm.sample(1000, tune=1000, cores=2)
            logging.info("Bayesian sampling completed.")

        tau_samples = trace['tau']
        change_point_index = int(np.bincount(tau_samples).argmax())  # Use the most frequent tau value
        self.change_points['Bayesian'] = [change_point_index]
        logging.info(f"Bayesian method detected change point at index: {change_point_index}")
        
        self._plot_bayesian_results(trace, change_point_index)

    def _plot_bayesian_results(self, trace, change_point_index):
        pm.plot_posterior(trace, var_names=['tau'])
        plt.title('Posterior Distribution of Change Point')
        plt.show()

        plt.figure(figsize=(14, 7))
        plt.plot(self.data['Date'], self.data['Price'], label='Brent Oil Price')
        plt.axvline(x=self.data['Date'].iloc[change_point_index], color='red', linestyle='--', label='Change Point')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.title('Brent Oil Prices with Detected Change Point')
        plt.legend()
        plt.show()

    def detect_statistical(self):
        """Detect change points using statistical tests (t-test)."""
        logging.info("Starting statistical change point detection using t-test.")
        price_data = self.data['Price'].values

        p_values = Parallel(n_jobs=-1)(
            delayed(self.perform_ttest)(price_data[:i], price_data[i:]) for i in range(1, len(price_data))
        )

        change_points = [i for i, p in enumerate(p_values) if p < 0.05]
        self.change_points['Statistical'] = change_points
        logging.info(f"Statistical method detected change points at indices: {change_points}")

    def perform_ttest(self, segment_1, segment_2):
        """Perform t-test and return the p-value."""
        if len(segment_1) < 2 or len(segment_2) < 2:
            return 1.0  # High p-value if segments are too small
        return ttest_ind(segment_1, segment_2)[1]  # Only return the p-value

    def detect_ruptures(self):
        """Detect change points using the ruptures package."""
        logging.info("Starting ruptures change point detection.")
        price_array = self.data['Price'].values
        model = "rbf"
        algo = rpt.Pelt(model=model).fit(price_array)
        change_points = algo.predict(pen=20)
        self.change_points['Ruptures'] = change_points
        logging.info(f"Ruptures method detected change points at indices: {change_points}")

        self._plot_ruptures(change_points)

    def _plot_ruptures(self, change_points):
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['Date'], self.data['Price'], label='Price')
        for cp in change_points[:-1]:
            plt.axvline(x=self.data['Date'].iloc[cp], color='red', linestyle='--', label='Change Point' if cp == change_points[0] else "")
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.title('Price Data with Detected Change Points (Ruptures)')
        plt.legend()
        plt.show()

    
    def get_change_point_dates(self, filename='../Data/change_point_dates.csv'):
        """Retrieve and save dates of detected change points to a CSV file."""
        change_point_dates = {}
        
        # Collect change point dates for each method
        for method, indices in self.change_points.items():
            change_point_dates[method] = [self.data['Date'].iloc[cp] for cp in indices if 0 <= cp < len(self.data)]
            logging.info(f"Change points detected by {method}: {change_point_dates[method]}")
            print(f"Change points detected by {method}: {change_point_dates[method]}")  # Print each method's dates

        # Convert the dictionary to a DataFrame
        change_point_df = pd.DataFrame.from_dict(change_point_dates, orient='index').transpose()

        # Save the DataFrame to a CSV file
        change_point_df.to_csv(filename, index=False)
        print(f"Change point dates saved to {filename}")

    def plot_results(self):
        """Plot the price data along with the detected change points."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Date'], self.data['Price'], label='Price', marker='o')

        for method, cps in self.change_points.items():
            for cp in cps:
                if 0 <= cp < len(self.data):
                    plt.axvline(x=self.data['Date'].iloc[cp], color='red', linestyle='--',
                                label=f'Change Point ({method})' if cp == cps[0] else "")

        plt.title('Change Point Analysis of Price Data')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid()
        plt.tight_layout()
        plt.show()