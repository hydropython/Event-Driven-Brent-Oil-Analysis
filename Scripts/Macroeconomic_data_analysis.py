import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
class EconomicDataProcessor:
    def __init__(self, macro_path, exchange_path, brent_path):
        self.macro_path = macro_path
        self.exchange_path = exchange_path
        self.brent_path = brent_path
        self.daily_macro_df = None
        self.daily_exchange_df = None
        self.final_merged_df = None

        # Set aesthetic parameters for seaborn
        sns.set(style="whitegrid", palette="muted")

    def load_macro_data(self):
        # Load macroeconomic data
        self.macro_df = pd.read_csv(self.macro_path)
        # Clean column names
        self.macro_df.columns = self.macro_df.columns.str.strip()
        # Process macroeconomic data into daily format
        daily_data = []
        for index, row in self.macro_df.iterrows():
            year = int(row['Year'])
            inflation = row['Infliation']
            gdp = row['GDP']
            unemployment = row['Unemployment']
            daily_month_range = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
            for date in daily_month_range:
                daily_data.append((date, inflation, gdp, unemployment))
        self.daily_macro_df = pd.DataFrame(daily_data, columns=['Date', 'Inflation', 'GDP', 'Unemployment'])
        self.daily_macro_df.to_csv('../Data/daily_macroeconomic_data.csv', index=False)

    def load_exchange_data(self):
        # Load exchange rate data
        self.exchange_df = pd.read_csv(self.exchange_path)
        # Clean column names
        self.exchange_df.columns = self.exchange_df.columns.str.strip()
        daily_exchange_data = []
        for index, row in self.exchange_df.iterrows():
            month_year_str = row['Date']
            date = pd.to_datetime(month_year_str, format='%b-%y')
            usd_to_uk = row['USD for United Kingdom']
            usd_to_euro = row['USD FOR EURO']
            usd_to_canada = row['USD FOR CANADA']
            daily_month_range = pd.date_range(start=date.replace(day=1), end=date + pd.offsets.MonthEnd(1) - pd.Timedelta(days=1), freq='D')
            for date in daily_month_range:
                daily_exchange_data.append((date, usd_to_uk, usd_to_euro, usd_to_canada))
        self.daily_exchange_df = pd.DataFrame(daily_exchange_data, columns=['Date', 'USD to UK', 'USD to Euro', 'USD to Canada'])
        self.daily_exchange_df.to_csv('../Data/daily_exchange_rate_data.csv', index=False)

    def merge_data(self):
        # Load daily macroeconomic data
        macro_df = pd.read_csv('../Data/daily_macroeconomic_data.csv')
        macro_df['Date'] = pd.to_datetime(macro_df['Date'])

        # Load daily exchange rate data
        exchange_df = pd.read_csv('../Data/daily_exchange_rate_data.csv')
        exchange_df['Date'] = pd.to_datetime(exchange_df['Date'])

        # Load Brent oil prices data
        brent_df = pd.read_csv(self.brent_path)
        brent_df['Date'] = pd.to_datetime(brent_df['Date'])

        # Merge datasets
        merged_df = pd.merge(macro_df, exchange_df, on='Date', how='inner')
        self.final_merged_df = pd.merge(merged_df, brent_df, on='Date', how='inner')

        # Save final merged DataFrame to a CSV file
        self.final_merged_df.to_csv('../Data/final_merged_data.csv', index=False)

    

    def visualize_data(self):
        # Ensure the directory exists
        output_dir = '../Images/'
        os.makedirs(output_dir, exist_ok=True)

        # Debugging: Print column names and the first few rows
        print("Columns in final_merged_df:", self.final_merged_df.columns.tolist())
        print("Data in final_merged_df:\n", self.final_merged_df.head())

        plt.figure(figsize=(12, 10))

        # Plot GDP
        plt.subplot(3, 1, 1)
        try:
            sns.lineplot(data=self.final_merged_df, x='Date', y='GDP', color='blue')
            plt.title('GDP Over Time')
            plt.xlabel('Date')
            plt.savefig(os.path.join(output_dir, 'gdp_over_time.png'), bbox_inches='tight')  # Save the GDP plot
        except ValueError as e:
            print("Error plotting GDP:", e)

        # Plot Inflation
        plt.subplot(3, 1, 2)
        try:
            sns.lineplot(data=self.final_merged_df, x='Date', y='Inflation', color='orange')
            plt.title('Inflation Over Time')
            plt.xlabel('Date')
            plt.savefig(os.path.join(output_dir, 'inflation_over_time.png'), bbox_inches='tight')  # Save the Inflation plot
        except ValueError as e:
            print("Error plotting Inflation:", e)

        # Plot Price (Brent Oil Prices)
        plt.subplot(3, 1, 3)
        try:
            sns.lineplot(data=self.final_merged_df, x='Date', y='Price', color='teal')
            plt.title('Brent Oil Prices Over Time')
            plt.xlabel('Date')
            plt.savefig(os.path.join(output_dir, 'brent_oil_prices_over_time.png'), bbox_inches='tight')  # Save the Price plot
        except ValueError as e:
            print("Error plotting Price:", e)

        plt.tight_layout()
        plt.show()

        # Correlation Plot
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.final_merged_df.corr()

        # Create a mask to show only the upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(correlation_matrix, mask=mask, cmap='Blues', annot=True, fmt='.2f', 
                    linewidths=.5, cbar_kws={"shrink": .8}, square=True)

        plt.title('Correlation Matrix')
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), bbox_inches='tight')  # Save the Correlation plot
        plt.show()
    def display_data(self):
        # Display the first few rows of the final merged DataFrame
        print(self.final_merged_df.head())