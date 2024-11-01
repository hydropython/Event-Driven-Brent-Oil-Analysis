import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

class BrentOilPricesAnalysis:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, parse_dates=['Date'])
        self.data.sort_values('Date', inplace=True)
        self.events_df = None  # Initialize events_df

    def display_statistics(self):
        print(self.data.describe())

    def plot_time_series(self):
        plt.figure(figsize=(14, 7))
        
        # Plotting the time series data
        plt.plot(self.data['Date'], self.data['Price'], color='royalblue', linewidth=2)

        # Formatting the date on the x-axis to show every 5 years
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(5))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_minor_locator(mdates.YearLocator(1))

        plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.title('Brent Oil Prices Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price (USD)', fontsize=14)
        plt.style.use('ggplot')

        # Saving the plot
        image_path = '../Images/brent_oil_prices.png'
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.savefig(image_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Plot saved as {image_path}")

    def add_events_data(self):
        events_data = {
            'Date': [
                '1990-08-02',
                '1991-01-17',
                '1998-08-20',
                '2001-09-11',
                '2003-03-20',
                '2008-09-15',
                '2011-03-11',
                '2014-03-01',
                '2016-06-23',
                '2020-03-11',
                '2022-02-24',
            ],
            'Event': [
                'Gulf War starts',
                'Gulf War ends',
                'Russian financial crisis',
                '9/11 attacks',
                'Iraq War starts',
                'Financial crisis begins',
                'Fukushima disaster',
                'Annexation of Crimea',
                'Brexit referendum',
                'COVID-19 declared a pandemic',
                'Russia invades Ukraine',
            ]
        }
        
        events_df = pd.DataFrame(events_data)
        events_df['Date'] = pd.to_datetime(events_df['Date'])
        
        # Merge the events DataFrame with the original Brent data
        merged_data = pd.merge(self.data, events_df, on='Date', how='left')
        merged_data['Event'] = merged_data['Event'].fillna('No Event')

        # Save the new CSV
        updated_file_path = '../Data/BrentOilPrices_with_Events.csv'
        merged_data.to_csv(updated_file_path, index=False)
        print(f"Updated CSV saved as {updated_file_path}")

        # Store events_df as an instance attribute
        self.events_df = events_df

    def plot_with_events(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['Date'], self.data['Price'], color='royalblue', linewidth=2, label='Brent Oil Price')

        # Highlight events
        if self.events_df is not None and not self.events_df.empty:
            for i, event in enumerate(self.events_df['Event']):
                event_date = self.events_df['Date'].iloc[i]
                if event_date in self.data['Date'].values:
                    price_at_event = self.data.loc[self.data['Date'] == event_date, 'Price'].values[0]
                    plt.annotate(event, (event_date, price_at_event),
                                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10,
                                 color='red', arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        # Format the x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.title('Brent Oil Prices with Significant Events', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price (USD)', fontsize=14)
        plt.legend()
        plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Saving the plot
        image_path = '../Images/brent_oil_prices_with_events.png'
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.savefig(image_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Plot with events saved as {image_path}")