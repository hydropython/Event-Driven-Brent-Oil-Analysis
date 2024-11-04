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
        # Your provided data
        data = {
            'Start Date': [
                '1990-01-01', '2003-01-01', '2010-01-01', '2014-01-01', 
                '2014-01-01', '2007-01-01', '2020-01-01', '2021-01-01', 
                '2022-01-01', '2016-01-01', '2015-01-01', '2015-01-01', 
                '2018-01-01', '2019-01-01', '2021-01-01'
            ],
            'End Date': [
                '1991-01-01', '2003-12-31', '2011-12-31', '2016-12-31', 
                '2022-12-31', '2008-12-31', '2020-04-30', '2021-12-31', 
                '2022-12-31', '2016-12-31', '2015-12-31', '2022-12-31', 
                '2018-12-31', '2019-12-31', '2022-12-31'
            ],
            'Category': [
                'Geopolitical Tensions', 'Geopolitical Tensions', 'Geopolitical Tensions', 
                'Geopolitical Tensions', 'Economic Events', 'Economic Events', 
                'Economic Events', 'Economic Events', 'Geopolitical Tensions', 
                'Policy Changes', 'Policy Changes', 'Policy Changes', 
                'Economic Events', 'Geopolitical Tensions', 'Economic Events'
            ],
            'Event': [
                'Gulf War', 'Iraq War', 'Arab Spring', 'ISIS Insurgency', 
                'U.S. Shale Boom', 'Global Financial Crisis', 
                'COVID-19 Pandemic', 'Economic Recovery Post-COVID', 
                'Russia-Ukraine Conflict', 'OPEC Production Cuts', 
                'Paris Agreement', 'Renewable Energy Policies', 
                'U.S. Economic Boom', 'Tensions with Iran', 'Global Supply Chain Issues'
            ],
            'Description': [
                "Iraq's invasion of Kuwait in August 1990 led to fears of supply disruptions, causing Brent oil prices to spike from $20 to over $40 per barrel.",
                "The U.S.-led invasion of Iraq in March 2003 resulted in instability in the Middle East, leading to fluctuations in oil prices, peaking at $37.73 in April 2003.",
                "Political unrest across several Middle Eastern countries caused concerns over oil production disruptions, contributing to a rise in prices during 2011, reaching $125 per barrel.",
                "The rise of ISIS in Iraq and Syria created instability in the region, contributing to fluctuating oil prices.",
                "The increase in shale oil production led to a significant rise in U.S. oil output, causing a drop in prices around 2014-2015, from over $100 to below $30 per barrel.",
                "The financial crisis led to a severe recession, significantly decreasing oil demand and causing prices to fall from $147 in July 2008 to around $32 in December 2008.",
                "Lockdowns and reduced travel due to the pandemic resulted in a historic drop in oil demand, causing prices to plummet to negative values in April 2020.",
                "As economies reopened, a surge in demand led to a rapid increase in oil prices, reaching $70 per barrel by mid-2021.",
                "Russia's invasion of Ukraine in February 2022 led to significant sanctions on Russian oil, causing Brent prices to surge, hitting $130 per barrel in March 2022.",
                "OPEC agreed to cut production to stabilize prices after the oil price crash of 2014-2015. This decision led to price increases in 2016.",
                "The global commitment to reduce carbon emissions may lead to long-term decreases in oil consumption, influencing pricing dynamics.",
                "Increased emphasis on renewable energy and environmental regulations impacts long-term oil demand and prices, pushing prices lower due to reduced reliance on fossil fuels.",
                "A strong economy boosted oil demand, with prices rising in late 2018, reaching around $80 per barrel.",
                "Increased tensions and threats to oil shipping routes in the Strait of Hormuz affected oil prices, contributing to volatility in 2019.",
                "Ongoing disruptions in supply chains and logistics during the pandemic impacted oil availability, influencing prices upward as economies recovered."
            ],
        }

        # Create events DataFrame from the provided data
        events_df = pd.DataFrame(data)
        events_df['Start Date'] = pd.to_datetime(events_df['Start Date'])
        events_df['End Date'] = pd.to_datetime(events_df['End Date'])

        # Store events_df as an instance attribute
        self.events_df = events_df

        # Merge the events DataFrame with the original Brent data
        merged_data = pd.merge(self.data, events_df, how='left', left_on='Date', right_on='Start Date')
        merged_data['Event'] = merged_data['Event'].fillna('No Event')

        # Save the new CSV
        updated_file_path = '../Data/BrentOilPrices_with_Events.csv'
        merged_data.to_csv(updated_file_path, index=False)
        print(f"Updated CSV saved as {updated_file_path}")

    def plot_with_events(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['Date'], self.data['Price'], color='royalblue', linewidth=2, label='Brent Oil Price')

        # Highlight events
        if self.events_df is not None and not self.events_df.empty:
            for i, row in self.events_df.iterrows():
                start_date = row['Start Date']
                end_date = row['End Date']
                event_name = row['Event']
                # Draw broken green lines for the event duration
                plt.axvline(x=start_date, color='green', linestyle='--', linewidth=2)
                plt.axvline(x=end_date, color='green', linestyle='--', linewidth=2)
                
                # Label the event above the price line with vertical text
                mid_date = start_date + (end_date - start_date) / 2
                mid_price = self.data['Price'].max() + (self.data['Price'].max() - self.data['Price'].min()) * 0.05  # Position above the max price
                plt.annotate(event_name, (mid_date, mid_price), textcoords="offset points", xytext=(0, 15),  # Adjust y-offset
                            ha='center', fontsize=10, color='red', 
                            rotation=90,  # Set rotation for vertical text
                            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        # Format the x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.title('Brent Oil Prices with Significant Events', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price (USD)', fontsize=14)
        plt.legend()
        plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Set the plot to be journal-fitted
        plt.tight_layout()

        # Saving the plot
        image_path = '../Images/brent_oil_prices_with_events.png'
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.savefig(image_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Plot with events saved as {image_path}")