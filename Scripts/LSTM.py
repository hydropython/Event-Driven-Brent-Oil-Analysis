import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

class LSTMPricePredictor:
    def __init__(self, data, time_step=60):
        self.data = data
        self.time_step = time_step
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def feature_engineering(self):
        """Convert date to datetime and extract month and year."""
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Year'] = self.data['Date'].dt.year
        self.data.set_index('Date', inplace=True)

    def scale_data(self):
        """Scale the data and create the dataset for LSTM."""
        scaled_data = self.scaler.fit_transform(self.data[['Price', 'Month', 'Year']])
        self.X_train, self.X_test, self.y_train, self.y_test = self.create_dataset(scaled_data)

    def create_dataset(self, data):
        """Create dataset for LSTM from scaled data."""
        X, y = [], []
        for i in range(len(data) - self.time_step - 1):
            a = data[i:(i + self.time_step), :]
            X.append(a)
            y.append(data[i + self.time_step, 0])  # Price as target
        return np.array(X), np.array(y)

    def build_model(self):
        """Build and compile the LSTM model."""
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(25))
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self, epochs=10, batch_size=1):
        """Train the LSTM model."""
        self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=epochs)

    def predict(self):
        """Make predictions using the trained model."""
        predictions = self.model.predict(self.X_test)
        predictions = self.scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 2))), axis=1))
        return predictions

    def calculate_rmse(self, predictions):
        """Calculate the root mean squared error."""
        rmse = np.sqrt(np.mean((predictions[:, 0] - self.scaler.inverse_transform(self.y_test.reshape(-1, 1))) ** 2))
        return rmse

    def plot_predictions(self, predictions):
        """Plot the predictions against the true prices."""
        plt.figure(figsize=(14, 5))
        plt.plot(self.data.index[-len(self.y_test):], self.scaler.inverse_transform(self.y_test.reshape(-1, 1)), label='True Price', color='blue')
        plt.plot(self.data.index[-len(predictions):], predictions[:, 0], label='Predicted Price', color='red')
        plt.title('Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def run(self):
        """Run the full process."""
        self.feature_engineering()
        self.scale_data()
        self.build_model()
        self.train_model()
        predictions = self.predict()
        rmse = self.calculate_rmse(predictions)
        print(f'Root Mean Squared Error: {rmse}')
        self.plot_predictions(predictions)