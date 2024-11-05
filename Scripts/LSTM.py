import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shap
import mlflow
import mlflow.keras
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LSTMPricePredictor:
    def __init__(self, data, time_step=60):
        self.data = data
        self.time_step = time_step
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def feature_engineering(self):
        """Convert date to datetime and extract month and year."""
        logging.info("Starting feature engineering.")
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Year'] = self.data['Date'].dt.year
        self.data.set_index('Date', inplace=True)

    def create_dataset(self, data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), 0])  # Use the first column (Price)
            y.append(data[i + time_step, 0])      # Next value as target
        X = np.array(X)
        y = np.array(y)
        return X, y

    def scale_data(self):
        """Scale the data and create the dataset for LSTM."""
        logging.info("Scaling data and creating dataset.")
        scaled_data = self.scaler.fit_transform(self.data[['Price']])
        X, y = self.create_dataset(scaled_data, time_step=10)  # Using 10 time steps
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Reshape input to be [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def build_model(self):
        """Build and compile the LSTM model."""
        logging.info("Building the LSTM model.")
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
        logging.info("Training the LSTM model.")
        self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=epochs)

    def predict(self):
        """Make predictions using the trained model."""
        logging.info("Making predictions.")
        predictions = self.model.predict(self.X_test)
        predictions = self.scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 2))), axis=1))
        return predictions

    def calculate_metrics(self, predictions):
        """Calculate evaluation metrics."""
        logging.info("Calculating evaluation metrics.")
        y_true = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        mse = mean_squared_error(y_true, predictions[:, 0])
        mae = mean_absolute_error(y_true, predictions[:, 0])
        rmse = np.sqrt(mse)
        return mse, mae, rmse

    def plot_predictions(self, predictions):
        """Plot the predictions against the true prices and save the figure."""
        plt.figure(figsize=(14, 5))
        plt.plot(self.data.index[-len(self.y_test):], self.scaler.inverse_transform(self.y_test.reshape(-1, 1)), label='True Price', color='blue')
        plt.plot(self.data.index[-len(predictions):], predictions[:, 0], label='Predicted Price', color='red')
        plt.title('Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        
        # Save the figure
        plt.savefig('../Images/price_prediction_plot.png')  # Save as a PNG file
        plt.close()  # Close the plot to free up memory

    def explain_predictions(self):
        """Explain model predictions using SHAP and save the summary plot and force plot."""
        X_test_array = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1])  # Reshape for KernelExplainer
        predictions = self.model.predict(self.X_test)

        # Initialize the SHAP KernelExplainer
        explainer = shap.KernelExplainer(self.model.predict, X_test_array)
        shap_values = explainer.shap_values(X_test_array)

        # Feature names for each time step in the sequence
        feature_names = [f"Price (t-{i})" for i in range(X_test_array.shape[1])]

        # Summary plot
        shap.summary_plot(shap_values, X_test_array, feature_names=feature_names, show=False)
        plt.savefig('../Images/shap_summary_plot.png')  # Save the SHAP summary plot
        plt.close()  # Close the plot to free up memory

        # Force plot for the first prediction
        shap.initjs()
        force_plot = shap.force_plot(explainer.expected_value, shap_values[0], X_test_array[0], feature_names=feature_names)
        shap.save_html('../Images/shap_force_plot.html', force_plot)

    def save_model(self, model_name):
        """Save the model as a .pkl file and log it using MLflow."""
        os.makedirs('models', exist_ok=True)  # Create a directory for models if it doesn't exist
        model_path = f'models/{model_name}.pkl'
        self.model.save(model_path)  # Save the Keras model
        logging.info(f'Model saved to {model_path}')
        
        # Log model to MLflow
        mlflow.start_run()
        mlflow.log_param("epochs", 10)
        mlflow.log_param("batch_size", 1)
        mlflow.log_metric("mean_squared_error", self.calculate_metrics(self.predict())[0])
        mlflow.keras.log_model(self.model, "model")
        mlflow.end_run()

    def run(self):
        """Run the full process."""
        self.feature_engineering()
        self.scale_data()
        self.build_model()
        self.train_model()
        predictions = self.predict()
        
        # Calculate and print metrics
        mse, mae, rmse = self.calculate_metrics(predictions)
        print(f'Mean Squared Error: {mse}')
        print(f'Mean Absolute Error: {mae}')
        print(f'Root Mean Squared Error: {rmse}')
        
        self.plot_predictions(predictions)
        self.explain_predictions()
        self.save_model("lstm_price_predictor")  # Save model with versioning