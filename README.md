# Event-Driven-Brent-Oil-Analysis

Birhan Energies is dedicated to providing strategic insights to help clients navigate the volatile global oil market. This project aims to analyze and quantify the impact of significant political and economic events on Brent oil prices over the past decade. By leveraging data-driven risk management and decision-making intelligence, the project delivers clear, actionable insights that enhance understanding of price movements, support strategic investment decisions, and inform policy and operational planning.

# Overview
Birhan Energies is dedicated to providing strategic insights to help clients navigate the volatile global oil market. This project aims to analyze and quantify the impact of significant political and economic events on Brent oil prices over the past decade. By leveraging data-driven risk management and decision-making intelligence, the project delivers clear, actionable insights that enhance understanding of price movements, support strategic investment decisions, and inform policy and operational planning.
# Table of Contents
•	Objective
•	Constraints
•	Data Collection
•	Data Preprocessing
•	Exploratory Data Analysis (EDA)
•	Feature Engineering
•	Model Building & Evaluation
•	Deployment
•	Interactive Dashboard
•	Installation
•	Usage
•	Contributing
•	License
# Objective
The primary goal of this project is to analyze historical Brent oil prices and understand how various political, economic, and technological factors influence these prices. This will help investors, policymakers, and energy companies make informed decisions in a complex market environment.
# Constraints
•	Data Limitations: Availability and accuracy of historical event data may impact analysis.
•	Time Constraints: Timely delivery is crucial for effective strategy implementation.
•	Complex Market Factors: Isolating event-specific impacts amid broader market influences presents challenges.
# Data Collection
Data collection involves gathering historical daily Brent oil prices and significant events to identify patterns influenced by external factors.
# Data Requirements
•	Historical daily Brent oil prices from May 20, 1987, to November 14, 2022.
•	Event data covering major occurrences such as:
o	Geopolitical tensions (e.g., wars, sanctions)
o	Economic events (e.g., recessions, booms)
o	Policy changes (e.g., OPEC decisions, environmental regulations)
# Data Sources
•	Historical price data (Brent oil prices).
•	Event data from credible sources such as the World Bank, IMF, and industry reports.
# Data Preprocessing
The data preprocessing stage includes transforming raw data into a consistent, structured format for accurate analysis.
•	Handling missing values: Identify and impute or remove gaps in the data.
•	Outlier management: Detect and assess outliers that could distort trends.
•	Date format consistency: Standardize date entries.
•	Scaling: Normalize price data for comparability in trend analysis.
# Exploratory Data Analysis (EDA)
In this stage, we analyze the data's main characteristics to uncover patterns and refine the project scope.
•	Examine the distribution of Brent oil prices over time.
•	Identify long-term trends and seasonal patterns.
•	Explore correlations with significant events.
•	Spot unusual price fluctuations linked to key events.
# Feature Engineering
Creating and selecting relevant features to enhance model accuracy and insight extraction.
•	Event-Based Features: Indicators for political, economic, and OPEC-related events.
•	Temporal Features: Month, year, or day-of-week features to capture seasonal effects.
•	Lagged Variables: Past price values as predictors to capture historical influence.
•	Feature Selection: Retain only impactful features to improve model performance.
# Model Building & Evaluation
# Model Selection
Utilize various models to predict Brent oil prices:
•	Time series models (e.g., ARIMA, VAR)
•	Machine learning models (e.g., LSTM networks)
# Model Evaluation
Evaluate models using performance metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared to ensure accuracy.
# Deployment
Backend (Flask)
•	Develop RESTful APIs to serve analysis results.
•	Handle requests for different datasets, model outputs, and performance metrics.
Frontend (React)
•	Build an intuitive interface for displaying results.
•	Utilize libraries like Recharts or D3.js for engaging visualizations.
# Interactive Dashboard
An interactive dashboard will be developed using Flask and React to visualize the results of the analysis. This will allow stakeholders to explore how various events affect Brent oil prices.

# Installation
To set up the project, follow these steps:

Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/brent-oil-analysis.git
Navigate to the project directory:
bash


git clone https://github.com/yourusername/brent-oil-analysis.git



Copy code
pip install -r requirements.txt
Usage
To run the application, execute the following command in your terminal:

bash
Copy code
flask run
Then open your browser and go to http://localhost:5000 to access the dashboard.

Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any improvements or suggestions.

License
This project is licensed under the MIT License. See the LICENSE file for details.