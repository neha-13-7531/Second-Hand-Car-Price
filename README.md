# Second-Hand-Car-Price

# Objective
To build a machine learning model that accurately predicts the price of second-hand cars based on various factors such as brand, model, year, mileage, fuel type, and transmission.

# Data Source 
"https://www.kaggle.com/datasets/sujithmandala/second-hand-car-price-prediction"

# Features in Data set
- Car_ID: A unique identifier for each car listing.
- Brand: The brand or manufacturer of the car (e.g., Toyota, Honda, Ford, etc.).
- Model: The model of the car (e.g., Camry, Civic, Mustang, etc.).
- Year: The manufacturing year of the car.
- Kilometers_Driven: The total kilometers driven by the car.
- Fuel_Type: The type of fuel used by the car (e.g., Petrol, Diesel, Electric, etc.).
- Transmission: The transmission type of the car (e.g., Manual, Automatic).
- Owner_Type: The number of previous owners of the car (e.g., First, Second, Third).
- Mileage: The fuel efficiency of the car in kilometers per liter.
- Engine: The engine capacity of the car in CC (Cubic Centimeters).
- Power: The maximum power output of the car in bhp (Brake Horsepower).
- Seats: The number of seats available in the car.
- Price: The selling price of the car in INR (Indian Rupees), which is the target variable to predict.

# Steps perform 
1. Data Collection & Loading
- Import libraries like numpy,seaborn,matplotlib.
- Load the dataset using pandas library.
- Inspect the first few rows using df.head() to understand the structure of data set.
  
2. EDA
- Check dataset structure: Use df.info() and df.describe() to understand data types and summary statistics.
- Identify missing values: Use df.isnull().sum() to detect any missing data.
- Check duplicates: Use df.duplicated().sum() to identify duplicate entries.
- Feature distribution: Visualize numeric features (e.g., price, mileage) using histograms and box plots.
- Categorical features: Analyze unique values in categorical columns (e.g., brand, fuel type).

3. Data Cleaning & Preprocessing
- Convert categorical data: Use One-Hot Encoding or Label Encoding for categorical columns (e.g., fuel type, transmission).
- Outlier detection: Identify and handle extreme values using box plots and IQR method.
- Feature engineering: Create new relevant features (e.g., car age from manufacture year).

4. Data Visualization
- Correlation heatmaps: Identify relationships between numerical features.
- Bar charts & Count plots: Analyze categorical features like fuel type, brand popularity.
- Price trends: Check how price varies with year, mileage, and other factors.

5. Model Selection & Training
- Split dataset into training and testing sets (e.g., 80(train)-20(test) split).
- use model like Random Forest,Gradient Boosting,KNN, XGBoost
- Train the models and evaluate performance using RMSE, R² Score, and MAE.

6. Model Evaluation & Optimization
- hyper tunning parameter using GridSearchCV or RandomizedSearchCV.
- Feature selection : use shap.

7. Model Deployment
- Save the trained model using pickle. and create new pickle file (.pkl) extension.
