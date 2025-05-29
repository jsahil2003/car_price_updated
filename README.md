# Car Price Prediction

This project predicts the **selling price of used cars** using machine learning models. It uses a dataset containing details like year, kilometers driven, fuel type, transmission, engine size, and more.

## What the project does

- Cleans and processes the car dataset
- Extracts useful features (e.g., car make, engine size, mileage)
- Encodes categorical data and scales numerical features
- Splits the data into training and test sets
- Trains and tunes three models using GridSearchCV:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor
- Evaluates each model using R² score and RMSE

## Which model worked best?

**XGBoost Regressor** gave the best results with:
- Highest R² Score: **0.967**
- Lowest RMSE: **146,631**

This means XGBoost was the most accurate at predicting car prices among the three models tested.
