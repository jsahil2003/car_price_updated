# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/Car details v3.csv')

# Check for missing values and shape of the dataset
df.isna().sum(), df.shape

# Preview the dataset
df.head()

# Extract numerical values from string columns
df['make'] = df['name'].str.extract(r'(\w+)')
df['engine_cc'] = df['engine'].str.extract(r'(\d*\.\d+|\d+)')
df['mileage_kmpl'] = df['mileage'].str.extract(r'(\d*\.\d+|\d+)')
df['max_power_bhp'] = df['max_power'].str.extract(r'(\d*\.\d+|\d+)')

# Check extracted car brand frequency
df['make'].value_counts()

# Define known brand names
names = ['Maruti', 'Hyundai', 'Mahindra', 'Tata', 'Toyota', 'Honda',
         'Ford', 'Chevrolet', 'Renault', 'Volkswagen', 'BMW', 'Skoda']

# Label rare brands as 'other'
df['make'] = df['make'].apply(lambda x: x if x in names else 'other')

# Check updated brand distribution
df['make'].value_counts()

# View dataset columns
df.columns

# Select relevant features
df_clean = df[['year', 'selling_price', 'km_driven', 'fuel', 'seller_type',
               'transmission', 'owner', 'seats', 'make', 'mileage_kmpl', 'engine_cc', 'max_power_bhp']]
df_clean.head()

# Review selected feature columns
df_clean.columns

# Define column types for encoding
ord_col = ['transmission', 'owner']
nom_col = ['fuel', 'seller_type', 'make']
num_col = ['year', 'km_driven', 'seats', 'mileage_kmpl', 'engine_cc', 'max_power_bhp']

# Define preprocessing objects
oe = OrdinalEncoder(categories=[['Manual', 'Automatic'],
                                ['First Owner', 'Second Owner', 'Test Drive Car',
                                 'Third Owner', 'Fourth & Above Owner']])
ohe = OneHotEncoder()
knn = KNNImputer(weights='distance')
ss = StandardScaler()

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df_clean.drop(columns='selling_price', axis=1),
                                                    df_clean['selling_price'], test_size=0.2, random_state=42)

# Create a pipeline for numeric columns
num_pipeline = Pipeline(steps=[
    ('knn', knn),
    ('standard_scaler', ss)
])

# Combine all preprocessing steps
preprocessor = ColumnTransformer(transformers=[
    ('ordinal', oe, ord_col),
    ('nominal', ohe, nom_col),
    ('num', num_pipeline, num_col)
])

# Define models and their hyperparameter grids
model_params = {
    'LinearRegression': {
        'model': LinearRegression(),
        'params': {
            # No hyperparameters to tune for basic LinearRegression
        }
    },
    'RandomForest': {
        'model': RandomForestRegressor(),
        'params': {
            'model__n_estimators': [100, 150, 200],
            'model__max_depth': [None, 5, 10, 15, 20]
        }
    },
    'XGBoost': {
        'model': XGBRegressor(),
        'params': {
            'model__n_estimators': [100, 150, 200],
            'model__learning_rate': [0.05, 0.1, 0.15],
            'model__max_depth': [3, 5, 7, 9]
        }
    }
}

# Prepare to store GridSearchCV results
grid_results = {}

# Loop through models, perform GridSearchCV, evaluate performance
for name, mp in model_params.items():
    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', mp['model'])
    ])

    grid = GridSearchCV(pipeline,
                        param_grid=mp['params'],
                        cv=5,
                        scoring='r2',
                        n_jobs=-1)
    
    grid.fit(x_train, y_train)

    y_pred = grid.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    grid_results[name] = {
        'Best Parameters': grid.best_params_,
        'CV R2 Score': grid.best_score_,
        'Test R2': r2,
        'Test RMSE': rmse
    }

# Convert results to DataFrame
results_df = pd.DataFrame(grid_results)

# Display model performance results
results_df

#RandomForest and XGBoost gave consistent results.
