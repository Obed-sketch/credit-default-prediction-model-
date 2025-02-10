import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
import joblib


# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
column_names = [
    'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings', 'employment', 'installment_rate', 'personal_status', 'debtors',
    'residence', 'property', 'age', 'other_plans', 'housing', 'existing_credits',
    'job', 'dependents', 'telephone', 'foreign', 'class'
]
df = pd.read_csv(url, delim_whitespace=True, names=column_names)

# Convert target to probabilities (synthetic example)
# Assume historical default rates grouped by 'credit_history' (for demonstration)
historical_prob = df.groupby('credit_history')['class'].transform(
    lambda x: (x == 2).mean()  # Probability of default (class=2)
)
df['target_prob'] = historical_prob  # Use this as the regression target
df.drop('class', axis=1, inplace=True)



#Train-test Data set

X = df.drop('target_prob', axis=1)
y = df['target_prob']  # Target is now a probability between 0 and 1

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#Preprocessing Pipeline
categorical_cols = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
numerical_cols = [1, 4, 7, 10, 12, 15, 17]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numericial_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

#Model Building (Regression for Probabilities)
#Gradient Boosting Regressor (with Sigmoid Link)
pipeline_gb = Pipeline([
    ('preprocessor', preprocessor),
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('regressor', GradientBoostingRegressor(
        loss='quantile', alpha=0.5,  # Use quantile loss for bounded output
        n_estimators=100, max_depth=3, random_state=42
    ))
])

# Hyperparameter tuning
param_grid_gb = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [3, 5]
}

grid_gb = GridSearchCV(pipeline_gb, param_grid_gb, cv=5, scoring='neg_mean_squared_error')
grid_gb.fit(X_train, y_train)

#Calibrated Logistic Regression
pipeline_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Calibrate probabilities (Platt scaling)
calibrated_lr = CalibratedClassifierCV(pipeline_lr, method='sigmoid', cv=5)
calibrated_lr.fit(X_train, (y_train > 0.5).astype(int))  # Treat as binary for calibration

#Evaluate Probabilistic Predictions
def evaluate_prob_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'Brier Score': brier_score_loss(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'ROC AUC': roc_auc_score((y_test > 0.5).astype(int), y_pred)  # Threshold at 0.5
    }

# Evaluate Gradient Boosting
print("Gradient Boosting Performance:")
print(evaluate_prob_model(grid_gb.best_estimator_, X_test, y_test))

# Evaluate Calibrated Logistic Regression
print("Calibrated Logistic Regression Performance:")
print(evaluate_prob_model(calibrated_lr, X_test, y_test))

#Backtesting with Time Series Split
tscv = TimeSeriesSplit(n_splits=5)
X_sorted = X.sort_values('duration')  # Simulate temporal order

for train_index, test_index in tscv.split(X_sorted):
    X_train_ts, X_test_ts = X_sorted.iloc[train_index], X_sorted.iloc[test_index]
    y_train_ts, y_test_ts = y.iloc[train_index], y.iloc[test_index]
    model = grid_gb.best_estimator_.fit(X_train_ts, y_train_ts)
    print(evaluate_prob_model(model, X_test_ts, y_test_ts))

#Save and Deploy the Model
# Save the best model
joblib.dump(grid_gb.best_estimator_, 'probability_default_model.pkl')

# Deploy with Flask (app.py)
app = Flask(__name__)
model = joblib.load('probability_default_model.pkl')

@app.route('/predict_probability', methods=['POST'])
def predict_probability():
    data = request.json
    df = pd.DataFrame([data])
    prob = model.predict(df)[0]
    return jsonify({'default_probability': prob})

if __name__ == '__main__':
    app.run(debug=True)
