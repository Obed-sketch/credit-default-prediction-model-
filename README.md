# credit-default-prediction-model-
# To create a model where the target variable is the probability of default (instead of a binary classification), we need to reframe the problem as a regression task where the output is a continuous probability value between 0 and 1. This requires a different approach, including calibration of probabilities and using metrics suited for probabilistic outputs. Below is a step-by-step implementation:
# Key Differences from Classification:
# Target Variable: The target is now a continuous probability (e.g., historical default rates) instead of a binary label.

# Evaluation Metrics:

# Brier Score: Measures accuracy of probabilistic predictions (lower is better).

# RMSE: Root Mean Squared Error for regression.

# ROC AUC: Evaluates ranking of probabilities (thresholded at 0.5).

# Models:

# Use regression models (e.g., GradientBoostingRegressor) or calibrated classifiers (e.g., CalibratedClassifierCV).

# Calibration: Ensures predicted probabilities align with true frequencies (e.g., Platt scaling).

# This approach is useful for scenarios where the goal is to predict continuous default risk scores (e.g., PD for Basel III compliance).


