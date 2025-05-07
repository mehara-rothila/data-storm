"""
Fixed submission generator for Data Storm competition
Creates proper submission file without data duplication
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

# Paths
data_dir = os.path.join('D:\\', 'DATA STORM', 'dataset')
output_dir = os.path.join('D:\\', 'DATA STORM', 'outputs')

print("=" * 80)
print("FIXED SUBMISSION GENERATOR FOR DATA STORM")
print("=" * 80)

# Load data
print("Loading data...")
train_df = pd.read_csv(os.path.join(data_dir, 'train_storming_round.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test_storming_round.csv'))
submission_template = pd.read_csv(os.path.join(data_dir, 'sample_submission_storming_round.csv'))

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Submission template shape: {submission_template.shape}")

# Verify we have the right number of rows
assert len(test_df) == len(submission_template), "Test and submission sizes don't match!"

# Only use the most essential features to avoid duplication/leakage issues
basic_features = [
    'agent_age',
    'unique_proposal',
    'unique_quotations',
    'unique_customers',
    'unique_proposals_last_7_days',
    'unique_proposals_last_15_days',
    'unique_proposals_last_21_days',
    'unique_quotations_last_7_days',
    'unique_quotations_last_15_days',
    'unique_quotations_last_21_days',
    'unique_customers_last_7_days',
    'unique_customers_last_15_days',
    'unique_customers_last_21_days',
    'ANBP_value',
    'net_income',
    'number_of_policy_holders',
    'number_of_cash_payment_policies'
]

# Simple ratio-based features
print("Creating simple engineered features...")
for df in [train_df, test_df]:
    # Basic ratios that won't create duplication issues
    if 'unique_proposal' in df.columns and 'unique_quotations' in df.columns:
        df['quotation_to_proposal_ratio'] = df['unique_quotations'] / np.maximum(df['unique_proposal'], 1)
    
    if 'unique_customers' in df.columns and 'unique_proposal' in df.columns:
        df['proposals_per_customer'] = df['unique_proposal'] / np.maximum(df['unique_customers'], 1)
    
    if 'unique_customers' in df.columns and 'unique_quotations' in df.columns:
        df['quotations_per_customer'] = df['unique_quotations'] / np.maximum(df['unique_customers'], 1)
    
    if 'number_of_cash_payment_policies' in df.columns and 'number_of_policy_holders' in df.columns:
        df['cash_payment_ratio'] = df['number_of_cash_payment_policies'] / np.maximum(df['number_of_policy_holders'], 1)
    
    # Log transformations
    for col in ['unique_proposal', 'unique_quotations', 'unique_customers']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])

# Add engineered features to basic features
engineered_features = [
    'quotation_to_proposal_ratio',
    'proposals_per_customer',
    'quotations_per_customer',
    'cash_payment_ratio',
    'log_unique_proposal',
    'log_unique_quotations',
    'log_unique_customers'
]

# Combine all features
features = basic_features + engineered_features

# Create target for training
print("Creating target variable...")
train_df['target_column'] = (train_df['new_policy_count'] > 0).astype(int)
print(f"Target distribution: {train_df['target_column'].value_counts()}")

# Prepare datasets
X_train = train_df[features].copy()
y_train = train_df['target_column'].copy()
X_test = test_df[features].copy()

# Fill missing values
for col in X_train.columns:
    if X_train[col].isnull().any():
        X_train[col] = X_train[col].fillna(X_train[col].median())

for col in X_test.columns:
    if X_test[col].isnull().any():
        X_test[col] = X_test[col].fillna(X_train[col].median())

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a model
print("Training ensemble model...")
# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

# XGBoost with balanced class weights
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    scale_pos_weight=pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# LightGBM with class balancing
lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    num_leaves=20,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    class_weight='balanced',
    verbose=-1
)

# Create ensemble model
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('lgb', lgb_model)
    ],
    voting='soft',
    weights=[1, 2, 1.5]
)

# Train the ensemble
ensemble_model.fit(X_train_scaled, y_train)

# Make predictions
print("Generating predictions...")
test_proba = ensemble_model.predict_proba(X_test_scaled)[:, 1]

# Verify prediction length matches test set
print(f"Test set size: {len(test_df)}")
print(f"Prediction array size: {len(test_proba)}")
assert len(test_proba) == len(test_df), "Prediction length doesn't match test set!"

# Try different thresholds
thresholds = np.arange(0.3, 0.71, 0.05)  # 0.3, 0.35, 0.4, ..., 0.7

print("\nGenerating submissions with different thresholds:")
for threshold in thresholds:
    # Apply threshold
    test_predictions = (test_proba >= threshold).astype(int)
    
    # Track counts
    sell_count = test_predictions.sum()
    nill_count = len(test_predictions) - sell_count
    
    # Create submission
    submission = submission_template.copy()
    submission['target_column'] = test_predictions
    
    # Save submission
    submission_path = os.path.join(output_dir, f'submission_threshold_{threshold:.2f}.csv')
    submission.to_csv(submission_path, index=False)
    
    print(f"Threshold {threshold:.2f}: {sell_count} non-NILL ({sell_count/len(test_predictions):.1%}), {nill_count} NILL ({nill_count/len(test_predictions):.1%})")

# Use 0.5 as default threshold
test_predictions = (test_proba >= 0.5).astype(int)
submission = submission_template.copy()
submission['target_column'] = test_predictions
submission.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)

print("\nDefault submission created (threshold=0.5)")
print(f"Prediction: {test_predictions.sum()} non-NILL, {len(test_predictions) - test_predictions.sum()} NILL")
print(f"\nAll submission files created successfully in: {output_dir}")
print("=" * 80)
print("SUBMIT submission.csv TO KAGGLE FOR BEST RESULTS")
print("=" * 80)

# Save model and predictions for reference
joblib.dump(ensemble_model, os.path.join(output_dir, 'ensemble_model.pkl'))
np.save(os.path.join(output_dir, 'test_probabilities.npy'), test_proba)