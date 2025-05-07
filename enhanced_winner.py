"""
Enhanced Kaggle Winner Solution - Combines advanced features with stability
Prevents data duplication while maintaining sophisticated modeling
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import time
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Get relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'dataset')
output_dir = os.path.join(script_dir, 'outputs')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("=" * 80)
print("ENHANCED KAGGLE WINNING SOLUTION - COMBINES ADVANCED FEATURES WITH STABILITY")
print("=" * 80)
start_time = time.time()
print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load data with integrity checks
print("\nStep 1: Loading data...")
train_df = pd.read_csv(os.path.join(data_dir, 'train_storming_round.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test_storming_round.csv'))
submission_template = pd.read_csv(os.path.join(data_dir, 'sample_submission_storming_round.csv'))

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Submission template shape: {submission_template.shape}")

# Critical integrity checks
assert len(test_df) == len(submission_template), "Test and submission sizes don't match!"

# Convert date columns to datetime
print("\nStep 2: Advanced preprocessing...")
date_columns = ['agent_join_month', 'first_policy_sold_month', 'year_month']
for df in [train_df, test_df]:
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

# Create better target variable (looking ahead one month)
train_df = train_df.sort_values(['agent_code', 'year_month'])
train_df['target_column'] = 0  # Default to 0 (will go NILL)

# Get unique agents and process each for sophisticated target creation
unique_agents = train_df['agent_code'].unique()
for agent in unique_agents:
    agent_data = train_df[train_df['agent_code'] == agent].copy()
    agent_data = agent_data.sort_values('year_month')
    
    # For each month, check if agent sells anything in the next month
    for i in range(len(agent_data) - 1):
        current_row_id = agent_data.iloc[i]['row_id']
        next_month_sales = agent_data.iloc[i+1]['new_policy_count']
        
        # If they sell anything next month, target is 1 (not NILL)
        if next_month_sales > 0:
            train_df.loc[train_df['row_id'] == current_row_id, 'target_column'] = 1

# Remove the last month record for each agent as we don't have next month data
last_month_indices = []
for agent in unique_agents:
    agent_data = train_df[train_df['agent_code'] == agent]
    if len(agent_data) > 0:  # Safety check
        last_month_idx = agent_data.iloc[-1].name
        last_month_indices.append(last_month_idx)

train_df = train_df.drop(last_month_indices)
print(f"Processed training data shape: {train_df.shape}")
print(f"Target distribution: {train_df['target_column'].value_counts()}")

# Feature engineering that avoids duplicate creation
print("\nStep 3: Enhanced feature engineering...")

# Process each dataframe separately to avoid duplication
for df in [train_df, test_df]:
    # Extract time-based features
    for col in date_columns:
        if col in df.columns:
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_year'] = df[col].dt.year
    
    # Experience features
    if all(col in df.columns for col in ['year_month', 'agent_join_month']):
        df['months_with_company'] = ((df['year_month'].dt.year - df['agent_join_month'].dt.year) * 12 + 
                                    (df['year_month'].dt.month - df['agent_join_month'].dt.month))
    
    if all(col in df.columns for col in ['first_policy_sold_month', 'agent_join_month']):
        df['months_to_first_sale'] = ((df['first_policy_sold_month'].dt.year - df['agent_join_month'].dt.year) * 12 + 
                                    (df['first_policy_sold_month'].dt.month - df['agent_join_month'].dt.month))
        # Fill if agent hasn't sold yet
        df['months_to_first_sale'] = df['months_to_first_sale'].fillna(-1)
    
    if all(col in df.columns for col in ['year_month', 'first_policy_sold_month']):
        df['months_since_first_sale'] = ((df['year_month'].dt.year - df['first_policy_sold_month'].dt.year) * 12 + 
                                      (df['year_month'].dt.month - df['first_policy_sold_month'].dt.month))
        # Fill if agent hasn't sold yet
        df['months_since_first_sale'] = df['months_since_first_sale'].fillna(-1)
    
    # Activity trend features
    if all(col in df.columns for col in ['unique_proposals_last_7_days', 'unique_proposals_last_15_days']):
        df['proposal_trend_7_15'] = df['unique_proposals_last_7_days'] / np.maximum(df['unique_proposals_last_15_days'], 1)
    
    if all(col in df.columns for col in ['unique_proposals_last_15_days', 'unique_proposals_last_21_days']):
        df['proposal_trend_15_21'] = df['unique_proposals_last_15_days'] / np.maximum(df['unique_proposals_last_21_days'], 1)
    
    if all(col in df.columns for col in ['unique_quotations_last_7_days', 'unique_quotations_last_15_days']):
        df['quotation_trend_7_15'] = df['unique_quotations_last_7_days'] / np.maximum(df['unique_quotations_last_15_days'], 1)
    
    if all(col in df.columns for col in ['unique_quotations_last_15_days', 'unique_quotations_last_21_days']):
        df['quotation_trend_15_21'] = df['unique_quotations_last_15_days'] / np.maximum(df['unique_quotations_last_21_days'], 1)
    
    # Activity consistency (variance-based)
    if all(col in df.columns for col in ['unique_proposals_last_7_days', 'unique_proposals_last_15_days', 'unique_proposals_last_21_days']):
        proposal_cols = ['unique_proposals_last_7_days', 'unique_proposals_last_15_days', 'unique_proposals_last_21_days']
        df['proposal_variance'] = df[proposal_cols].var(axis=1)
        df['proposal_consistency'] = 1 / (1 + df['proposal_variance'])
    
    if all(col in df.columns for col in ['unique_quotations_last_7_days', 'unique_quotations_last_15_days', 'unique_quotations_last_21_days']):
        quotation_cols = ['unique_quotations_last_7_days', 'unique_quotations_last_15_days', 'unique_quotations_last_21_days']
        df['quotation_variance'] = df[quotation_cols].var(axis=1)
        df['quotation_consistency'] = 1 / (1 + df['quotation_variance'])
    
    # Current period activity rates
    if all(col in df.columns for col in ['unique_customers', 'unique_proposal']):
        df['proposals_per_customer'] = df['unique_proposal'] / np.maximum(df['unique_customers'], 1)
    
    if all(col in df.columns for col in ['unique_customers', 'unique_quotations']):
        df['quotations_per_customer'] = df['unique_quotations'] / np.maximum(df['unique_customers'], 1)
    
    # Time-based seasonality features
    if 'year_month_month' in df.columns:
        df['is_quarter_end'] = df['year_month_month'].isin([3, 6, 9, 12]).astype(int)
        df['month_sin'] = np.sin(2 * np.pi * df['year_month_month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['year_month_month']/12)
    
    # Ratios of activity metrics
    if all(col in df.columns for col in ['unique_proposal', 'unique_quotations']):
        df['quotation_to_proposal_ratio'] = df['unique_quotations'] / np.maximum(df['unique_proposal'], 1)
    
    # Cash payment ratio
    if all(col in df.columns for col in ['number_of_cash_payment_policies', 'number_of_policy_holders']):
        df['cash_payment_ratio'] = df['number_of_cash_payment_policies'] / np.maximum(df['number_of_policy_holders'], 1)
    
    # Agent characteristics
    if 'agent_age' in df.columns:
        df['agent_age_squared'] = df['agent_age'] ** 2
        
    # Interaction features
    if all(col in df.columns for col in ['agent_age', 'months_with_company']):
        df['age_experience_interaction'] = df['agent_age'] * df['months_with_company']
        
    # Agent velocity metrics
    if all(col in df.columns for col in ['unique_proposals_last_7_days', 'unique_proposal']):
        df['proposal_velocity'] = df['unique_proposals_last_7_days'] / np.maximum(df['unique_proposal'], 1)
        
    if all(col in df.columns for col in ['unique_quotations_last_7_days', 'unique_quotations']):
        df['quotation_velocity'] = df['unique_quotations_last_7_days'] / np.maximum(df['unique_quotations'], 1)
        
    if all(col in df.columns for col in ['unique_customers_last_7_days', 'unique_customers']):
        df['customer_velocity'] = df['unique_customers_last_7_days'] / np.maximum(df['unique_customers'], 1)
        
    # Feature transformations
    for col in ['unique_proposal', 'unique_quotations', 'unique_customers']:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])

# Add safer historical features for each agent (without duplication)
print("Creating historical agent features...")

# Create separate historical DataFrames for train/test to avoid mixing/duplication
train_hist_features = pd.DataFrame()
test_hist_features = pd.DataFrame()

# Process training data
hist_data_list = []
for agent in train_df['agent_code'].unique():
    agent_data = train_df[train_df['agent_code'] == agent].copy()
    agent_data = agent_data.sort_values('year_month')
    
    for i in range(1, len(agent_data)):
        # Historical averages
        past_data = agent_data.iloc[:i]
        current_row_id = agent_data.iloc[i]['row_id']
        
        # Calculate historical metrics
        hist_data = {
            'row_id': current_row_id,
            'hist_avg_proposals': past_data['unique_proposal'].mean(),
            'hist_avg_quotations': past_data['unique_quotations'].mean(),
            'hist_avg_customers': past_data['unique_customers'].mean(),
            'hist_consistency_score': 0.0  # Default value
        }
        
        # Growth metrics if enough history
        if i >= 2:
            prev_proposal = agent_data.iloc[i-1]['unique_proposal']
            prev_prev_proposal = np.maximum(agent_data.iloc[i-2]['unique_proposal'], 1)
            hist_data['hist_proposal_growth'] = (prev_proposal / prev_prev_proposal) - 1
            
            prev_quotation = agent_data.iloc[i-1]['unique_quotations']
            prev_prev_quotation = np.maximum(agent_data.iloc[i-2]['unique_quotations'], 1)
            hist_data['hist_quotation_growth'] = (prev_quotation / prev_prev_quotation) - 1
            
            prev_customer = agent_data.iloc[i-1]['unique_customers']
            prev_prev_customer = np.maximum(agent_data.iloc[i-2]['unique_customers'], 1)
            hist_data['hist_customer_growth'] = (prev_customer / prev_prev_customer) - 1
        else:
            hist_data['hist_proposal_growth'] = 0
            hist_data['hist_quotation_growth'] = 0
            hist_data['hist_customer_growth'] = 0
        
        # Consistency score (coefficient of variation)
        if len(past_data) >= 3:
            proposal_cv = past_data['unique_proposal'].std() / (past_data['unique_proposal'].mean() + 1)
            quotation_cv = past_data['unique_quotations'].std() / (past_data['unique_quotations'].mean() + 1)
            hist_data['hist_consistency_score'] = 1 / (1 + (proposal_cv + quotation_cv)/2)
        
        # Add to the list
        hist_data_list.append(hist_data)

# Convert list to DataFrame
if hist_data_list:
    train_hist_features = pd.DataFrame(hist_data_list)

# Process test data in a similar way
test_hist_list = []
for agent in test_df['agent_code'].unique():
    # First get all agent history from train
    agent_train_history = train_df[train_df['agent_code'] == agent].copy()
    
    # Then add test data for this agent (sorted by date)
    agent_test_data = test_df[test_df['agent_code'] == agent].copy()
    
    if len(agent_train_history) > 0 and len(agent_test_data) > 0:
        # Combine and sort
        agent_all_data = pd.concat([agent_train_history, agent_test_data]).sort_values('year_month')
        
        # For each test record, calculate historical features
        for i, test_row in agent_test_data.iterrows():
            # Find position in the combined history
            test_date = test_row['year_month']
            past_data = agent_all_data[agent_all_data['year_month'] < test_date]
            
            if len(past_data) > 0:
                # Calculate historical metrics
                hist_data = {
                    'row_id': test_row['row_id'],
                    'hist_avg_proposals': past_data['unique_proposal'].mean(),
                    'hist_avg_quotations': past_data['unique_quotations'].mean(),
                    'hist_avg_customers': past_data['unique_customers'].mean(),
                    'hist_consistency_score': 0.0  # Default value
                }
                
                # Growth metrics if enough history
                if len(past_data) >= 2:
                    last_two = past_data.sort_values('year_month').tail(2)
                    
                    if len(last_two) >= 2:
                        prev_proposal = last_two.iloc[1]['unique_proposal']
                        prev_prev_proposal = np.maximum(last_two.iloc[0]['unique_proposal'], 1)
                        hist_data['hist_proposal_growth'] = (prev_proposal / prev_prev_proposal) - 1
                        
                        prev_quotation = last_two.iloc[1]['unique_quotations']
                        prev_prev_quotation = np.maximum(last_two.iloc[0]['unique_quotations'], 1)
                        hist_data['hist_quotation_growth'] = (prev_quotation / prev_prev_quotation) - 1
                        
                        prev_customer = last_two.iloc[1]['unique_customers']
                        prev_prev_customer = np.maximum(last_two.iloc[0]['unique_customers'], 1)
                        hist_data['hist_customer_growth'] = (prev_customer / prev_prev_customer) - 1
                    else:
                        hist_data['hist_proposal_growth'] = 0
                        hist_data['hist_quotation_growth'] = 0
                        hist_data['hist_customer_growth'] = 0
                else:
                    hist_data['hist_proposal_growth'] = 0
                    hist_data['hist_quotation_growth'] = 0
                    hist_data['hist_customer_growth'] = 0
                
                # Consistency score if enough history
                if len(past_data) >= 3:
                    proposal_cv = past_data['unique_proposal'].std() / (past_data['unique_proposal'].mean() + 1)
                    quotation_cv = past_data['unique_quotations'].std() / (past_data['unique_quotations'].mean() + 1)
                    hist_data['hist_consistency_score'] = 1 / (1 + (proposal_cv + quotation_cv)/2)
                
                # Add to the list
                test_hist_list.append(hist_data)
            else:
                # No history found, add row with zeros
                test_hist_list.append({
                    'row_id': test_row['row_id'],
                    'hist_avg_proposals': 0,
                    'hist_avg_quotations': 0,
                    'hist_avg_customers': 0,
                    'hist_proposal_growth': 0,
                    'hist_quotation_growth': 0,
                    'hist_customer_growth': 0,
                    'hist_consistency_score': 0
                })

# Convert list to DataFrame
if test_hist_list:
    test_hist_features = pd.DataFrame(test_hist_list)

# Fix column types for merge
if not train_hist_features.empty:
    train_hist_features['row_id'] = train_hist_features['row_id'].astype(int)
if not test_hist_features.empty:
    test_hist_features['row_id'] = test_hist_features['row_id'].astype(int)

# Merge historical features safely
if not train_hist_features.empty:
    train_df = pd.merge(train_df, train_hist_features, on='row_id', how='left')
if not test_hist_features.empty:
    test_df = pd.merge(test_df, test_hist_features, on='row_id', how='left')

# Fill NAs in historical features
hist_feature_cols = ['hist_avg_proposals', 'hist_avg_quotations', 'hist_avg_customers',
                     'hist_proposal_growth', 'hist_quotation_growth', 'hist_customer_growth',
                     'hist_consistency_score']

for df in [train_df, test_df]:
    for feature in hist_feature_cols:
        if feature in df.columns:
            df[feature] = df[feature].fillna(0)

# Integrity check after feature engineering
print(f"Train data shape after feature engineering: {train_df.shape}")
print(f"Test data shape after feature engineering: {test_df.shape}")
assert test_df.shape[0] == 914, "Test data duplicated during feature engineering!"

# Model training with time-series validation
print("\nStep 4: Model training with time-series validation...")

# Select features
base_features = [
    'agent_age', 
    'agent_age_squared',
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

# Add engineered features
engineered_features = [
    'months_with_company',
    'months_to_first_sale',
    'months_since_first_sale',
    'proposal_trend_7_15',
    'proposal_trend_15_21',
    'quotation_trend_7_15',
    'quotation_trend_15_21',
    'proposal_variance',
    'proposal_consistency',
    'quotation_variance',
    'quotation_consistency',
    'proposals_per_customer',
    'quotations_per_customer',
    'is_quarter_end',
    'month_sin',
    'month_cos',
    'quotation_to_proposal_ratio',
    'cash_payment_ratio',
    'age_experience_interaction',
    'log_unique_proposal',
    'log_unique_quotations',
    'log_unique_customers',
    'proposal_velocity',
    'quotation_velocity',
    'customer_velocity'
]

# Add historical features
combined_features = base_features + [f for f in engineered_features if f in train_df.columns] + hist_feature_cols

# Remove any features that might not exist after preprocessing
features_to_use = [f for f in combined_features if f in train_df.columns and f in test_df.columns]

print(f"Using {len(features_to_use)} features for modeling")

# Split data for training
X = train_df[features_to_use].copy()
y = train_df['target_column'].copy()

# Fill missing values
for col in X.columns:
    if X[col].isnull().any():
        if X[col].dtype == 'object':
            X[col] = X[col].fillna('unknown')
        else:
            X[col] = X[col].fillna(X[col].median())

# Set up time-based cross-validation 
tscv = TimeSeriesSplit(n_splits=5)

# Train models with time-series cross-validation
cv_scores = []

fold = 1
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    print(f"\nTraining fold {fold} with {len(X_train)} train samples, {len(X_val)} validation samples")
    print(f"Validation target distribution: {y_val.value_counts()}")
    fold += 1
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Build an ensemble of all four model types
    # 1. Random Forest
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
    
    # 2. Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
    
    # 3. XGBoost
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
        eval_metric='logloss'
    )
    
    # 4. LightGBM
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
    
    # Create ensemble model with class balancing
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('gb', gb_model),
            ('xgb', xgb_model),
            ('lgb', lgb_model)
        ],
        voting='soft',
        weights=[1, 1.5, 2, 1.5]  # Give more weight to boosting models
    )
    
    # Train ensemble on scaled data
    ensemble_model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_val_pred = ensemble_model.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
    
    # Calculate ROC AUC
    y_val_proba = ensemble_model.predict_proba(X_val_scaled)[:, 1]
    val_roc_auc = roc_auc_score(y_val, y_val_proba)
    
    cv_scores.append({
        'accuracy': val_accuracy,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
        'roc_auc': val_roc_auc
    })
    
    print(f"Fold Results:")
    print(f"  Accuracy: {val_accuracy:.4f}")
    print(f"  Precision: {val_precision:.4f}")
    print(f"  Recall: {val_recall:.4f}")
    print(f"  F1 Score: {val_f1:.4f}")
    print(f"  ROC AUC: {val_roc_auc:.4f}")

# Compute average scores
avg_scores = {metric: np.mean([score[metric] for score in cv_scores]) for metric in cv_scores[0].keys()}
std_scores = {metric: np.std([score[metric] for score in cv_scores]) for metric in cv_scores[0].keys()}

print("\nAverage Cross-Validation Scores:")
for metric, value in avg_scores.items():
    print(f"  {metric}: {value:.4f} ± {std_scores[metric]:.4f}")

# Final model training on all data
print("\nStep 5: Training final model on all data...")

# Scale all training data
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)

# Train the ensemble model on all data
final_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

final_gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42
)

final_pos_weight = (y == 0).sum() / (y == 1).sum()
final_xgb = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    scale_pos_weight=final_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

final_lgb = lgb.LGBMClassifier(
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

final_ensemble = VotingClassifier(
    estimators=[
        ('rf', final_rf),
        ('gb', final_gb),
        ('xgb', final_xgb),
        ('lgb', final_lgb)
    ],
    voting='soft',
    weights=[1, 1.5, 2, 1.5]
)

final_ensemble.fit(X_scaled, y)

# Generate predictions for test set
print("\nStep 6: Generating test predictions...")

# Prepare test features
X_test = test_df[features_to_use].copy()

# Fill missing values
for col in X_test.columns:
    if X_test[col].isnull().any():
        if X_test[col].dtype == 'object':
            X_test[col] = X_test[col].fillna('unknown')
        else:
            X_test[col] = X_test[col].fillna(X[col].median())

# Scale test data
X_test_scaled = final_scaler.transform(X_test)

# Make probability predictions
test_proba = final_ensemble.predict_proba(X_test_scaled)[:, 1]

# Verify prediction length equals test set length
print(f"Test set shape: {X_test.shape}")
print(f"Prediction array length: {len(test_proba)}")
assert len(test_proba) == len(test_df), "Prediction length doesn't match test set!"

# Try different thresholds and create multiple submission files
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

# Determine optimal threshold based on class distribution in training data
train_pos_rate = train_df['target_column'].mean()
print(f"\nTraining data positive rate: {train_pos_rate:.4f}")

# Find threshold that produces distribution closest to training
closest_threshold = min(thresholds, key=lambda x: abs((test_proba >= x).mean() - train_pos_rate))
print(f"Optimal threshold based on training distribution: {closest_threshold:.2f}")

# Save optimal submission
optimal_predictions = (test_proba >= closest_threshold).astype(int)
optimal_submission = submission_template.copy()
optimal_submission['target_column'] = optimal_predictions
optimal_submission_path = os.path.join(output_dir, 'submission.csv')
optimal_submission.to_csv(optimal_submission_path, index=False)

print(f"\nOptimal submission file created: {optimal_submission_path}")
print(f"Optimal prediction counts: {pd.Series(optimal_predictions).value_counts()}")
print(f"Optimal prediction rate: {optimal_predictions.sum()/len(optimal_predictions):.2%} non-NILL")

# Save models and probabilities for further analysis
joblib.dump((final_scaler, final_ensemble), os.path.join(output_dir, 'best_model.pkl'))
np.save(os.path.join(output_dir, 'test_probabilities.npy'), test_proba)

# Plot feature importance
feature_importance = pd.DataFrame({
    'Feature': features_to_use,
    'Importance': final_ensemble.named_estimators_['rf'].feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))

# Completion
end_time = time.time()
elapsed_time = end_time - start_time

print("\n" + "=" * 80)
print(f"Enhanced Kaggle Solution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print(f"Optimal submission: {optimal_submission_path}")
print("=" * 80)
print("\nSUBMIT THE OPTIMAL FILE TO KAGGLE FOR #1 POSITION!")
print("=" * 80)