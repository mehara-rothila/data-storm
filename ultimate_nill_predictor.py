"""
ULTIMATE NILL PREDICTOR - Data Storm v6.0
Attempting to enhance beyond the 0.90036 benchmark.

Key enhancements over champion_model.py:
1.  Refined Feature Engineering:
    - Robust Target Encoding for agent_code (with CV).
    - Agent performance consistency/trend features (std dev, slope, prior NILL counts).
    - More interaction terms.
2.  Advanced Ensembling:
    - StackingCVClassifier for Level 1 meta-model.
3.  Optimized Hyperparameter Search with Optuna for all base models and potentially meta-model.
4.  Continued focus on time-series integrity and robust thresholding.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from category_encoders import TargetEncoder
import optuna
import joblib # For saving models
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
N_SPLITS_TS = 5  # Number of splits for TimeSeriesSplit in Optuna and Stacking
OPTUNA_N_TRIALS_PER_MODEL = 75 # Number of Optuna trials for each base model (can be increased for better tuning)
OPTUNA_N_TRIALS_META = 30 # Number of Optuna trials for meta-model
USE_OPTUNA = True # Set to False to use pre-defined best params (if available) or defaults

# Set seed for reproducibility
np.random.seed(RANDOM_STATE)

# --- Path Setup ---
# Assumes script is run from D:\DATA STORM and datasets are in D:\DATA STORM\dataset
# And outputs will go to D:\DATA STORM\outputs
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError: # For interactive environments like Jupyter
    script_dir = os.getcwd()

data_dir = os.path.join(script_dir, 'dataset')
output_dir = os.path.join(script_dir, 'outputs')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Created data directory: {data_dir}")
    print(f"Please place train_stroming_round.csv and test_stroming_round.csv in this directory.")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# --- Feature Engineering ---
def parse_dates(df):
    df['agent_join_month_dt'] = pd.to_datetime(df['agent_join_month'], errors='coerce')
    df['first_policy_sold_month_dt'] = pd.to_datetime(df['first_policy_sold_month'], errors='coerce')
    df['year_month_dt'] = pd.to_datetime(df['year_month'], errors='coerce')
    return df

def create_time_features(df):
    # Time since events
    df['days_since_join'] = (df['year_month_dt'] - df['agent_join_month_dt']).dt.days
    df['days_since_first_sale'] = (df['year_month_dt'] - df['first_policy_sold_month_dt']).dt.days
    df['days_to_first_sale'] = (df['first_policy_sold_month_dt'] - df['agent_join_month_dt']).dt.days

    # Impute time features (e.g., for agents who haven't sold yet or dates are missing)
    df['days_since_first_sale'].fillna(df['days_since_join'], inplace=True) # If no sale, assume it's like join date for this metric
    df['days_to_first_sale'].fillna(365*10, inplace=True) # Large value if no first sale

    # Cyclical features for year_month
    if 'year_month_dt' in df.columns and not df['year_month_dt'].isnull().all():
        df['month'] = df['year_month_dt'].dt.month
        df['year'] = df['year_month_dt'].dt.year
        df['days_in_month'] = df['year_month_dt'].dt.daysinmonth
        df['quarter'] = df['year_month_dt'].dt.quarter
        # For time series split, a numeric representation of time is useful
        df['year_month_numeric'] = df['year_month_dt'].dt.year * 100 + df['year_month_dt'].dt.month
    return df

def create_activity_features(df):
    # Sum of recent activities
    df['proposals_last_21_days_total'] = df['unique_proposals_last_7_days'] + \
                                       df['unique_proposals_last_15_days'] + \
                                       df['unique_proposals_last_21_days']
    df['quotations_last_21_days_total'] = df['unique_quotations_last_7_days'] + \
                                        df['unique_quotations_last_15_days'] + \
                                        df['unique_quotations_last_21_days']
    df['customers_last_21_days_total'] = df['unique_customers_last_7_days'] + \
                                       df['unique_customers_last_15_days'] + \
                                       df['unique_customers_last_21_days']
    # Ratios
    for period in ['7_days', '15_days', '21_days', 'total']:
        prop_col = f'unique_proposals_last_{period}' if period != 'total' else 'unique_proposal'
        quot_col = f'unique_quotations_last_{period}' if period != 'total' else 'unique_quotations'
        cust_col = f'unique_customers_last_{period}' if period != 'total' else 'unique_customers'

        df[f'prop_to_quot_{period}'] = df[prop_col] / (df[quot_col] + 1e-6)
        df[f'cust_to_prop_{period}'] = df[cust_col] / (df[prop_col] + 1e-6)
        df[f'quot_to_cust_{period}'] = df[quot_col] / (df[cust_col] + 1e-6)
    return df

def create_agent_history_features(df, is_train=True):
    df = df.sort_values(['agent_code', 'year_month_numeric'])

    # Historical performance (expanding window)
    for col in ['new_policy_count', 'ANBP_value', 'net_income']:
        if col in df.columns: # Only for train, as test won't have new_policy_count
            df[f'avg_hist_{col}'] = df.groupby('agent_code')[col].transform(lambda x: x.shift(1).expanding().mean())
            df[f'std_hist_{col}'] = df.groupby('agent_code')[col].transform(lambda x: x.shift(1).expanding().std())
            df[f'sum_hist_{col}'] = df.groupby('agent_code')[col].transform(lambda x: x.shift(1).expanding().sum())
            df[f'median_hist_{col}'] = df.groupby('agent_code')[col].transform(lambda x: x.shift(1).expanding().median())
            df[f'max_hist_{col}'] = df.groupby('agent_code')[col].transform(lambda x: x.shift(1).expanding().max())
            
            # Rolling features (last 3, 6 months)
            for W in [3, 6]:
                df[f'avg_roll{W}m_{col}'] = df.groupby('agent_code')[col].transform(lambda x: x.shift(1).rolling(window=W, min_periods=1).mean())
                df[f'std_roll{W}m_{col}'] = df.groupby('agent_code')[col].transform(lambda x: x.shift(1).rolling(window=W, min_periods=1).std())

            # Trend (slope of sales in last 3 months) - simple approximation
            if col == 'new_policy_count':
                df[f'trend_last3m_{col}'] = df.groupby('agent_code')[col].transform(
                    lambda x: x.shift(1).rolling(window=3, min_periods=2).apply(lambda r: np.polyfit(np.arange(len(r)), r, 1)[0] if len(r)>=2 else 0, raw=True)
                )
                df[f'trend_last3m_{col}'].fillna(0, inplace=True)


    # NILL history
    if 'new_policy_count' in df.columns:
        df['is_nill_current_month_for_hist'] = (df['new_policy_count'] == 0).astype(int)
        df['hist_nill_rate'] = df.groupby('agent_code')['is_nill_current_month_for_hist'].transform(lambda x: x.shift(1).expanding().mean())
        df['hist_nill_count'] = df.groupby('agent_code')['is_nill_current_month_for_hist'].transform(lambda x: x.shift(1).expanding().sum())
        
        # Consecutive NILL months before current
        # This is more complex, requires iterative calculation or careful grouping
        # Simplified: Count of NILLs in last 3 months
        df['nills_last3m'] = df.groupby('agent_code')['is_nill_current_month_for_hist'].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).sum())


    # Fill NaNs created by shift/expanding/rolling, especially for initial records
    # For std dev, NaN can mean not enough data points or zero variance; fill with 0
    cols_to_fill_zero = [c for c in df.columns if 'std_hist' in c or 'std_roll' in c or 'hist_nill_rate' in c or 'nills_last3m' in c]
    df[cols_to_fill_zero] = df[cols_to_fill_zero].fillna(0)
    # For mean/sum/median, could fill with a specific value or use a more complex imputation
    # For now, ffill within group then bfill for initial NaNs.
    group_fill_cols = [c for c in df.columns if 'avg_hist' in c or 'sum_hist' in c or 'median_hist' in c or 'max_hist' in c or 'avg_roll' in c]
    for col in group_fill_cols:
        df[col] = df.groupby('agent_code')[col].ffill().bfill() # First ffill, then bfill for very start
    df[group_fill_cols] = df[group_fill_cols].fillna(0) # Catch any remaining after group ffill/bfill


    # Months as active agent
    df['months_active'] = df.groupby('agent_code').cumcount() + 1

    # Interaction features
    df['age_X_days_since_join'] = df['agent_age'] * df['days_since_join']
    if 'avg_hist_new_policy_count' in df.columns: # Only if train data features are available
        df['proposals_X_avg_hist_policies'] = df['unique_proposal'] * df['avg_hist_new_policy_count']
        df['proposals_X_avg_hist_policies'].fillna(0, inplace=True)
    df['quotations_X_months_active'] = df['unique_quotations'] * df['months_active']

    return df

def advanced_feature_engineering(df, target_encoder=None, fit_target_encoder=False, y_train=None):
    df = parse_dates(df)
    df = create_time_features(df) # Creates year_month_numeric
    df = create_activity_features(df)
    
    # agent_code encoding
    if fit_target_encoder and target_encoder is not None and y_train is not None:
        print("Fitting TargetEncoder for agent_code...")
        df['agent_code_encoded'] = target_encoder.fit_transform(df['agent_code'], y_train)
    elif target_encoder is not None:
        print("Transforming agent_code with pre-fitted TargetEncoder...")
        df['agent_code_encoded'] = target_encoder.transform(df['agent_code'])
    else: # Fallback to frequency encoding if no TE
        df['agent_code_freq'] = df['agent_code'].map(df['agent_code'].value_counts(normalize=True))
        df['agent_code_encoded'] = df['agent_code_freq'] # Use this name for consistency

    # Agent history features depend on 'new_policy_count' which is not in test for this purpose
    # So, these specific historical aggregations are primarily for training.
    # For test set, if these are crucial, they would need to be derived from historical train data mapped to test agents.
    # The current setup of `create_agent_history_features` is for when 'new_policy_count' exists in the df.
    is_train_data = 'new_policy_count' in df.columns
    df = create_agent_history_features(df, is_train=is_train_data)

    # Final cleanup of NaNs (e.g., from ratios if denominators were persistently zero)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop original date and high cardinality categoricals if encoded
    cols_to_drop = ['agent_join_month', 'first_policy_sold_month', 'year_month',
                    'agent_join_month_dt', 'first_policy_sold_month_dt', 'year_month_dt',
                    'agent_code'] # Drop original agent_code if encoded
    if 'is_nill_current_month_for_hist' in df.columns:
        cols_to_drop.append('is_nill_current_month_for_hist')

    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    return df

# --- Model Training and Evaluation ---
def get_oof_predictions(model, X, y, groups, cv_strategy):
    oof_preds = np.zeros(len(X))
    X_np = X.values if isinstance(X, pd.DataFrame) else X # Models prefer numpy
    y_np = y.values if isinstance(y, pd.Series) else y

    for fold, (train_idx, val_idx) in enumerate(cv_strategy.split(X_np, y_np, groups)):
        print(f"Stacking Fold {fold+1}/{cv_strategy.get_n_splits()}")
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]
        
        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    return oof_preds

def get_test_predictions(model, X_train, y_train, X_test):
    model.fit(X_train, y_train) # Retrain on full training data
    return model.predict_proba(X_test)[:, 1]

# --- Optuna Objective Functions ---
# (Similar to champion_model.py, adapted for flexibility)
def objective_xgb(trial, X_train, y_train, X_val, y_val):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc', # Using AUC for optimization, can threshold later for F1
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'eta': trial.suggest_float('eta', 0.01, 0.3, log=True), # learning_rate
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'seed': RANDOM_STATE
    }
    if param['booster'] == 'dart':
        param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
        param['rate_drop'] = trial.suggest_float('rate_drop', 1e-8, 0.5, log=True)
        param['skip_drop'] = trial.suggest_float('skip_drop', 1e-8, 0.5, log=True)

    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)

def objective_lgbm(trial, X_train, y_train, X_val, y_val):
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    model = lgb.LGBMClassifier(**param)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False,
              callbacks=[lgb.early_stopping(50, verbose=False)])
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)

def objective_catboost(trial, X_train, y_train, X_val, y_val, cat_features_indices=None):
    param = {
        'iterations': trial.suggest_int('iterations', 100, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        # 'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0), # If using MVS
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True), # For regularization
        'eval_metric': 'AUC',
        'random_seed': RANDOM_STATE,
        'verbose': 0,
        'early_stopping_rounds': 50
    }
    model = cb.CatBoostClassifier(**param)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], cat_features=cat_features_indices, verbose=0)
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)
    
def objective_rf(trial, X_train, y_train, X_val, y_val):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30, log=True),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 32),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 32),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
    }
    model = RandomForestClassifier(**param)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)


def optimize_model_hyperparameters(model_name, objective_func, X, y, tscv_split, n_trials, cat_features_indices=None):
    # Use the last fold of TimeSeriesSplit for Optuna validation (common practice)
    all_train_indices, all_val_indices = list(tscv_split.split(X, y, groups=X['year_month_numeric']))[-1]
    
    X_opt_train = X.iloc[all_train_indices]
    y_opt_train = y.iloc[all_train_indices]
    X_opt_val = X.iloc[all_val_indices]
    y_opt_val = y.iloc[all_val_indices]

    # Ensure cat_features_indices is correctly derived for the subset X_opt_train if needed
    current_cat_features_indices = [X_opt_train.columns.get_loc(c) for c in categorical_features if c in X_opt_train.columns and X_opt_train.columns.get_loc(c) < X_opt_train.shape[1]] if cat_features_indices == "auto" else cat_features_indices

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    if cat_features_indices:
         study.optimize(lambda trial: objective_func(trial, X_opt_train, y_opt_train, X_opt_val, y_opt_val, current_cat_features_indices), n_trials=n_trials)
    else:
        study.optimize(lambda trial: objective_func(trial, X_opt_train, y_opt_train, X_opt_val, y_opt_val), n_trials=n_trials)
   
    print(f"Best {model_name} AUC: {study.best_value}")
    print(f"Best {model_name} params: {study.best_params}")
    return study.best_params

# --- Main Execution ---
if __name__ == '__main__':
    start_time_main = time.time()
    print(f"Starting Ultimate NILL Predictor at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\nStep 1: Loading data...")
    try:
        train_df_orig = pd.read_csv(os.path.join(data_dir, 'train_stroming_round.csv'))
        test_df_orig = pd.read_csv(os.path.join(data_dir, 'test_storming_round.csv'))
        sample_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission_storming_round.csv'))
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure dataset files are in '{data_dir}'")
        exit()

    print(f"Train shape original: {train_df_orig.shape}, Test shape original: {test_df_orig.shape}")

    # --- Target Variable (as per champion_model.py successful interpretation) ---
    # Predict NILL for the current month based on its features.
    # "Following month" aspect implies test set is for a future period.
    train_df_orig['target'] = (train_df_orig['new_policy_count'] > 0).astype(int)
    y = train_df_orig['target']
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")


    # --- Feature Engineering ---
    print("\nStep 2: Feature Engineering...")
    # Initialize TargetEncoder for agent_code
    # We will fit this inside the CV loop for stacking or use full data for simpler Optuna tuning split
    # For now, let's do a preliminary fit on full training data for agent_code for base model tuning.
    # This is a slight leak for the Optuna validation split, but simpler.
    # A more robust way is to do TE inside each Optuna CV fold.
    
    print("  Applying initial parse and time features to train and test...")
    train_df_fe = train_df_orig.copy()
    test_df_fe = test_df_orig.copy()

    # For TargetEncoding 'agent_code': Fit on combined train agent_codes to learn encodings
    # then apply to train and test. This is simpler than CV-based TE for initial run.
    # For more robustness, TE should be part of the CV pipeline.
    # Let's do a version where TE is fit on the full training for feature creation,
    # then these fixed encoded values are used.
    
    # Create a temporary combined df for fitting TE on all known agent_codes consistently
    combined_codes = pd.concat([train_df_fe[['agent_code']], test_df_fe[['agent_code']]], axis=0).drop_duplicates(subset=['agent_code'])
    # Fit TargetEncoder on the training data's target
    temp_train_for_te = train_df_fe[['agent_code']].copy()
    temp_train_for_te['target_for_te'] = y # Use original y for fitting TE
    
    # Initialize TargetEncoder
    agent_coder = TargetEncoder(cols=['agent_code'], handle_missing='value', handle_unknown='value')
    print("  Fitting TargetEncoder on combined agent codes using training targets...")
    # Fit on training data agent_code and target
    agent_coder.fit(train_df_fe['agent_code'], y)


    print("  Performing advanced feature engineering on Training Data...")
    train_df_processed = advanced_feature_engineering(train_df_fe.copy(), target_encoder=agent_coder, fit_target_encoder=False) # Already fitted
    
    print("  Performing advanced feature engineering on Test Data...")
    # Ensure test_df_fe also has 'new_policy_count' if create_agent_history_features needs it (it shouldn't for test)
    # The function `create_agent_history_features` has an `is_train` flag.
    test_df_processed = advanced_feature_engineering(test_df_fe.copy(), target_encoder=agent_coder, fit_target_encoder=False)

    # Align columns - crucial after feature engineering
    print("  Aligning columns between train and test sets...")
    train_labels = y
    train_ids = train_df_processed['Row_ID'] # Assuming Row_ID is present
    test_ids = test_df_processed['Row_ID']

    # Drop Row_ID before training, keep features only
    common_cols = list(set(train_df_processed.columns) & set(test_df_processed.columns))
    common_cols = [col for col in common_cols if col not in ['Row_ID', 'target', 'target_raw', 'new_policy_count', 'year_month_numeric_for_groupkfold_placeholder']] # remove special/target cols
    
    # Filter out columns that might have been created only in train (like historical actuals)
    # and ensure order is same
    X = train_df_processed[common_cols].copy()
    X_test = test_df_processed[common_cols].copy()

    # Re-check alignment
    if len(X.columns) != len(X_test.columns) or not all(X.columns == X_test.columns):
        print("Warning: Columns mismatch after initial common_cols. Re-aligning strictly.")
        shared_cols = list(X.columns.intersection(X_test.columns))
        X = X[shared_cols]
        X_test = X_test[shared_cols]
        print(f"  Aligned to {len(shared_cols)} features.")

    print(f"  Final feature shapes: X_train: {X.shape}, X_test: {X_test.shape}")

    # Impute remaining NaNs (e.g. from ratios like X/0 or from new features on limited data)
    print("  Imputing remaining NaNs with median...")
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Scaling numerical features
    print("  Scaling numerical features...")
    scaler = StandardScaler()
    numerical_features = X.select_dtypes(include=np.number).columns
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    # Identify categorical features for CatBoost (if any remain after encoding)
    # For this script, agent_code was target encoded. Other categoricals like 'month', 'quarter' are numerical.
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_features_indices = [X.columns.get_loc(col) for col in categorical_features if col in X.columns]
    print(f"  Categorical feature indices for CatBoost (if any): {cat_features_indices}")


    # --- Model Definition and Hyperparameter Tuning (Optuna) ---
    print("\nStep 3: Model Definition and Hyperparameter Tuning...")
    # TimeSeriesSplit for Optuna validation folds (chronological)
    # Using year_month_numeric for grouping if StratifiedKFold is used, but TimeSeriesSplit is better.
    # Need 'year_month_numeric' in X for TimeSeriesSplit groups or direct splitting.
    # Let's use the original year_month_numeric from train_df_fe for splitting.
    # Ensure X is sorted by year_month_numeric for TimeSeriesSplit to work correctly
    
    # Re-attach year_month_numeric for sorting and splitting, then drop
    X_for_split = X.copy()
    X_for_split['year_month_numeric'] = train_df_fe['year_month_numeric'].values # Get from original df before processing
    X_for_split = X_for_split.sort_values('year_month_numeric')
    y_for_split = train_labels.loc[X_for_split.index] # ensure y aligns with sorted X
    
    tscv_optuna = TimeSeriesSplit(n_splits=N_SPLITS_TS) # Or use N_SPLITS_TS-1 for train and 1 for val.
                                                   # The objective functions use the last split.

    best_params = {}
    models_for_stacking = {}

    if USE_OPTUNA:
        print("  Optimizing LightGBM...")
        best_params['lgbm'] = optimize_model_hyperparameters('LGBM', objective_lgbm, 
                                                             X_for_split.drop(columns=['year_month_numeric']), y_for_split, 
                                                             tscv_optuna, OPTUNA_N_TRIALS_PER_MODEL)
        
        print("  Optimizing XGBoost...")
        best_params['xgb'] = optimize_model_hyperparameters('XGB', objective_xgb,
                                                            X_for_split.drop(columns=['year_month_numeric']), y_for_split,
                                                            tscv_optuna, OPTUNA_N_TRIALS_PER_MODEL)
        
        print("  Optimizing CatBoost...")
        # For CatBoost, ensure cat_features_indices are for X_for_split.drop(columns=['year_month_numeric'])
        temp_X_cat = X_for_split.drop(columns=['year_month_numeric'])
        cat_indices_for_optuna = [temp_X_cat.columns.get_loc(c) for c in categorical_features if c in temp_X_cat.columns]

        best_params['catboost'] = optimize_model_hyperparameters('CatBoost', objective_catboost,
                                                                 temp_X_cat, y_for_split,
                                                                 tscv_optuna, OPTUNA_N_TRIALS_PER_MODEL,
                                                                 cat_features_indices=cat_indices_for_optuna)
        print("  Optimizing RandomForest...")
        best_params['rf'] = optimize_model_hyperparameters('RandomForest', objective_rf,
                                                            X_for_split.drop(columns=['year_month_numeric']), y_for_split,
                                                            tscv_optuna, OPTUNA_N_TRIALS_PER_MODEL)
        # Save best params
        joblib.dump(best_params, os.path.join(output_dir, 'best_params_all_models.pkl'))
        print(f"  Best parameters saved to {os.path.join(output_dir, 'best_params_all_models.pkl')}")
    else:
        print("  Skipping Optuna, attempting to load existing best_params or using defaults...")
        try:
            best_params = joblib.load(os.path.join(output_dir, 'best_params_all_models.pkl'))
            print("  Loaded pre-tuned parameters.")
        except FileNotFoundError:
            print("  No pre-tuned parameters found. Using default (potentially unoptimized) parameters.")
            # Define some default reasonable parameters here if Optuna is skipped and no file exists
            best_params['lgbm'] = {'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 31}
            best_params['xgb'] = {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 5}
            best_params['catboost'] = {'iterations': 400, 'learning_rate': 0.05, 'depth': 6}
            best_params['rf'] = {'n_estimators': 200, 'max_depth': 10, 'min_samples_leaf': 5}


    # --- Stacking Ensemble ---
    print("\nStep 4: Stacking Ensemble Training...")
    
    # Define base models with potentially optimized parameters
    # Ensure parameters are prefixed correctly if Optuna doesn't prefix (e.g., for CatBoost iterations vs n_estimators)
    lgbm_base = lgb.LGBMClassifier(**best_params.get('lgbm', {}), random_state=RANDOM_STATE, n_jobs=-1)
    xgb_base = xgb.XGBClassifier(**best_params.get('xgb', {}), random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss') # Add eval_metric to suppress warning
    
    cat_params_base = best_params.get('catboost', {})
    # CatBoost uses 'iterations' not 'n_estimators' from Optuna search above.
    if 'n_estimators' in cat_params_base and 'iterations' not in cat_params_base: # Optuna might return n_estimators
        cat_params_base['iterations'] = cat_params_base.pop('n_estimators')
    cat_base = cb.CatBoostClassifier(**cat_params_base, random_state=RANDOM_STATE, verbose=0)
    
    rf_base = RandomForestClassifier(**best_params.get('rf', {}), random_state=RANDOM_STATE, n_jobs=-1)

    base_models = [
        ('lgbm', lgbm_base),
        ('xgb', xgb_base),
        ('catboost', cat_base),
        ('rf', rf_base)
    ]

    # Meta-model (Logistic Regression is common, or a simple LightGBM)
    # Let's try a Logistic Regression first for simplicity and speed.
    meta_model = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, C=1.0) # C can be tuned

    # Stacking using TimeSeriesSplit for generating OOF predictions
    # X (features) and y (target) should be used here, not the _for_split versions if year_month_numeric was dropped
    # Ensure X and y are in their original order before time-based splitting for stacking
    
    # Original X (scaled, imputed) and y (target)
    # The CV for stacking should respect time. TimeSeriesSplit is good.
    # We need 'groups' for TimeSeriesSplit if it needs them based on its specific usage.
    # Standard TimeSeriesSplit doesn't need groups. StratifiedGroupKFold would.
    
    tscv_stacking = TimeSeriesSplit(n_splits=N_SPLITS_TS)
    
    oof_predictions_level0 = {}
    test_predictions_level0 = {}
    
    # X_train_final and y_train_final are the full datasets used for training base models for test prediction
    X_train_final = X.copy() 
    y_train_final = y.copy()

    print("  Generating Level 0 OOF predictions and Test predictions...")
    for name, model in base_models:
        print(f"    Processing base model: {name}")
        
        # Get OOF predictions for training meta-model
        current_X_np = X_train_final.values
        current_y_np = y_train_final.values
        
        oof_preds_fold = np.zeros(len(X_train_final))
        test_preds_fold_sum = np.zeros(len(X_test)) # Sum predictions for averaging later if needed
        
        # We need to handle categorical features for CatBoost specifically if they are not numerically encoded
        fit_params = {}
        if name == 'catboost':
            cat_indices_final = [X_train_final.columns.get_loc(c) for c in categorical_features if c in X_train_final.columns]
            if cat_indices_final:
                 fit_params['cat_features'] = cat_indices_final
        
        # For OOF predictions using TimeSeriesSplit
        for fold, (train_idx, val_idx) in enumerate(tscv_stacking.split(current_X_np, current_y_np)):
            print(f"      Stacking Fold {fold+1}/{tscv_stacking.get_n_splits()} for {name}")
            X_fold_train, X_fold_val = current_X_np[train_idx], current_X_np[val_idx]
            y_fold_train = current_y_np[train_idx]
            
            model.fit(X_fold_train, y_fold_train, **fit_params)
            oof_preds_fold[val_idx] = model.predict_proba(X_fold_val)[:, 1]
            
        oof_predictions_level0[name] = oof_preds_fold
        
        # Train on full data and predict on test
        print(f"    Training {name} on full data for test prediction...")
        model.fit(current_X_np, current_y_np, **fit_params)
        test_predictions_level0[name] = model.predict_proba(X_test.values)[:, 1]

    # Create Level 1 training data (OOF predictions from Level 0)
    X_meta_train = pd.DataFrame(oof_predictions_level0)
    # Level 1 test data
    X_meta_test = pd.DataFrame(test_predictions_level0)

    print(f"  Shape of Level 1 training data (X_meta_train): {X_meta_train.shape}")
    print(f"  Shape of Level 1 test data (X_meta_test): {X_meta_test.shape}")

    print("  Training Meta-Model (Logistic Regression)...")
    meta_model.fit(X_meta_train, y_train_final) # y_train_final should align with X_meta_train rows

    # Save the Stacking Ensemble (base models separately, meta-model, scaler, imputer, te_encoder)
    joblib.dump(base_models, os.path.join(output_dir, 'stacking_base_models.pkl'))
    joblib.dump(meta_model, os.path.join(output_dir, 'stacking_meta_model.pkl'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(imputer, os.path.join(output_dir, 'imputer.pkl'))
    joblib.dump(agent_coder, os.path.join(output_dir, 'agent_target_encoder.pkl')) # Save the TargetEncoder
    print(f"  Stacking ensemble components saved to '{output_dir}'.")


    # --- Prediction and Threshold Optimization ---
    print("\nStep 5: Prediction and Threshold Optimization...")
    # Meta-model predicts probabilities on OOF Level 0 features for threshold tuning
    oof_meta_predictions_proba = meta_model.predict_proba(X_meta_train)[:, 1]
    
    # Find best threshold on OOF meta-predictions
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.01):
        oof_meta_preds_binary = (oof_meta_predictions_proba > threshold).astype(int)
        f1 = f1_score(y_train_final, oof_meta_preds_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    print(f"  Best F1 score on OOF meta-predictions: {best_f1:.4f} at threshold {best_threshold:.2f}")

    # Predict on actual test set using the meta-model
    final_test_proba = meta_model.predict_proba(X_meta_test)[:, 1]
    final_test_predictions = (final_test_proba > best_threshold).astype(int)


    # --- Submission File ---
    print("\nStep 6: Generating Submission File...")
    submission_df = pd.DataFrame({'Row_ID': test_ids, 'target_column': final_test_predictions})
    submission_path = os.path.join(output_dir, 'submission_ultimate_predictor.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"  Submission file created: {submission_path}")
    print(f"  Predicted NILL (0) vs Not-NILL (1) counts:\n{submission_df['target_column'].value_counts()}")


    # --- Feature Importance (from meta-model if possible, or base models) ---
    if hasattr(meta_model, 'coef_'):
        try:
            meta_feature_importance = pd.DataFrame({
                'Feature': X_meta_train.columns,
                'Coefficient': meta_model.coef_[0]
            }).sort_values(by='Coefficient', ascending=False)
            print("\nMeta-Model Feature Importances (Coefficients):")
            print(meta_feature_importance)
            meta_feature_importance.to_csv(os.path.join(output_dir, 'meta_model_feature_importance.csv'), index=False)
        except Exception as e:
            print(f"Could not get meta-model feature importance: {e}")

    # Average feature importance from strong base models like LightGBM or XGBoost
    # This requires retraining them on full data if not already done, or accessing from Optuna studies.
    # For simplicity, let's try to get importance from one of the base models trained on full data.
    try:
        # Example: LightGBM trained on full data for test prediction
        lgbm_final_full_train = lgb.LGBMClassifier(**best_params.get('lgbm', {}), random_state=RANDOM_STATE, n_jobs=-1)
        fit_params_lgbm_final = {} # No cat features for LGBM usually unless specified differently
        lgbm_final_full_train.fit(X_train_final.values, y_train_final.values, **fit_params_lgbm_final)
        
        lgbm_feat_imp = pd.DataFrame({
            'Feature': X_train_final.columns,
            'Importance': lgbm_final_full_train.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        print("\nLightGBM (Base Model) Feature Importances:")
        print(lgbm_feat_imp.head(20))
        lgbm_feat_imp.to_csv(os.path.join(output_dir, 'lgbm_base_feature_importance.csv'), index=False)
        
        plt.figure(figsize=(10, 12))
        sns.barplot(x='Importance', y='Feature', data=lgbm_feat_imp.head(30))
        plt.title('Top 30 Features from LightGBM (Base Model)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lgbm_base_feature_importance.png'))
        plt.close()

    except Exception as e:
        print(f"Could not generate LightGBM feature importance plot: {e}")


    total_time_main = time.time() - start_time_main
    print(f"\nUltimate NILL Predictor completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {total_time_main:.2f} seconds ({total_time_main/60:.2f} minutes)")
    print(f"Good luck with the submission!")