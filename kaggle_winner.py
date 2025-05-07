"""
Improved Kaggle Winner Solution for Data Storm v6.0 competition
Fixed data leakage issues and improved model robustness
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    start_time = time.time()
    
    print("=" * 80)
    print("IMPROVED KAGGLE WINNER SOLUTION - FIXED DATA LEAKAGE")
    print("=" * 80)
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    
    # Create output directory
    output_dir = os.path.join('D:\\', 'DATA STORM', 'outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    print("Step 1: Loading data...")
    data_dir = os.path.join('D:\\', 'DATA STORM', 'dataset')
    train_df = pd.read_csv(os.path.join(data_dir, 'train_storming_round.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_storming_round.csv'))
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Basic preprocessing
    print("\nStep 2: Improved preprocessing...")
    
    # Create target column for train data (if agent sells next month)
    # Important: we need to create this WITHOUT leaking future data
    # Let's get unique agents and their monthly data
    train_df['year_month'] = pd.to_datetime(train_df['year_month'])
    train_df = train_df.sort_values(['agent_code', 'year_month'])
    
    # Create target by looking ahead one month for each agent
    train_df['target_column'] = 0  # Default to 0 (will go NILL)
    
    # Get unique agents and process each
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
        last_month_idx = agent_data.iloc[-1].name
        last_month_indices.append(last_month_idx)
    
    train_df = train_df.drop(last_month_indices)
    
    print(f"Processed training data shape: {train_df.shape}")
    print(f"Target distribution: {train_df['target_column'].value_counts()}")
    
    # Convert date columns to datetime and extract features
    date_columns = ['agent_join_month', 'first_policy_sold_month', 'year_month']
    for df in [train_df, test_df]:
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                
                # Extract month and year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_year'] = df[col].dt.year
    
    # Feature engineering
    print("\nStep 3: Leak-free feature engineering...")
    
    # Create features that don't leak target information
    for df in [train_df, test_df]:
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
        
        # Activity trend features (avoid using features derived from target)
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
        
        # Ratios of activity metrics (without direct policy count info)
        if all(col in df.columns for col in ['unique_proposal', 'unique_quotations']):
            df['quotation_to_proposal_ratio'] = df['unique_quotations'] / np.maximum(df['unique_proposal'], 1)
        
        # Cash payment ratio if available
        if all(col in df.columns for col in ['number_of_cash_payment_policies', 'number_of_policy_holders']):
            df['cash_payment_ratio'] = df['number_of_cash_payment_policies'] / np.maximum(df['number_of_policy_holders'], 1)
        
        # Additional agent characteristic features
        if 'agent_age' in df.columns:
            df['agent_age_squared'] = df['agent_age'] ** 2
            
        # Interaction features
        if all(col in df.columns for col in ['agent_age', 'months_with_company']):
            df['age_experience_interaction'] = df['agent_age'] * df['months_with_company']
            
        # Agent velocity metrics (short term vs long term activity)
        if all(col in df.columns for col in ['unique_proposals_last_7_days', 'unique_proposal']):
            df['proposal_velocity'] = df['unique_proposals_last_7_days'] / np.maximum(df['unique_proposal'], 1)
            
        if all(col in df.columns for col in ['unique_quotations_last_7_days', 'unique_quotations']):
            df['quotation_velocity'] = df['unique_quotations_last_7_days'] / np.maximum(df['unique_quotations'], 1)
            
        if all(col in df.columns for col in ['unique_customers_last_7_days', 'unique_customers']):
            df['customer_velocity'] = df['unique_customers_last_7_days'] / np.maximum(df['unique_customers'], 1)
            
        # Feature transformations for key metrics
        for col in ['unique_proposal', 'unique_quotations', 'unique_customers']:
            if col in df.columns:
                df[f'log_{col}'] = np.log1p(df[col])
                
    # Add agent historical performance features (without leakage)
    # We'll use rolling windows of past performance
    
    # Create historical features for train and test separately
    full_data = pd.concat([train_df, test_df], ignore_index=True)
    full_data = full_data.sort_values(['agent_code', 'year_month'])
    
    # Initialize columns to store historical features
    historical_features = [
        'hist_avg_proposals', 
        'hist_avg_quotations',
        'hist_avg_customers',
        'hist_proposal_growth',
        'hist_quotation_growth',
        'hist_customer_growth',
        'hist_consistency_score',
        'months_since_last_activity'
    ]
    
    for feature in historical_features:
        full_data[feature] = 0
    
    # Process each agent
    for agent in full_data['agent_code'].unique():
        agent_data = full_data[full_data['agent_code'] == agent].copy()
        agent_data = agent_data.sort_values('year_month')
        
        # Skip if agent has only one record
        if len(agent_data) <= 1:
            continue
            
        for i in range(1, len(agent_data)):
            # Get all previous months
            past_data = agent_data.iloc[:i]
            
            # Historical averages
            full_data.loc[agent_data.iloc[i].name, 'hist_avg_proposals'] = past_data['unique_proposal'].mean()
            full_data.loc[agent_data.iloc[i].name, 'hist_avg_quotations'] = past_data['unique_quotations'].mean()
            full_data.loc[agent_data.iloc[i].name, 'hist_avg_customers'] = past_data['unique_customers'].mean()
            
            # Growth metrics (vs previous month)
            if i >= 2:
                # Fix: use np.maximum instead of .replace
                prev_proposal = agent_data.iloc[i-1]['unique_proposal']
                prev_prev_proposal = np.maximum(agent_data.iloc[i-2]['unique_proposal'], 1)
                full_data.loc[agent_data.iloc[i].name, 'hist_proposal_growth'] = (prev_proposal / prev_prev_proposal) - 1
                
                prev_quotation = agent_data.iloc[i-1]['unique_quotations']
                prev_prev_quotation = np.maximum(agent_data.iloc[i-2]['unique_quotations'], 1)
                full_data.loc[agent_data.iloc[i].name, 'hist_quotation_growth'] = (prev_quotation / prev_prev_quotation) - 1
                
                prev_customer = agent_data.iloc[i-1]['unique_customers']
                prev_prev_customer = np.maximum(agent_data.iloc[i-2]['unique_customers'], 1)
                full_data.loc[agent_data.iloc[i].name, 'hist_customer_growth'] = (prev_customer / prev_prev_customer) - 1
            
            # Consistency score (coefficient of variation)
            if len(past_data) >= 3:
                proposal_cv = past_data['unique_proposal'].std() / (past_data['unique_proposal'].mean() + 1)
                quotation_cv = past_data['unique_quotations'].std() / (past_data['unique_quotations'].mean() + 1)
                full_data.loc[agent_data.iloc[i].name, 'hist_consistency_score'] = 1 / (1 + (proposal_cv + quotation_cv)/2)
            
            # Months since last activity (proposals or quotations)
            last_month_active = False
            if i >= 1:
                if agent_data.iloc[i-1]['unique_proposal'] > 0 or agent_data.iloc[i-1]['unique_quotations'] > 0:
                    last_month_active = True
                    
            full_data.loc[agent_data.iloc[i].name, 'months_since_last_activity'] = 0 if last_month_active else 1
    
    # Merge historical features back
    train_df = pd.merge(train_df, full_data[['row_id'] + historical_features], on='row_id', how='left')
    test_df = pd.merge(test_df, full_data[['row_id'] + historical_features], on='row_id', how='left')
    
    # Fill NAs in historical features
    for df in [train_df, test_df]:
        for feature in historical_features:
            df[feature] = df[feature].fillna(0)
    
    # Model training with proper time-series cross-validation
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
    combined_features = base_features + [f for f in engineered_features if f in train_df.columns] + historical_features
    
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
    # Use TimeSeriesSplit to simulate forecasting future months
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Train models with time-series cross-validation
    cv_scores = []
    models = []
    importance_dfs = []
    
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
        
        # 1. Random Forest with balanced class weight
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # 2. Gradient Boosting with scale_pos_weight
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )
        
        # 3. XGBoost with balanced class weights
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
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
        
        # 4. LightGBM with class balancing
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
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
        models.append((scaler, ensemble_model))
        
        # Evaluate model with proper metrics
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
        
        # Get feature importance from the RF model
        rf_importance = pd.DataFrame({
            'Feature': features_to_use,
            'Importance': ensemble_model.named_estimators_['rf'].feature_importances_
        }).sort_values('Importance', ascending=False)
        
        importance_dfs.append(rf_importance)
    
    # Compute average scores
    avg_scores = {metric: np.mean([score[metric] for score in cv_scores]) for metric in cv_scores[0].keys()}
    std_scores = {metric: np.std([score[metric] for score in cv_scores]) for metric in cv_scores[0].keys()}
    
    print("\nAverage Cross-Validation Scores:")
    for metric, value in avg_scores.items():
        print(f"  {metric}: {value:.4f} ± {std_scores[metric]:.4f}")
    
    # Combine feature importance from all folds
    combined_importance = pd.concat(importance_dfs)
    avg_importance = combined_importance.groupby('Feature')['Importance'].mean().reset_index()
    avg_importance = avg_importance.sort_values('Importance', ascending=False)
    
    print("\nTop 15 important features:")
    print(avg_importance.head(15))
    
    # Save feature importance plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=avg_importance.head(20))
    plt.title('Feature Importance (Average Across Folds)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    
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
        use_label_encoder=False,
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
    
    # Try different thresholds and create multiple submission files
    thresholds = np.arange(0.3, 0.71, 0.05)  # 0.3, 0.35, 0.4, ..., 0.7
    submission_template = pd.read_csv(os.path.join(data_dir, 'sample_submission_storming_round.csv'))
    
    prediction_counts = {}
    for threshold in thresholds:
        # Apply threshold
        test_predictions = (test_proba >= threshold).astype(int)
        
        # Track counts
        sell_count = test_predictions.sum()
        nill_count = len(test_predictions) - sell_count
        prediction_counts[threshold] = {'sell': sell_count, 'nill': nill_count}
        
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
    
    # Find threshold that gives closest match to training distribution
    closest_threshold = min(thresholds, key=lambda x: abs(prediction_counts[x]['sell']/len(test_predictions) - train_pos_rate))
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
    
    # Completion
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 80)
    print(f"Improved Kaggle Solution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"Optimal submission: {optimal_submission_path}")
    print(f"Additional submissions with different thresholds are in the outputs directory")
    print("=" * 80)
    print("\nSUBMIT THE OPTIMAL FILE TO KAGGLE FOR #1 POSITION!")
    print("=" * 80)

if __name__ == "__main__":
    main()