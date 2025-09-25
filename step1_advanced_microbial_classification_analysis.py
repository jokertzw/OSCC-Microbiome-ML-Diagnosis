#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Microbial Classification Analysis Script

Important Notes:
- SRP097643 cohort is exclusively used for model validation and does not participate in the training process
- Training and feature selection are performed only on other cohorts to ensure objective evaluation of model generalization ability

Main Optimization Features:
1. Advanced feature selection: Combines differential analysis, mutual information, model selection and other methods
2. Enhanced data preprocessing: Outlier detection, missing value handling, data transformation, batch correction
3. Hyperparameter optimization: Uses Optuna for Bayesian optimization
4. Ensemble learning: Multi-model fusion to improve performance
5. Advanced sampling strategies: Handles class imbalance
6. Training/validation separation: SRP097643 dedicated for validation to avoid data leakage
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC  # Removed for 4-model analysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import EditedNearestNeighbours
import xgboost as xgb
import optuna
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests
import json
import os
from datetime import datetime
import logging

# Set random seed
np.random.seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def enhanced_batch_correction(X, batch_info, method='combat'):
    """
    Enhanced batch effect correction
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    batch_info : array-like
        Batch information
    method : str
        Correction method ('combat', 'zscore', 'quantile')
    
    Returns:
    --------
    X_corrected : array-like
        Corrected feature matrix
    """
    X_corrected = X.copy()
    
    if method == 'combat':
        # Simplified ComBat method
        unique_batches = np.unique(batch_info)
        for batch in unique_batches:
            batch_mask = batch_info == batch
            if np.sum(batch_mask) > 1:
                batch_data = X_corrected[batch_mask]
                batch_mean = np.mean(batch_data, axis=0)
                batch_std = np.std(batch_data, axis=0) + 1e-8
                
                # Standardize to global distribution
                global_mean = np.mean(X_corrected, axis=0)
                global_std = np.std(X_corrected, axis=0) + 1e-8
                
                X_corrected[batch_mask] = (batch_data - batch_mean) / batch_std * global_std + global_mean
    
    elif method == 'zscore':
        # Z-score standardization
        unique_batches = np.unique(batch_info)
        for batch in unique_batches:
            batch_mask = batch_info == batch
            if np.sum(batch_mask) > 1:
                batch_data = X_corrected[batch_mask]
                X_corrected[batch_mask] = stats.zscore(batch_data, axis=0, nan_policy='omit')
    
    elif method == 'quantile':
        # Quantile normalization
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        X_corrected = qt.fit_transform(X_corrected)
    
    return X_corrected

def ultra_advanced_feature_selection(X, y, batch_info=None, n_features=50, alpha=0.05):
    """
    Ultra advanced feature selection method
    
    Combines multiple feature selection strategies:
    1. Differential analysis (Mann-Whitney U test)
    2. Mutual information
    3. Model-based feature importance
    4. Variance analysis
    5. Correlation analysis
    6. Multiple testing correction
    """
    logger.info("Starting ultra advanced feature selection...")
    
    feature_scores = pd.DataFrame(index=range(X.shape[1]))
    
    # 1. Differential analysis (Mann-Whitney U test)
    logger.info("Performing differential analysis...")
    mw_pvalues = []
    mw_statistics = []
    
    for i in range(X.shape[1]):
        try:
            group0 = X[y == 0, i]
            group1 = X[y == 1, i]
            if len(group0) > 0 and len(group1) > 0:
                stat, pval = mannwhitneyu(group0, group1, alternative='two-sided')
                mw_pvalues.append(pval)
                mw_statistics.append(stat)
            else:
                mw_pvalues.append(1.0)
                mw_statistics.append(0.0)
        except:
            mw_pvalues.append(1.0)
            mw_statistics.append(0.0)
    
    # Multiple testing correction
    _, mw_pvalues_corrected, _, _ = multipletests(mw_pvalues, alpha=alpha, method='fdr_bh')
    feature_scores['mw_pvalue'] = mw_pvalues_corrected
    feature_scores['mw_statistic'] = mw_statistics
    feature_scores['mw_score'] = -np.log10(np.array(mw_pvalues_corrected) + 1e-10)
    
    # 2. Mutual information
    logger.info("Calculating mutual information...")
    try:
        mi_scores = mutual_info_classif(X, y, random_state=42)
        feature_scores['mi_score'] = mi_scores
    except:
        feature_scores['mi_score'] = 0
    
    # 3. Random Forest-based feature importance
    logger.info("Calculating Random Forest feature importance...")
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        feature_scores['rf_importance'] = rf.feature_importances_
    except:
        feature_scores['rf_importance'] = 0
    
    # 4. XGBoost-based feature importance
    logger.info("Calculating XGBoost feature importance...")
    try:
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        xgb_model.fit(X, y)
        feature_scores['xgb_importance'] = xgb_model.feature_importances_
    except:
        feature_scores['xgb_importance'] = 0
    
    # 5. Variance analysis
    logger.info("Performing variance analysis...")
    try:
        f_scores, f_pvalues = f_classif(X, y)
        _, f_pvalues_corrected, _, _ = multipletests(f_pvalues, alpha=alpha, method='fdr_bh')
        feature_scores['f_score'] = f_scores
        feature_scores['f_pvalue'] = f_pvalues_corrected
    except:
        feature_scores['f_score'] = 0
        feature_scores['f_pvalue'] = 1
    
    # 6. Calculate combined score
    logger.info("Calculating combined feature scores...")
    
    # Standardize each score
    score_columns = ['mw_score', 'mi_score', 'rf_importance', 'xgb_importance', 'f_score']
    for col in score_columns:
        if feature_scores[col].std() > 0:
            feature_scores[f'{col}_norm'] = (feature_scores[col] - feature_scores[col].mean()) / feature_scores[col].std()
        else:
            feature_scores[f'{col}_norm'] = 0
    
    # Combined score (weighted average)
    weights = {'mw_score_norm': 0.3, 'mi_score_norm': 0.2, 'rf_importance_norm': 0.2, 
               'xgb_importance_norm': 0.2, 'f_score_norm': 0.1}
    
    feature_scores['combined_score'] = 0
    for score_col, weight in weights.items():
        feature_scores['combined_score'] += feature_scores[score_col] * weight
    
    # Select top features
    top_features = feature_scores.nlargest(n_features, 'combined_score').index.tolist()
    
    logger.info(f"Selected {len(top_features)} features")
    
    return top_features, feature_scores

def advanced_preprocessing_pipeline(X, y, batch_info=None):
    """
    Advanced data preprocessing pipeline
    
    Includes:
    1. Outlier detection and handling
    2. Missing value imputation
    3. Data transformation
    4. Batch effect correction
    """
    logger.info("Starting advanced data preprocessing...")
    
    X_processed = X.copy()
    
    # 1. Outlier detection
    logger.info("Detecting outliers...")
    try:
        # Use Isolation Forest for outlier detection
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(X_processed)
        
        # Use Elliptic Envelope for outlier detection
        elliptic_env = EllipticEnvelope(contamination=0.1, random_state=42)
        outlier_labels_2 = elliptic_env.fit_predict(X_processed)
        
        # Combine outlier detection results
        outlier_mask = (outlier_labels == -1) | (outlier_labels_2 == -1)
        logger.info(f"Detected {np.sum(outlier_mask)} outliers")
        
        # Apply Winsorization to outliers
        for i in range(X_processed.shape[1]):
            col_data = X_processed[:, i]
            q1, q99 = np.percentile(col_data, [1, 99])
            X_processed[:, i] = np.clip(col_data, q1, q99)
    
    except Exception as e:
        logger.warning(f"Outlier detection failed: {e}")
    
    # 2. Missing value imputation
    logger.info("Handling missing values...")
    if np.any(np.isnan(X_processed)):
        try:
            imputer = KNNImputer(n_neighbors=5)
            X_processed = imputer.fit_transform(X_processed)
        except:
            # If KNN imputation fails, use median imputation
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X_processed = imputer.fit_transform(X_processed)
    
    # 3. Data transformation
    logger.info("Performing data transformation...")
    try:
        # Use Yeo-Johnson transformation for skewed distributions
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        X_processed = pt.fit_transform(X_processed)
    except Exception as e:
        logger.warning(f"Data transformation failed: {e}")
    
    # 4. Batch effect correction
    if batch_info is not None:
        logger.info("Performing batch effect correction...")
        try:
            X_processed = enhanced_batch_correction(X_processed, batch_info, method='combat')
        except Exception as e:
            logger.warning(f"Batch effect correction failed: {e}")
    
    logger.info("Data preprocessing completed")
    return X_processed

def ultra_optimize_model(X, y, model_type='rf', n_trials=100, cv_folds=5):
    """
    Hyperparameter optimization using Optuna
    """
    logger.info(f"Starting optimization for {model_type} model...")
    
    def objective(trial):
        if model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'random_state': 42
            }
            model = RandomForestClassifier(**params)
        
        elif model_type == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'eval_metric': 'logloss'
            }
            model = xgb.XGBClassifier(**params)
        
        elif model_type == 'lr':
            params = {
                'C': trial.suggest_float('C', 0.001, 100, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                'solver': 'saga',
                'max_iter': 1000,
                'random_state': 42
            }
            if params['penalty'] == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)
            model = LogisticRegression(**params)
        
        elif model_type == 'gb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42
            }
            model = GradientBoostingClassifier(**params)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()
    
    # Create Optuna study
    study = optuna.create_study(direction='maximize', 
                               sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"Best AUC: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    # Create model with best parameters
    best_params = study.best_params.copy()
    best_params['random_state'] = 42
    
    if model_type == 'rf':
        best_model = RandomForestClassifier(**best_params)
    elif model_type == 'xgb':
        best_params['eval_metric'] = 'logloss'
        best_model = xgb.XGBClassifier(**best_params)
    elif model_type == 'lr':
        best_params['solver'] = 'saga'
        best_params['max_iter'] = 1000
        best_model = LogisticRegression(**best_params)
    elif model_type == 'gb':
        best_model = GradientBoostingClassifier(**best_params)
    
    # Fit best model on complete training data
    best_model.fit(X, y)
    
    return best_model, study.best_value, study.best_params

def create_ultra_ensemble_model(models_dict, X, y, cv_folds=5):
    """
    Create ultra ensemble model
    """
    logger.info("Creating ultra ensemble model...")
    
    # Select best performing models for ensemble
    model_scores = {}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for name, model in models_dict.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        model_scores[name] = scores.mean()
        logger.info(f"{name} average AUC: {scores.mean():.4f}")
    
    # Select top models for ensemble
    top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    ensemble_models = [(name, models_dict[name]) for name, _ in top_models]
    
    # Create voting classifier
    ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
    
    # Evaluate ensemble model
    ensemble_scores = cross_val_score(ensemble, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    logger.info(f"Ensemble model average AUC: {ensemble_scores.mean():.4f}")
    
    # Fit ensemble model on complete training data
    ensemble.fit(X, y)
    
    return ensemble, ensemble_scores.mean()

def advanced_sampling_strategy(X, y, strategy='auto'):
    """
    Advanced sampling strategy for handling class imbalance
    """
    logger.info(f"Performing advanced sampling strategy: {strategy}")
    
    class_counts = np.bincount(y)
    logger.info(f"Original class distribution: {class_counts}")
    
    if strategy == 'auto':
        # Automatically select strategy based on imbalance degree
        imbalance_ratio = max(class_counts) / min(class_counts)
        if imbalance_ratio > 3:
            strategy = 'smoteenn'
        elif imbalance_ratio > 2:
            strategy = 'smote'
        else:
            return X, y  # No sampling needed
    
    try:
        if strategy == 'smote':
            sampler = SMOTE(random_state=42, k_neighbors=min(5, min(class_counts)-1))
        elif strategy == 'adasyn':
            sampler = ADASYN(random_state=42, n_neighbors=min(5, min(class_counts)-1))
        elif strategy == 'borderline':
            sampler = BorderlineSMOTE(random_state=42, k_neighbors=min(5, min(class_counts)-1))
        elif strategy == 'smoteenn':
            sampler = SMOTEENN(random_state=42, 
                              smote=SMOTE(k_neighbors=min(5, min(class_counts)-1)))
        elif strategy == 'smotetomek':
            sampler = SMOTETomek(random_state=42,
                               smote=SMOTE(k_neighbors=min(5, min(class_counts)-1)))
        else:
            logger.warning(f"Unknown sampling strategy: {strategy}, using SMOTE")
            sampler = SMOTE(random_state=42, k_neighbors=min(5, min(class_counts)-1))
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        new_class_counts = np.bincount(y_resampled)
        logger.info(f"Class distribution after sampling: {new_class_counts}")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        logger.warning(f"Sampling failed: {e}, returning original data")
        return X, y

def ultra_comprehensive_analysis(data_type='genus'):
    """
    Ultra comprehensive analysis function
    
    Note: SRP097643 cohort is excluded from training process and dedicated for final model validation
    """
    logger.info(f"Starting ultra comprehensive analysis for {data_type} data")
    
    # Load data
    if data_type == 'genus':
        abundance_file = 'filtered_normalized_genus_abundance.csv'
    else:
        abundance_file = 'filtered_normalized_species_abundance.csv'
    
    metadata_file = 'filtered_HC_OSCC_final.csv'
    
    try:
        abundance_data = pd.read_csv(abundance_file, index_col=0)
        metadata = pd.read_csv(metadata_file)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None
    
    # Data preprocessing
    logger.info("Merging data...")
    merged_data = abundance_data.merge(metadata[['SampleID', 'group', 'Cohort']], 
                                     left_index=True, right_on='SampleID', how='inner')
    
    # Separate training and validation data
    # SRP097643 is dedicated for validation and does not participate in training and feature selection
    logger.info("Separating training and validation data...")
    validation_mask = merged_data['Cohort'] == 'SRP097643'
    train_mask = ~validation_mask
    
    # Training data (excluding SRP097643)
    train_data = merged_data[train_mask].copy()
    validation_data = merged_data[validation_mask].copy()
    
    logger.info(f"Training data sample count: {len(train_data)}")
    logger.info(f"Validation data sample count: {len(validation_data)}")
    logger.info(f"Training data cohort distribution: {train_data['Cohort'].value_counts().to_dict()}")
    logger.info(f"Validation data cohort: {validation_data['Cohort'].unique()}")
    
    # Prepare training features and labels
    feature_columns = [col for col in train_data.columns if col not in ['SampleID', 'group', 'Cohort']]
    X_train = train_data[feature_columns].values
    y_train = (train_data['group'] == 'OSCC').astype(int).values
    batch_info_train = train_data['Cohort'].values
    
    # Prepare validation features and labels
    X_validation = validation_data[feature_columns].values
    y_validation = (validation_data['group'] == 'OSCC').astype(int).values
    
    logger.info(f"Training data shape: {X_train.shape}, label distribution: {np.bincount(y_train)}")
    logger.info(f"Validation data shape: {X_validation.shape}, label distribution: {np.bincount(y_validation)}")
    
    # Advanced data preprocessing (only on training data)
    X_train_processed = advanced_preprocessing_pipeline(X_train, y_train, batch_info_train)
    
    # Ultra advanced feature selection (only on training data)
    selected_features, feature_scores = ultra_advanced_feature_selection(
        X_train_processed, y_train, batch_info_train, n_features=min(50, X_train.shape[1]//2)
    )
    
    X_train_selected = X_train_processed[:, selected_features]
    logger.info(f"Number of features after selection: {X_train_selected.shape[1]}")
    
    # Advanced sampling (only on training data)
    X_train_sampled, y_train_sampled = advanced_sampling_strategy(X_train_selected, y_train, strategy='auto')
    
    # Data standardization (fit on training data)
    scaler = RobustScaler()
    X_train_final = scaler.fit_transform(X_train_sampled)
    
    # Apply same preprocessing steps to validation data
    logger.info("Applying same preprocessing to validation data...")
    # Apply same data preprocessing (excluding batch correction since validation set has only one cohort)
    X_validation_processed = advanced_preprocessing_pipeline(X_validation, y_validation, batch_info=None)
    # Apply same feature selection
    X_validation_selected = X_validation_processed[:, selected_features]
    # Apply same standardization
    X_validation_final = scaler.transform(X_validation_selected)
    
    # Model optimization (only on training data)
    models = {}
    optimal_params = {}  # Save optimal parameters
    model_scores = {}    # Save model scores
    model_types = ['rf', 'xgb', 'lr', 'gb']  # Removed 'svm' for 4-model analysis
    
    for model_type in model_types:
        try:
            logger.info(f"Optimizing {model_type} model...")
            model, score, params = ultra_optimize_model(X_train_final, y_train_sampled, 
                                                       model_type=model_type, 
                                                       n_trials=50)
            models[model_type] = model
            optimal_params[model_type] = params  # Save optimal parameters
            model_scores[model_type] = score     # Save model scores
            logger.info(f"{model_type} training set best AUC: {score:.4f}")
            logger.info(f"{model_type} optimal parameters: {params}")
        except Exception as e:
            logger.error(f"Model {model_type} optimization failed: {e}")
    
    # Create ensemble model (only on training data)
    if len(models) > 1:
        ensemble_model, ensemble_score = create_ultra_ensemble_model(models, X_train_final, y_train_sampled)
        models['ensemble'] = ensemble_model
        logger.info(f"Ensemble model training set AUC: {ensemble_score:.4f}")
    
    # Evaluate all models on validation dataset (SRP097643)
    logger.info("="*50)
    logger.info("Evaluating model generalization ability on validation dataset (SRP097643)")
    logger.info("="*50)
    
    validation_results = {}
    
    for model_name, model in models.items():
        try:
            # Predict on validation set
            y_pred_proba = model.predict_proba(X_validation_final)[:, 1]
            validation_auc = roc_auc_score(y_validation, y_pred_proba)
            validation_results[model_name] = validation_auc
            
            logger.info(f"{model_name} validation set AUC: {validation_auc:.4f}")
            
            # Calculate other metrics
            y_pred = model.predict(X_validation_final)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_validation, y_pred)
            precision = precision_score(y_validation, y_pred, zero_division=0)
            recall = recall_score(y_validation, y_pred, zero_division=0)
            f1 = f1_score(y_validation, y_pred, zero_division=0)
            
            logger.info(f"{model_name} validation set detailed metrics:")
            logger.info(f"  - Accuracy: {accuracy:.4f}")
            logger.info(f"  - Precision: {precision:.4f}")
            logger.info(f"  - Recall: {recall:.4f}")
            logger.info(f"  - F1-score: {f1:.4f}")
            
        except Exception as e:
            logger.error(f"Model {model_name} evaluation on validation set failed: {e}")
            validation_results[model_name] = 0.0
    
    # Find best performing model on validation set
    if validation_results:
        best_model_name = max(validation_results.items(), key=lambda x: x[1])[0]
        best_validation_auc = validation_results[best_model_name]
        logger.info(f"\nBest performing model on validation set: {best_model_name} (AUC: {best_validation_auc:.4f})")
    
    # Save results
    results = {
        'data_type': data_type,
        'train_data_info': {
            'original_shape': X_train.shape,
            'final_shape': X_train_final.shape,
            'cohorts': list(np.unique(batch_info_train)),
            'label_distribution': np.bincount(y_train).tolist()
        },
        'validation_data_info': {
            'shape': X_validation.shape,
            'final_shape': X_validation_final.shape,
            'cohort': 'SRP097643',
            'label_distribution': np.bincount(y_validation).tolist()
        },
        'selected_features': selected_features,
        'selected_features_count': len(selected_features),
        'models': {name: type(model).__name__ for name, model in models.items()},
        'optimal_parameters': optimal_params,  # Add optimization parameters
        'model_training_scores': model_scores,  # Add training scores
        'validation_results': validation_results,
        'best_model_on_validation': {
            'model_name': best_model_name if validation_results else None,
            'validation_auc': best_validation_auc if validation_results else None
        },
        'timestamp': datetime.now().isoformat(),
        'note': 'SRP097643 cohort is dedicated for validation and did not participate in training and feature selection process. Contains Optuna-optimized best parameters.'
    }
    
    # Save feature importance
    feature_scores.to_csv(f'{data_type}_ultra_feature_scores.csv')
    
    # Save results
    with open(f'{data_type}_ultra_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"{data_type} analysis completed")
    logger.info(f"Results saved to: {data_type}_ultra_results.json")
    logger.info(f"Feature importance saved to: {data_type}_ultra_feature_scores.csv")
    
    return models, X_train_final, y_train_sampled, X_validation_final, y_validation, validation_results, results

def run_ultra_optimized_analysis():
    """
    Run ultra optimized analysis
    """
    logger.info("Starting ultra optimized analysis...")
    
    # Analyze genus data
    logger.info("=" * 50)
    logger.info("Analyzing GENUS data")
    logger.info("=" * 50)
    
    try:
        genus_results = ultra_comprehensive_analysis('genus')
        if genus_results:
            logger.info("Genus analysis completed successfully")
    except Exception as e:
        logger.error(f"Genus analysis failed: {e}")
    
    # Analyze species data
    logger.info("=" * 50)
    logger.info("Analyzing SPECIES data")
    logger.info("=" * 50)
    
    try:
        species_results = ultra_comprehensive_analysis('species')
        if species_results:
            logger.info("Species analysis completed successfully")
    except Exception as e:
        logger.error(f"Species analysis failed: {e}")
    
    logger.info("Ultra optimized analysis completed!")

if __name__ == "__main__":
    # Ignore warnings
    warnings.filterwarnings('ignore')
    
    # Run analysis
    run_ultra_optimized_analysis()