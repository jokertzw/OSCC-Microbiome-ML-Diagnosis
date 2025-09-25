#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Visualization Analysis Script

Features include:
1. Model performance evaluation visualization
2. Feature importance analysis charts
3. Feature selection process visualization
4. Model generalization validation
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneGroupOut
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
# SVM removed - using only 4 models for paper
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix, auc
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
import logging
from itertools import combinations
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set random seed
np.random.seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set Chinese font and plotting style
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create output directory
output_dir = 'visualization_analysis_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def main():
    """
    Main function - run complete visualization analysis workflow
    """
    logger.info("Starting comprehensive visualization analysis...")
    
    try:
        print("\n" + "="*60)
        print("Starting Microbial Classification Model Comprehensive Visualization Analysis")
        print("="*60)
        
        # 1. Genus level analysis
        print("\nPerforming Genus level analysis...")
        genus_results = analyze_data_level('genus')
        
        # 2. Species level analysis
        print("\nPerforming Species level analysis...")
        species_results = analyze_data_level('species')
        

        
        print("\n" + "="*60)
        print("Microbial Classification Model Comprehensive Visualization Analysis Completed!")
        print("="*60)
        print(f"Visualization files saved to: {output_dir}/")

        if genus_results and 'best_cohort' in genus_results:
            print(f"Genus best validation cohort: {genus_results['best_cohort']}")
        if species_results and 'best_cohort' in species_results:
            print(f"Species best validation cohort: {species_results['best_cohort']}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error occurred during analysis: {e}")
        print(f"Analysis failed: {e}")
        raise

def analyze_data_level(data_type):
    """
    Analyze specific data level (genus or species)
    """
    logger.info(f"Starting {data_type} level analysis...")
    
    try:
        # 1. Create model performance visualization
        print(f"  Creating {data_type} model performance visualization...")
        model_results = create_model_performance_visualization(data_type)
        
        # 2. Create feature importance visualization
        print(f"  Creating {data_type} feature importance visualization...")
        feature_results = create_feature_importance_visualization(data_type)
        
        # 3. Create feature selection process visualization
        print(f"  Creating {data_type} feature selection process visualization...")
        selection_results = create_feature_selection_process_visualization(data_type)
        
        # 4. Create cohort generalization validation
        print(f"  Creating {data_type} cohort generalization validation...")
        validation_results = create_cohort_generalization_validation(data_type)
        
        # Find best performing cohort
        best_cohort = None
        best_auc = 0
        if validation_results:
            for cohort, results in validation_results.items():
                if results:
                    cohort_aucs = [result['auc'] for result in results.values() if 'auc' in result]
                    if cohort_aucs:
                        avg_auc = np.mean(cohort_aucs)
                        if avg_auc > best_auc:
                            best_auc = avg_auc
                            best_cohort = cohort
        
        return {
            'data_type': data_type,
            'model_results': model_results,
            'feature_results': feature_results,
            'selection_results': selection_results,
            'validation_results': validation_results,
            'best_cohort': best_cohort,
            'best_auc': best_auc
        }
        
    except Exception as e:
        logger.error(f"{data_type} level analysis failed: {e}")
        return None



def load_and_prepare_data(data_type='genus', exclude_validation_cohort=True):
    """
    Load and prepare data for analysis
    """
    logger.info(f"Loading {data_type} level data...")
    
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
        return None, None, None, None, None, None
    
    # Merge data
    logger.info("Merging data...")
    merged_data = abundance_data.merge(metadata[['SampleID', 'group', 'Cohort']], 
                                     left_index=True, right_on='SampleID', how='inner')
    
    # Separate training and validation data
    if exclude_validation_cohort:
        # Exclude SRP097643 for training, use for validation only
        train_mask = merged_data['Cohort'] != 'SRP097643'
        train_data = merged_data[train_mask].copy()
        validation_data = merged_data[~train_mask].copy()
    else:
        # Use all data for training
        train_data = merged_data.copy()
        validation_data = None
    
    # Prepare features and labels
    feature_columns = [col for col in train_data.columns if col not in ['SampleID', 'group', 'Cohort']]
    X_train = train_data[feature_columns].values
    y_train = (train_data['group'] == 'OSCC').astype(int).values
    batch_info_train = train_data['Cohort'].values
    
    if validation_data is not None and len(validation_data) > 0:
        X_validation = validation_data[feature_columns].values
        y_validation = (validation_data['group'] == 'OSCC').astype(int).values
        batch_info_validation = validation_data['Cohort'].values
    else:
        X_validation = None
        y_validation = None
        batch_info_validation = None
    
    logger.info(f"Training data shape: {X_train.shape}")
    if X_validation is not None:
        logger.info(f"Validation data shape: {X_validation.shape}")
    
    return X_train, y_train, batch_info_train, X_validation, y_validation, feature_columns

def enhanced_batch_correction(X, batch_info, method='combat'):
    """
    Enhanced batch effect correction
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

def ultra_advanced_feature_selection(X, y, batch_info=None, n_features=50, alpha=0.05, feature_names=None):
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
    
    # Add feature names if provided
    if feature_names is not None:
        feature_scores['feature_name'] = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in range(len(feature_scores))]
    
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

def create_model_performance_visualization(data_type='genus'):
    """
    Create model performance comparison visualization
    """
    logger.info(f"Creating {data_type} model performance visualization...")
    
    # Load data
    X_train, y_train, batch_info_train, X_validation, y_validation, feature_columns = load_and_prepare_data(data_type)
    
    if X_train is None:
        logger.error("Failed to load data")
        return None
    
    # Data preprocessing
    X_processed = advanced_preprocessing_pipeline(X_train, y_train, batch_info_train)
    
    # Feature selection
    selected_features, feature_scores = ultra_advanced_feature_selection(
        X_processed, y_train, batch_info_train, n_features=min(50, X_train.shape[1]//2), feature_names=feature_columns
    )
    
    X_selected = X_processed[:, selected_features]
    
    # Advanced sampling
    from imblearn.over_sampling import SMOTE
    try:
        smote = SMOTE(random_state=42, k_neighbors=min(5, np.bincount(y_train).min()-1))
        X_final, y_sampled = smote.fit_resample(X_selected, y_train)
    except:
        X_final, y_sampled = X_selected, y_train
    
    # Data standardization
    scaler = RobustScaler()
    X_final = scaler.fit_transform(X_final)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{data_type.capitalize()} Level Model Performance Comparison', fontsize=16, fontweight='bold')
    
    model_results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for idx, (name, model) in enumerate(models.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Cross-validation ROC curves
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_final, y_sampled)):
            X_train_fold, X_val_fold = X_final[train_idx], X_final[val_idx]
            y_train_fold, y_val_fold = y_sampled[train_idx], y_sampled[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
            
            fpr, tpr, _ = roc_curve(y_val_fold, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            
            # Interpolate to common FPR grid
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            
            ax.plot(fpr, tpr, alpha=0.3, label=f'Fold {fold+1} (AUC = {roc_auc:.3f})')
        
        # Calculate mean ROC curve
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        ax.plot(mean_fpr, mean_tpr, color='b', linewidth=2,
                label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
        
        # Fill standard deviation area
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2)
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.6)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{name}')
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Save results
        model_results[name] = {
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'aucs': aucs
        }
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{data_type}_model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/{data_type}_model_performance_comparison.pdf', bbox_inches='tight')
    plt.close()
    
    # Save results
    with open(f'{output_dir}/{data_type}_model_performance_results.json', 'w', encoding='utf-8') as f:
        json.dump(model_results, f, ensure_ascii=False, indent=2)
    
    # Save ROC curve plotting data
    roc_data_for_plotting = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        model_roc_data = {
            'folds': [],
            'mean_curve': {},
            'std_curve': {}
        }
        
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        # Recalculate each fold's ROC data for saving
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_final, y_sampled)):
            X_train_fold, X_val_fold = X_final[train_idx], X_final[val_idx]
            y_train_fold, y_val_fold = y_sampled[train_idx], y_sampled[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
            
            fpr, tpr, thresholds = roc_curve(y_val_fold, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Interpolate
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)
            
            model_roc_data['folds'].append({
                'fold': fold + 1,
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc
            })
        
        # Calculate mean and std curves
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        
        model_roc_data['mean_curve'] = {
            'fpr': mean_fpr.tolist(),
            'tpr': mean_tpr.tolist(),
            'auc': auc(mean_fpr, mean_tpr),
            'std_auc': np.std(aucs)
        }
        
        model_roc_data['std_curve'] = {
            'tpr_upper': np.minimum(mean_tpr + std_tpr, 1).tolist(),
            'tpr_lower': np.maximum(mean_tpr - std_tpr, 0).tolist()
        }
        
        roc_data_for_plotting[name] = model_roc_data
    
    # Save plotting data
    with open(f'{output_dir}/{data_type}_roc_plotting_data.json', 'w', encoding='utf-8') as f:
        json.dump(roc_data_for_plotting, f, ensure_ascii=False, indent=2)
    
    logger.info(f"{data_type} model performance visualization completed")
    return model_results

def create_feature_importance_visualization(data_type='genus'):
    """
    Create feature importance analysis visualization
    """
    logger.info(f"Creating {data_type} feature importance visualization...")
    
    # Load data
    X_train, y_train, batch_info_train, X_validation, y_validation, feature_columns = load_and_prepare_data(data_type)
    
    if X_train is None:
        logger.error("Failed to load data")
        return None
    
    # Data preprocessing
    X_processed = advanced_preprocessing_pipeline(X_train, y_train, batch_info_train)
    
    # Feature selection
    selected_features, feature_scores = ultra_advanced_feature_selection(
        X_processed, y_train, batch_info_train, n_features=min(50, X_train.shape[1]//2), feature_names=feature_columns
    )
    
    # Create visualization
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'{data_type.capitalize()} Level Feature Importance Analysis', fontsize=20, fontweight='bold')
    
    # 1. Top 20 features by combined score
    ax1 = fig.add_subplot(gs[0, :2])
    top_20_indices = feature_scores.nlargest(20, 'combined_score').index
    top_20_scores = feature_scores.loc[top_20_indices, 'combined_score']
    top_20_names = [feature_columns[i] if i < len(feature_columns) else f'Feature_{i}' for i in top_20_indices]
    
    bars = ax1.barh(range(len(top_20_names)), top_20_scores, color='skyblue', alpha=0.8)
    ax1.set_yticks(range(len(top_20_names)))
    ax1.set_yticklabels(top_20_names, fontsize=10)
    ax1.set_xlabel('Combined Feature Score')
    ax1.set_title('Top 20 Features by Combined Score')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, top_20_scores)):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=8)
    
    # 2. Feature score distribution
    ax2 = fig.add_subplot(gs[0, 2])
    score_columns = ['mw_score', 'mi_score', 'rf_importance', 'xgb_importance', 'f_score']
    score_data = [feature_scores[col].values for col in score_columns]
    
    bp = ax2.boxplot(score_data, labels=[col.replace('_', '\n') for col in score_columns], patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_title('Feature Score Distribution')
    ax2.set_ylabel('Score Value')
    ax2.grid(True, alpha=0.3)
    
    # 3. Random Forest feature importance
    ax3 = fig.add_subplot(gs[1, 0])
    rf_top_indices = feature_scores.nlargest(15, 'rf_importance').index
    rf_top_scores = feature_scores.loc[rf_top_indices, 'rf_importance']
    rf_top_names = [feature_columns[i][:20] if i < len(feature_columns) else f'Feature_{i}' for i in rf_top_indices]
    
    ax3.barh(range(len(rf_top_names)), rf_top_scores, color='lightgreen', alpha=0.8)
    ax3.set_yticks(range(len(rf_top_names)))
    ax3.set_yticklabels(rf_top_names, fontsize=8)
    ax3.set_xlabel('RF Importance')
    ax3.set_title('Top 15 RF Important Features')
    ax3.grid(True, alpha=0.3)
    
    # 4. XGBoost feature importance
    ax4 = fig.add_subplot(gs[1, 1])
    xgb_top_indices = feature_scores.nlargest(15, 'xgb_importance').index
    xgb_top_scores = feature_scores.loc[xgb_top_indices, 'xgb_importance']
    xgb_top_names = [feature_columns[i][:20] if i < len(feature_columns) else f'Feature_{i}' for i in xgb_top_indices]
    
    ax4.barh(range(len(xgb_top_names)), xgb_top_scores, color='lightcoral', alpha=0.8)
    ax4.set_yticks(range(len(xgb_top_names)))
    ax4.set_yticklabels(xgb_top_names, fontsize=8)
    ax4.set_xlabel('XGB Importance')
    ax4.set_title('Top 15 XGB Important Features')
    ax4.grid(True, alpha=0.3)
    
    # 5. Mutual information scores
    ax5 = fig.add_subplot(gs[1, 2])
    mi_top_indices = feature_scores.nlargest(15, 'mi_score').index
    mi_top_scores = feature_scores.loc[mi_top_indices, 'mi_score']
    mi_top_names = [feature_columns[i][:20] if i < len(feature_columns) else f'Feature_{i}' for i in mi_top_indices]
    
    ax5.barh(range(len(mi_top_names)), mi_top_scores, color='lightyellow', alpha=0.8)
    ax5.set_yticks(range(len(mi_top_names)))
    ax5.set_yticklabels(mi_top_names, fontsize=8)
    ax5.set_xlabel('MI Score')
    ax5.set_title('Top 15 MI Important Features')
    ax5.grid(True, alpha=0.3)
    
    # 6. Statistical significance (Mann-Whitney U)
    ax6 = fig.add_subplot(gs[2, 0])
    mw_top_indices = feature_scores.nlargest(15, 'mw_score').index
    mw_top_scores = feature_scores.loc[mw_top_indices, 'mw_score']
    mw_top_names = [feature_columns[i][:20] if i < len(feature_columns) else f'Feature_{i}' for i in mw_top_indices]
    
    ax6.barh(range(len(mw_top_names)), mw_top_scores, color='lightpink', alpha=0.8)
    ax6.set_yticks(range(len(mw_top_names)))
    ax6.set_yticklabels(mw_top_names, fontsize=8)
    ax6.set_xlabel('-log10(p-value)')
    ax6.set_title('Top 15 Statistically Significant Features')
    ax6.grid(True, alpha=0.3)
    
    # 7. Feature score correlation heatmap
    ax7 = fig.add_subplot(gs[2, 1:])
    score_corr = feature_scores[score_columns].corr()
    
    im = ax7.imshow(score_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax7.set_xticks(range(len(score_columns)))
    ax7.set_yticks(range(len(score_columns)))
    ax7.set_xticklabels([col.replace('_', ' ').title() for col in score_columns], rotation=45, ha='right')
    ax7.set_yticklabels([col.replace('_', ' ').title() for col in score_columns])
    ax7.set_title('Feature Score Method Correlation')
    
    # Add correlation values
    for i in range(len(score_columns)):
        for j in range(len(score_columns)):
            text = ax7.text(j, i, f'{score_corr.iloc[i, j]:.2f}',
                           ha="center", va="center", color="white" if abs(score_corr.iloc[i, j]) > 0.5 else "black")
    
    plt.colorbar(im, ax=ax7, shrink=0.8)
    
    plt.savefig(f'{output_dir}/{data_type}_feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/{data_type}_feature_importance_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    # Save feature importance results
    feature_importance_results = {
        'top_features': [feature_columns[i] if i < len(feature_columns) else f'Feature_{i}' for i in selected_features],
        'feature_scores': feature_scores.to_dict(),
        'selected_feature_indices': selected_features,
        'total_features': len(feature_columns),
        'selected_count': len(selected_features)
    }
    
    with open(f'{output_dir}/{data_type}_feature_importance_results.json', 'w', encoding='utf-8') as f:
        json.dump(feature_importance_results, f, ensure_ascii=False, indent=2)
    
    # Save feature scores CSV
    feature_scores_with_names = feature_scores.copy()
    feature_scores_with_names['feature_name'] = [feature_columns[i] if i < len(feature_columns) else f'Feature_{i}' for i in range(len(feature_scores))]
    feature_scores_with_names.to_csv(f'{output_dir}/{data_type}_feature_scores.csv')
    
    logger.info(f"{data_type} feature importance visualization completed")
    return feature_importance_results

def create_feature_selection_process_visualization(data_type='genus'):
    """
    Create feature selection process visualization
    """
    logger.info(f"Creating {data_type} feature selection process visualization...")
    
    # Load data
    X_train, y_train, batch_info_train, X_validation, y_validation, feature_columns = load_and_prepare_data(data_type)
    
    if X_train is None:
        logger.error("Failed to load data")
        return None
    
    # Data preprocessing
    X_processed = advanced_preprocessing_pipeline(X_train, y_train, batch_info_train)
    
    # Feature selection with different numbers of features
    feature_counts = [10, 20, 30, 40, 50, 75, 100, 150, 200]
    feature_counts = [n for n in feature_counts if n <= X_processed.shape[1]]
    
    selection_results = {}
    
    for n_features in feature_counts:
        selected_features, feature_scores = ultra_advanced_feature_selection(
            X_processed, y_train, batch_info_train, n_features=n_features, feature_names=feature_columns
        )
        
        # Quick model evaluation
        X_selected = X_processed[:, selected_features]
        
        # Simple sampling
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42, k_neighbors=min(5, np.bincount(y_train).min()-1))
            X_sampled, y_sampled = smote.fit_resample(X_selected, y_train)
        except:
            X_sampled, y_sampled = X_selected, y_train
        
        # Standardization
        scaler = RobustScaler()
        X_final = scaler.fit_transform(X_sampled)
        
        # Quick RF evaluation
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        cv_scores = cross_val_score(rf, X_final, y_sampled, cv=5, scoring='roc_auc')
        
        selection_results[n_features] = {
            'mean_auc': cv_scores.mean(),
            'std_auc': cv_scores.std(),
            'selected_features': [feature_columns[i] if i < len(feature_columns) else f'Feature_{i}' for i in selected_features]
        }
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{data_type.capitalize()} Level Feature Selection Process Analysis', fontsize=16, fontweight='bold')
    
    # 1. Performance vs number of features
    ax1 = axes[0, 0]
    n_features_list = list(selection_results.keys())
    mean_aucs = [selection_results[n]['mean_auc'] for n in n_features_list]
    std_aucs = [selection_results[n]['std_auc'] for n in n_features_list]
    
    ax1.errorbar(n_features_list, mean_aucs, yerr=std_aucs, marker='o', capsize=5, capthick=2)
    ax1.set_xlabel('Number of Selected Features')
    ax1.set_ylabel('Cross-Validation AUC')
    ax1.set_title('Model Performance vs Feature Count')
    ax1.grid(True, alpha=0.3)
    
    # Find optimal number of features
    optimal_idx = np.argmax(mean_aucs)
    optimal_n_features = n_features_list[optimal_idx]
    optimal_auc = mean_aucs[optimal_idx]
    
    ax1.axvline(x=optimal_n_features, color='red', linestyle='--', alpha=0.7)
    ax1.text(optimal_n_features, optimal_auc + 0.01, f'Optimal: {optimal_n_features}', 
             ha='center', va='bottom', color='red', fontweight='bold')
    
    # 2. Feature selection method comparison
    ax2 = axes[0, 1]
    
    # Get feature scores for optimal number of features
    _, feature_scores = ultra_advanced_feature_selection(
        X_processed, y_train, batch_info_train, n_features=optimal_n_features, feature_names=feature_columns
    )
    
    score_methods = ['mw_score', 'mi_score', 'rf_importance', 'xgb_importance', 'f_score']
    method_names = ['Mann-Whitney', 'Mutual Info', 'RF Importance', 'XGB Importance', 'F-Score']
    
    # Calculate correlation between methods
    method_correlations = []
    for method in score_methods:
        corr_with_combined = feature_scores['combined_score'].corr(feature_scores[method])
        method_correlations.append(corr_with_combined)
    
    bars = ax2.bar(method_names, method_correlations, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
    ax2.set_ylabel('Correlation with Combined Score')
    ax2.set_title('Feature Selection Method Contribution')
    ax2.set_xticklabels(method_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, corr in zip(bars, method_correlations):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Top features stability across different selection sizes
    ax3 = axes[1, 0]
    
    # Count how often each feature appears in top selections
    feature_stability = {}
    for n_features in n_features_list:
        if n_features >= 20:  # Only consider selections with at least 20 features
            top_features = selection_results[n_features]['selected_features'][:20]  # Top 20
            for feature in top_features:
                if feature not in feature_stability:
                    feature_stability[feature] = 0
                feature_stability[feature] += 1
    
    # Get most stable features
    stable_features = sorted(feature_stability.items(), key=lambda x: x[1], reverse=True)[:15]
    feature_names = [item[0][:25] for item in stable_features]  # Truncate long names
    stability_counts = [item[1] for item in stable_features]
    
    bars = ax3.barh(range(len(feature_names)), stability_counts, color='lightblue', alpha=0.8)
    ax3.set_yticks(range(len(feature_names)))
    ax3.set_yticklabels(feature_names, fontsize=9)
    ax3.set_xlabel('Selection Frequency')
    ax3.set_title('Most Stable Features Across Selection Sizes')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, stability_counts):
        ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                str(count), va='center', fontsize=8)
    
    # 4. Feature score distribution for optimal selection
    ax4 = axes[1, 1]
    
    optimal_features, optimal_scores = ultra_advanced_feature_selection(
        X_processed, y_train, batch_info_train, n_features=optimal_n_features, feature_names=feature_columns
    )
    
    # Plot distribution of combined scores
    all_scores = optimal_scores['combined_score'].values
    selected_scores = optimal_scores.loc[optimal_features, 'combined_score'].values
    
    ax4.hist(all_scores, bins=50, alpha=0.5, label='All Features', color='lightgray')
    ax4.hist(selected_scores, bins=20, alpha=0.7, label='Selected Features', color='red')
    ax4.axvline(x=np.min(selected_scores), color='red', linestyle='--', alpha=0.7, 
                label=f'Selection Threshold: {np.min(selected_scores):.3f}')
    
    ax4.set_xlabel('Combined Feature Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Feature Score Distribution (Optimal: {optimal_n_features} features)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{data_type}_feature_selection_process.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/{data_type}_feature_selection_process.pdf', bbox_inches='tight')
    plt.close()
    
    # Save results
    process_results = {
        'optimal_n_features': optimal_n_features,
        'optimal_auc': optimal_auc,
        'selection_results': selection_results,
        'feature_stability': feature_stability,
        'method_correlations': dict(zip(method_names, method_correlations))
    }
    
    with open(f'{output_dir}/{data_type}_feature_selection_process_results.json', 'w', encoding='utf-8') as f:
        json.dump(process_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"{data_type} feature selection process visualization completed")
    return process_results

def load_validation_data(data_type='genus'):
    """
    Load validation data and trained models
    """
    logger.info(f"Loading validation data for {data_type}...")
    
    try:
        # Load results from step1 analysis
        with open(f'{data_type}_ultra_results.json', 'r', encoding='utf-8') as f:
            step1_results = json.load(f)
        
        # Load feature scores
        feature_scores = pd.read_csv(f'{data_type}_ultra_feature_scores.csv', index_col=0)
        
        return step1_results, feature_scores
    
    except FileNotFoundError as e:
        logger.warning(f"Could not load step1 results: {e}")
        return None, None

def create_cohort_generalization_validation(data_type='genus'):
    """
    Create cohort generalization validation visualization
    """
    logger.info(f"Creating {data_type} cohort generalization validation...")
    
    # Load step1 results
    step1_results, feature_scores = load_validation_data(data_type)
    
    if step1_results is None:
        logger.warning("No step1 results found, creating mock validation...")
        return create_mock_validation(data_type)
    
    # Load data for validation
    X_train, y_train, batch_info_train, X_validation, y_validation, feature_columns = load_and_prepare_data(data_type)
    
    if X_train is None or X_validation is None:
        logger.error("Failed to load validation data")
        return None
    
    # Recreate preprocessing pipeline from step1
    X_processed = advanced_preprocessing_pipeline(X_train, y_train, batch_info_train)
    
    # Use same feature selection as step1
    selected_features = step1_results.get('selected_features', list(range(min(50, X_train.shape[1]))))
    X_selected = X_processed[:, selected_features]
    
    # Apply same sampling
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42, k_neighbors=min(5, np.bincount(y_train).min()-1))
        X_sampled, y_sampled = smote.fit_resample(X_selected, y_train)
    except:
        X_sampled, y_sampled = X_selected, y_train
    
    # Apply same standardization
    scaler = RobustScaler()
    X_final = scaler.fit_transform(X_sampled)
    
    # Train models (recreate from step1)
    trained_models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Train all models
    for name, model in trained_models.items():
        model.fit(X_final, y_sampled)
    
    # Prepare validation data with same preprocessing
    X_val_processed = advanced_preprocessing_pipeline(X_validation, y_validation, batch_info=None)
    X_val_selected = X_val_processed[:, selected_features]
    X_val_final = scaler.transform(X_val_selected)  # Use training scaler
    
    # Validation results
    validation_results = {}
    
    logger.info(f"Validating trained models on SRP097643 (validation samples: {len(y_validation)})...")
    
    for model_name, trained_model in trained_models.items():
        try:
            # Use trained model to predict on SRP097643
            y_pred_proba = trained_model.predict_proba(X_val_final)[:, 1]
            
            # Calculate AUC and ROC curve
            test_auc = roc_auc_score(y_validation, y_pred_proba)
            fpr, tpr, _ = roc_curve(y_validation, y_pred_proba)
            
            validation_results[model_name] = {
                'auc': test_auc,
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'train_samples': 499,  # Training samples (from other 4 cohorts)
                'test_samples': len(y_validation),
                'test_positive_rate': np.mean(y_validation)
            }
            
            logger.info(f"{model_name} AUC on SRP097643: {test_auc:.4f}")
            
        except Exception as e:
            logger.error(f"Model {model_name} validation on SRP097643 failed: {e}")
    
    # Cohort results format (maintain compatibility with original function)
    cohort_results = {'SRP097643': validation_results}
    
    # Create visualization
    n_cohorts = len(cohort_results)
    n_models = len(trained_models)
    
    fig, axes = plt.subplots(n_cohorts, 2, figsize=(16, 4*n_cohorts))
    if n_cohorts == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'{data_type.capitalize()} Level Model Validation on SRP097643 (Holdout Cohort)', fontsize=16, fontweight='bold')
    
    # Create ROC curve plot for each validation cohort
    for idx, (cohort, results) in enumerate(cohort_results.items()):
        # ROC curves
        ax_roc = axes[idx, 0]
        
        for model_name, result in results.items():
            if 'fpr' in result and 'tpr' in result:
                ax_roc.plot(result['fpr'], result['tpr'], 
                           label=f"{model_name} (AUC = {result['auc']:.3f})", 
                           linewidth=2)
        
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'ROC Curves - {cohort}')
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True, alpha=0.3)
        
        # Performance metrics bar plot
        ax_metrics = axes[idx, 1]
        models = list(results.keys())
        aucs = [results[model]['auc'] for model in models]
        
        bars = ax_metrics.bar(models, aucs, alpha=0.7)
        ax_metrics.set_ylabel('AUC Score')
        ax_metrics.set_title(f'Model Performance - {cohort}')
        ax_metrics.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar, auc in zip(bars, aucs):
            height = bar.get_height()
            ax_metrics.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{auc:.3f}', ha='center', va='bottom')
        
        plt.setp(ax_metrics.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{data_type}_cohort_validation_results.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'validation_results': cohort_results,
        'summary': {
            'total_cohorts': len(cohort_results),
            'models_tested': len(next(iter(cohort_results.values()))),
            'average_performance': {
                model: np.mean([cohort_results[cohort][model]['auc'] 
                               for cohort in cohort_results.keys()])
                for model in next(iter(cohort_results.values())).keys()
            }
        }
    }


def create_comprehensive_summary_visualization(genus_results, species_results):
    """
    Create comprehensive summary visualization combining genus and species results
    
    Args:
        genus_results: Results from genus-level analysis
        species_results: Results from species-level analysis
    
    Returns:
        dict: Summary visualization results
    """
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Extract performance data
    genus_performance = genus_results.get('model_performance', {})
    species_performance = species_results.get('model_performance', {})
    
    # 1. Model Performance Comparison (Genus vs Species)
    ax1 = fig.add_subplot(gs[0, :2])
    models = list(genus_performance.keys())
    genus_aucs = [genus_performance[model]['auc'] for model in models]
    species_aucs = [species_performance[model]['auc'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, genus_aucs, width, label='Genus', alpha=0.8)
    bars2 = ax1.bar(x + width/2, species_aucs, width, label='Species', alpha=0.8)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('AUC Score')
    ax1.set_title('Model Performance Comparison: Genus vs Species')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Feature Importance Heatmap
    ax2 = fig.add_subplot(gs[0, 2:])
    genus_features = genus_results.get('feature_importance', {})
    species_features = species_results.get('feature_importance', {})
    
    # Get top features from both analyses
    all_features = set()
    for model_features in [genus_features, species_features]:
        for model, features in model_features.items():
            if isinstance(features, dict):
                all_features.update(list(features.keys())[:10])  # Top 10 features per model
    
    if all_features:
        feature_matrix = []
        feature_labels = []
        
        for data_type, features_dict in [('Genus', genus_features), ('Species', species_features)]:
            for model, features in features_dict.items():
                if isinstance(features, dict):
                    row = []
                    for feature in sorted(all_features):
                        row.append(features.get(feature, 0))
                    feature_matrix.append(row)
                    feature_labels.append(f'{data_type}_{model}')
        
        if feature_matrix:
            feature_df = pd.DataFrame(feature_matrix, 
                                    index=feature_labels, 
                                    columns=sorted(all_features))
            
            sns.heatmap(feature_df, ax=ax2, cmap='viridis', 
                       cbar_kws={'label': 'Feature Importance'})
            ax2.set_title('Feature Importance Heatmap')
            ax2.set_xlabel('Features')
            ax2.set_ylabel('Model_DataType')
    
    # 3. Cross-validation Performance Distribution
    ax3 = fig.add_subplot(gs[1, :2])
    cv_data = []
    cv_labels = []
    
    for data_type, results in [('Genus', genus_results), ('Species', species_results)]:
        cv_scores = results.get('cross_validation', {})
        for model, scores in cv_scores.items():
            if isinstance(scores, list):
                cv_data.extend(scores)
                cv_labels.extend([f'{data_type}_{model}'] * len(scores))
    
    if cv_data:
        cv_df = pd.DataFrame({'AUC': cv_data, 'Model': cv_labels})
        sns.boxplot(data=cv_df, x='Model', y='AUC', ax=ax3)
        ax3.set_title('Cross-validation Performance Distribution')
        ax3.set_xlabel('Model_DataType')
        ax3.set_ylabel('AUC Score')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # 4. Validation Cohort Performance
    ax4 = fig.add_subplot(gs[1, 2:])
    genus_validation = genus_results.get('validation_results', {})
    species_validation = species_results.get('validation_results', {})
    
    validation_data = []
    for data_type, validation_results in [('Genus', genus_validation), ('Species', species_validation)]:
        for cohort, cohort_results in validation_results.items():
            for model, result in cohort_results.items():
                validation_data.append({
                    'DataType': data_type,
                    'Cohort': cohort,
                    'Model': model,
                    'AUC': result.get('auc', 0)
                })
    
    if validation_data:
        val_df = pd.DataFrame(validation_data)
        pivot_df = val_df.pivot_table(values='AUC', index=['DataType', 'Model'], 
                                     columns='Cohort', aggfunc='mean')
        sns.heatmap(pivot_df, ax=ax4, annot=True, fmt='.3f', cmap='RdYlBu_r')
        ax4.set_title('Validation Performance Across Cohorts')
    
    # 5. Feature Selection Process
    ax5 = fig.add_subplot(gs[2, :])
    genus_selection = genus_results.get('feature_selection_process', {})
    species_selection = species_results.get('feature_selection_process', {})
    
    selection_data = []
    for data_type, selection_results in [('Genus', genus_selection), ('Species', species_selection)]:
        for method, scores in selection_results.items():
            if isinstance(scores, list):
                for i, score in enumerate(scores):
                    selection_data.append({
                        'DataType': data_type,
                        'Method': method,
                        'Step': i,
                        'Score': score
                    })
    
    if selection_data:
        sel_df = pd.DataFrame(selection_data)
        for data_type in sel_df['DataType'].unique():
            data_subset = sel_df[sel_df['DataType'] == data_type]
            for method in data_subset['Method'].unique():
                method_data = data_subset[data_subset['Method'] == method]
                ax5.plot(method_data['Step'], method_data['Score'], 
                        marker='o', label=f'{data_type}_{method}')
        
        ax5.set_xlabel('Selection Step')
        ax5.set_ylabel('Performance Score')
        ax5.set_title('Feature Selection Process')
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.grid(True, alpha=0.3)
    
    # 6. Summary Statistics
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    # Create summary text
    summary_text = []
    summary_text.append("COMPREHENSIVE ANALYSIS SUMMARY")
    summary_text.append("=" * 50)
    
    for data_type, results in [('GENUS LEVEL', genus_results), ('SPECIES LEVEL', species_results)]:
        summary_text.append(f"\n{data_type} ANALYSIS:")
        summary_text.append("-" * 30)
        
        # Best performing model
        performance = results.get('model_performance', {})
        if performance:
            best_model = max(performance.keys(), key=lambda x: performance[x]['auc'])
            best_auc = performance[best_model]['auc']
            summary_text.append(f"Best Model: {best_model} (AUC: {best_auc:.3f})")
        
        # Number of features selected
        features = results.get('feature_importance', {})
        if features:
            avg_features = np.mean([len(f) for f in features.values() if isinstance(f, dict)])
            summary_text.append(f"Average Features Selected: {avg_features:.1f}")
        
        # Validation performance
        validation = results.get('validation_results', {})
        if validation:
            avg_val_performance = np.mean([
                np.mean([model_result['auc'] for model_result in cohort_results.values()])
                for cohort_results in validation.values()
            ])
            summary_text.append(f"Average Validation AUC: {avg_val_performance:.3f}")
    
    # Add timestamp
    summary_text.append(f"\nAnalysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    ax6.text(0.05, 0.95, '\n'.join(summary_text), transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.savefig(f'{output_dir}/comprehensive_summary_visualization.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'summary_created': True,
        'timestamp': datetime.now().isoformat(),
        'output_file': f'{output_dir}/comprehensive_summary_visualization.png'
    }


if __name__ == "__main__":
    main()