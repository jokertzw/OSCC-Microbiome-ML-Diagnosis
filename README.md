# Machine Learning Models for Oral Squamous Cell Carcinoma Diagnosis Based on Cross-Cohort Oral Microbial Signatures

This repository contains the complete code implementation for the research paper "Machine learning models diagnoses oral squamous cell carcinoma based on cross-cohort oral microbial signatures". The project develops machine learning models to diagnose oral squamous cell carcinoma (OSCC) using oral microbial abundance data from multiple cohorts.

## Project Overview

This study implements a comprehensive machine learning pipeline for OSCC diagnosis using oral microbiome signatures. The approach includes advanced feature selection, cross-cohort validation, and ensemble learning to ensure robust and generalizable diagnostic models.

## Files Description

### 1. step1_advanced_microbial_classification_analysis.py
**Core Machine Learning Pipeline**

This script implements the main machine learning pipeline for OSCC diagnosis with the following key features:

- **Cross-Cohort Validation Strategy**: SRP097643 cohort is exclusively reserved for independent validation to ensure unbiased model evaluation
- **Ultra-Advanced Feature Selection**: Integrates multiple feature selection methods including:
  - Differential abundance analysis (Mann-Whitney U test)
  - Mutual information scoring
  - Model-based feature importance (Random Forest, XGBoost)
  - ANOVA F-test
  - Correlation analysis with multiple testing correction
- **Enhanced Data Preprocessing**: 
  - Outlier detection using Isolation Forest and Elliptic Envelope
  - Missing value imputation with KNN
  - Data transformation (log, square root, Box-Cox)
  - Batch effect correction using ComBat-like methods
- **Hyperparameter Optimization**: Bayesian optimization using Optuna framework
- **Ensemble Learning**: Multi-model fusion including:
  - Random Forest Classifier
  - XGBoost Classifier  
  - Logistic Regression
  - Gradient Boosting Classifier
- **Advanced Sampling Strategies**: Handles class imbalance using SMOTE, ADASYN, BorderlineSMOTE, and hybrid methods

**Key Methodological Features:**
- Strict training/validation separation to prevent data leakage
- Comprehensive feature scoring and selection pipeline
- Multi-level analysis (genus and species level)
- Cross-validation with stratified sampling
- Ensemble model optimization and evaluation

### 2. step2_comprehensive_visualization_analysis.py
**Visualization and Results Analysis**

This script generates publication-ready visualizations and comprehensive analysis reports:

- **Model Performance Visualization**: 
  - ROC curves with confidence intervals
  - AUC comparison across models and cohorts
  - Performance metrics heatmaps
- **Feature Importance Analysis**: 
  - Multi-method feature importance rankings
  - Feature selection process visualization
  - Correlation matrices and clustering
- **Cross-Cohort Validation Results**:
  - Generalization performance across different cohorts
  - Cohort-specific model performance analysis
  - Sample distribution and batch effect visualization
- **Comprehensive Summary Reports**: 
  - Integrated analysis reports with statistical summaries
  - Feature selection process documentation
  - Model comparison and recommendation

**Visualization Features:**
- Publication-quality figures with customizable styling
- Interactive performance comparison charts
- Feature importance heatmaps with hierarchical clustering
- Cross-validation performance distributions
- Comprehensive summary dashboards

## Requirements

### Python Packages
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
optuna
scipy
statsmodels
imbalanced-learn
```

### Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna scipy statsmodels imbalanced-learn
```

## Data Requirements

The analysis requires the following input files:

### Required Input Files
1. **`filtered_normalized_genus_abundance.csv`**: Genus-level microbial abundance data
   - Rows: Sample IDs
   - Columns: Genus-level microbial features
   - Values: Normalized abundance values

2. **`filtered_normalized_species_abundance.csv`**: Species-level microbial abundance data
   - Rows: Sample IDs  
   - Columns: Species-level microbial features
   - Values: Normalized abundance values

3. **`filtered_HC_OSCC_final.csv`**: Sample metadata
   - `SampleID`: Unique sample identifier
   - `group`: Sample classification (HC for healthy controls, OSCC for oral squamous cell carcinoma)
   - `Cohort`: Cohort information (including SRP097643 for independent validation)

### Data Format Requirements
- All abundance data should be pre-normalized and filtered
- Sample IDs must be consistent across abundance and metadata files
- Missing values should be minimal (handled by imputation pipeline)
- Cohort information is essential for cross-cohort validation

## Usage

### Step 1: Run Core Machine Learning Analysis
```bash
python step1_advanced_microbial_classification_analysis.py
```

**This script will:**
- Load and preprocess oral microbial abundance data (genus and species levels)
- Perform advanced feature selection using multiple methods
- Optimize machine learning models using Bayesian optimization
- Train ensemble models on training cohorts (excluding SRP097643)
- Validate models on independent SRP097643 cohort
- Generate comprehensive analysis results for both taxonomic levels
- Save trained models and results to JSON files

### Step 2: Generate Comprehensive Visualizations
```bash
python step2_comprehensive_visualization_analysis.py
```

**This script will:**
- Load analysis results from Step 1
- Generate publication-ready visualization reports
- Create model performance comparison charts
- Visualize feature importance and selection processes
- Perform cross-cohort validation analysis
- Save all visualizations to the `visualization_analysis_results/` directory

## Output Files

### Analysis Results (from Step 1)
- **`genus_ultra_results.json`**: Complete genus-level analysis results including:
  - Selected features and their importance scores
  - Optimized model parameters
  - Cross-validation performance metrics
  - Independent validation results on SRP097643
  
- **`species_ultra_results.json`**: Complete species-level analysis results
- **`genus_ultra_feature_scores.csv`**: Detailed feature scoring results for genus level
- **`species_ultra_feature_scores.csv`**: Detailed feature scoring results for species level

### Visualization Results (from Step 2)
All visualizations are saved in `visualization_analysis_results/` directory:

#### Model Performance Visualizations
- **`{level}_model_performance_comparison.pdf/png`**: Comprehensive model comparison charts
- **`{level}_cohort_validation_results.png`**: Cross-cohort validation performance
- **`{level}_roc_plotting_data.json`**: ROC curve data for external plotting

#### Feature Analysis Visualizations  
- **`{level}_feature_importance_analysis.pdf/png`**: Feature importance heatmaps and rankings
- **`{level}_feature_selection_process.pdf/png`**: Step-by-step feature selection visualization
- **`{level}_feature_scores.csv`**: Comprehensive feature scoring results

#### Summary Reports
- **`{level}_model_performance_results.json`**: Detailed performance metrics
- **`{level}_feature_importance_results.json`**: Feature importance analysis results
- **`{level}_feature_selection_process_results.json`**: Feature selection process documentation

*Note: `{level}` refers to either `genus` or `species` taxonomic levels*

## Technical Specifications

### Machine Learning Models
- **Random Forest Classifier**: Ensemble tree-based method with feature importance ranking
- **XGBoost Classifier**: Gradient boosting with advanced regularization
- **Logistic Regression**: Linear model with L1/L2 regularization
- **Gradient Boosting Classifier**: Sequential ensemble learning

### Feature Selection Methods
- **Statistical Tests**: Mann-Whitney U test, ANOVA F-test
- **Information Theory**: Mutual information scoring
- **Model-Based**: Random Forest and XGBoost feature importance
- **Correlation Analysis**: Pearson/Spearman correlation with multiple testing correction

### Validation Strategy
- **Cross-Validation**: Stratified k-fold cross-validation on training data
- **Independent Validation**: SRP097643 cohort reserved for final model evaluation
- **Performance Metrics**: AUC, sensitivity, specificity, precision, recall, F1-score
