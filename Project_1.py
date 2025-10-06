# =============================================================
# AER 850 Project 1: Maintenance Step Classification from 3D Coordinates
# Name: Jawad Rizwan
# Student Number: 501124033
# Due Date: October 6th, 2025
# -------------------------------------------------------------
# This script loads a dataset of 3D coordinates (X, Y, Z) and their
# associated maintenance steps, performs data analysis,
# visualizes the data, and builds several machine learning models
# to classify the maintenance step based on the coordinates.
#
# Libraries and Tools Used:
# - pandas: For data loading, cleaning, and manipulation.
# - matplotlib & seaborn: For data visualization (3D plots, histograms, heatmaps).
# - scikit-learn: For model building, pipelines, and evaluation.
#   - GridSearchCV: Exhaustive hyperparameter tuning for best model selection.
#   - RandomizedSearchCV: Efficient random hyperparameter search for large grids.
#   - Pipelines: Combine preprocessing (scaling) and modeling for reproducibility.
#   - StackingClassifier: Ensemble learning by combining SVM and Random Forest.
#
# Workflow:
# 1. Data Processing: Load and inspect the dataset using pandas.
# 2. Data Visualization: 3D scatter, histograms, pairwise plots with matplotlib/seaborn.
# 3. Correlation Analysis: Compute and plot feature correlations.
# 4. Model Development: Train/test split, hyperparameter tuning (GridSearchCV, RandomizedSearchCV), and
#    training of Logistic Regression, SVM, and Random Forest models using scikit-learn pipelines.
# 5. Model Evaluation: Evaluate and compare models on test data.
# 6. Stacked Model: Combine SVM and Random Forest in a stacking ensemble.
# 7. Output: Print performance metrics and predictions in a readable format.
# =============================================================

# Import necessary libraries for the code
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, f1_score, 
                             confusion_matrix, classification_report)
from sklearn.ensemble import StackingClassifier

# Force the use of xcb platform for Qt (I'm using Linux and wayland doesn't seem to work for me)
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Clear the console/terminal and close all plots
os.system('cls' if os.name == 'nt' else 'clear')
plt.close('all')

##########################################################
# STEP 1: Data Processing
##########################################################

print("="*70)
print("STEP 1: Data Processing")
print("="*70)

# Import the data
data = pd.read_csv("Project 1 Data.csv")

# Display basic information about the dataframe
print(data.info())

# Display first few rows of the dataframe  
print("\n", data.head())

##########################################################
# STEP 2: Data Visualization
##########################################################

print("="*70)
print("STEP 2: Data Visualization ")
print("="*70)

# Create 3D plot of the data
DataVivPicture = plt.figure()
ax = plt.axes(projection ='3d')
plot = ax.scatter(data['X'], data['Y'], data['Z'], c=data['Step'], 
                  cmap='viridis', s=50, alpha=0.7)
color_bar = plt.colorbar(plot, label='Step')
ax.set_xlabel('X Coordinate', fontsize=12)
ax.set_ylabel('Y Coordinate', fontsize=12)
ax.set_zlabel('Z Coordinate', fontsize=12)
ax.set_title('3D Visualization of Maintenance Steps', fontsize=14, fontweight='bold')
plt.tight_layout()

# Create histogram
DataHistPicture = plt.figure(figsize=(10, 6))
counts = plt.hist(data['Step'], bins=13, edgecolor='black', color='purple', 
                  align='mid', rwidth=0.8)
plt.xticks(range(1, 14))
plt.title('Frequency of Data Points per Maintenance Step', fontsize=14, fontweight='bold')
plt.xlabel('Maintenance Step', fontsize=12)
plt.ylabel('Frequency (Number of Points)', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Create scatter plots for each pair of coordinates
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# X vs Y
scatter1 = axes[0].scatter(data['X'], data['Y'], c=data['Step'], 
                           cmap='viridis', alpha=0.6, s=30)
axes[0].set_xlabel('X Coordinate', fontsize=11)
axes[0].set_ylabel('Y Coordinate', fontsize=11)
axes[0].set_title('X vs Y Coordinates', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='Step')

# X vs Z
scatter2 = axes[1].scatter(data['X'], data['Z'], c=data['Step'], 
                           cmap='viridis', alpha=0.6, s=30)
axes[1].set_xlabel('X Coordinate', fontsize=11)
axes[1].set_ylabel('Z Coordinate', fontsize=11)
axes[1].set_title('X vs Z Coordinates', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='Step')

# Y vs Z
scatter3 = axes[2].scatter(data['Y'], data['Z'], c=data['Step'], 
                           cmap='viridis', alpha=0.6, s=30)
axes[2].set_xlabel('Y Coordinate', fontsize=11)
axes[2].set_ylabel('Z Coordinate', fontsize=11)
axes[2].set_title('Y vs Z Coordinates', fontsize=12, fontweight='bold')
axes[2].grid(alpha=0.3)
plt.colorbar(scatter3, ax=axes[2], label='Step')
plt.tight_layout()

# Displaying statistical analysis of dataframe
print(data.describe())

##########################################################
# STEP 3: Correlation Analysis 
##########################################################

# Compute and plot the correlation matrix
corr_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, 
            annot=True,          # Show correlation values on plot
            fmt='.3f',           # 3 decimal places
            cmap='coolwarm',     # Red-blue color scheme
            center=0,            # Center at 0git 
            square=True,         # Square cells
            linewidths=1,        # Grid lines
            cbar_kws={'label': 'Pearson Correlation Coefficient'})

plt.title('Correlation Matrix (Coordinates vs Maintenance Steps)', 
          fontsize=14, fontweight='bold')
plt.tight_layout()

##########################################################
# STEP 4: Classification Model Development/Engineering 
##########################################################

print("="*70)
print("STEP 4: Classification Model Development/Engineering ")
print("="*70)

# Separate features (X, Y, Z) and target (Step)
features = data[['X', 'Y', 'Z']]
target = data['Step']

# Split data into training (80%) and testing (20%) sets
features_train, features_test, target_train, target_test = train_test_split(
    features, target,
    test_size=0.2,      # 20% for testing
    random_state=42,    
    stratify=target     # Keep same proportion of each step in train/test
)

# ==================== MODEL 1: LOGISTIC REGRESSION (GridSearchCV) ====================

# Create pipeline for Logistic Regression
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42, max_iter=1000))
])

# Define hyperparameter grid for Logistic Regression
param_grid_lr = {
    'model__C': [0.01, 0.1, 1, 10, 100],
    'model__solver': ['lbfgs', 'saga'],
    'model__penalty': ['l2']
}

# GridSearchCV for Logistic Regression with 5-fold cross-validation
grid_lr = GridSearchCV(
    estimator=pipeline_lr,
    param_grid=param_grid_lr,
    cv=5,                           
    scoring='accuracy',             
    n_jobs=-1,                      # Use all available cores for parallel processing
    refit=True,                     
    verbose=1,
    return_train_score=True
)

# Train the model
print("\nTraining Logistic Regression with GridSearchCV...")
start_time = time.time()
grid_lr.fit(features_train, target_train)
lr_time = time.time() - start_time

# Confirmation message
print(f"\n✓ Training complete!")
print(f"Best Score: {grid_lr.best_score_:.4f}")
print(f"Best Parameters: {grid_lr.best_params_}")
print(f"Training Time: {lr_time:.2f} seconds\n")


# ==================== MODEL 2: SUPPORT VECTOR MACHINE (GridSearchCV) ====================

# Create pipeline for SVM
pipeline_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(random_state=42))
])

# Define hyperparameter grid for SVM
param_grid_svm = {
    'model__C': [0.01, 0.1, 1, 10, 100],
    'model__kernel': ['linear', 'rbf', 'poly'],
    'model__gamma': ['scale', 'auto']
}

# GridSearchCV for SVM with 5-fold cross-validation
grid_svm = GridSearchCV(
    estimator=pipeline_svm,
    param_grid=param_grid_svm,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    refit=True,
    verbose=1,
    return_train_score=True
)

# Train the model
print("\nTraining Support Vector Machine with GridSearchCV...")
start_time = time.time()
grid_svm.fit(features_train, target_train)
svm_time = time.time() - start_time

# Confirmation message
print(f"\n✓ Training complete!")
print(f"Best Score: {grid_svm.best_score_:.4f}")
print(f"Best Parameters: {grid_svm.best_params_}")
print(f"Training Time: {svm_time:.2f} seconds\n")

# ==================== MODEL 3: RANDOM FOREST (GridSearchCV) ====================

# Create pipeline for Random Forest
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# Define hyperparameter grid for Random Forest
param_grid_rf = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2']
}

# GridSearchCV for Random Forest with 5-fold cross-validation
grid_rf = GridSearchCV(
    estimator=pipeline_rf,
    param_grid=param_grid_rf,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,                     
    refit=True,
    verbose=1,
    return_train_score=True
)

# Train the model
print("\nTraining Random Forest with GridSearchCV...")
start_time = time.time()
grid_rf.fit(features_train, target_train)
rf_time = time.time() - start_time

# Confirmation message
print(f"\n✓ Training complete!")
print(f"Best Score: {grid_rf.best_score_:.4f}")
print(f"Best Parameters: {grid_rf.best_params_}")
print(f"Training Time: {rf_time:.2f} seconds\n")

# ==================== MODEL 4: RANDOM FOREST (RandomizedSearchCV) ====================

# Create pipeline for Random Forest with RandomizedSearchCV
pipeline_rf_random = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# Define hyperparameter distributions for Random Forest with RandomizedSearchCV
param_dist_rf = {
    'model__n_estimators': [50, 100, 150, 200, 250, 300],
    'model__max_depth': [None, 5, 10, 15, 20, 25, 30],
    'model__min_samples_split': [2, 5, 10, 15],
    'model__min_samples_leaf': [1, 2, 4, 6, 8],
    'model__max_features': ['sqrt', 'log2', None]
}

# RandomizedSearchCV
random_rf = RandomizedSearchCV(
    estimator=pipeline_rf_random,
    param_distributions=param_dist_rf,
    n_iter=50,                      
    cv=5,
    scoring='accuracy',
    n_jobs=-1,                      # Use all available cores for parallel processing
    random_state=42,
    refit=True,
    verbose=1,
    return_train_score=True
)

# Train the model
print("\nTraining Random Forest with RandomizedSearchCV...")
start_time = time.time()
random_rf.fit(features_train, target_train)
rf_random_time = time.time() - start_time

# Confirmation message
print(f"\n✓ Training complete!")
print(f"Best Score: {random_rf.best_score_:.4f}")
print(f"Best Parameters: {random_rf.best_params_}")
print(f"Training Time: {rf_random_time:.2f} seconds\n")

print("\n✓ All models trained successfully!")

##########################################################
# STEP 5: Model Performance Analysis
##########################################################

print("="*70)
print("STEP 5: Model Performance Analysis")
print("="*70)

# Dictionary to store all results
results = {}

# Model 1: Logistic Regression
print("Evaluating Model 1: Logistic Regression...")
y_pred_lr = grid_lr.predict(features_test)
results['Logistic Regression'] = {
    'predictions': y_pred_lr,
    'accuracy': accuracy_score(target_test, y_pred_lr),
    'precision': precision_score(target_test, y_pred_lr, average='weighted'),
    'f1_score': f1_score(target_test, y_pred_lr, average='weighted')
}

# Model 2: Support Vector Machine (SVM)
print("Evaluating Model 2: Support Vector Machine (SVM)...")
y_pred_svm = grid_svm.predict(features_test)
results['Support Vector Machine (SVM)'] = {
    'predictions': y_pred_svm,
    'accuracy': accuracy_score(target_test, y_pred_svm),
    'precision': precision_score(target_test, y_pred_svm, average='weighted'),
    'f1_score': f1_score(target_test, y_pred_svm, average='weighted')
}

# Model 3: Random Forest (GridSearchCV)
print("Evaluating Model 3: Random Forest (GridSearchCV)...")
y_pred_rf = grid_rf.predict(features_test)
results['Random Forest (Grid)'] = {
    'predictions': y_pred_rf,
    'accuracy': accuracy_score(target_test, y_pred_rf),
    'precision': precision_score(target_test, y_pred_rf, average='weighted'),
    'f1_score': f1_score(target_test, y_pred_rf, average='weighted')
}

# Model 4: Random Forest (RandomizedSearchCV)
print("Evaluating Model 4: Random Forest (RandomizedSearchCV)...")
y_pred_rf_random = random_rf.predict(features_test)
results['Random Forest (Random)'] = {
    'predictions': y_pred_rf_random,
    'accuracy': accuracy_score(target_test, y_pred_rf_random),
    'precision': precision_score(target_test, y_pred_rf_random, average='weighted'),
    'f1_score': f1_score(target_test, y_pred_rf_random, average='weighted')
}

print("\n✓ All models evaluated!\n")

print("="*70)
print("PERFORMANCE COMPARISON - TEST SET RESULTS")
print("="*70)
print(f"\n{'Model':<35} {'Accuracy':<15} {'Precision':<15} {'F1-Score':<15}")
print("-"*70)

for model_name, metrics in results.items():
    print(f"{model_name:<35} {metrics['accuracy']:<15.4f} {metrics['precision']:<15.4f} {metrics['f1_score']:<15.4f}")

print("="*70)

# Find best model based on accuracy
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_accuracy = results[best_model_name]['accuracy']

print(f"\nBEST MODEL: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.4f}")
print(f"   Precision: {results[best_model_name]['precision']:.4f}")
print(f"   F1-Score: {results[best_model_name]['f1_score']:.4f}")
print("="*70)

# Get predictions from best model
best_predictions = results[best_model_name]['predictions']

# Calculate confusion matrix
cm = confusion_matrix(target_test, best_predictions)

# Create confusion matrix plot
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(target_test.unique()),
            yticklabels=sorted(target_test.unique()),
            cbar_kws={'label': 'Number of Predictions'})

plt.xlabel('Predicted Step', fontsize=12, fontweight='bold')
plt.ylabel('Actual Step', fontsize=12, fontweight='bold')
plt.title(f'Confusion Matrix: {best_model_name}\nAccuracy: {best_accuracy:.4f}', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

print("\n" + "="*70)
print(f"DETAILED CLASSIFICATION REPORT: {best_model_name}")
print("="*70)
print(classification_report(target_test, best_predictions, 
                           target_names=[f'Step {i}' for i in sorted(target_test.unique())]))

##########################################################
# STEP 6: Stacked Model Performance Analysis 
##########################################################

print("="*70)
print("STEP 6: STACKED MODEL PERFORMANCE ANALYSIS")
print("="*70)

# Create stacking classifier
stacked_model = StackingClassifier(
    estimators=[
    ('svm', grid_svm.best_estimator_),
        ('random_forest', grid_rf.best_estimator_)
    ],
    final_estimator=LogisticRegression(random_state=42, max_iter=1000),
    cv=5  # 5-fold cross-validation for training the meta-model
)

 # Train the stacked model
print("Training stacked model (SVM + Random Forest)...")
start_time = time.time()
stacked_model.fit(features_train, target_train)
stacked_time = time.time() - start_time
print(f"✓ Training complete! Time: {stacked_time:.2f} seconds\n")

# Make predictions with the stacked model
target_pred_stacked = stacked_model.predict(features_test)

# Calculate metrics for stacked model
stacked_accuracy = accuracy_score(target_test, target_pred_stacked)
stacked_precision = precision_score(target_test, target_pred_stacked, average='weighted')
stacked_f1 = f1_score(target_test, target_pred_stacked, average='weighted')

# Print stacked model performance
print("="*70)
print("STACKED MODEL PERFORMANCE")
print("="*70)
print(f"Accuracy:  {stacked_accuracy:.4f}")
print(f"Precision: {stacked_precision:.4f}")
print(f"F1-Score:  {stacked_f1:.4f}")
print("="*70)

# Calculate confusion matrix
cm_stacked = confusion_matrix(target_test, target_pred_stacked)

# Create confusion matrix plot
plt.figure(figsize=(12, 10))
sns.heatmap(cm_stacked, annot=True, fmt='d', cmap='Blues', 
            xticklabels=sorted(target_test.unique()),
            yticklabels=sorted(target_test.unique()),
            cbar_kws={'label': 'Number of Predictions'})

plt.xlabel('Predicted Step', fontsize=12, fontweight='bold')
plt.ylabel('Actual Step', fontsize=12, fontweight='bold')
plt.title(f'Confusion Matrix: Stacked Model\nAccuracy: {stacked_accuracy:.4f}', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

##########################################################
# STEP 7:  Model Evaluation 
##########################################################

print("="*70)
print("STEP 7: Model Evaluation ")
print("="*70)

# Package the stacked model 
joblib.dump(stacked_model, 'SVM.joblib')
loaded_model = joblib.load('SVM.joblib')

# Coordinates for evaluation
Eval_Coordinates = [
                [9.375, 3.0625, 1.51],
                [6.995, 5.125, 0.3875],
                [0, 3.0625, 1.93],
                [9.4, 3, 1.8],
                [9.4, 3, 1.3]
]

# Predict maintenance steps for the given coordinates
coordinates_df = pd.DataFrame(Eval_Coordinates, columns=['X', 'Y', 'Z'])
predictions = loaded_model.predict(coordinates_df)
results_df = coordinates_df.copy()
results_df['Predicted Step'] = predictions
print("\nPredictions for the given coordinates:")
print(results_df.to_string(index=False))

# Show all figures at once
plt.show(block=False)
input("\nPress Enter to exit and close all plots...")
plt.close('all')