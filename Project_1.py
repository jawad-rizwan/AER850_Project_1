# Import necessary libraries for the code
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, f1_score, 
                             confusion_matrix, classification_report)
import time

# Force the use of xcb platform for Qt (I'm using Linux and wayland doesn't seem to work for me)
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Clear the console/terminal and close all plots
os.system('cls' if os.name == 'nt' else 'clear')
plt.close('all')

##########################################################
# STEP 1: Data Processing
##########################################################

# Import the data
data = pd.read_csv("Project 1 Data.csv")

# Display basic information about the dataframe
print(data.info())

##########################################################
# STEP 2: Data Visualization
##########################################################

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
plt.show(block=False)

##########################################################
# STEP 4: Classification Model Development/Engineering 
##########################################################

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

# ==================== MODEL 2: GRADIENT BOOSTING (GridSearchCV) ====================

# Create pipeline for Gradient Boosting
pipeline_gbm = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier(random_state=42))
])

# Define hyperparameter grid for Gradient Boosting
param_grid_gbm = {
    'model__n_estimators': [50, 100, 200],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [3, 5, 7],
    'model__subsample': [0.8, 1.0]
}

# GridSearchCV for Gradient Boosting with 5-fold cross-validation
grid_gbm = GridSearchCV(
    estimator=pipeline_gbm,
    param_grid=param_grid_gbm,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    refit=True,
    verbose=1,
    return_train_score=True
)

# Train the model
print("\nTraining Gradient Boosting with GridSearchCV...")
start_time = time.time()
grid_gbm.fit(features_train, target_train)
gbm_time = time.time() - start_time

# Confirmation message
print(f"\n✓ Training complete!")
print(f"Best Score: {grid_gbm.best_score_:.4f}")
print(f"Best Parameters: {grid_gbm.best_params_}")
print(f"Training Time: {gbm_time:.2f} seconds\n")

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

# Model 2: Gradient Boosting
print("Evaluating Model 2: Gradient Boosting...")
y_pred_gbm = grid_gbm.predict(features_test)
results['Gradient Boosting'] = {
    'predictions': y_pred_gbm,
    'accuracy': accuracy_score(target_test, y_pred_gbm),
    'precision': precision_score(target_test, y_pred_gbm, average='weighted'),
    'f1_score': f1_score(target_test, y_pred_gbm, average='weighted')
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

from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("STEP 6: STACKED MODEL PERFORMANCE ANALYSIS")
print("="*70)

print("\n[Step 6.1] Creating stacked model by combining two best models...\n")

# Select the two best performing models from Step 5
# Based on typical performance, we'll combine Gradient Boosting and Random Forest
# You can change these based on your Step 5 results

# Create stacking classifier
# The final_estimator makes the final prediction based on the base models' outputs
stacked_model = StackingClassifier(
    estimators=[
        ('gradient_boosting', grid_gbm.best_estimator_),
        ('random_forest', grid_rf.best_estimator_)
    ],
    final_estimator=LogisticRegression(random_state=42, max_iter=1000),
    cv=5  # 5-fold cross-validation for training the meta-model
)

# Train the stacked model
print("Training stacked model (Gradient Boosting + Random Forest)...")
start_time = time.time()
stacked_model.fit(features_train, target_train)
stacked_time = time.time() - start_time
print(f"✓ Training complete! Time: {stacked_time:.2f} seconds\n")

print("[Step 6.2] Evaluating stacked model performance...\n")

# Make predictions with the stacked model
y_pred_stacked = stacked_model.predict(features_test)

# Calculate metrics for stacked model
stacked_accuracy = accuracy_score(target_test, y_pred_stacked)
stacked_precision = precision_score(target_test, y_pred_stacked, average='weighted')
stacked_f1 = f1_score(target_test, y_pred_stacked, average='weighted')

print("="*70)
print("STACKED MODEL PERFORMANCE")
print("="*70)
print(f"Accuracy:  {stacked_accuracy:.4f}")
print(f"Precision: {stacked_precision:.4f}")
print(f"F1-Score:  {stacked_f1:.4f}")
print("="*70)

# Get individual model performances for comparison
gbm_accuracy = accuracy_score(target_test, grid_gbm.predict(features_test))
rf_accuracy = accuracy_score(target_test, grid_rf.predict(features_test))

print("\n[Step 6.3] Comparing stacked model to individual models...\n")

print("="*70)
print("PERFORMANCE COMPARISON")
print("="*70)
print(f"Gradient Boosting (Individual):  {gbm_accuracy:.4f}")
print(f"Random Forest (Individual):      {rf_accuracy:.4f}")
print(f"Stacked Model (Combined):        {stacked_accuracy:.4f}")
print("="*70)

# Calculate improvement
best_individual = max(gbm_accuracy, rf_accuracy)
improvement = stacked_accuracy - best_individual
improvement_pct = (improvement / best_individual) * 100

print(f"\nImprovement over best individual model: {improvement:+.4f} ({improvement_pct:+.2f}%)")

# Create comparison visualization
plt.figure(figsize=(12, 6))

models_comparison = ['Gradient Boosting', 'Random Forest', 'Stacked Model']
accuracies = [gbm_accuracy, rf_accuracy, stacked_accuracy]
colors = ['steelblue', 'steelblue', 'green' if stacked_accuracy > best_individual else 'orange']

bars = plt.bar(models_comparison, accuracies, color=colors, edgecolor='black', alpha=0.7, width=0.6)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylabel('Accuracy', fontsize=12)
plt.title('Stacked Model vs Individual Models Performance', fontsize=14, fontweight='bold')
plt.ylim(min(accuracies) - 0.05, 1.0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('step6_stacked_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Comparison chart saved: step6_stacked_comparison.png")

print("\n[Step 6.4] Creating confusion matrix for stacked model...\n")

# Calculate confusion matrix
cm_stacked = confusion_matrix(target_test, y_pred_stacked)

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
plt.savefig('step6_stacked_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix saved: step6_stacked_confusion_matrix.png")

print("\n[Step 6.5] Detailed metrics comparison...\n")

# Create detailed comparison table
comparison_data = {
    'Model': ['Gradient Boosting', 'Random Forest', 'Stacked Model'],
    'Accuracy': [
        accuracy_score(target_test, grid_gbm.predict(features_test)),
        accuracy_score(target_test, grid_rf.predict(features_test)),
        stacked_accuracy
    ],
    'Precision': [
        precision_score(target_test, grid_gbm.predict(features_test), average='weighted'),
        precision_score(target_test, grid_rf.predict(features_test), average='weighted'),
        stacked_precision
    ],
    'F1-Score': [
        f1_score(target_test, grid_gbm.predict(features_test), average='weighted'),
        f1_score(target_test, grid_rf.predict(features_test), average='weighted'),
        stacked_f1
    ]
}

print("="*70)
print("DETAILED METRICS COMPARISON")
print("="*70)
print(f"{'Model':<25} {'Accuracy':<15} {'Precision':<15} {'F1-Score':<15}")
print("-"*70)
for i in range(len(comparison_data['Model'])):
    print(f"{comparison_data['Model'][i]:<25} "
          f"{comparison_data['Accuracy'][i]:<15.4f} "
          f"{comparison_data['Precision'][i]:<15.4f} "
          f"{comparison_data['F1-Score'][i]:<15.4f}")
print("="*70)

print("\n[Step 6.6] Analysis and interpretation...\n")

print("="*70)
print("STACKING ANALYSIS")
print("="*70)

if improvement > 0.01:  # Significant improvement (>1%)
    print("\n✓ SIGNIFICANT IMPROVEMENT OBSERVED")
    print("\nThe stacked model shows meaningful improvement over individual models.")
    print("\nPossible reasons for improvement:")
    print("1. Complementary Strengths: The two models may excel at different")
    print("   maintenance steps or coordinate regions, and stacking combines their")
    print("   expertise effectively.")
    print("\n2. Error Diversity: When individual models make different types of")
    print("   mistakes, the meta-learner can learn to trust the correct predictions")
    print("   and ignore the errors.")
    print("\n3. Pattern Complexity: The combination captures both the tree-based")
    print("   patterns from Random Forest and the sequential learning from")
    print("   Gradient Boosting, providing a more robust classification.")
    
elif improvement > 0:  # Minimal improvement
    print("\n≈ MINIMAL IMPROVEMENT OBSERVED")
    print("\nThe stacked model shows slight improvement, but the gain is marginal.")
    print("\nPossible reasons for limited effectiveness:")
    print("1. High Individual Performance: Both base models already achieve")
    print("   excellent accuracy, leaving little room for improvement.")
    print("\n2. Similar Predictions: The two models may be making very similar")
    print("   predictions, providing little diversity for the meta-learner to")
    print("   exploit.")
    print("\n3. Simple Problem: The maintenance step classification may be")
    print("   straightforward enough that a single well-tuned model is sufficient.")
    
else:  # No improvement or worse
    print("\n⚠ NO IMPROVEMENT OR SLIGHT DEGRADATION")
    print("\nThe stacked model does not outperform the best individual model.")
    print("\nPossible reasons:")
    print("1. Overfitting in Meta-Learner: The final estimator may be overfitting")
    print("   to the training predictions.")
    print("\n2. Redundant Information: Both base models capture the same patterns,")
    print("   so stacking adds complexity without benefit.")
    print("\n3. Small Dataset: Limited test data may not provide enough evidence")
    print("   for the meta-learner to learn effective combination strategies.")

print("\n" + "="*70)
print("✓ Step 6 Complete! Stacked model analysis finished.")
print("="*70)