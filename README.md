# AER 850 Project 1: Maintenance Step Classification from 3D Coordinates

**Author:** Jawad Rizwan  
**Student Number:** 501124033  
**Due Date:** October 6th, 2025

## Overview
This project predicts maintenance steps based on 3D coordinates (X, Y, Z) using supervised machine learning. The workflow includes data analysis, visualization, model training, hyperparameter tuning, and model evaluation.

## Features
- Loads and processes a CSV dataset of 3D coordinates and maintenance steps.
- Visualizes data with 3D scatter plots, histograms, and correlation heatmaps.
- Splits data into training and testing sets with stratification.
- Trains and tunes multiple models using scikit-learn pipelines:
	- Logistic Regression (with GridSearchCV)
	- Support Vector Machine (SVM, with GridSearchCV)
	- Random Forest (with GridSearchCV and RandomizedSearchCV)
- Compares model performance using accuracy, precision, and F1-score.
- Implements a stacked ensemble (SVM + Random Forest) for advanced performance.
- Presents results and predictions in a clear, tabular format.

## Libraries Used
- pandas: Data loading and manipulation
- matplotlib, seaborn: Data visualization
- scikit-learn: Model building, pipelines, hyperparameter tuning, and evaluation

## How to Run
1. Place `Project 1 Data.csv` in the project directory.
2. Run `Project_1.py` with Python 3.8+ and the required libraries installed.
3. Follow the prompts and review the printed and plotted results.

## File Structure
- `Project_1.py`: Main script for data analysis, modeling, and evaluation
- `Project 1 Data.csv`: Input data file (not included)

## Notes
- All model hyperparameters are tuned using cross-validation.

Results are printed and visualized for easy interpretation.




