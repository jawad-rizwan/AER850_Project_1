# Import necessary libraries for the code
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Show all figures at once
plt.show()



