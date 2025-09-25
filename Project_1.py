# Importing necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Forcing the use of xcb platform for Qt (I'm using Linux and wayland doesn't seem to work for me)
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Importing the data
data = pd.read_csv("Project 1 Data.csv")

# Displaying statistical summary of data frame
print(data.describe())

# Plotting histograms for each column in the data frame
data.hist()
plt.show()