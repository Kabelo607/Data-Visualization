from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt

  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
  
# metadata 
print(iris.metadata) 
  
# variable information 
print(iris.variables) 

# Fetch the dataset
iris = fetch_ucirepo(id=53)

# Extract features and targets
X = iris.data.features
y = iris.data.targets

# Display the first 5 rows of the features
print(X.head())

# Check data types and missing values
print(X.info())
print(X.isnull().sum())

# Handle missing values - example: filling with mean
X_clean = X.fillna(X.mean())

# Basic descriptive statistics
stats_summary = X.describe()
print("Basic statistics using .describe():")
print(stats_summary)

# Compute medians separately
median_values = X.median()
print("\nMedian values:")
print(median_values)



# Fetch Iris dataset
iris = fetch_ucirepo(id=53)

# Extract features DataFrame and target labels
X = iris.data.features
y = iris.data.targets

# Line plot of SepalLength over sample index
plt.figure(figsize=(8, 5))
plt.plot(X.index, X['SepalLength'], marker='o', linestyle='-')
plt.title('Sepal Length values across samples')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.grid(True)
plt.show()

