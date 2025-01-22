import pandas as pd

data = pd.read_csv('wine-dataset.txt', header=None, sep='\t')

print(data.head())
print(data.shape)

y_wine = data.iloc[:, 0].values
X_wine = data.iloc[:, 1:].values

print("Shape of X_wine (features):", X_wine.shape)
print("Shape of y_wine (labels):", y_wine.shape)