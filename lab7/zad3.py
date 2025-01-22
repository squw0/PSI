import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
data = np.genfromtxt("input.txt", delimiter=",")

# Najlepsza dokładność: 53.10% dla learningRate = 0.005
# Najlepsza dokładność: 53.20% dla learningRate = 0.001
# Najlepsza dokładność: 53.90% dla learningRate = 0.1

learningRate = 0.1  
use_zero_weights = False  

for i in range(data.shape[1] - 1):
    data[:, i] = (data[:, i] - np.mean(data[:, i])) / np.std(data[:, i])

data = np.delete(data, [3], axis=1)
new_feature_sum = data[:, 0] + data[:, 1]
new_feature_product = data[:, 0] * data[:, 1]
new_features = np.vstack((new_feature_sum, new_feature_product)).T


data = np.hstack((data, new_features))

Y = data[:, -3]  
X_train = np.delete(data, -3, axis=1)
oneVector = np.ones((X_train.shape[0], 1))
X_train = np.concatenate((oneVector, X_train), axis=1)

if use_zero_weights:
    weights = np.zeros((X_train.shape[1], 1))
else:
    weights = np.random.rand(X_train.shape[1], 1)

plotData = []
misClassifications = 1
minMisclassifications = 10000
noChangeCounter = 0
maxNoChangeIterations = 500

iteration = 0
while (misClassifications != 0 and (iteration < 10000)):
    iteration += 1
    misClassifications = 0
    for i in range(0, len(X_train)):
        currentX = X_train[i].reshape(-1, X_train.shape[1])
        currentY = Y[i]
        wTx = np.dot(currentX, weights)[0][0]
        if currentY == 1 and wTx < 0:
            misClassifications += 1
            weights = weights + learningRate * np.transpose(currentX)
        elif currentY == -1 and wTx > 0:
            misClassifications += 1
            weights = weights - learningRate * np.transpose(currentX)
    plotData.append(misClassifications)
    if misClassifications < minMisclassifications:
        minMisclassifications = misClassifications
        noChangeCounter = 0 
    else:
        noChangeCounter += 1

    if noChangeCounter >= maxNoChangeIterations:
        print("Liczba błędnych klasyfikacji nie zmienia się przez 500 iteracji. Przerywam działanie.")
        break

    print(f"Iteracja {iteration}, Błędne klasyfikacje {misClassifications}")

print(f"Dla learningRate = {learningRate}:")
print(f"Błędnych klasyfikacji: {minMisclassifications}")
print(f"Wagi: {weights.T}")
print(f"Najlepsza dokładność: {((X_train.shape[0] - minMisclassifications) / X_train.shape[0]) * 100:.2f}%")

plt.plot(np.arange(0, len(plotData)), plotData)
plt.xlabel("Liczba iteracji")
plt.ylabel("Liczba błędnych klasyfikacji")
plt.title("Konwergencja algorytmu kieszonkowego z nowymi cechami")
plt.show()
