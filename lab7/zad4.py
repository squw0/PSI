import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(1)
data = np.genfromtxt("input.txt", delimiter=",")

learningRate = 0.01
data = np.delete(data, [3], axis=1)  
Y = data[:, -1]
X_train = np.delete(data, [3], axis=1)
oneVector = np.ones((X_train.shape[0], 1))
X_train = np.concatenate((oneVector, X_train), axis=1)

sizes = [100, 1000, 10000]  
execution_times = []

for size in sizes:
    if size > X_train.shape[0]:
        size = X_train.shape[0]

    indices = np.random.choice(range(X_train.shape[0]), size, replace=False)
    X_sample = X_train[indices]
    Y_sample = Y[indices]

    weights = np.random.rand(4, 1)
    misClassifications = 1
    minMisclassifications = 10000
    noChangeCounter = 0
    maxNoChangeIterations = 500

    iteration = 0
    plotData = []

    start_time = time.time()
    while (misClassifications != 0 and (iteration < 7000)):
        iteration += 1
        misClassifications = 0
        for i in range(0, len(X_sample)):
            currentX = X_sample[i].reshape(-1, X_sample.shape[1])
            currentY = Y_sample[i]
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

    end_time = time.time()
    execution_times.append(end_time - start_time)

# Wykres czasu działania
plt.plot(sizes, execution_times, marker='o')
plt.xlabel("Rozmiar danych")
plt.ylabel("Czas działania (s)")
plt.title("Czas działania algorytmu kieszonkowego")
plt.show()

for size, exec_time in zip(sizes, execution_times):
    print(f"Rozmiar danych: {size}, Czas: {exec_time:.4f} s")
