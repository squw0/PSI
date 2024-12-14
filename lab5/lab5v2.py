import numpy as np
import matplotlib.pyplot as plt

def trimf(x, params):
    a, b, c = params
    return np.maximum(0, np.minimum((x - a) / np.maximum(1e-10, b - a), (c - x) / np.maximum(1e-10, c - b)))

def trapezoidal(x, params):
    a, b, c, d = params
    return np.maximum(0, np.minimum((x - a) / np.maximum(1e-10, b - a), 
                                     np.minimum(1, (d - x) / np.maximum(1e-10, d - c))))

def gaussmf(x, mean, sigma):
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

def approximated_sin(x, membership_functions):
    aggregated_values = np.zeros_like(x)
    total_activation = np.zeros_like(x)

    for mf, weight in membership_functions:
        activation = mf(x)
        aggregated_values += activation * weight(x)
        total_activation += activation

    return aggregated_values / np.maximum(total_activation, 1e-10)

# Zadanie 1
def zadanie1(x):
    mf_small = lambda x: trimf(x, [-np.pi, -np.pi, -np.pi / 6])
    mf_medium = lambda x: trimf(x, [-np.pi / 6, 0, np.pi / 6])
    mf_large = lambda x: trimf(x, [np.pi / 6, np.pi / 3, np.pi])

    membership_functions = [
        (mf_small, lambda x: np.sin(x)),
        (mf_medium, lambda x: np.sin(x)),
        (mf_large, lambda x: np.sin(x)),
    ]

    approx = approximated_sin(x, membership_functions)

    plt.figure()
    plt.plot(x, np.sin(x), label='True Sin')
    plt.plot(x, approx, label='Approximated Sin')
    plt.title("Zadanie 1: Zmiana liczby reguł")
    plt.legend()
    plt.show()

# Zadanie 2
def zadanie2(x):
    mf_small = trimf(x, [-np.pi, -np.pi, -np.pi / 6])
    mf_medium = trimf(x, [-np.pi / 6, 0, np.pi / 6])
    mf_large = trimf(x, [np.pi / 6, np.pi / 3, np.pi])

    aggregated = (mf_small * np.sin(x)) + (mf_medium * np.sin(x)) + (mf_large * np.sin(x))
    total_activation = mf_small + mf_medium + mf_large

    aggregated /= np.maximum(total_activation, 1e-10)

    plt.figure()
    plt.plot(x, np.sin(x), label='True Sin')
    plt.plot(x, aggregated, label='Approximated Sin (Optimized)')
    plt.title("Zadanie 2: Optymalizacja")
    plt.legend()
    plt.show()

# Zadanie 3
def zadanie3(x):
    mf_small = lambda x: trimf(x, [-np.pi, -np.pi, -np.pi / 6])
    mf_medium = lambda x: trimf(x, [-np.pi / 6, 0, np.pi / 6])
    mf_large = lambda x: trimf(x, [np.pi / 6, np.pi / 3, np.pi])
    mf_very_large = lambda x: trimf(x, [np.pi / 2, np.pi, np.pi])

    membership_functions = [
        (mf_small, lambda x: np.sin(x)),
        (mf_medium, lambda x: np.sin(x)),
        (mf_large, lambda x: np.sin(x)),
        (mf_very_large, lambda x: np.sin(x)),
    ]

    approx = approximated_sin(x, membership_functions)

    plt.figure()
    plt.plot(x, np.sin(x), label='True Sin')
    plt.plot(x, approx, label='Approximated Sin (New Category)')
    plt.title("Zadanie 3: Nowa kategoria")
    plt.legend()
    plt.show()

x = np.linspace(-np.pi, np.pi, 200)

zadanie1(x)
zadanie2(x)
zadanie3(x)
