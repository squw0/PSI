import numpy as np
import matplotlib.pyplot as plt

def trapmf(x, params):
    a, b, c, d = params
    return np.maximum(0, np.minimum((x - a) / (b - a), (d - x) / (d - c)))

def gaussmf(x, params):
    mean, sigma = params
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

def main():
    x = np.linspace(-np.pi, np.pi, 200)

    # 2. Użycie funkcji trapezoidalnej
    small_deviation = trapmf(x, [-np.pi, -np.pi/2, -np.pi/4, 0])
    medium_deviation = trapmf(x, [-np.pi/4, -np.pi/8, 0, np.pi/8])
    large_deviation = trapmf(x, [-np.pi/2, -np.pi/4, 0, np.pi/4])
    very_large_deviation = trapmf(x, [-np.pi, -np.pi/2, 0, np.pi])

    # 2. Aktywacja
    sin_x = np.sin(x)
    activation_small = np.minimum(small_deviation, sin_x)
    activation_medium = np.minimum(medium_deviation, sin_x)
    activation_large = np.minimum(large_deviation, sin_x)
    activation_very_large = np.minimum(very_large_deviation, sin_x)

    # Agregacja
    aggregated = np.maximum.reduce([activation_small, activation_medium, activation_large, activation_very_large])
    
    # Dyfuzja
    approximated_sin = np.where(aggregated != 0, 
                                 np.sum(x * aggregated) / np.sum(aggregated), 
                                 0)

    # Wykres 1: Aproksymacja funkcji sinus
    plt.subplot(2, 1, 1)
    plt.plot(x, np.sin(x), 'b', x, approximated_sin, 'r', linewidth=2)
    plt.title('Aproksymacja funkcji sinus (Trapezoidalne)')
    plt.legend(['sin(x)', 'Aproksymacja'], loc='upper right')

    # Wykres 2: Zbiory rozmyte dla odchylenia
    plt.subplot(2, 1, 2)
    plt.plot(x, small_deviation, 'b', x, medium_deviation, 'g', 
             x, large_deviation, 'r', x, very_large_deviation, 'm', linewidth=2)
    plt.title('Zbiory rozmyte dla odchylenia (Trapezoidalne)')
    plt.legend(['Małe odchylenie', 'Średnie odchylenie', 'Duże odchylenie', 'Bardzo duże odchylenie'], loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
