import numpy as np
import matplotlib.pyplot as plt

def trimf(x, params):
    a, b, c = params
    return np.maximum(0, np.minimum((x - a) / np.maximum(1e-10, b - a), (c - x) / np.maximum(1e-10, c - b)))

def main():
    x = np.linspace(-np.pi, np.pi, 200)

    # zad2. Zmiana w liczbie reguł i dodanie "bardzo dużego odchylenia"
    small_deviation = np.maximum(
        trimf(x, [-np.pi, -np.pi, -np.pi/4]), 
        np.maximum(trimf(x, [-np.pi/4, 0, np.pi/4]), trimf(x, [0, np.pi/4, np.pi]))
    )
    
    medium_deviation = np.maximum(
        trimf(x, [-np.pi/4, -np.pi/8, 0]), 
        trimf(x, [0, np.pi/8, np.pi/4])
    )
    
    large_deviation = np.maximum(
        trimf(x, [-np.pi/2, -np.pi/4, 0]), 
        np.maximum(trimf(x, [np.pi/4, np.pi/2, np.pi]), 
                   trimf(x, [-np.pi, -np.pi/2, -np.pi/4]))
    )

    # zad3. Dodanie bardzo dużego odchylenia
    very_large_deviation = np.maximum(
        trimf(x, [-np.pi, -np.pi/2, -np.pi/4]), 
        trimf(x, [np.pi/2, np.pi, np.pi])
    )

    # zad2. Optymalizacja obliczeń poprzez usunięcie pętli for
    sin_x = np.sin(x)
    activation_small = np.minimum(small_deviation, sin_x)
    activation_medium = np.minimum(medium_deviation, sin_x)
    activation_large = np.minimum(large_deviation, sin_x)
    activation_very_large = np.minimum(very_large_deviation, sin_x)

    # Agregacja
    aggregated = np.maximum.reduce([activation_small, activation_medium, activation_large, activation_very_large])
    
    # Dyfuzja
    approximated_sin = np.where(aggregated != 0, np.sum(x * aggregated) / np.sum(aggregated), 0)

    # Wykres 1: Aproksymacja funkcji sinus
    plt.subplot(2, 1, 1)
    plt.plot(x, np.sin(x), 'b', x, approximated_sin, 'r', linewidth=2)
    plt.title('Aproksymacja funkcji sinus')
    plt.legend(['sin(x)', 'Aproksymacja'], loc='upper right')

    # Wykres 2: Zbiory rozmyte dla odchylenia
    plt.subplot(2, 1, 2)
    plt.plot(x, small_deviation, 'b', x, medium_deviation, 'g', 
             x, large_deviation, 'r', x, very_large_deviation, 'm', linewidth=2)
    plt.title('Zbiory rozmyte dla odchylenia')
    plt.legend(['Małe odchylenie', 'Średnie odchylenie', 'Duże odchylenie', 'Bardzo duże odchylenie'], loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
