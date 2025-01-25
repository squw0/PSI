import numpy as np
import matplotlib.pyplot as plt

#zapisywac jakie sie wykonalo zmiany
#zad1 - zmniejszenie liczby regul o 1
#zad2 -
#zad3 - nowa kategoria odchylenia "bardzo duze odchylenie"


def trimf(x, params):
    a, b, c = params
    
    return np.maximum(0, np.minimum((x - a) / np.maximum(1e-10, b - a), (c - x) / np.maximum(1e-10, c - b)))

def trapezoidal(x, params):
    a, b, c, d = params
    return np.maximum(0, np.minimum((x - a) / np.maximum(1e-10, b - a), 
                            np.minimum(1, (d - x) / np.maximum(1e-10, d - c))))

def main():
    x = np.linspace(-np.pi, np.pi, 200)

    small_deviation = np.maximum(trimf(x, [-np.pi, -np.pi, -np.pi/6]), trimf(x, [0, np.pi/6, np.pi]))
    #medium_deviation = trimf(x, [-np.pi/6, 0, np.pi/6])
    large_deviation = np.maximum(trimf(x, [-np.pi/3, -np.pi/6, 0]), trimf(x, [np.pi/6, np.pi/3, np.pi/2]))
    very_large_deviation = np.maximum(trimf(x, [-np.pi/6, -np.pi/12, 0]), trimf(x, [np.pi/3, np.pi/2, np.pi]))
    approximated_sin = np.zeros_like(x)

    large_deviationT = np.maximum(trapezoidal(x, [-np.pi/2, -np.pi/2, -np.pi/3, -np.pi/6]), trapezoidal(x, [0, np.pi/3, np.pi/2, np.pi]))
    small_deviationT = np.maximum(trapezoidal(x, [-np.pi/3, -np.pi/3, -np.pi/6, -np.pi/12]), trapezoidal(x, [0, np.pi/6, np.pi/3, np.pi/2]))

    for i in range(len(x)):
        # activate rules:
        activation_small = np.minimum(small_deviation[i], np.sin(x[i]))
        #activation_medium = np.minimum(medium_deviation[i], np.sin(x[i]))
        activation_large = np.minimum(large_deviation[i], np.sin(x[i]))
        #activation_very_large = np.minimum(very_large_deviation[i], np.sin(x[i]))
        
        # agregation
        aggregated = np.max([activation_small, activation_large])
        #aggregated = np.max([activation_small, activation_large,activation_very_large])
        
        # diffusion
        if aggregated != 0:
            approximated_sin[i] = np.sum(x * aggregated) / np.sum(aggregated)
        else:
            approximated_sin[i] = 0

    plt.subplot(3, 1, 1)
    plt.plot(x, np.sin(x), 'b', x, approximated_sin, 'r', linewidth=2)
    plt.title('Aproksymacja funkcji sinus')
    plt.legend(['sin(x)', 'Aproksymacja'], loc='upper right')

    # # Wykres zbiorow rozmytych dla odchylenia
    # plt.subplot(2, 1, 2)
    # #plt.plot(x, small_deviation, 'b', x, medium_deviation, 'g', x, large_deviation, 'r', linewidth=2)
    # plt.plot(x, small_deviation, 'b', x, large_deviation, 'r', linewidth=2)
    # plt.title('Zbiory rozmyte dla odchylenia')
    # #plt.legend(['Małe odchylenie', 'Średnie odchylenie', 'Duże odchylenie'], loc='upper right')
    # plt.legend(['Małe odchylenie', 'Duże odchylenie'], loc='upper right')

    plt.subplot(3, 1, 2)
    plt.plot(x, small_deviation, 'b', x, large_deviation, 'r', linewidth=2)
    plt.title('Zbiory rozmyte dla odchylenia')
    plt.legend(['Małe odchylenie', 'Duże odchylenie'], loc='upper right')

    plt.subplot(3,1,3)
    plt.plot(x, small_deviationT, 'b', x, large_deviationT, 'r', linewidth=2)
    plt.title('Zbiory rozmyte dla odchylenia(f. trapezoidalna)')
    plt.legend(['male Odchylenie', 'duze odchylenie'], loc='upper right')

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()