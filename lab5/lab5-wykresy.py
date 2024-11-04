import numpy as np
import matplotlib.pyplot as plt

# Zakres wartości x
x = np.linspace(-10, 10, 400)

# Funkcja trapezoidalna
def trapezoidal(x, a, b, c, d):
    return np.maximum(0, np.minimum((x - a) / np.maximum(1e-10, b - a), 
                                    np.minimum(1, (d - x) / np.maximum(1e-10, d - c))))

# Funkcja trójkątna
def triangular(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / np.maximum(1e-10, b - a), 
                                    (c - x) / np.maximum(1e-10, c - b)))

# Funkcja sinusoidalna
def sinusoidal(x, a, b):
    return 1 / (1 + np.exp(-(x - a) / b))

def main():
    # Parametry dla funkcji
    params_trap = (-5, -2, 2, 5)  # (a, b, c, d) dla trapezoidalnej
    params_tri = (-5, 0, 5)       # (a, b, c) dla trójkątnej
    params_sin = (0, 2)           # (a, b) dla sinusoidalnej
    
    # Generowanie danych
    trap_y = trapezoidal(x, *params_trap)
    tri_y = triangular(x, *params_tri)
    sin_y = sinusoidal(x, *params_sin)
    
    # Wykres funkcji trapezoidalnej
    plt.figure(figsize=(8, 4))
    plt.plot(x, trap_y, label='Funkcja trapezoidalna', color='blue')
    plt.title('Funkcja trapezoidalna')
    plt.xlabel('x')
    plt.ylabel('Przynależność')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid()
    plt.legend()
    plt.savefig('funkcja_trapezoidalna.png')
    plt.show()  # Wyświetlenie figury
    
    # Wykres funkcji trójkątnej
    plt.figure(figsize=(8, 4))
    plt.plot(x, tri_y, label='Funkcja trójkątna', color='green')
    plt.title('Funkcja trójkątna')
    plt.xlabel('x')
    plt.ylabel('Przynależność')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid()
    plt.legend()
    plt.savefig('funkcja_trojkatna.png')
    plt.show()  # Wyświetlenie figury
    
    # Wykres funkcji sinusoidalnej
    plt.figure(figsize=(8, 4))
    plt.plot(x, sin_y, label='Funkcja sinusoidalna', color='red')
    plt.title('Funkcja sinusoidalna')
    plt.xlabel('x')
    plt.ylabel('Przynależność')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid()
    plt.legend()
    plt.savefig('funkcja_sinusoidalna.png')
    plt.show()  # Wyświetlenie figury

if __name__ == "__main__":
    main()