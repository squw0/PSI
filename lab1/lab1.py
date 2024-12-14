import numpy as np
import matplotlib.pyplot as plt

# ZAD1
def zad1():
    my_list = list(range(201))
    modified_list = [(x + 6) for x in my_list]
    reversed_list = modified_list[::-1]
    return reversed_list

# ZAD2
def zad2():
    random_array = np.random.choice([0, 1], size=100)
    bool_array = random_array.astype(bool)
    return bool_array

# ZAD3
def zad3():
    my_list = list(range(100, 201))
    first_half, second_half = my_list[:25], my_list[25:50]
    avg_first_half = sum(first_half) / len(first_half)
    avg_second_half = sum(second_half) / len(second_half)
    return first_half, avg_first_half, second_half, avg_second_half

# ZAD4
def zad4():
    data = np.random.normal(0, 1, 1000)
    plt.figure()
    plt.hist(data, bins=30, density=True, alpha=0.5, color='blue')
    plt.title("Density Plot of Normal Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.savefig("density_plot.png")
    plt.close()

# ZAD5
def zad5():
    x = np.linspace(-10, 10, 200)
    y = x ** 3
    plt.figure()
    plt.plot(x, y, label="f(x) = x^3", color="green")
    plt.title("Cubic Function")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid()
    plt.legend()
    plt.savefig("cubic_function.png")
    plt.close()

# ZAD6
def zad6(list1, list2):
    merged_list = [val for pair in zip(list1, list2) for val in pair]
    return merged_list


def main():

    zad1_result = zad1()
    
    zad2_result = zad2()
    
    zad3_first, zad3_avg1, zad3_second, zad3_avg2 = zad3()
    
    zad4()
    
    zad5()
    
    zad6_result = zad6([1, 3, 5], [2, 4, 6])

    return {
        "zad1": zad1_result,
        "zad2": zad2_result,
        "zad3": (zad3_first, zad3_avg1, zad3_second, zad3_avg2),
        "zad6": zad6_result
    }

if __name__ == "__main__":
    results = main()
