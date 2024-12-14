import random
import numpy as np
from matplotlib import pyplot as plt


def fitness_function(x):
    return x ** 2

def binary_to_decimal(binary):
    return int("".join(map(str, binary)), 2)

def generate_population(size, chrom_length):
    return [np.random.randint(0, 2, chrom_length).tolist() for _ in range(size)]

def evaluate_population(population):
    return [fitness_function(binary_to_decimal(ind)) for ind in population]

def select_parents(population, fitness):
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]
    return random.choices(population, weights=probabilities, k=2)

def single_point_crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def two_point_crossover(parent1, parent2):
    point1, point2 = sorted(random.sample(range(len(parent1)), 2))
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return child1, child2

def mutate(individual, mutation_rate):
    return [bit ^ 1 if random.random() < mutation_rate else bit for bit in individual]

# ZAD1
def zad1():
    chrom_length = 5
    population_size = 10
    generations = 20
    mutation_rate = 0.1

    population = generate_population(population_size, chrom_length)

    for _ in range(generations):
        fitness = evaluate_population(population)
        new_population = []

        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness)
            child1, child2 = single_point_crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population

    best_individual = max(population, key=lambda ind: fitness_function(binary_to_decimal(ind)))
    return binary_to_decimal(best_individual), fitness_function(binary_to_decimal(best_individual))

# ZAD2
def zad2():
    items = [(2, 3), (3, 4), (4, 5), (5, 6)]  # (weight, value)
    max_weight = 8
    population_size = 10
    generations = 20
    mutation_rate = 0.1

    def knapsack_fitness(chromosome):
        total_weight = sum(items[i][0] for i in range(len(chromosome)) if chromosome[i] == 1)
        total_value = sum(items[i][1] for i in range(len(chromosome)) if chromosome[i] == 1)
        return total_value if total_weight <= max_weight else 0

    population = generate_population(population_size, len(items))

    for _ in range(generations):
        fitness = [knapsack_fitness(ind) for ind in population]
        new_population = []

        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness)
            child1, child2 = single_point_crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population

    best_individual = max(population, key=knapsack_fitness)
    return best_individual, knapsack_fitness(best_individual)

# ZAD 3
def zad3():
    chrom_length = 5
    population_size = 10
    generations = 20
    mutation_rate = 0.1

    def run_experiment(crossover_function):
        population = generate_population(population_size, chrom_length)
        for _ in range(generations):
            fitness = evaluate_population(population)
            new_population = []

            for _ in range(population_size // 2):
                parent1, parent2 = select_parents(population, fitness)
                child1, child2 = crossover_function(parent1, parent2)
                child1 = mutate(child1, mutation_rate)
                child2 = mutate(child2, mutation_rate)
                new_population.extend([child1, child2])

            population = new_population

        best_individual = max(population, key=lambda ind: fitness_function(binary_to_decimal(ind)))
        return binary_to_decimal(best_individual), fitness_function(binary_to_decimal(best_individual))

    result_single = run_experiment(single_point_crossover)
    result_two = run_experiment(two_point_crossover)

    return result_single, result_two

# ZAD4
def zad4():
    def equation_fitness(x):
        return -abs(x ** 3 - 4 * x ** 2 + 6 * x - 24)

    chrom_length = 8
    population_size = 10
    generations = 20
    mutation_rate = 0.1

    population = generate_population(population_size, chrom_length)

    for _ in range(generations):
        fitness = [equation_fitness(binary_to_decimal(ind)) for ind in population]
        new_population = []

        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness)
            child1, child2 = single_point_crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population

    best_individual = max(population, key=lambda ind: equation_fitness(binary_to_decimal(ind)))
    return binary_to_decimal(best_individual), equation_fitness(binary_to_decimal(best_individual))

if __name__ == "__main__":
    print("Zadanie 1:", zad1())
    print("Zadanie 2:", zad2())
    print("Zadanie 3:", zad3())
    print("Zadanie 4:", zad4())
