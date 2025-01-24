import random
import time

def fitness_function(chromosome):
    x = int(chromosome, 2)  
    return x ** 2  # x^2

def generate_population(population_size, chromosome_length):
    population = []
    for _ in range(population_size):
        chromosome = ''.join(random.choice('01') for _ in range(chromosome_length))
        population.append(chromosome)
    return population

def roulette_wheel_selection(population):
    total_fitness = sum(fitness_function(chrom) for chrom in population)
    weights = [fitness_function(chrom) / total_fitness for chrom in population]
    selected = random.choices(population, weights=weights, k=len(population))
    return selected

def crossover(chromosome1, chromosome2):
    point = random.randint(1, len(chromosome1) - 1)
    offspring1 = chromosome1[:point] + chromosome2[point:]
    offspring2 = chromosome2[:point] + chromosome1[point:] 
    return offspring1, offspring2

def mutation(chromosome, mutation_prob):
    chromosome = list(chromosome)
    for i in range(len(chromosome)):
        if random.random() < mutation_prob:
            chromosome[i] = '1' if chromosome[i] == '0' else '0'
    return ''.join(chromosome)

def genetic_algorithm(population_size, chromosome_length, generations, mutation_prob):
    population = generate_population(population_size, chromosome_length)
    start_time = time.time()

    for generation in range(generations):
        population = roulette_wheel_selection(population)
        new_population = []

        for i in range(0, len(population) - 1, 2):
            parent1, parent2 = population[i], population[i+1]
            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.extend([mutation(offspring1, mutation_prob), mutation(offspring2, mutation_prob)])

        if len(population) % 2 == 1:
            new_population.append(population[-1])

        population = new_population
        best_chromosome = max(population, key=fitness_function)
        print(f"Generation {generation + 1}: Best = {best_chromosome}, f(x) = {fitness_function(best_chromosome)}")

    best_chromosome = max(population, key=fitness_function)
    end_time = time.time()
    print(f"Best solution: x = {int(best_chromosome, 2)}, f(x) = {fitness_function(best_chromosome)}")
    print(f"Execution Time: {end_time - start_time:.2f} seconds")

population_sizes = [2, 3]
mutation_prob = 0.01
chromosome_length = 5
generations = 10

for population_size in population_sizes:
    print(f"\npopulacja = {population_size}")
    genetic_algorithm(population_size, chromosome_length, generations, mutation_prob)
