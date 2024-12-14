import random
import matplotlib.pyplot as plt

def fitness_function(chromosome):
    x = int(chromosome, 2)  
    return x ** 2 

def generate_population(population_size, chromosome_length):
    return [''.join(random.choice('01') for _ in range(chromosome_length)) for _ in range(population_size)]

def roulette_wheel_selection(population):
    total_fitness = sum(fitness_function(chrom) for chrom in population)
    weights = [fitness_function(chrom) / total_fitness for chrom in population]
    return random.choices(population, weights=weights, k=len(population))

def crossover(chromosome1, chromosome2, crossover_prob):
    if random.random() < crossover_prob:
        point = random.randint(1, len(chromosome1) - 1)
        offspring1 = chromosome1[:point] + chromosome2[point:]
        offspring2 = chromosome2[:point] + chromosome1[point:]
        return offspring1, offspring2
    return chromosome1, chromosome2

def mutation(chromosome, mutation_prob):
    chromosome = list(chromosome)
    for i in range(len(chromosome)):
        if random.random() < mutation_prob:
            chromosome[i] = '1' if chromosome[i] == '0' else '0'
    return ''.join(chromosome)

def genetic_algorithm(population_size, chromosome_length, generations, mutation_prob, crossover_prob):
    population = generate_population(population_size, chromosome_length)
    best_fitness_over_time = []

    for generation in range(generations):
        population = roulette_wheel_selection(population)
        new_population = []

        for i in range(0, len(population), 2):
            parent1, parent2 = population[i], population[i + 1]
            offspring1, offspring2 = crossover(parent1, parent2, crossover_prob)
            new_population.append(mutation(offspring1, mutation_prob))
            new_population.append(mutation(offspring2, mutation_prob))

        population = new_population
        best_chromosome = max(population, key=fitness_function)
        best_fitness_over_time.append(fitness_function(best_chromosome))

    best_solution = max(population, key=fitness_function)
    return int(best_solution, 2), fitness_function(best_solution), best_fitness_over_time

# Eksperyment 1
def experiment_varying_parameters():
    population_sizes = [5, 10, 20]
    mutation_probs = [0.01, 0.05, 0.1]
    crossover_probs = [0.6, 0.8, 1.0]

    results = {}
    for pop_size in population_sizes:
        for mut_prob in mutation_probs:
            for cross_prob in crossover_probs:
                best_x, best_fitness, _ = genetic_algorithm(
                    pop_size, 5, 30, mut_prob, cross_prob
                )
                key = f"Pop={pop_size}, Mut={mut_prob}, Cross={cross_prob}"
                results[key] = (best_x, best_fitness)

    return results

# Eksperyment 2
def experiment_mutation_effect():
    mutation_probs = [0.001, 0.01, 0.05, 0.1, 0.2]
    fitness_over_time = {}

    for mut_prob in mutation_probs:
        _, _, best_fitness = genetic_algorithm(10, 5, 30, mut_prob, 0.8)
        fitness_over_time[mut_prob] = best_fitness

    return fitness_over_time

# Eksperyment 3
def experiment_small_population():
    _, best_fitness, fitness_over_time = genetic_algorithm(2, 5, 30, 0.05, 0.8)
    return best_fitness, fitness_over_time

# Eksperyment 4
def experiment_generations():
    generations = [10, 20, 50, 100]
    fitness_by_generations = {}

    for gen in generations:
        _, best_fitness, _ = genetic_algorithm(10, 5, gen, 0.05, 0.8)
        fitness_by_generations[gen] = best_fitness

    return fitness_by_generations

# Eksperyment 5
def experiment_linear_function():
    def fitness_linear(chromosome):
        x = int(chromosome, 2)
        return 2 * x + 3

    def genetic_algorithm_linear(population_size, chromosome_length, generations, mutation_prob, crossover_prob):
        population = generate_population(population_size, chromosome_length)
        best_solution_over_time = []

        for _ in range(generations):
            population = roulette_wheel_selection(population)
            new_population = []

            for i in range(0, len(population), 2):
                parent1, parent2 = population[i], population[i + 1]
                offspring1, offspring2 = crossover(parent1, parent2, crossover_prob)
                new_population.append(mutation(offspring1, mutation_prob))
                new_population.append(mutation(offspring2, mutation_prob))

            population = new_population
            best_solution = max(population, key=fitness_linear)
            best_solution_over_time.append(fitness_linear(best_solution))

        best_solution = max(population, key=fitness_linear)
        return int(best_solution, 2), fitness_linear(best_solution), best_solution_over_time

    return genetic_algorithm_linear(10, 5, 30, 0.05, 0.8)

if __name__ == "__main__":
    print("Experiment 1:", experiment_varying_parameters())
    print("Experiment 2:", experiment_mutation_effect())
    print("Experiment 3:", experiment_small_population())
    print("Experiment 4:", experiment_generations())
    print("Experiment 5:", experiment_linear_function())
