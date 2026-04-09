from os import name

import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split
from pathlib import Path
"""
Main implementation of evolutionary and search algorithms for Random Forest hyperparameter tuning.
Includes Random Search, Grid Search, and an Adaptive Genetic Algorithm.
"""


def generate_random_params():
    return [
        random.randint(10, 300),          # params[0]: n_estimators
        random.randint(2, 30),           # params[1]: max_depth
        random.randint(2, 20),           # params[2]: min_samples_split
        random.randint(1, 20),           # params[3]: min_samples_leaf
        random.uniform(0.1, 1.0),        # params[4]: max_features (real)
        random.randint(0, 1),            # params[5]: bootstrap (0=False, 1=True)
        random.randint(0, 1),            # params[6]: criterion (0=gini, 1=entropy)
        random.randint(0, 1),            # params[7]: class_weight (0=None, 1=balanced)
        random.randint(10, 200),         # params[8]: max_leaf_nodes
        random.uniform(0, 0.1)           # params[9]: min_impurity_decrease (real)
    ]

def evaluate_solution(params):
    model = RandomForestClassifier(
    n_estimators=int(params[0]),
    max_depth=int(params[1]),
    min_samples_split=int(params[2]),
    min_samples_leaf=int(params[3]),
    max_features=float(params[4]),
    bootstrap=bool(params[5]),
    criterion="gini" if params[6] == 0 else "entropy",
    class_weight=None if params[7] == 0 else "balanced",
    max_leaf_nodes=int(params[8]),
    min_impurity_decrease=float(params[9]),
    random_state=42     
    )
    # Fixed CV splits guarantee that equal chromosomes always get equal scores,
    # eliminating noise in the fitness function and making the cache 100% reliable.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return scores.mean()


def random_search():
    best_score      = None
    best_parameters = None
    all_scores      = []  # every individual score evaluated (for histogram analysis)

    for _ in range(100):
        parameters = generate_random_params()
        score      = evaluate_solution(parameters)
        all_scores.append(score)
        if best_score is None or score > best_score:
            best_score      = score
            best_parameters = parameters.copy()

    return best_parameters, best_score, all_scores

def grid_search():
    best_score = None
    best_parameters = None
    heatmap = {}  # {(n_estimators, max_depth): best_accuracy}
    
    # Loop over influential hyperparameters for forest size and depth.
    for n_estimators in [25, 75, 200]:
        for max_depth in [5, 15, 25]:
            for min_samples_split in [4, 12]:
                for max_features in [0.3, 0.7]:
                    for criterion in ["gini", "entropy"]:
                        
                        # Fixed parameters with lower impact to keep search space manageable.
                        min_samples_leaf = 1
                        bootstrap = 1
                        class_weight = 0
                        max_leaf_nodes = 200
                        min_impurity_decrease = 0.0
                        
                        parameters = [
                            n_estimators,
                            max_depth,
                            min_samples_split,
                            min_samples_leaf,
                            max_features,
                            bootstrap,
                            int(criterion == "entropy"),
                            class_weight,
                            max_leaf_nodes,
                            min_impurity_decrease
                        ]
                        
                        score = evaluate_solution(parameters)
                        
                        if best_score is None or score > best_score:
                            best_score = score
                            best_parameters = parameters.copy()

                        key = (n_estimators, max_depth)
                        if key not in heatmap or score > heatmap[key]:
                            heatmap[key] = score
                            
    return best_parameters, best_score, heatmap

def is_diverse(candidate, population, min_diff=2):
    """
    Calculates the Hamming distance between a candidate and all individuals in the current population.
    A candidate is considered diverse if it differs from every existing individual in at least 'min_diff' genes.
    
    Args:
        candidate (list): Hyperparameter combination to check.
        population (list): List of already accepted individuals.
        min_diff (int): Minimum number of parameters that must be different.
        
    Returns:
        bool: True if the candidate meets the diversity requirement, False otherwise.
    """
    for existing in population:
        diff_count = sum(1 for i in range(len(candidate)) if candidate[i] != existing[i])
        if diff_count < min_diff:
            return False
    return True

def init_population(pop_size=20):
    """
    Generates the initial population ensuring a minimum separation (diversity) 
    between individuals. This prevents starting with redundant regions of the search space.
    
    Args:
        pop_size (int): The number of individuals in the population.
        
    Returns:
        list: A list of diverse individuals (hyperparameter combinations).
    """
    population = []
    max_attempts = pop_size * 20
    attempts = 0
    
    while len(population) < pop_size and attempts < max_attempts:
        new_ind = generate_random_params()
        
        # Check if the candidate is sufficiently diverse (at least 2 parameters different)
        if is_diverse(new_ind, population, min_diff=2):
            population.append(new_ind)
        attempts += 1
        
    # If the requirement is too strict to find enough individuals, fill the remaining spots.
    while len(population) < pop_size:
        population.append(generate_random_params())
        
    return population

def evaluate_population(population, fitness_cache=None):
    """
    Evaluates all individuals in the current population.
    Uses a cache dictionary to avoid re-evaluating known individuals.
    
    Args:
        population (list): The list of individuals (hyperparameter combinations).
        fitness_cache (dict, optional): Dictionary storing previously calculated fitness scores.
        
    Returns:
        list: A list of fitness scores corresponding to each individual.
    """
    fitness_scores = []
    
    # If no cache is provided, create a temporary empty one
    if fitness_cache is None:
        fitness_cache = {}
        
    for individual in population:
        # Convert the parameter list to a tuple to use it as a dictionary key
        param_key = tuple(individual)
        
        # Check if this identical configuration was already evaluated
        if param_key in fitness_cache:
            score = fitness_cache[param_key]
        else:
            # If it is completely new, evaluate it and store it in the cache
            score = evaluate_solution(individual)
            fitness_cache[param_key] = score
            
        fitness_scores.append(score)
        
    return fitness_scores

def tournament_selection(population, fitness_scores, k=3):
    """
    Selects an individual from the population using tournament selection.
    
    Args:
        population (list): List containing all individuals of the current generation.
        fitness_scores (list): List with the evaluation scores (e.g., accuracy) of each individual.
        k (int): Number of participants in the tournament (3 is a good balance).
        
    Returns:
        list: A copy of the winning individual's hyperparameters.
    """
    # Pick 'k' random participants
    participant_indexes = random.sample(range(len(population)), k)
    
    # Tournament logic
    best_index = participant_indexes[0]
    best_score = fitness_scores[best_index]
    
    for i in range(1, len(participant_indexes)):
        idx = participant_indexes[i]
        if fitness_scores[idx] > best_score:
            best_score = fitness_scores[idx]
            best_index = idx
            
    return population[best_index].copy()


def crossover_two_point(parent1, parent2, crossover_rate=0.8):
    """
    Performs a two-point crossover between two parents.
    
    Args:
        parent1 (list): Hyperparameters of the first parent.
        parent2 (list): Hyperparameters of the second parent.
        crossover_rate (float): Probability of crossover occurring.
        
    Returns:
        tuple: Two child chromosomes (list, list).
    """
    if random.random() < crossover_rate:
        # 1. Get 2 unique cut points and sort them
        cut_points = sorted(random.sample(range(1, len(parent1)), 2))
        pt1 = cut_points[0]
        pt2 = cut_points[1]
        
        # 2. Swap the middle segment to create children
        child1 = parent1[:pt1] + parent2[pt1:pt2] + parent1[pt2:]
        child2 = parent2[:pt1] + parent1[pt1:pt2] + parent2[pt2:]
    else:
        # No crossover: children are clones
        child1 = parent1.copy()
        child2 = parent2.copy()
        
    return child1, child2


def mutate(child, mutation_rate, gene_space):
    """
    With a probability of 'mutation_rate', the individual undergoes mutation.
    When mutated, 1 to 3 random genes are altered to introduce controlled diversity.
    
    Args:
        child (list): The chromosome to be mutated.
        mutation_rate (float): Probability for the individual to mutate (e.g., 0.2 to 0.9).
        gene_space (list): Gene space constraints.
        
    Returns:
        list: The individual (mutated or intact).
    """
    if random.random() < mutation_rate:
        # Determine number of genes to mutate (1 is highly probable, 3 is rare)
        prob = random.random()
        if prob < 0.75:
            num_mutations = 1   # 75% chance to mutate 1 gene
        elif prob < 0.95:
            num_mutations = 2   # 20% chance to mutate 2 genes
        else:
            num_mutations = 3   # 5% chance to mutate 3 genes
            
        # Select unique genes to mutate concurrently
        indices_to_mutate = random.sample(range(len(child)), num_mutations)
        
        for index in indices_to_mutate:
            gene_type = gene_space[index]['type']
            min_val = gene_space[index]['min']
            max_val = gene_space[index]['max']
            if gene_type == 'int':
                child[index] = random.randint(min_val, max_val)
            elif gene_type == 'float':
                child[index] = random.uniform(min_val, max_val)
                
    return child

def genetic_algorithm(pop_size=40, generations=40, elite_size=3):

    """
    Main function to run the Genetic Algorithm for hyperparameter optimization.
    
    Args:
        pop_size (int): Number of individuals in the population.
        generations (int): Number of generations to evolve.
        elite_size (int): Number of top individuals to preserve in each generation.
    Returns:
        tuple: Best hyperparameters found and their corresponding fitness score.
    """

    # 1. Define the gene space for each hyperparameter
    gene_space = [
        {'min': 10, 'max': 300, 'type': 'int'},          # n_estimators
        {'min': 2, 'max': 30, 'type': 'int'},           # max_depth
        {'min': 2, 'max': 20, 'type': 'int'},           # min_samples_split
        {'min': 1, 'max': 20, 'type': 'int'},           # min_samples_leaf
        {'min': 0.1, 'max': 1.0, 'type': 'float'},      # max_features
        {'min': 0, 'max': 1, 'type': 'int'},            # bootstrap (binary)
        {'min': 0, 'max': 1, 'type': 'int'},            # criterion (binary)
        {'min': 0, 'max': 1, 'type': 'int'},            # class_weight (binary)
        {'min': 10, 'max': 200, 'type': 'int'},         # max_leaf_nodes
        {'min': 0.0, 'max': 0.1, 'type': 'float'}      # min_impurity_decrease
    ]
    
    # 2. Initialize the population with random individuals
    population = init_population(pop_size)
    
    best_individual = None
    best_fitness    = None
    history         = []  # best fitness found so far, recorded once per generation
    pc_history      = []  # Pc value used each generation (after adaptive update)
    pm_history      = []  # Pm value used each generation (after adaptive update)
    
    # Adaptive parameters tracking
    Pc = 0.65
    Pm = 0.35
    delta       = 0.05           # Small step for exploitation boosts
    escape_delta = 0.10          # Larger step for stagnation escape
    stagnation_limit = 5         # React faster to stagnation
    stagnation_count = 0
    epsilon = 0.001
    prev_elite_mean = 0.0
    Pc_min = 0.40          # Crossover always has at least 40% weight
    Pm_max = 0.60          # Mutation never exceeds 60%
    Pm_min = 0.15          # Mutation always has at least 15% — prevents premature convergence
    
    # Cache to avoid recalculating fitness for repeated individuals (elitism/clones)
    fitness_cache = {}
    
    for generation in range(generations):
        print(f"Generation {generation+1}/{generations} | Pc: {Pc:.2f} | Pm: {Pm:.2f}")
        
        # Evaluate population using cache
        fitness_scores = evaluate_population(population, fitness_cache)
        
        # Sort population by fitness to identify the elite
        pop_with_fitness = list(zip(population, fitness_scores))
        pop_with_fitness.sort(key=lambda x: x[1], reverse=True)
        
        elite = [ind.copy() for ind, fit in pop_with_fitness[:elite_size]]
        elite_fitness = [fit for ind, fit in pop_with_fitness[:elite_size]]
        
        # Calculate mean fitness of the elite to evaluate stagnation
        current_elite_mean = sum(elite_fitness) / len(elite_fitness)
        
        if generation > 0:
            if current_elite_mean <= prev_elite_mean + epsilon:
                stagnation_count += 1
            else:
                # Improvement detected: favor exploitation by increasing Pc
                stagnation_count = 0
                Pc = min(0.9, Pc + delta)
                Pm = max(Pm_min, round(1.0 - Pc, 2))  # clamp to Pm_min
                
        if stagnation_count >= stagnation_limit:
            # Stagnation detected: aggressive exploration boost — respect hard bounds
            Pm = min(Pm_max, Pm + escape_delta)
            Pc = max(Pc_min, round(1.0 - Pm, 2))
            stagnation_count = 0
            
        prev_elite_mean = current_elite_mean
        
        # Update the best solution found so far (global)
        best_of_gen = elite[0]
        fitness_of_gen = elite_fitness[0]
        
        if best_fitness is None or fitness_of_gen > best_fitness:
            best_fitness = fitness_of_gen
            best_individual = best_of_gen.copy()
        
        # Record convergence history (best cumulative fitness per generation)
        history.append(best_fitness)
        # Record adaptive probability values actually used this generation
        pc_history.append(Pc)
        pm_history.append(Pm)
        
        # The new population starts with the elite to guarantee their survival (Elitism)
        new_population = elite.copy()
        
        while len(new_population) < pop_size:
            # 4. Select parents using tournament selection
            parent1 = tournament_selection(population, fitness_scores)
            
            # --- Incest Prevention ---
            # We force parent2 to be different from parent1 to ensure crossover 
            # always generates new genetic combinations.
            attempts = 0
            parent2 = tournament_selection(population, fitness_scores)
            while parent1 == parent2 and attempts < 10:
                parent2 = tournament_selection(population, fitness_scores)
                attempts += 1
                
            # 5. Perform crossover to produce offspring
            child1, child2 = crossover_two_point(parent1, parent2, Pc)
            
            # 6. Mutate the offspring            
            child1 = mutate(child1, Pm, gene_space)
            child2 = mutate(child2, Pm, gene_space)
            
            # 7. Add the new children only if they are UNIQUE (Diversity Enforcement)
            # This ensures that every individual in the population is an explorer.
            for child in [child1, child2]:
                if len(new_population) < pop_size:
                    if child not in new_population:
                        new_population.append(child)
                    else:
                        # If it's a clone, we discard it and the loop continues,
                        # effectively forcing the discovery of a new individual.
                        pass
                
        # 8. Replace the old population with the new one
        population = new_population      
        
    n_evaluations = len(fitness_cache)  # unique model trainings (cache hits excluded)
    return best_individual, best_fitness, history, pc_history, pm_history, n_evaluations

if __name__ == "__main__":

    # Obtain the current directory of the script to build the path to the dataset
    current_dir = Path(__file__).resolve().parent

    # Construct the path to the dataset located in the "data" folder, ensuring compatibility across different operating systems
    csv_path = current_dir.parent / "data" / "winequality-red.csv"
    # Load the dataset using pandas, specifying the correct separator for the CSV file
    data = pd.read_csv(csv_path,sep=";")
    # Convert the target variable "quality" into a binary classification problem, where wines with a quality score of 6 or higher are labeled as 1 (good quality) and those below 6 are labeled as 0 (bad quality).
    data["quality"] = (data["quality"] >= 6).astype(int)
    X = data.drop("quality", axis=1)
    y = data["quality"]

    # --- Full Comparative Benchmark ---
    best_random_score = 0.0
    best_grid_score = 0.0
    
    if False:
        print("\n[1/3] Running Random Search...")
        best_random_parameters, best_random_score, _ = random_search()
        print("Best parameters from Random Search:", best_random_parameters, "with accuracy:", best_random_score)

        print("\n[2/3] Running Grid Search...")
        best_grid_parameters, best_grid_score, _ = grid_search()
        print("Best parameters from Grid Search:", best_grid_parameters, "with accuracy:", best_grid_score)

    print("\n[3/3] Running Adaptive Genetic Algorithm...")
    best_ga_parameters, best_ga_score, _, _, _, _ = genetic_algorithm()
    print("Best parameters from Genetic Algorithm:", best_ga_parameters, "with accuracy:", best_ga_score)

    print("\n===== FINAL COMPARISON =====")
    print(f"  Random Search : {best_random_score:.6f}")
    print(f"  Grid Search   : {best_grid_score:.6f}")
    print(f"  Genetic Alg.  : {best_ga_score:.6f}")
    winner = max(
        [("Random Search", best_random_score),
         ("Grid Search",   best_grid_score),
         ("Genetic Alg.",  best_ga_score)],
        key=lambda x: x[1]
    )
    print(f"  WINNER        : {winner[0]} ({winner[1]:.6f})")