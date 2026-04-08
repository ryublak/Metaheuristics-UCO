from os import name

import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from pathlib import Path
#TODO:
#- Implement the Genetic Algorithm for hyperparameter optimization of the Random Forest Classifier.
#- Define the gene space for each hyperparameter, ensuring that the values are within reasonable ranges based on the Random Forest's characteristics.


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
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    return scores.mean()


def RandomSearch():
    best_score=None
    best_parameters=None
    for i in range(0,100):
        score = None
        parameters=generate_random_params()
        if best_score is None:
            best_score = evaluate_solution(parameters)
            best_parameters = parameters.copy()

        elif (score:=evaluate_solution(parameters)) > best_score:
            best_score = score
            best_parameters = parameters.copy()
    return best_parameters, best_score

def gridSearch():
    best_score = None
    best_parameters = None
    
    # 1. Iteration loop over the most influential hyperparameters.
    # Exploring different combinations of forest size, depth, and splitting criteria.
    for n_estimators in [50, 100, 300]:          # Number of trees in the forest
        for max_depth in [10, 20, 30]:           # Maximum depth of each tree
            for min_samples_split in [2, 10]:    # Minimum number of samples required to split an internal node
                for max_features in [0.1, 0.5, 1.0]: # Fraction of features to consider when looking for the best split
                    for criterion in ["gini", "entropy"]: # Function to measure the quality of a split
                        
                        # 2. Fixed hyperparameters (lower relative impact or redundant).
                        # Kept constant to avoid a combinatorial explosion in execution time.
                        min_samples_leaf = 1         # Allows leaves to be of any size
                        bootstrap = 1                # Maintain sampling with replacement (core to Random Forest)
                        class_weight = 0             # No additional class weights
                        max_leaf_nodes = 200         # Broad limit, actual control is managed by 'max_depth'
                        min_impurity_decrease = 0.0  # No strict purity gain threshold for splitting
                        
                        # 3. Chromosome/individual construction according to the assignment's defined order
                        parameters = [
                            n_estimators,
                            max_depth,
                            min_samples_split,
                            min_samples_leaf,
                            max_features,
                            bootstrap,
                            int(criterion == "entropy"), # Binary conversion (0=gini, 1=entropy)
                            class_weight,
                            max_leaf_nodes,
                            min_impurity_decrease
                        ]
                        
                        # 4. Evaluation using 5-fold Cross-Validation
                        score = evaluate_solution(parameters)
                        
                        # 5. Update the best model found
                        if best_score is None or score > best_score:
                            best_score = score
                            best_parameters = parameters.copy()
                            
    return best_parameters, best_score

def init_population(pop_size=20):
    """
    Generates the initial population of individuals for the Genetic Algorithm.
    Each individual represents a random set of hyperparameters.
    
    Args:
        pop_size (int): The number of individuals in the population.
        
    Returns:
        list: A list of dictionaries, where each dictionary is an individual.
    """
    population = []
    
    for _ in range(pop_size):
        # We assume generate_random_params() is already defined and 
        # returns a valid dictionary of random hyperparameters.
        individual = generate_random_params()
        population.append(individual)
        
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
    # 1. Choose 'k' random participants from our population
    indexes=random.sample(range(len(population)),k)
    # 2. Find which of these participants has the best score
    best_index=indexes[0]
    best_score=fitness_scores[best_index]
    
    for index in range (1,(len(indexes))):
        if (fitness_scores[index]>best_score):
            best_score=fitness_scores[index]
            best_index=index
    # 3. Return the winner (copy to avoid accidental modifications)
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

def genetic_algorithm(pop_size=20, generations=50, elite_size=2):

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
    best_fitness = None
    
    # Adaptive parameters tracking
    Pc = 0.8
    Pm = 0.2
    delta = 0.05
    stagnation_limit = 5
    stagnation_count = 0
    epsilon = 0.001
    prev_elite_mean = 0.0
    
    # Cache to avoid recalculating fitness for repeated individuals (elitism/clones)
    fitness_cache = {}
    
    for generation in range(generations):
        print(f"Generation {generation+1}/{generations} | Pc: {Pc:.2f} | Pm: {Pm:.2f}")
        
        # 3. Evaluate the fitness of the current population using the cache
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
                Pm = round(1.0 - Pc, 2)
                
        if stagnation_count >= stagnation_limit:
            # Stagnation detected: favor exploration by increasing Pm
            Pm = min(0.9, Pm + delta)
            Pc = round(1.0 - Pm, 2)
            stagnation_count = 0
            
        prev_elite_mean = current_elite_mean
        
        # Update the best solution found so far (global)
        best_of_gen = elite[0]
        fitness_of_gen = elite_fitness[0]
        
        if best_fitness is None or fitness_of_gen > best_fitness:
            best_fitness = fitness_of_gen
            best_individual = best_of_gen.copy()
        
        # The new population starts with the elite to guarantee their survival (Elitism)
        new_population = elite.copy()
        
        while len(new_population) < pop_size:
            # 4. Select parents using tournament selection
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            # 5. Perform crossover to produce offspring
            child1, child2 = crossover_two_point(parent1, parent2, Pc)
            # 6. Mutate the offspring            
            child1 = mutate(child1, Pm, gene_space)
            child2 = mutate(child2, Pm, gene_space)
            # 7. Add the new children to the next generation
            if len(new_population) < pop_size:
                new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
                
        # 8. Replace the old population with the new one
        population = new_population      
        
    return best_individual, best_fitness  

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

    if(True):
        best_random_parameters, best_random_score = RandomSearch()
        best_grid_parameters, best_grid_score = gridSearch()
        print("Best parameters from Random Search:", best_random_parameters, "with accuracy:", best_random_score)
        print("Best parameters from Grid Search:", best_grid_parameters, "with accuracy:", best_grid_score)
    if(False):
        best_ga_parameters, best_ga_score = genetic_algorithm()
        print("Best parameters from Genetic Algorithm:", best_ga_parameters, "with accuracy:", best_ga_score)