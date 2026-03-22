from os import name

import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from pathlib import Path


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


if name == "main":
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
    
    best_random_parameters, best_random_score = RandomSearch()
    best_grid_parameters, best_grid_score = gridSearch()
    print("Best parameters from Random Search:", best_random_parameters, "with accuracy:", best_random_score)
    print("Best parameters from Grid Search:", best_grid_parameters, "with accuracy:", best_grid_score)