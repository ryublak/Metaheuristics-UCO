import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from pathlib import Path

# 1. Obtener la ruta de la carpeta donde está este script (src)
current_dir = Path(__file__).resolve().parent

# 2. Construir la ruta al CSV: subir un nivel y entrar en data
# Esto genera: .../Practice 2/data/winequality-red.csv
csv_path = current_dir.parent / "data" / "winequality-red.csv"
# cargar dataset
data = pd.read_csv(csv_path,sep=";")
# convertir problema a clasificación binaria
data["quality"] = (data["quality"] >= 6).astype(int)
X = data.drop("quality", axis=1)
y = data["quality"]

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


'''def RandomSearch():
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

best_random_parameters, best_random_score = RandomSearch()

print("Best parameters:", best_random_parameters, "with accuracy:", best_random_score)
'''
def gridSearch():
    best_score=None
    best_parameters=None
    for n_estimators in [10, 100, 300]:
        for max_depth in [2, 10, 30]:
            for min_samples_split in [2, 10, 20]:
                for min_samples_leaf in [1, 4, 16]:
                    for max_features in [0.1, 0.5, 1.0]:
                        for bootstrap in [False, True]:
                            for criterion in ["gini", "entropy"]:
                                for class_weight in [None, "balanced"]:
                                    for max_leaf_nodes in [10, 100, 200]:
                                        for min_impurity_decrease in [0.0, 0.01, 0.05, 0.1]:
                                            parameters = [
                                                n_estimators,
                                                max_depth,
                                                min_samples_split,
                                                min_samples_leaf,
                                                max_features,
                                                int(bootstrap),
                                                int(criterion == "entropy"),
                                                int(class_weight == "balanced"),
                                                max_leaf_nodes,
                                                min_impurity_decrease
                                            ]
                                            score = evaluate_solution(parameters)
                                            if best_score is None or score > best_score:
                                                best_score = score
                                                best_parameters = parameters.copy()
    return best_parameters, best_score
best_grid_parameters, best_grid_score = gridSearch()
print("Best parameters:", best_grid_parameters, "with accuracy:", best_grid_score)