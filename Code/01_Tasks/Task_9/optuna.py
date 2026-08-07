"""
Task 2: Use Optuna to find the best max_depth for a Decision Tree
Dataset: Breast Cancer Wisconsin (classic Kaggle/UCI dataset, built into sklearn)
Goal: Search over max_depth values and find the one that maximizes
      cross-validation accuracy using Optuna.
"""

import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Load dataset

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define the Optuna objective function

def objective(trial):
    max_depth = trial.suggest_int("max_depth", 1, 30)

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

    # 5-fold cross-validation on the training set
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    return scores.mean()

# Run the Optuna study

optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep output clean

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("=" * 50)
print("OPTUNA RESULTS - Decision Tree max_depth tuning")
print("=" * 50)
print(f"Best max_depth: {study.best_params['max_depth']}")
print(f"Best CV accuracy: {study.best_value:.4f}")

# Train final model with best max_depth and evaluate on test set

best_depth = study.best_params["max_depth"]
final_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print(f"\nTest set accuracy with best max_depth ({best_depth}): "
      f"{accuracy_score(y_test, y_pred):.4f}")


# Show top trials for reference

print("\nTop 5 trials:")
trials_df = study.trials_dataframe().sort_values("value", ascending=False).head(5)
print(trials_df[["number", "params_max_depth", "value"]].to_string(index=False))
