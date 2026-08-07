"""
===============================================================================
Task 3 : Random Forest + Optuna Hyperparameter Optimization
===============================================================================
Goal:
    Train a Random Forest classifier and use Optuna to automatically search
    for the best combination of hyperparameters (instead of guessing them
    by hand or doing a slow brute-force GridSearch).

What is Random Forest, quickly?
    - An "ensemble" of many Decision Trees (Task 0), each trained on a
      random bootstrap sample of the data and a random subset of features.
    - Every tree "votes" for a class, and the forest predicts the majority
      vote. Averaging many trees reduces the overfitting that a single deep
      tree suffers from.

What is Optuna, and why use it instead of manual tuning?
    - Optuna is a hyperparameter optimization framework. You define an
      "objective function" that trains+evaluates a model for a given set of
      hyperparameters and returns a score. Optuna then intelligently
      searches the hyperparameter space (using Bayesian-style sampling, by
      default the Tree-structured Parzen Estimator / TPE sampler) to find
      the combination that maximizes that score — far more efficient than
      randomly guessing or exhaustively trying every combination.
===============================================================================
"""
from xml.parsers.expat import model

import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        bootstrap=bootstrap,
        random_state=42,
        n_jobs=-1,         
    )

    scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring='f1', n_jobs=-1
    )

    return scores.mean() 


study = optuna.create_study(direction='maximize', study_name='rf_f1_optimization')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\nOptuna Search Finished")
print(f"Best CV F1-score (train, 5-fold): {study.best_value:.4f}")
print("Best hyperparameters found:")
for param_name, param_value in study.best_params.items():
    print(f"  {param_name:<20}: {param_value}")


best_model = RandomForestClassifier(
    **study.best_params,
    random_state=42,
    n_jobs=-1
)
best_model.fit(X_train, y_train)

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("\nFinal Tuned Random Forest")
print(f"Train F1-score : {train_f1:.4f}")
print(f"Test  F1-score : {test_f1:.4f}")
print(f"Test  Accuracy : {test_acc:.4f}")

print("\nConfusion matrix (test set):")
print(confusion_matrix(y_test, y_test_pred))

print("\nFull classification report (test set):")
print(classification_report(y_test, y_test_pred, target_names=data.target_names))



print("\nTop 5 most important features:")
importances = best_model.feature_importances_
top_indices = importances.argsort()[::-1][:5]
for idx in top_indices:
    print(f"  {data.feature_names[idx]:<25} importance = {importances[idx]:.4f}")

import matplotlib.pyplot as plt

trial_numbers = [t.number for t in study.trials]
trial_values = [t.value for t in study.trials]

running_best = []
current_best = float('-inf')
for v in trial_values:
    current_best = max(current_best, v)
    running_best.append(current_best)

plt.figure(figsize=(10, 6))
plt.scatter(trial_numbers, trial_values, alpha=0.5, label='Trial F1-score (5-fold CV)')
plt.plot(trial_numbers, running_best, color='#d62728', linewidth=2, label='Best F1-score so far')
plt.xlabel('Trial number')
plt.ylabel('F1-score (cross-validated on train set)')
plt.title('Optuna Hyperparameter Search — Random Forest')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('optuna_optimization_history.png', dpi=150)
print("\nPlot saved: optuna_optimization_history.png")
