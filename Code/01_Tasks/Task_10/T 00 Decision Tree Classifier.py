"""
Task 0 : Decision Tree Classifier
Goal:
    1) Train a Decision Tree Classifier on a chosen dataset.
    2) Study how the "max_depth" hyperparameter affects performance.
    3) Plot F1-score (Y axis) vs max_depth (X axis) for max_depth = 3 to 25
    4) Produce TWO curves on the same figure: one for the TRAIN set and one
       for the TEST set, so we can visually detect overfitting/underfitting

Dataset chosen:
    Breast Cancer Wisconsin (Diagnostic) dataset, shipped built-in with
    scikit-learn (sklearn.datasets.load_breast_cancer)
    - It's a binary classification problem (malignant vs benign tumor)
    - 569 samples, 30 numeric features (computed from digitized images of a
      breast mass, e.g. radius, texture, perimeter, area, smoothness...)
    - It's a well-known, clean dataset that is perfect for a first
      classification exercise (no missing values, all numeric, balanced-ish).

Why F1-score (and not just accuracy)?
    - F1-score = harmonic mean of Precision and Recall.
    - It gives a more honest picture than plain accuracy, especially when the
      two classes are not perfectly balanced (here it's roughly 63% / 37%)
"""

import numpy as np   
import matplotlib.pyplot as plt                      
from sklearn.datasets import load_breast_cancer      
from sklearn.model_selection import train_test_split   
from sklearn.tree import DecisionTreeClassifier      
from sklearn.metrics import f1_score               
from sklearn.preprocessing import StandardScaler   


data = load_breast_cancer()
X = data.data
y = data.target

print(f"Dataset shape       : {X.shape}")
print(f"Number of classes   : {len(np.unique(y))}")
print(f"Class distribution  : {np.bincount(y)}  (0=malignant, 1=benign)")


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)        

depth_range = range(3, 26)  
train_f1_scores = []   
test_f1_scores = []    

for depth in depth_range:

    tree_model = DecisionTreeClassifier(max_depth=depth, random_state=42)

    tree_model.fit(X_train_scaled, y_train)

    train_predictions = tree_model.predict(X_train_scaled)
    test_predictions = tree_model.predict(X_test_scaled)

    
    train_f1 = f1_score(y_train, train_predictions)
    test_f1 = f1_score(y_test, test_predictions)

    train_f1_scores.append(train_f1)
    test_f1_scores.append(test_f1)

    print(f"max_depth={depth:>2} | train F1 = {train_f1:.4f} | test F1 = {test_f1:.4f}")


# plot

# Two curves on the SAME chart:
#    Train F1 curve  usually keeps climbing towards 1.0 (overfitting)
#  Test F1 curve    usually plateaus or drops after some depth
#     (that "gap" between the two curves IS the overfitting signal)
plt.figure(figsize=(10, 6))

plt.plot(
    list(depth_range), train_f1_scores,
    marker='o', linewidth=2, label='Train F1-score', color='#1f77b4'
)
plt.plot(
    list(depth_range), test_f1_scores,
    marker='s', linewidth=2, label='Test F1-score', color='#d62728'
)

plt.xlabel('max_depth')
plt.ylabel('F1-score')
plt.title('Decision Tree: F1-score vs max_depth (Train vs Test)')
plt.xticks(list(depth_range))          
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

output_path = 'f1_vs_max_depth.png'
plt.savefig(output_path, dpi=150)
print(f"\nPlot saved to: {output_path}")

# plt.show()  

# summary

best_test_depth = list(depth_range)[int(np.argmax(test_f1_scores))]
best_test_f1 = max(test_f1_scores)

print("\nSumary")
print(f"Best TEST F1 score = {best_test_f1:.4f} achieved at max_depth = {best_test_depth}")
print("Tip: once train F1 keeps rising while test F1 flattens/declines  the tree is starting to overfit past that depth.")
