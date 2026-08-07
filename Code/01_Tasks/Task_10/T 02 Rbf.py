"""
===============================================================================
Task 2 : Support Vector Machine (SVM) with an RBF kernel
===============================================================================
Goal:
    Train an SVM classifier that uses the RBF (Radial Basis Function) kernel
    instead of a plain straight-line (linear) boundary, and evaluate it.

What is the RBF kernel, intuitively?
    - RBF = Gaussian kernel. It measures "similarity" between two points as
      a function of the distance between them: K(x, x') = exp(-gamma * ||x - x'||^2)
    - Instead of drawing ONE straight hyperplane like the linear kernel, RBF
      implicitly maps the data into an infinite-dimensional space where
      complex, curved (non-linear) decision boundaries become possible.
    - This makes RBF much more flexible than the linear kernel, at the cost
      of being harder to interpret and more prone to overfitting if not
      tuned properly (see 'gamma' below).

Key hyperparameters for RBF:
    - C     : same regularization idea as in the linear SVM (Task 1).
    - gamma : controls how far the influence of a single training example
              reaches.
                * small gamma -> far reach -> smoother/simpler boundary
                * large gamma -> short reach -> boundary hugs the training
                  points tightly -> higher risk of overfitting
===============================================================================
"""


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)



data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


svm_rbf_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

svm_rbf_model.fit(X_train_scaled, y_train)

y_train_pred = svm_rbf_model.predict(X_train_scaled)
y_test_pred = svm_rbf_model.predict(X_test_scaled)

train_f1 = f1_score(y_train, y_train_pred)
train_acc = accuracy_score(y_train, y_train_pred)

test_f1 = f1_score(y_test, y_test_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)

print("SVM (RBF kernel)")
print(f"Train is Accuracy: {train_acc:.4f}  |  F1-score: {train_f1:.4f}")
print(f"Test is Accuracy: {test_acc:.4f}  |  F1-score: {test_f1:.4f}")
print(f"Test is Precision: {test_precision:.4f}  |  Recall: {test_recall:.4f}")

print("\nConfusion matrix (test set):")
print(confusion_matrix(y_test, y_test_pred))

print("\nFull classification report (test set):")
print(classification_report(y_test, y_test_pred, target_names=data.target_names))


print(f"\nSupport vectors per class: {svm_rbf_model.n_support_} "
      f"(classes = {list(data.target_names)})")

# comparison

# Note: compare these numbers with 'Task 1 - SVM linear' results")
# If RBF's test F1 is close to (or lower than) linear's, it usually means the data is close to linearly separable already so the extra
# flexibility of RBF isn't buying us much (and could even overfit)
