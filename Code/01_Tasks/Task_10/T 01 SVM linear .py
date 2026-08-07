"""
===============================================================================
Task 1 : Support Vector Machine (SVM) with a LINEAR kernel
===============================================================================
Goal:
    Train an SVM classifier that uses a *linear* decision boundary
    (kernel='linear') on a chosen dataset, and evaluate it.

Dataset chosen:
    Same dataset as Task 0 (Breast Cancer Wisconsin) so the results are
    directly comparable across all 4 models (Decision Tree, SVM linear,
    SVM RBF, Random Forest).

Why feature scaling matters A LOT for SVM (unlike Decision Trees):
    SVM tries to find the maximum-margin hyperplane using distances between
    points. If one feature ranges 0-1000 and another ranges 0-1, the large
    feature will completely dominate the distance calculation and the model
    will basically ignore the small-scale feature. StandardScaler fixes this
    by making every feature have mean=0 and std=1.
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

print(f"Dataset shape       : {X.shape}")
print(f"Class distribution  : benign={sum(y==1)}, malignant={sum(y==0)}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y   )

# feuture scalling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# train the model

svm_linear_model = SVC(kernel='linear', C=1.0, random_state=42)

# Train the model on the SCALED training data.
svm_linear_model.fit(X_train_scaled, y_train)


y_train_pred = svm_linear_model.predict(X_train_scaled)
y_test_pred = svm_linear_model.predict(X_test_scaled)

train_f1 = f1_score(y_train, y_train_pred)
train_acc = accuracy_score(y_train, y_train_pred)

test_f1 = f1_score(y_test, y_test_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)

print("\nSVM (linear kernel) Results")
print(f"Train is Accuracy: {train_acc:.4f}  |  F1-score: {train_f1:.4f}")
print(f"Test is Accuracy: {test_acc:.4f}  |  F1-score: {test_f1:.4f}")
print(f"Test is Precision: {test_precision:.4f}  |  Recall: {test_recall:.4f}")

print("\nConfusion matrix (test set):")
print(confusion_matrix(y_test, y_test_pred))

print("\nFull classification report (test set):")
print(classification_report(y_test, y_test_pred, target_names=data.target_names))

print("\nTop 5 most influential features (by |coefficient|):")
coefficients = svm_linear_model.coef_[0]
top_indices = abs(coefficients).argsort()[::-1][:5]
for idx in top_indices:
    print(f"  {data.feature_names[idx]:<25} weight = {coefficients[idx]:+.4f}")
