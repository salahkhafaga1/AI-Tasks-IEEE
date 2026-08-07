"""
Task 1: Stacking Ensemble (Logistic Regression + kNN + Decision Tree)
Dataset: Titanic (Kaggle-style dataset, loaded via seaborn)
Goal: Predict 'survived' using a StackingClassifier that combines
      Logistic Regression, kNN, and Decision Tree as base learners,
      with Logistic Regression as the meta-learner.
"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset

df = sns.load_dataset("titanic")

# Keep relevant columns
cols = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
df = df[cols].copy()

target = "survived"
X = df.drop(columns=[target])
y = df[target]

numeric_features = ["age", "sibsp", "parch", "fare", "pclass"]
categorical_features = ["sex", "embarked"]

# Preprocessing (impute + scale + one-hot encode)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", pd.get_dummies)  # placeholder, replaced below
])

# Use sklearn's OneHotEncoder instead of pd.get_dummies inside pipeline
from sklearn.preprocessing import OneHotEncoder
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Base learners for stacking

base_learners = [
    ("logreg", LogisticRegression(max_iter=1000)),
    ("knn", KNeighborsClassifier(n_neighbors=7)),
    ("dtree", DecisionTreeClassifier(max_depth=5, random_state=42))
]

meta_learner = LogisticRegression(max_iter=1000)

stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    passthrough=False
)

# Full pipeline: preprocessing + stacking model

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("stacking", stacking_model)
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


# Evaluation

print("=" * 50)
print("STACKING MODEL RESULTS (Titanic dataset)")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Compare against each individual base model

print("\n" + "=" * 50)
print("COMPARISON WITH INDIVIDUAL BASE MODELS")
print("=" * 50)
for name, clf in base_learners:
    solo_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", clf)
    ])
    solo_pipeline.fit(X_train, y_train)
    solo_pred = solo_pipeline.predict(X_test)
    acc = accuracy_score(y_test, solo_pred)
    print(f"{name:10s} accuracy: {acc:.4f}")
