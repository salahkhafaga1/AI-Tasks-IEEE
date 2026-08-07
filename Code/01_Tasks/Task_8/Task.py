#IEEE SCSS
# KNN (Task 1) + Logistic Regression learning rate (Task 2) + iterations (Task 3)

#link of the dataset https://www.kaggle.com/datasets/uciml/mushroom-classification


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

RANDOM_STATE = 42


#loading + preprocessing
def load_and_prep(path="mushrooms.csv"):
    df = pd.read_csv(path)

    # checked earlier only column with missing values is stalk-root (marked as '?' so im gonna just drop it filling it doesnt really make sense for this feature
    df = df.drop(columns=["stalk-root"], errors="ignore")

    # every column here is categorical evven the target so label encode everything not using one hot on purpose the dataset would blow up to 100 columns and KNN distance gets weird with that many sparse dims
  # label encoding is not "ideal"for nominal features but it's simpler and works fine here, keeping it real
    encoders = {}
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=["class"])
    y = df["class"]

    return X, y


X, y = load_and_prep()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# scaling AFTER split fit only on train so we don't leak test info into the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# task 1 KNN       K from 2 to 150
train_f1_knn = []
test_f1_knn = []
k_values = range(2, 151)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    train_pred = knn.predict(X_train_scaled)
    test_pred = knn.predict(X_test_scaled)

    train_f1_knn.append(f1_score(y_train, train_pred))
    test_f1_knn.append(f1_score(y_test, test_pred))

best_k = list(k_values)[int(np.argmax(test_f1_knn))]
print(f"[KNN] best K based on test f1 = {best_k}, f1 = {max(test_f1_knn):.4f}")

plt.figure(figsize=(10, 6))
plt.plot(k_values, train_f1_knn, label="Train F1")
plt.plot(k_values, test_f1_knn, label="Test F1")
plt.axvline(best_k, color="gray", linestyle="--", alpha=0.6, label=f"best K = {best_k}")
plt.xlabel("K")
plt.ylabel("F1-score")
plt.title("KNN: F1-score vs K")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("task1_knn_f1_vs_k.png", dpi=150)
plt.close()


# Tasj 2: learning rate analysis

lr_values = np.arange(0.001, 1.001, 0.001)

train_f1_lr = []
test_f1_lr = []

# 100k SGDClassifier fits one by one would take forever sampling to keep it while still covering the full range asked in the task
lr_sample = lr_values[::5]  # every 5th value    still 200 points across the range

for eta in lr_sample:
    clf = SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=eta,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train_scaled, y_train)

    train_f1_lr.append(f1_score(y_train, clf.predict(X_train_scaled)))
    test_f1_lr.append(f1_score(y_test, clf.predict(X_test_scaled)))

best_lr = lr_sample[int(np.argmax(test_f1_lr))]
print(f"[LR - learning rate] best eta0 = {best_lr:.3f}, f1 = {max(test_f1_lr):.4f}")

plt.figure(figsize=(10, 6))
plt.plot(lr_sample, train_f1_lr, label="Train F1")
plt.plot(lr_sample, test_f1_lr, label="Test F1")
plt.xlabel("Learning Rate (eta0)")
plt.ylabel("F1-score")
plt.title("Logistic Regression (SGD): F1-score vs Learning Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("task2_lr_f1_vs_learning_rate.png", dpi=150)
plt.close()


#Tasl 3 number of iterations analysis


#fixing eta0 at the best value found in task 2 only varying max_iter now
iteration_values = list(range(1000, 100001, 5000))  # step of 5000      100k one by one is overkill

train_f1_iter = []
test_f1_iter = []

for n_iter in iteration_values:
    clf = SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=best_lr,
        max_iter=n_iter,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train_scaled, y_train)

    train_f1_iter.append(f1_score(y_train, clf.predict(X_train_scaled)))
    test_f1_iter.append(f1_score(y_test, clf.predict(X_test_scaled)))

best_iter = iteration_values[int(np.argmax(test_f1_iter))]
print(f"[LR - iterations] best max_iter = {best_iter}, f1 = {max(test_f1_iter):.4f}")

plt.figure(figsize=(10, 6))
plt.plot(iteration_values, train_f1_iter, label="Train F1")
plt.plot(iteration_values, test_f1_iter, label="Test F1")
plt.xlabel("Number of Iterations")
plt.ylabel("F1-score")
plt.title("Logistic Regression (SGD): F1-score vs Number of Iterations")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("task3_lr_f1_vs_iterations.png", dpi=150)
plt.close()

print("done, 3 png files saved")
