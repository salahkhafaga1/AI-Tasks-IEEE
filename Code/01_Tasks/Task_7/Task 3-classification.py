"""
Task 3 - classification.py
Contains the LogisticRegression class implemented from scratch with NumPy only.
No ML libraries (sklearn, etc.) are used.
"""
import numpy as np


def sigmoid(z):
    # Sigmoid activation it clipped to avoid overflow in exp().
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


class LogisticRegression:

    def __init__(self, iterations=10000, lr=0.1):
      
        # iterations : number of gradient descent steps
        # lr         : learning rate
        
        self.iterations = iterations
        self.lr = lr
        self.w = None  # weights includes bias as w[0]

    def _add_bias(self, x):
        # Prepend a column of ones to x for the bias term
        ones = np.ones((x.shape[0], 1))
        return np.hstack((ones, x))

    def fit(self, x_train, y_train):
        
        # x_train : (n_samples, n_features)
        # y_train : (n_samples, 1)
        
        x_train = np.array(x_train, dtype=float)
        y_train = np.array(y_train, dtype=float).reshape(-1, 1)

        x_b = self._add_bias(x_train)
        n_samples, n_features = x_b.shape

        # initialize weights (bias + one weight per feature) at zero
        self.w = np.zeros((n_features, 1))

        for _ in range(self.iterations):
            z = x_b @ self.w
            h = sigmoid(z)
            gradient = (1 / n_samples) * (x_b.T @ (h - y_train))
            self.w -= self.lr * gradient

        return self

    def predict_proba(self, x_test):
        # Returns the raw sigmoid probabilities for x_test
        x_test = np.array(x_test, dtype=float)
        x_b = self._add_bias(x_test)
        return sigmoid(x_b @ self.w)

    def predict(self, x_test, threshold=0.5):
        # Returns hard 0/1 predictions for x_test.
        probs = self.predict_proba(x_test)
        return (probs >= threshold).astype(int)

    def evaluate(self, x_test, y_test):
        """Returns accuracy of the model on x_test / y_test."""
        y_test = np.array(y_test, dtype=float).reshape(-1, 1)
        preds = self.predict(x_test)
        accuracy = np.mean(preds == y_test)
        return accuracy
