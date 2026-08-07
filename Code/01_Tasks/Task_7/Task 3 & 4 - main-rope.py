"""
Task 3 & 4 - main.py
Imports the LogisticRegression class from classification.py and trains y tests it on the 4D XOR truth table dataset.
"""

import numpy as np
from classification import LogisticRegression

# 4D XOR truth table (a, b, c, d -> Out)

x_data = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
])
# collecting data from that image was so rekhem

y_data = np.array([
    [0], [1], [1], [0],
    [1], [0], [0], [1],
    [1], [0], [0], [1],
    [0], [1], [1], [0],
])

# train and test on the whole set (it's the full truth table, only 16 rows)
model = LogisticRegression(iterations=10000, lr=0.5)
model.fit(x_data, y_data)

predictions = model.predict(x_data)
accuracy = model.evaluate(x_data, y_data)

print("Predictions:")
print(predictions.flatten())
print("\nGround truth:")
print(y_data.flatten())
print(f"\nAccuracy: {accuracy * 100:.2f}%")

if accuracy < 0.8:
    print(
        "\nNote: XOR is not linearly separable, so a plain Logistic Regression model (a linear decision boundary) can't fully learn it.\nGetting 50% accuracy here is expected and shows the
        limitation of linear models on XOR-type problems."
    )
