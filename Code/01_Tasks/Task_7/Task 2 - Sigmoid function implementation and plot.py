"""
Task 2 - Sigmoid function implementation and plot
IEEE SSCS - ML Task 7
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    
    # z : numpy array
    # hreturn : sigmoid(z) in range (0,1)
    
    return 1 / (1 + np.exp(-z))


if __name__ == "__main__":
    # suitable range to show the S-shape clearly (saturates near +-10)
    z = np.linspace(-10, 10, 400)
    s = sigmoid(z)

    plt.figure(figsize=(7, 5))
    plt.plot(z, s, color="#2B5748", linewidth=2, label="sigmoid(z)")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("Sigmoid Function")
    plt.xlabel("z")
    plt.ylabel("sigmoid(z)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("sigmoid_plot.png", dpi=150)
    plt.show()
