import numpy as np
from sklearn.metrics import precision_score, recall_score


def read_int(prompt: str) -> int:
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
            print("Value must be positive.")
        except ValueError:
            print("Invalid integer.")


def read_label(prompt: str) -> int:
    while True:
        try:
            value = int(input(prompt))
            if value in (0, 1):
                return value
            print("Label must be 0 or 1.")
        except ValueError:
            print("Invalid label.")


def main():

    # Read N
    N = read_int("Enter N (number of data points): ")

    # Preallocate NumPy arrays
    true_labels = np.zeros(N, dtype=np.int32)
    pred_labels = np.zeros(N, dtype=np.int32)

    # Read N pairs
    for i in range(N):
        print(f"\nPoint {i+1}:")
        x = read_label("  Enter X (true label 0/1): ")
        y = read_label("  Enter Y (predicted label 0/1): ")
        true_labels[i] = x
        pred_labels[i] = y

    # Compute precision and recall
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)

    # Output
    print("\n=== Results ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")


if __name__ == "__main__":
    main()
