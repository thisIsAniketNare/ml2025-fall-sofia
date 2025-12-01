import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class KNNClassificationExperiment:
    def __init__(self):
        self.X_train = None  # shape (N, 1)
        self.y_train = None  # shape (N,)
        self.X_test = None   # shape (M, 1)
        self.y_test = None   # shape (M,)

    # ---------- I/O helpers ----------

    @staticmethod
    def read_positive_int(prompt: str) -> int:
        while True:
            try:
                value = int(input(prompt))
                if value > 0:
                    return value
                print("Value must be a positive integer.")
            except ValueError:
                print("Invalid input. Please enter a positive integer.")

    @staticmethod
    def read_float(prompt: str) -> float:
        while True:
            try:
                return float(input(prompt))
            except ValueError:
                print("Invalid input. Please enter a real number.")

    @staticmethod
    def read_nonnegative_int(prompt: str) -> int:
        while True:
            try:
                value = int(input(prompt))
                if value >= 0:
                    return value
                print("Label must be a non-negative integer.")
            except ValueError:
                print("Invalid input. Please enter a non-negative integer.")

    # ---------- Data reading ----------

    def read_training_data(self):
        """Read N and then N (x, y) pairs into preallocated NumPy arrays."""
        N = self.read_positive_int("Enter N (number of training samples): ")

        # Preallocate arrays for speed
        self.X_train = np.zeros((N, 1), dtype=float)
        self.y_train = np.zeros(N, dtype=int)

        print("\nEnter training samples (x, y):")
        for i in range(N):
            x_val = self.read_float(f"  Train sample {i+1} - x: ")
            y_val = self.read_nonnegative_int(f"  Train sample {i+1} - y (class): ")
            self.X_train[i, 0] = x_val
            self.y_train[i] = y_val

    def read_test_data(self):
        """Read M and then M (x, y) pairs into preallocated NumPy arrays."""
        M = self.read_positive_int("\nEnter M (number of test samples): ")

        self.X_test = np.zeros((M, 1), dtype=float)
        self.y_test = np.zeros(M, dtype=int)

        print("\nEnter test samples (x, y):")
        for i in range(M):
            x_val = self.read_float(f"  Test sample {i+1} - x: ")
            y_val = self.read_nonnegative_int(f"  Test sample {i+1} - y (class): ")
            self.X_test[i, 0] = x_val
            self.y_test[i] = y_val

    # ---------- Core ML logic ----------

    def find_best_k(self, k_min: int = 1, k_max: int = 10):
        """
        Search for best k in [k_min, k_max], but k <= number of training samples.
        Returns (best_k, best_accuracy).
        """
        if self.X_train is None or self.X_test is None:
            raise ValueError("Training and test data must be loaded before calling find_best_k().")

        N = self.X_train.shape[0]
        max_k = min(k_max, N)

        if max_k < k_min:
            raise ValueError("Not enough training samples to try any k in the given range.")

        best_k = None
        best_acc = -1.0

        for k in range(k_min, max_k + 1):
            # k-NN classifier with Euclidean distance
            clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)

            acc = accuracy_score(self.y_test, y_pred)

            # Keep best k (if tie, keep first/smaller k)
            if acc > best_acc:
                best_acc = acc
                best_k = k

        return best_k, best_acc

    # ---------- Orchestration ----------

    def run(self):
        print("=== k-NN Classification: NumPy + Scikit-learn ===")
        self.read_training_data()
        self.read_test_data()

        best_k, best_acc = self.find_best_k(k_min=1, k_max=10)

        print("\n=== Results ===")
        print(f"Best k: {best_k}")
        print(f"Test accuracy for best k: {best_acc:.4f}")


def main():
    experiment = KNNClassificationExperiment()
    experiment.run()


if __name__ == "__main__":
    main()
