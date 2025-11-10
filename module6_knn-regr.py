
import numpy as np


class KNNRegressor:

    def __init__(self):
        self.x_data = np.array([])
        self.y_data = np.array([])

    def read_float(self, prompt: str) -> float:
        while True:
            try:
                return float(input(prompt))
            except ValueError:
                print("Please enter a valid real number.")

    def read_int(self, prompt: str) -> int:
        while True:
            try:
                val = int(input(prompt))
                if val > 0:
                    return val
                else:
                    print("Value must be a positive integer.")
            except ValueError:
                print("Please enter a valid integer.")

    def insert_points(self, n: int):
        x_vals = []
        y_vals = []

        for i in range(n):
            x = self.read_float(f"Enter x value for point {i+1}: ")
            y = self.read_float(f"Enter y value for point {i+1}: ")
            x_vals.append(x)
            y_vals.append(y)

        self.x_data = np.array(x_vals)
        self.y_data = np.array(y_vals)

    def predict(self, x_input: float, k: int) -> float:
        if len(self.x_data) == 0:
            raise ValueError("No data points available for prediction.")
        if k > len(self.x_data):
            raise ValueError("k cannot be greater than the number of training points.")

        # Compute Euclidean distances (L2)
        distances = np.abs(self.x_data - x_input)

        # Get indices of k smallest distances
        nearest_indices = np.argsort(distances)[:k]

        # Compute mean Y of k nearest neighbors
        y_pred = np.mean(self.y_data[nearest_indices])
        return y_pred


def main():
    model = KNNRegressor()

    N = model.read_int("Enter N (number of points): ")
    k = model.read_int("Enter k (number of neighbors): ")

    model.insert_points(N)

    if k > N:
        print("Error: k cannot be greater than N.")
        return

    X_input = model.read_float("Enter X (value to predict Y): ")

    try:
        Y_pred = model.predict(X_input, k)
        print(f"Predicted Y for X = {X_input:.3f} is: {Y_pred:.3f}")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
