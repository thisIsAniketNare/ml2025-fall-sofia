import numpy as np
from sklearn.neighbors import KNeighborsRegressor


def read_int(prompt: str) -> int:
    while True:
        try:
            val = int(input(prompt))
            if val > 0:
                return val
            else:
                print("Value must be a positive integer.")
        except ValueError:
            print("Invalid input. Enter a valid integer.")


def read_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Enter a valid real number.")


def main():
    # Step 1: Read N and k
    N = read_int("Enter N (number of points): ")
    k = read_int("Enter k (number of neighbors): ")

    # Step 2: Read N (x, y) pairs
    x_vals = np.zeros(N)
    y_vals = np.zeros(N)

    for i in range(N):
        x_vals[i] = read_float(f"Enter x value for point {i+1}: ")
        y_vals[i] = read_float(f"Enter y value for point {i+1}: ")

    # Step 3: Check k validity
    if k > N:
        print("Error: k cannot be greater than N.")
        return

    # Step 4: Read test X
    X_test = read_float("Enter X (value to predict Y): ")

    # Step 5: Prepare data for sklearn (reshape for 1-dimensional feature)
    X_train = x_vals.reshape(-1, 1)
    y_train = y_vals

    # Step 6: Train k-NN regressor
    model = KNeighborsRegressor(n_neighbors=k, metric='euclidean')
    model.fit(X_train, y_train)

    # Step 7: Predict
    Y_pred = model.predict(np.array([[X_test]]))[0]

    # Step 8: Output results
    label_variance = np.var(y_train)

    print("\n=== Results ===")
    print(f"Predicted Y for X = {X_test:.3f}: {Y_pred:.3f}")
    print(f"Variance of training labels: {label_variance:.6f}")


if __name__ == "__main__":
    main()
