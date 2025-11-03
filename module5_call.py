from module5_mod import NumberStore

def main():
    store = NumberStore()

    # Read N (positive integer)
    while True:
        N = store.read_int("Enter N (positive integer): ")
        if N > 0:
            break
        print("N must be a positive integer.")

    # Read N numbers one by one
    for i in range(1, N + 1):
        num = store.read_int(f"Enter number {i}: ")
        store.insert(num)

    # Read X and search
    X = store.read_int("Enter X (integer to search): ")
    idx = store.find_first(X)
    print(idx)

if __name__ == "__main__":
    main()
