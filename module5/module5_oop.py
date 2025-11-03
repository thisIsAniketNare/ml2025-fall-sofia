class NumberStore:
    
    def __init__(self):
        self._data = []

    def read_int(self, prompt: str) -> int:
        while True:
            try:
                return int(input(prompt))
            except ValueError:
                print("Please enter a valid integer.")

    def insert(self, value: int) -> None:
        self._data.append(value)

    def find_first(self, target: int) -> int:
        try:
            return self._data.index(target) + 1
        except ValueError:
            return -1

    def get_data_length(self) -> int:
        return len(self._data)


def main():
    store = NumberStore()

    # Read N (positive integer)
    while True:
        N = store.read_int("Enter N (positive integer): ")
        if N > 0:
            break
        print("N must be a positive integer.")

    # Read N numbers
    for i in range(1, N + 1):
        num = store.read_int(f"Enter number {i}: ")
        store.insert(num)

    # Read X and search
    X = store.read_int("Enter X (integer to search): ")
    idx = store.find_first(X)
    print(idx)


if __name__ == "__main__":
    main()
