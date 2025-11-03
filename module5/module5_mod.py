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
