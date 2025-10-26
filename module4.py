# Read N (number of elements)
N = int(input("Enter N (positive integer): "))

# Read N numbers one by one
numbers = []
for i in range(N):
    num = int(input(f"Enter number {i + 1}: "))
    numbers.append(num)

# Read X
X = int(input("Enter X (integer to search for): "))

# Search for X in the list
if X in numbers:
    # Output the index (1-based)
    print(numbers.index(X) + 1)
else:
    # Output -1 if not found
    print(-1)
