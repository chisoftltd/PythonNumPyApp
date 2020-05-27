# Random Number in Python
from numpy import random


# Generate Random Number
x = random.randint(100)

print(x)
print()


# Generate Random Array
x = random.randint(100, size = (5))

print(x)
print()

x = random.randint(100, size = (10))

print(x)
print()

x = random.randint(100, size = (15))

print(x)
print()

x = random.randint(100, size=(3, 5))

print(x)
print()

# Floats
x = random.rand(5)
print(x)

# Floats
x = random.rand(5)
print(x)

x = random.rand(3, 5)
print(x)
print()


# Generate Random Number From Array
x = random.choice([3, 5, 7, 9])
print(x)
print()

x = random.choice([3, 5, 7, 9, 10, 11, 12, 13], size=(3, 5))

print(x)