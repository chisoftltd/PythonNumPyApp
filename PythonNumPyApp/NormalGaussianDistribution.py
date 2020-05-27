# NormalGaussianDistribution by ChisoftMedia
from numpy import random

x = random.normal(size=(2, 3))
print(x)
print()

x = random.normal(loc=1, scale=2, size=(2, 3))

print(x)

