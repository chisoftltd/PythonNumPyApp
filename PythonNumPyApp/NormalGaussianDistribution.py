# NormalGaussianDistribution by ChisoftMedia
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import random

x = random.normal(size=(2, 3))
print(x)
print()

x = random.normal(loc=1, scale=2, size=(2, 3))

print(x)
print()

sns.distplot(random.normal(size=1000), hist=False)

plt.show()
