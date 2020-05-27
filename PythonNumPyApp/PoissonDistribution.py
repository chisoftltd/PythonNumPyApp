# PoissonDistribution by ChisoftMedia
import seaborn as sns
from numpy import random
import matplotlib.pyplot as plt

x = random.poisson(lam=2, size=10)
print(x)

sns.distplot(random.poisson(lam=2, size=1000), kde=False)

plt.show()

# Difference Between Normal and Poisson Distribution
sns.distplot(random.normal(loc=50, scale=7, size=1000), hist=False, label='normal')
sns.distplot(random.poisson(lam=50,size=1000), hist=False, label='poisson')

plt.show()

sns.distplot(random.binomial(n=1000, p=0.01, size=1000),hist=False, label='binomial')
sns.distplot(random.poisson(lam=10, size=1000), hist=False, label='poisson')
plt.show()