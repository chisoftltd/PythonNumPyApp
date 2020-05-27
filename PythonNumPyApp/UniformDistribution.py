# UniformDistribution by ChisoftMedia
from numpy import random
from seaborn import sns
import matplotlib.pyplot as plt

x = random.uniform(size=(2, 3))

print(x)

sns.distplot(random.uniform(size=1000), hist=False)
plt.show()
