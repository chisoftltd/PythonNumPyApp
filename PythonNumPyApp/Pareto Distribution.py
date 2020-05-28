# Pareto Distribution by ChisoftMedia
from numpy import random
import seaborn as sns
import matplotlib.pyplot as plt

x = random.pareto(a=2, size=(2, 3))
print(x)

sns.distplot(random.pareto(a = 2, size=1000), kde=False)
plt.show()
