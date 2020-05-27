
# Logistic Distribution by ChisoftMedia
from numpy import random
import seaborn as sns
import matplotlib.pyplot as plt

x = random.logistic(loc=1, scale=2, size= (2,3))
print(x)
 
sns.distplot(random.logistic(size=1000), hist=False)
 
plt.show()

sns.distplot(random.normal(scale=2, size=1000), hist=False, label='normal')
sns.distplot(random.logistic(size=1000), hist=False, label='logistic')

plt.show()