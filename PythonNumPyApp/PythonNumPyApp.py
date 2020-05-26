# NumPy Project by ChisoftMedia
import numpy
import numpy as np

arr = numpy.array([1,2,3,4,5])

print(arr)


# NumPy as np

arr = np.array([6, 7, 8, 9, 10])

print(arr)

# Checking NumPy Version
print(np.__version__)

# Create a NumPy ndarray Object
arr = np.array([11,12,13,14,15])
print(arr)
print(type(arr))

arr1 = np.array((1, 2, 3, 4, 5))

print(arr1)
print(type(arr1))

# 0-D Arrays
arr = np.array(42)

print(arr)

# 2-D Arrays
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr)

# Check Number of Dimensions
a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(a.ndim)
print(b.ndim)
print(c.ndim)
print(d.ndim)

# Higher Dimensional Arrays
arr = np.array([1, 2, 3, 4], ndmin=5)

print(arr)
print('number of dimensions :', arr.ndim)

# Access Array Elements
arr = np.array([1, 2, 3, 4])

print(arr[0])

print(arr[2] + arr[3])

# Negative Indexing
arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])

print('Last element from 2nd dim: ', arr[1, -1])

# Slicing arrays
arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:5])

print(arr[:4])