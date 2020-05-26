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

# Negative Slicing
arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[-3:-1])

# STEP
arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(arr[1:5:2])
print(arr[1:5:4])
print(arr[1:6:3])
print(arr[::2])

# Slicing 2-D Arrays
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[1, 1:4])

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(arr[0:2, 1:4])

# Data Types in NumPy
arr = np.array([1, 2, 3, 4])

print(arr.dtype)

arr = np.array(['apple', 'banana', 'cherry'])

print(arr.dtype)

# Creating Arrays With a Defined Data Type
arr = np.array([1, 2, 3, 4], dtype='S')

print(arr)
print(arr.dtype)

arr = np.array([1, 2, 3, 4], dtype='i4')

print(arr)
print(arr.dtype)


# Converting Data Type on Existing Arrays
arr = np.array([1.1, 2.1, 3.1])

print(arr)
print(arr.dtype)


newarr = arr.astype('i')

print(newarr)
print(newarr.dtype)

arr = np.array([1, 0, 3])

newarr = arr.astype(bool)

print(newarr)
print(newarr.dtype)

# NumPy Array Copy vs View
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[0] = 42

print(arr)
print(x)

# Make Changes in the VIEW:
arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
x[0] = 31

print(arr)
print(x)

# Check if Array Owns it's Data
arr = np.array([1, 2, 3, 4, 5])

x = arr.copy()
y = arr.view()

print(x.base)
print(y.base)