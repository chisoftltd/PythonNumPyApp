# NumPy Project by ChisoftMedia
import numpy
import numpy as np
import matplotlib.pyplot as plt

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

# Histogram
# x = numpy.random.uniform(0.0, 5.0, 550)

# plt.hist(x, 10)
# plt.show()

# Get the Shape of an Array
arr = np.array(([[1,2,3,4],[5,6,7,8]]))

print(arr.shape)

arr = np.array([1, 2, 3, 4], ndmin=5)

print(arr)
print('shape of array :', arr.shape)

# NumPy Array Reshaping
# Reshape From 1-D to 2-D
arr = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

newarr = arr.reshape(4, 3)

print(newarr)
print(newarr.shape)

# Reshape From 1-D to 3-D
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(2, 3, 2)

print(newarr)

# Returns Copy or View
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

print(arr.reshape(2,4).base)

# Flattening the arrays
arr = np.array([[1, 2, 3], [4, 5, 6]])

newarr = arr.reshape(-1)

print(newarr)

# NumPy Array Iterating
arr = np.array([1, 2, 3])

for x in arr:
  print(x)


# Iterating 2-D Arrays
arr = np.array([[1, 2, 3], [4, 5, 6]])

for x in arr:
  print(x)

for x in arr:
  for y in x:
    print(y)


# Iterating 3-D Arrays
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

for x in arr:
  print(x)

for x in arr:
  for y in x:
    for z in y:
      print(z)


# Iterating Arrays Using nditer()
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

for x in np.nditer(arr):
  print(x)

for x in np.nditer(arr, flags=['buffered'], op_dtypes=['S']):
  print(x)

for x in np.nditer(arr[:, ::2]):
  print(x)

# Enumerated Iteration Using ndenumerate()
for idx, x in np.ndenumerate(arr):
  print(idx, x)
  

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

for idx, x in np.ndenumerate(arr):
  print(idx, x)

print()  
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9,10,11,12]])

for idx, x in np.ndenumerate(arr):
  print(idx, x)

print()
arr = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[9,10,11,12], [13,14,15,16]]])

print()
for idx, x in np.ndenumerate(arr):
  print(idx, x)


# Joining NumPy Arrays
arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.concatenate((arr1, arr2))

print(arr)

arr2 = np.concatenate((arr, arr))

print(arr2)

# Joining Arrays Using Stack Functions
arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.stack((arr1, arr2), axis=1)

print(arr)
print()

arr = np.hstack((arr1, arr2))

print(arr)
# Joining Arrays Using Stack Functions
arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.stack((arr1, arr2), axis=1)

print(arr)

print()

arr = np.hstack((arr1, arr2))

print(arr)
print()

arr = np.vstack((arr1, arr2))

print(arr)

print()

arr = np.dstack((arr1, arr2))

print(arr)

# Splitting NumPy Arrays
arr = np.array([1, 2, 3, 4, 5, 6])

newarr = np.array_split(arr, 3)

print(newarr)

newarr = np.array_split(arr, 6)

print(newarr)

# Split Into Arrays
arr = np.array([1, 2, 3, 4, 5, 6])

newarr = np.array_split(arr, 3)

print(newarr[0])
print(newarr[1])
print(newarr[2])

# Splitting 2-D Arrays
arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])

newarr = np.array_split(arr, 3)

print(newarr)


arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

newarr = np.array_split(arr, 3)

print(newarr)

# Searching Arrays
arr = np.array([1, 2, 3, 4, 5, 4, 4])

x = np.where(arr == 4)

print(x)

x = np.where(arr%2 == 0)

print(x)

x = np.where(arr%2 != 0)

print(x)

# Search Sorted
arr = np.array([1, 2, 3, 4, 6, 7, 8, 9,10])
x = np.searchsorted(arr, 10)

print(x)
print()
x = np.searchsorted(arr, 1)
print(x)
print()
x = np.where(arr%2 != 0)

print(x)
print()
x = np.where(arr%2 == 0)

print(x)
print()
x = np.searchsorted(arr, 7, side='right')

print(x)

# NumPy Sorting Arrays

print(np.sort(arr))
print()

arr4 = np.array(['banana', 'cherry', 'apple', 'orange', 'mango'])

print(np.sort(arr4))
print()

# Sorting a 2-D Array
arr = np.array([[3, 2, 4], [5, 0, 1]])

print(np.sort(arr))

# NumPy Filter Array
arr = np.array([41, 42, 43, 44, 45, 46, 47, 48])
x = [True, False, True, False, False, True, True, False]
newarr = arr[x]

print(newarr)
print(np.sort(x))
print(np.sort(arr))



filter_arr = []
for value in arr:
    if value >= 45:
        filter_arr.append(True)
    else:
        filter_arr.append(False)

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)
print()


# Create an empty list
filter_arr = []
for element in arr:
  # if the element is completely divisble by 2, set the value to True, otherwise False
  if element % 2 == 0:
    filter_arr.append(True)
  else:
    filter_arr.append(False)

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)
print()

# Create an empty list
filter_arr = []
for element in arr:
  # if the element is completely not divisble by 2, set the value to True, otherwise False
  if element % 2 != 0:
    filter_arr.append(True)
  else:
    filter_arr.append(False)

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)

# Creating Filter Directly From Array
arr = np.array([31,32,34,36,38,40,42,21,31,33,35,23,24,25,0,5])

filter_arr = arr > 20

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)
print(np.sort(newarr))
print()

filter_arr = arr % 2 == 0
newarr = arr[filter_arr]

print(filter_arr)
print(newarr)
print(np.sort(newarr))