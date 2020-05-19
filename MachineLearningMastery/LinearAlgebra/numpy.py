import numpy
from numpy import array

a = numpy.empty([3, 3])
print(a.shape)
print(a.dtype)
print(a)

top = numpy.array([1, 2, 3])
bottom = numpy.array([4, 5, 6])
v = numpy.vstack((top, bottom))
h = numpy.hstack((top, bottom))
print(v)
print(h)
print(v[1, 2])
print(v[0:2, 0:2])

# reshape 1D array to 2D

# define array
data = array([11, 22, 33, 44, 55])
print(data.shape)
# reshape
data = data.reshape((data.shape[0], 1))
print(data.shape)
print(data)


