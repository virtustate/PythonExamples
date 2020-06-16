import pandas
import numpy

data = pandas.read_csv('abalone.csv', header=None)
Y = numpy.asarray(data[data.columns[-1]])
mean = numpy.mean(Y)
std = numpy.std(Y)
print(mean, std)
print(numpy.mean(abs((Y-mean)/Y)))