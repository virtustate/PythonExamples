from sklearn import tree
import numpy
import pandas
import matplotlib.pyplot as plt
import graphviz
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

data = pandas.read_csv('iris.csv', header=None)
Y = numpy.asarray(data[data.columns[-1]])
X = numpy.asarray(data[data.columns[0:-1]])
#clf = tree.DecisionTreeClassifier(max_depth=4)
clf = GaussianNB()

scores = cross_val_score(clf, X, Y, cv=5)
print(scores)

clf.fit(X, Y)
print(clf.score(X, Y))