from sklearn import tree
import numpy
import pandas
import matplotlib.pyplot as plt
import graphviz
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB, ComplementNB, MultinomialNB

data = pandas.read_csv('iris.csv', header=None)
Y = numpy.asarray(data[data.columns[-1]])
X = numpy.asarray(data[data.columns[0:-1]])
clf = tree.DecisionTreeClassifier(max_depth=4)
GNB = GaussianNB()
MNB = MultinomialNB()
CNB = ComplementNB()

print('clf')
scores = cross_val_score(clf, X, Y, cv=5)
print(scores)
clf.fit(X, Y)
print(clf.score(X, Y))

print('GNB')
scores = cross_val_score(GNB, X, Y, cv=5)
print(scores)
GNB.fit(X, Y)
print(GNB.score(X, Y))

print('MNB')
scores = cross_val_score(MNB, X, Y, cv=5)
print(scores)
MNB.fit(X, Y)
print(MNB.score(X, Y))

print('CNB')
scores = cross_val_score(CNB, X, Y, cv=5)
print(scores)
CNB.fit(X, Y)
print(CNB.score(X, Y))