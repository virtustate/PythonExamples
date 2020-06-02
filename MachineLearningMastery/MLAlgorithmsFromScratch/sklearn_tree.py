from sklearn import tree
import numpy
import pandas
import matplotlib.pyplot as plt
import graphviz
from sklearn.model_selection import cross_val_score

data = pandas.read_csv('data_banknote_authentication.csv', header=None)
Y = numpy.asarray(data[data.columns[-1]])
X = numpy.asarray(data[data.columns[0:-1]])
clf = tree.DecisionTreeClassifier(max_depth=4)

scores = cross_val_score(clf, X, Y, cv=5)
print(scores)

clf.fit(X, Y)
tree.plot_tree(clf)
plt.show()
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("bank")
print(clf.score(X, Y))
