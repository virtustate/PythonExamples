from sklearn_lvq import GlvqModel
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit
import pandas
import numpy
from sklearn import preprocessing

df = pandas.read_csv('ionosphere.csv', header=None)
Y = df[df.columns[-1]]
X = df[df.columns[0:-1]]

glvq = GlvqModel()
means = []
for n in range(1,20):
    glvq.prototypes_per_class=n
    glvq.fit(X,Y)
    print(n,glvq.score(X,Y))
    scores=cross_val_score(glvq, X, Y, cv=5)
    print(scores, numpy.mean(scores))
    means.append(numpy.mean(scores))
print(means)


