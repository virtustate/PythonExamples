from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit
import pandas
import numpy
from sklearn import preprocessing

dfAbalone = pandas.read_csv('abalone.csv', header=None)
Y = dfAbalone[dfAbalone.columns[-1]]
X = dfAbalone[dfAbalone.columns[0:-1]]
le = preprocessing.LabelEncoder()
le.fit(X[0])
print(le.classes_)
X[0] = le.transform(X[0].array)

for algorithm in ('auto','ball_tree','kd_tree','brute'):
    knn = KNeighborsClassifier(n_neighbors=5, algorithm=algorithm)
    knn.fit(X,Y)
    print(algorithm,knn.score(X,Y))
    print(cross_val_score(knn, X, Y, cv=5))
    cv = ShuffleSplit(n_splits=5, test_size=0.2)
    print(numpy.mean(cross_val_score(knn, X, Y, cv=cv)))

