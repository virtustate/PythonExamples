import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# % matplotlib inline
from sklearn.linear_model import LogisticRegression, Perceptron
import joblib
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit
from sklearn.preprocessing import scale

dfSonar = pd.read_csv('sonar.all-data.csv', header=None)
labels = dfSonar[dfSonar.columns[-1]]
labels = map(lambda x: 1 if x=='R' else 0, dfSonar[dfSonar.columns[-1]])
dfSonar[dfSonar.columns[-1]] = list(labels)
allLabel = np.asarray(dfSonar[dfSonar.columns[-1]])
allData = np.asarray(dfSonar[dfSonar.columns[0:-1]])

perceptron = Perceptron()
perceptron.fit(allData,allLabel)
print(perceptron.score(allData,allLabel))
print(cross_val_score(perceptron, allData, allLabel, cv=2))

logistic = LogisticRegression()
logistic.fit(scale(allData),allLabel)
print(logistic.score(allData,allLabel))
print(cross_val_score(logistic, scale(allData), allLabel, cv=2))


cv = ShuffleSplit(n_splits=5, test_size=0.2)
print(np.mean(cross_val_score(logistic, scale(allData), allLabel, cv=cv)))
print(np.mean(cross_val_score(perceptron, scale(allData), allLabel, cv=cv)))


