import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# % matplotlib inline
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale

diabetesDF = pd.read_csv('pima-indians-diabetes.csv')
print(diabetesDF.head())

# correlation matrix
corr = diabetesDF.corr()
#print(corr)
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.show()

# k-fold cross validation
allLabel = np.asarray(diabetesDF['Outcome'])
allData = np.asarray(diabetesDF.drop('Outcome', 1))
allData = scale(allData)
diabetesCheckK = LogisticRegression()
scores = cross_val_score(diabetesCheckK, allData, allLabel, cv=10)
print(scores)

# train/test split validation
dfTrain = diabetesDF[:650]
dfTest = diabetesDF[650:750]
dfCheck = diabetesDF[750:]
trainLabel = np.asarray(dfTrain['Outcome'])
trainData = np.asarray(dfTrain.drop('Outcome', 1))
testLabel = np.asarray(dfTest['Outcome'])
testData = np.asarray(dfTest.drop('Outcome', 1))
means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)
trainData = (trainData - means) / stds
testData = (testData - means) / stds
diabetesCheck = LogisticRegression()
# print(np.mean(trainData, axis=0),np.std(trainData, axis=0))
diabetesCheck.fit(trainData, trainLabel)
accuracy = diabetesCheck.score(testData, testLabel)

# visualize feature significance
print("accuracy = ", accuracy * 100, "%")
coeff = list(diabetesCheck.coef_[0])
labels = list(diabetesDF.columns.drop('Outcome'))
features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 6), color=features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')
plt.show()

# save and check saved model
joblib.dump([diabetesCheck, means, stds], 'diabeteseModel.pkl')
diabetesLoadedModel, means, stds = joblib.load('diabeteseModel.pkl')
accuracyModel = diabetesLoadedModel.score(testData, testLabel)
print("accuracy = ", accuracyModel * 100, "%")
