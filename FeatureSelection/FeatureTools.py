# adapted from https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
from sklearn.datasets import make_regression, make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import f_regression
import pandas
from sklearn.linear_model import LinearRegression


def numeric_to_numeric():
    X, y = make_regression(n_samples=100, n_features=100, n_informative=10)
    # define feature selection
    fs = SelectKBest(score_func=f_regression, k=10)
    # apply feature selection
    X_selected = fs.fit_transform(X, y)
    # print(X_selected.shape)
    print_summary(fs)
    for k in range(1, 20):
        fs = SelectKBest(score_func=f_regression, k=k)
        # apply feature selection
        X_selected = fs.fit_transform(X, y)
        model = LinearRegression()
        model.fit(X_selected, y)
        print(k, model.score(X_selected, y))


def print_summary(fs):
    # collect data in dataframe and show top 20 features
    ds = pandas.DataFrame(columns=['selected', 'pvalue', 'score'])
    ds.selected = fs.get_support()
    ds.pvalue = fs.pvalues_
    ds.score = fs.scores_
    ds = ds.sort_values('score', ascending=False)
    print(ds.head(20))


# numeric to categorical
def numeric_to_categorical():
    X, y = make_classification(n_samples=100, n_features=20, n_informative=2)
    # define feature selection
    fs = SelectKBest(score_func=f_classif, k=2)
    # apply feature selection
    X_selected = fs.fit_transform(X, y)
    print_summary(fs)


if __name__ == "__main__":
    # numeric_to_numeric()
    numeric_to_categorical()
