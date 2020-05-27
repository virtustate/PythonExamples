from csv import reader
from random import randrange, seed

from numpy.ma import sqrt


def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    ds = list(lines)
    return ds


def str_column_to_float(ds, column):
    for row in ds:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(ds, column):
    class_values = [row[column] for row in ds]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in ds:
        row[column] = lookup[row[column]]
    return lookup


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# calculate column means
def column_means(dataset):
    means = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        means[i] = sum(col_values) / float(len(dataset))
    return means


# calculate column standard deviations
def column_stdevs(dataset, means):
    stdevs = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        variance = [pow(row[i] - means[i], 2) for row in dataset]
        stdevs[i] = sum(variance)
    stdevs = [sqrt(x / (float(len(dataset) - 1))) for x in stdevs]
    return stdevs


# standardize dataset
def standardize_dataset(dataset, means, stdevs):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - means[i]) / stdevs[i]


# Split a dataset into a train and test set
def train_test_split(dataset, split=0.60):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy


# Split a dataset into $k$ folds
def cross_validation_split(dataset, folds=3):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Load pima dataset
filename = 'pima-indians-diabetes.csv'
ds_pima = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(ds_pima), len(ds_pima[0])))
print(ds_pima[0])
# convert string columns to float
for i in range(len(ds_pima[0])):
    str_column_to_float(ds_pima, i)
print(ds_pima[0])
# Calculate min and max for each column
minmax = dataset_minmax(ds_pima)
print(minmax)
# Normalize columns
normalize_dataset(ds_pima, minmax)
print(ds_pima[0])
# Estimate mean and standard deviation
means = column_means(ds_pima)
stdevs = column_stdevs(ds_pima, means)
# standardize dataset
standardize_dataset(ds_pima, means, stdevs)
print(ds_pima[0])
print('-----------------------------')

# Load iris dataset
filename = 'iris.csv'
ds_iris = load_csv(filename)
print('Loaded data file {0} with {1} rows and {2} columns'.format(filename, len(ds_iris), len(ds_iris[0])))
print(ds_iris[0])
# convert string columns to float
for i in range(4):
    str_column_to_float(ds_iris, i)
# convert class column to int
lookup = str_column_to_int(ds_iris, 4)
print(ds_iris[0])
print(lookup)
seed(1)
train, test = train_test_split(ds_iris, split=.8)
print(len(train), len(test))
folds = cross_validation_split(ds_iris, 5)
fold_size = map(len, folds)
print(list(fold_size))
