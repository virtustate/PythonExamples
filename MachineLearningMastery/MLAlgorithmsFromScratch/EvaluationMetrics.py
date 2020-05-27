# Calculate accuracy percentage between two lists
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# calculate a confusion matrix
def confusion_matrix(actual, predicted):
    unique = set(actual)
    matrix = [list() for x in range(len(unique))]
    for i in range(len(unique)):
        matrix[i] = [0 for x in range(len(unique))]
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for i in range(len(actual)):
        x = lookup[actual[i]]
        y = lookup[predicted[i]]
        matrix[y][x] += 1
    return unique, matrix


# pretty print a confusion matrix
def print_confusion_matrix(unique, matrix):
    print('(A)' + ' '.join(str(x) for x in unique))
    print('(P)---')
    for i, x in enumerate(unique):
        print("%s| %s" % (x, ' '.join(str(x) for x in matrix[i])))


# Test accuracy
actual = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
predicted = [0, 1, 0, 0, 0, 1, 0, 1, 1, 1]
accuracy = accuracy_metric(actual, predicted)
print(accuracy)

# Test confusion matrix with integers
actual = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
predicted = [0, 1, 1, 0, 0, 1, 0, 1, 1, 1]
unique, matrix = confusion_matrix(actual, predicted)
print(unique)
print(matrix)
print_confusion_matrix(unique, matrix)
