a_tuple = ('a', 'b')
print(a_tuple)
a_list = [1, a_tuple]
print(a_list)
another_tuple = (a_list, a_tuple)
print(another_tuple)
a_set = set([1, 2, 3, 4,4,2])
print(a_set)
weird_set = set([a_tuple, 2])
print(weird_set)
print(type(a_set))
print(isinstance(a_set, set))