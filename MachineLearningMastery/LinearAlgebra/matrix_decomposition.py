# LU decomposition
import numpy
from scipy.linalg import lu
from numpy.linalg import eig, inv, svd, pinv

# define a square matrix
A = numpy.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]])
print(A)
# factorize
P, L, U = lu(A)
print(P)
print(L)
print(U)
# reconstruct
B = P.dot(L).dot(U)

# eigendecomposition
# define matrix
A = numpy.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]])
print(A)
# factorize
print('eigendecomposition')
values, vectors = eig(A)
print(values)
print(vectors)
print('----------------')
print(inv(vectors))
print('----------------')
print(vectors.dot(inv(vectors)))
print('----------------')
B = inv(vectors).dot(A.dot(vectors))
print(B)
print(vectors.dot(B.dot(inv(vectors))))
L = numpy.diag(values)
print(vectors.dot(L.dot(inv(vectors))))

# confirm eigenvector
# define matrix
print('confirm eigenvectors')
A = numpy.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 10]])
# factorize
values, vectors = eig(A)
# confirm first eigenvector
for i in range(0, 3):
    B = A.dot(vectors[:, i])
    print(B)
    C = vectors[:, i] * values[i]
    print(C)

# singular-value decomposition
# define a matrix
print('----------------')
A = numpy.array([
    [1, 2],
    [3, 4],
    [5, 6]])
print(A)
# factorize
U, s, VT = svd(A)
print(U)
print(s)
print(VT)
Sigma = numpy.zeros((A.shape[0], A.shape[1]))
Sigma[:A.shape[1], :A.shape[1]] = numpy.diag(s)
print(U.dot(Sigma.dot(VT)))

# pseudoinverse
# define matrix
print('------- psudoinverse---------')
A = numpy.array([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
    [0.7, 0.8]])
print(A)
# calculate pseudoinverse
B = pinv(A)
print(B)
print(B.dot(A))
# pseudoinverse via svd
# factorize
U, s, VT = svd(A)
# reciprocals of s
d = 1.0 / s
# create m x n D matrix
D = numpy.zeros(A.shape)
# populate D with n x n diagonal matrix
D[:A.shape[1], :A.shape[1]] = numpy.diag(d)
# calculate pseudoinverse
B2 = VT.T.dot(D.T).dot(U.T)
print(B2-B)