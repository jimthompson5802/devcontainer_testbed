#cython: language_level=3

cimport cython
from libc.stdlib cimport malloc, free

@cython.boundscheck(False)
@cython.wraparound(False)
def add_vectors(a, b):
    cdef int i, n = a.shape[0]
#    cdef double[:] result = <double[:n]> malloc(n * sizeof(double))
    cdef result = a.copy()

    for i in range(n):
        result[i] = a[i] + b[i]

    return result