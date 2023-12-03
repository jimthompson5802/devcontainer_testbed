#cython: language_level=3

cimport cython
from cython.view cimport array as cvarray
from cython.parallel cimport prange
from libc.stdlib cimport calloc, free
from libc.math cimport isnan


@cython.boundscheck(False)
@cython.wraparound(False)
def matrix_multiply_cp(A, B):
    # result = np.zeros((A.shape[0], B.shape[1]))
    cdef int i, j, k
    cdef int nrows = len(A)
    cdef int ncols = len(B[0])
    cdef int ncols_A = len(A[0])

    cdef double[:,:] result = <double[:nrows, :ncols]> calloc(nrows * ncols, sizeof(double))

    cdef double[:,:] A_view = A
    cdef double[:,:] B_view = B
    cdef double[:,:] result_view = result

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in prange(ncols_A, nogil=True):
                result[i][j] += A_view[i][k] * B_view[k][j]

    return result
