import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport abs

ctypedef np.uint32_t u32_t
ctypedef np.float64_t f64_t

@cython.boundscheck(False)
def sampen_scale2_f64f64(np.ndarray[f64_t, ndim=1] y, float r):
    cdef Py_ssize_t i, j
    cdef Py_ssize_t l, Nn, Nd
    
    l = len(y)
    Nn = 0
    Nd = 0
    for i in xrange(l - 2):
        for j in xrange(i + 1, l - 2):
            if abs(y[i] - y[j]) < r and abs(y[i + 1] - y[j + 1]) < r:
                Nn += 1

                if abs(y[i + 2] - y[j + 2]) < r:
                    Nd += 1

    return -np.log(float(Nd) / Nn)
