# cython: language_level=3

cdef class SuperClass:
    cdef int x
    def __cinit__(self, int x, int y=0):
        self.x = x
        print(f"SuperClass.__cinit__")

    cpdef void show(self):
        print(f"x={self.x}")

cdef class SubClass(SuperClass):
    cdef int y
    def __cinit__(self, int x, int y):
        print(f"SubClass.__cinit__")
        self.y = y

    cpdef void show(self):
        super(SubClass, self).show()
        print(f"y={self.y}")