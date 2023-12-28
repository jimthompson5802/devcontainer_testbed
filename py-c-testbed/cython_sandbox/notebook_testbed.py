#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cython


# In[2]:


# test inline compile
@cython.compile
def plus(a, b):
    return a + b


# In[3]:


print(plus('3', '5'))
print(plus(3, 5))


# In[4]:


print(plus('5x', '5y'))
print(plus(5, 5))


# In[5]:


# define naive matrix multiplication in numpy
import numpy as np

def matrix_multiply(A, B):
    result = np.zeros((A.shape[0], B.shape[1]))

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                result[i][j] += A[i][k] * B[k][j]
    return result

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])

print(matrix_multiply(A, B))


# In[6]:


# get cythonized version of the naive matrix multiplication
# values should match the above example
import matrix_multiply as mm
A = np.array([[1., 2, 3], [4, 5, 6]])
B = np.array([[7., 8], [9, 10], [11, 12]])

print(np.array(mm.matrix_multiply_cp(A, B)))


# In[7]:


# test naive numpy and cythonized numpy matrix multiplication 
# against each other
DIM_SIZE = 100
np.random.seed(0)
# Create two random square matrices
A = np.random.rand(DIM_SIZE, DIM_SIZE)
B = np.random.rand(DIM_SIZE, DIM_SIZE)



# In[8]:

C1 = matrix_multiply(A, B)


# In[9]:


C2 = np.array(mm.matrix_multiply_cp(A, B))
print(C2.shape)


# In[10]:


# check that the results are the same
print(np.all(np.isclose(C1, C2)))

