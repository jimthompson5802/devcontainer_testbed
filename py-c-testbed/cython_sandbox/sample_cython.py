import fib 
import add_vectors as av
import numpy as np


fib.fib(2000)

a = np.array([1, 2.5, 3])
b = np.array([4, 5, 6])

print(np.array(av.add_vectors(a, b)))