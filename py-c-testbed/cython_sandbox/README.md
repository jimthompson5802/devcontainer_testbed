# Cython Sandbox

## Development Environment Setup
Requires C/C++ compiler with OpenMP support and Cython. For details see the following on development environment setup:
- [`devcontainer.json`](../../.devcontainer/py-c-dev/devcontainer.json)
- [Dockerfile](../../.devcontainer/py-c-dev/Dockerfile)
- [requirements.txt](../../.devcontainer/py-c-dev/requirements.txt)


## Example Cython Files

1. `add_vectors.pyx`: This is a Cython file that defines a function `add_vectors(a, b)` which adds two vectors `a` and `b`. The function is implemented in a way that disables bounds checking and wraparound for performance.

1. `matrix_multiply.pyx`: This is another Cython file that defines a function `matrix_multiply_cp(A, B)` which multiplies two matrices `A` and `B`. The function is also implemented with bounds checking and wraparound disabled for performance.  Makes use of thread parallelism for the inner-most loop.

1. `fib.pyx`: file is a Cython file that computes the Fibonacci sequence.

1. `helloworld.pyx`: file is a Cython file that prints "Hello World!".

## Testing Cython Code

`notebook_testbed.ipynb`: This is a Jupyter notebook file that seems to be used for testing Cython code. It contains cells for importing Cython, testing inline compilation, and executing the compiled functions.


## Building Cython extension module 
Extension modulecan be built in one of two ways:

### Command Line
```bash
$ cythonize -i add_vectors.pyx
```

### setup script `setup.py`
`setup.py`: This is a Python script used for building the Cython extension module.

Example `setup.py` script:
```python
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("add_vectors.pyx"),
)
```

```bash
$ python setup.py build_ext --inplace
```

### Generated Cython extension module
```bash
$ ls -l *.so

-rwxr-xr-x 1 vscode vscode  265864 Dec  3 05:05 add_vectors.cpython-310-x86_64-linux-gnu.so

```

### To use generated extension module import into Python program
```python
import add_vectors
```
