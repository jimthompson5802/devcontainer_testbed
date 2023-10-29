# Initial protype for Python/C++ integration using pybind11

## To run this prototype:
```bash
$ make clean
rm -f main *.o cppmodule.cpython-310-x86_64-linux-gnu.so


$ make run_all
g++ -O3 -Wall -std=c++11 -fPIC -c message_lib.cpp
g++ -O3 -Wall -std=c++11 -fPIC -o main main.cpp message_lib.o
./main
echo Hello, world!
The sum of 2 and 3 is 5
The sum of 2.5 and 3.9 is 6.4
g++ -O3 -Wall -std=c++11 -fPIC -shared -fPIC -I/usr/local/include/python3.10 -I/home/vscode/.local/lib/python3.10/site-packages/pybind11/include binding.cpp message_lib.o -o cppmodule.cpython-310-x86_64-linux-gnu.so
python3 main.py
echo Hello from Python using C++!
echo Hello again from Python using C++!
echo Hello again again from Python using C++!
1 + 4 answer is: 5
2 + 5 answer is: 7
3 + 6 answer is: 9
floats: 3.5
pytest -v test_prototype.py
====================================================================== test session starts ======================================================================
platform linux -- Python 3.10.12, pytest-7.4.3, pluggy-1.3.0 -- /usr/local/bin/python
cachedir: .pytest_cache
rootdir: /workspaces/devcontainer_testbed/py-c-testbed/prototypes
collected 3 items                                                                                                                                               

test_prototype.py::test_print_message PASSED                                                                                                              [ 33%]
test_prototype.py::test_add_numbers_int PASSED                                                                                                            [ 66%]
test_prototype.py::test_add_numbers_float PASSED                                                                                                          [100%]

======================================================================= 3 passed in 0.01s =======================================================================
```