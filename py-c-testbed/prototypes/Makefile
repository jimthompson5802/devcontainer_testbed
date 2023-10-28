# Variables
CXX = g++
CXXFLAGS = -O3 -Wall -std=c++11 -fPIC
PYBINDFLAGS = -I/usr/local/include/python3.10 -I/home/vscode/.local/lib/python3.10/site-packages/pybind11/include
PYSUFFIX = .cpython-310-x86_64-linux-gnu.so

# Targets
all: main cppmodule$(PYSUFFIX)

# Compile the C++ program
main: main.cpp message_lib.o
	$(CXX) $(CXXFLAGS) -o $@ $< message_lib.o

# Compile the C++ library object
message_lib.o: message_lib.cpp message_lib.h
	$(CXX) $(CXXFLAGS) -c $<

# Compile the Python module
cppmodule$(PYSUFFIX): binding.cpp message_lib.o
	$(CXX) $(CXXFLAGS) -shared -fPIC $(PYBINDFLAGS) $< message_lib.o -o $@

run: main
	./main

run_python: cppmodule$(PYSUFFIX)
	python3 main.py

clean:
	rm -f main *.o cppmodule$(PYSUFFIX)

.PHONY: all run run_python clean
