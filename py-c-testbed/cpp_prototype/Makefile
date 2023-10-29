# Makefile

# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -std=c++17 -fPIC

# Source and object files
EXEC = matrix_test_pgm
LIB_OBJS = matrix.o
LIBRARY = libmatrix.so
OBJS = $(SRCS:.cpp=.o)
SRCS = matrix.cpp matrix_test_pgm.cpp
PYTHON_VERSION = 3.9
PYTHON_INC = /usr/include/python$(PYTHON_VERSION)


all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(LIBRARY): $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@


.PHONY: run
run: $(EXEC)
	./$(EXEC)

.PHONY: clean
clean:
	rm -f $(OBJS) $(EXEC) $(LIBRARY)  matrixboostlib.so

.PHONY: test
test: $(EXEC)
	./$(EXEC)
    
.PHONY: library
library: $(LIBRARY)