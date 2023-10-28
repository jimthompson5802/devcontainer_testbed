# Makefile

# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -std=c++17 -fPIC

# Source and object files
SRCS = matrix.cpp matrix_test_pgm.cpp
OBJS = $(SRCS:.cpp=.o)
EXEC = matrix_test_pgm
LIBRARY = libmatrix.so

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(LIBRARY): $(OBJS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: run
run: $(EXEC)
	./$(EXEC)

.PHONY: clean
clean:
	rm -f $(OBJS) $(EXEC) $(LIBRARY)

.PHONY: test
test: $(EXEC)
	./$(EXEC)
    
.PHONY: library
library: $(LIBRARY)