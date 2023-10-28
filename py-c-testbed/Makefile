# Makefile

# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -std=c++17

# Source and object files
SRCS = matrix.cpp matrix_test_pgm.cpp
OBJS = $(SRCS:.cpp=.o)
EXEC = matrix_app

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: run
run: $(EXEC)
	./$(EXEC)

.PHONY: clean
clean:
	rm -f $(OBJS) $(EXEC)
