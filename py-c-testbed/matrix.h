// matrix.h

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>

class Matrix {
public:
    using matrix_t = std::vector<std::vector<double>>;

    // Constructors
    Matrix() = default;
    Matrix(const matrix_t& data) : mat(data) {}

    // Operations
    Matrix add(const Matrix& other) const;
    Matrix subtract(const Matrix& other) const;

    // Display
    void display() const;

private:
    matrix_t mat;
};

#endif // MATRIX_H
