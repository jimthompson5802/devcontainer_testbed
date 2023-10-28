// matrix.cpp

#include "matrix.h"
#include <stdexcept>  // for std::invalid_argument

Matrix Matrix::add(const Matrix& other) const {
    if (mat.size() != other.mat.size() || mat[0].size() != other.mat[0].size()) {
        throw std::invalid_argument("Matrices must be of the same size for addition.");
    }

    matrix_t result(mat.size(), std::vector<double>(mat[0].size()));

    for (size_t i = 0; i < mat.size(); ++i) {
        for (size_t j = 0; j < mat[0].size(); ++j) {
            result[i][j] = mat[i][j] + other.mat[i][j];
        }
    }
    return Matrix(result);
}

Matrix Matrix::subtract(const Matrix& other) const {
    if (mat.size() != other.mat.size() || mat[0].size() != other.mat[0].size()) {
        throw std::invalid_argument("Matrices must be of the same size for subtraction.");
    }

    matrix_t result(mat.size(), std::vector<double>(mat[0].size()));

    for (size_t i = 0; i < mat.size(); ++i) {
        for (size_t j = 0; j < mat[0].size(); ++j) {
            result[i][j] = mat[i][j] - other.mat[i][j];
        }
    }
    return Matrix(result);
}

void Matrix::display() const {
    for (const auto& row : mat) {
        for (double val : row) {
            std::cout << val << "\t";
        }
        std::cout << std::endl;
    }
}
