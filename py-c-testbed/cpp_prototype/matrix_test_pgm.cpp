#include "matrix.h"

int main() {
    Matrix A({{1, 2}, {3, 4}});
    Matrix B({{5, 6}, {7, 8}});

    std::cout << "Matrix A:" << std::endl;
    A.display();

    std::cout << "\nMatrix B:" << std::endl;
    B.display();

    Matrix C = A.add(B);
    Matrix D = A.subtract(B);

    std::cout << "Matrix A + B:" << std::endl;
    C.display();

    std::cout << "\nMatrix A - B:" << std::endl;
    D.display();

    return 0;
}
