#include <boost/python.hpp>
#include "matrix.h"

using namespace boost::python;

BOOST_PYTHON_MODULE(matrixboostlib)
{
    Py_Initialize();
    
    class_<Matrix>("Matrix", init<const Matrix::matrix_t &>())
        .def("add", &Matrix::add)
        .def("subtract", &Matrix::subtract)
        .def("display", &Matrix::display);
}
