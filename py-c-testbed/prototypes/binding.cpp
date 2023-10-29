#include <pybind11/pybind11.h>
#include "message_lib.h"

namespace py = pybind11;

PYBIND11_MODULE(cppmodule, m) {
    m.doc() = "Python module with C++ binding";
    m.def("echo_message", &echoMessage, "A function to print a message from C++ to stdout");
    m.def("add_numbers", (int (*)(int, int)) &addNumbers, "A function to add two integers");
    m.def("add_numbers", (float (*)(float, float)) &addNumbers, "A function to add two floats");
}
