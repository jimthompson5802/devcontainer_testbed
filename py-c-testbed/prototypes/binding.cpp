#include <pybind11/pybind11.h>
#include "message_lib.h"

namespace py = pybind11;

void pythonPrintMessage(const std::string &message) {
    printMessage(message);
}

PYBIND11_MODULE(cppmodule, m) {
    m.doc() = "Python module with C++ binding";
    m.def("print_message", &pythonPrintMessage, "A function to print a message from C++ to stdout");
}
