#include <torch/extension.h>
#include <iostream>
#include <math.h>
#include "nms.cpp"
#include "overlap.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cpu_overlap", &cpu_overlap, "box cpu_overlap");
    m.def("cpu_nms", &cpu_nms, "box cpu_nms");
}
