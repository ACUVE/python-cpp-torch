#include <iostream>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <torch/script.h>

namespace p = boost::python;
namespace np = boost::python::numpy;

p::object obj{};

p::object greet(np::ndarray arr)
{
    p::object tmp = obj;
    obj = arr;
    return tmp;
}

BOOST_PYTHON_MODULE(test_module)
{
    np::initialize();
    p::def("greet", greet);
}
