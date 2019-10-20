cmake_minimum_required(VERSION 3.0)

project(gcc-module CXX)

# find_package(PythonLibs REQUIRED)

# string(REGEX MATCH "[0-9]+\\.[0-9]+" PYTHON_SIMPLE_VER "${PYTHONLIBS_VERSION_STRING}")
# string(REPLACE "." "" PYTHON_SIMPLE_VER "${PYTHON_SIMPLE_VER}")

# find_package(Boost REQUIRED COMPONENTS "python${PYTHON_SIMPLE_VER}" "numpy${PYTHON_SIMPLE_VER}")

find_package(Torch REQUIRED)

# message(STATUS "*** dump start cmake variables ***")
# get_cmake_property(_variableNames VARIABLES)
# foreach(_variableName ${_variableNames})
#         message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()
# message(STATUS "*** dump end ***")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# include_directories("${PROJECT_SOURCE_DIR}" "${Boost_INCLUDE_DIRS}" "${PYTHON_INCLUDE_DIRS}" "${TORCH_INCLUDE_DIRS}")
# link_directories("${Boost_LIBRARY_DIRS}")
# 
# add_library(test_module SHARED module.cpp)
# set_target_properties(test_module PROPERTIES PREFIX "")
# if(WIN32)
#     set_target_properties(test_module PROPERTIES SUFFIX ".dll")
# else()
#     set_target_properties(test_module PROPERTIES SUFFIX ".so")
# endif()
# set_property(TARGET test_module PROPERTY CXX_STANDARD 14)
# target_link_libraries(test_module "${Boost_LIBRARIES}" "${PYTHON_LIBRARIES}" "${TORCH_LIBRARIES}")

add_executable(test_pytorch main.cpp)
include_directories("${PROJECT_SOURCE_DIR}" "${TORCH_INCLUDE_DIRS}")
target_link_libraries(test_pytorch "${TORCH_LIBRARIES}")
