cmake_minimum_required (VERSION 2.6)
project (MetricLearning)

set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")
include(cotire)

list( APPEND CMAKE_CXX_FLAGS 
	"-std=c++1y -O3 -Wall -Wextra -Wno-unused-parameter ${CMAKE_CXX_FLAGS} ")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)

include_directories(/usr/local/include/eigen3)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(utils)
add_subdirectory(constraint)
add_subdirectory(gaussian)
add_subdirectory(emkmeans)
add_subdirectory(pckmeans)
add_subdirectory(mpckmeans)
add_subdirectory(globalMetric)

set(MAIN_SRCS
	ml.cpp)

add_executable (ml ${MAIN_SRCS})
target_link_libraries(ml propertyutil pckmeans mpckmeans globalMetricKMeans)
cotire(ml)

set_target_properties(ml PROPERTIES COTIRE_CXX_PRECOMPILED_HEADER
	"/home/vvminh/git/dml/metriclearning/src/utils/Eigen3.h.gch")

get_target_property(_precompiledHeader ml COTIRE_CXX_PRECOMPILED_HEADER)
message(STATUS "Eigen3 precompiled header: ${_precompiledHeader}")
