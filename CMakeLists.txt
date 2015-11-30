cmake_minimum_required(VERSION 3.3)
project(mnist)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules/")

find_package(CImg REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

link_libraries(X11 pthread)

set(SOURCE_FILES Dataset.cpp NeuralNetwork.cpp load.cpp util.cpp)
add_executable(mnist main.cpp ${SOURCE_FILES})
add_executable(mnist_test test.cpp ${SOURCE_FILES})
