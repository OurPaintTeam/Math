cmake_minimum_required(VERSION 3.10)

project(MinimizerOptimizer VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(min main.cpp MinimizeTask.cpp Optimizers/GradientMethod.cpp Optimizers/LeastSquereMethod.cpp Optimizers/LeastSquereMethod.cpp)