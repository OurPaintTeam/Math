cmake_minimum_required(VERSION 3.20)

set(CMAKE_OBJECT_PATH_MAX 300 CACHE INTERNAL "")

project(Math)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG         3.4.0
)
FetchContent_MakeAvailable(eigen)
include_directories(${eigen_SOURCE_DIR})
# Отключаем тестирование и сборку примеров Eigen (меньше будет warnings)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

include_directories(headers)
include_directories(headers/optimizers)
include_directories(headers/optimizers/MatrixOptimizers)
include_directories(headers/optimizers/EigenOptimizers)
include_directories(headers/optimizers)
include_directories(headers/decomposition)
include_directories(headers/graph)
include_directories(headers/Tasks/Matrix)
include_directories(headers/Tasks/Eigen)
include_directories(headers/Tasks)


# Сборка основной библиотеки
add_library(Math
        src/LevenbergMarquardtSolver.cc
        src/QR.cc
        src/Function.cc
        src/ErrorFunctions.cc
        src/GradientOptimizer.cc
        src/NewtonOptimizer.cc
        src/NewtonGaussSolver.cc
        )

add_subdirectory(tests)

set(TYPE_OF_COMPILING BOOL 1)
if (TYPE_OF_COMPILING)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(Math PRIVATE -Wall -Wextra -Wpedantic)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        #target_compile_options(Math PRIVATE /W4)
    endif ()
endif ()

set(TYPE_OF_OPTIMIZATION BOOL 1)
if (TYPE_OF_OPTIMIZATION)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
    endif ()
endif ()
