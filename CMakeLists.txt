cmake_minimum_required(VERSION 3.20)

set(CMAKE_OBJECT_PATH_MAX 300 CACHE INTERNAL "")

project(Math)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_LEAVE_TEST_IN_ALL OFF CACHE BOOL "" FORCE)

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

set(MATH_CORE_SOURCES
        src/core/Function.cc
        src/core/ErrorFunctions.cc
)

set(MATH_DECOMPOSITION_SOURCES
        src/decomposition/QR.cc
        src/decomposition/SparseQR.cc
)

set(MATH_CORE_HEADERS
        headers/core/Function.h
        headers/core/ErrorFunction.h
)

set(MATH_LINEAR_ALGEBRA_HEADERS
        headers/linear_algebra/Matrix.h
        headers/linear_algebra/SparseMatrix.h
)

set(MATH_DECOMPOSITION_HEADERS
        headers/decomposition/QR.h
        headers/decomposition/SparseQR.h
        headers/decomposition/ordering/SparseQrColamd.h
)

set(MATH_GRAPH_HEADERS
        headers/graph/Graph.h
        headers/graph/GraphObjects.h
        headers/graph/InheritanceGraph.h
        headers/graph/Politicians.h
)

set(MATH_OPTIMIZER_BASE_HEADERS
        headers/optimizers/base/Optimizer.h
)

set(MATH_MATRIX_OPTIMIZER_SOURCES
        src/optimizers/matrix/GradientOptimizer.cc
        src/optimizers/matrix/NewtonOptimizer.cc
        src/optimizers/matrix/NewtonGaussSolver.cc
        src/optimizers/matrix/LevenbergMarquardtSolver.cc
        src/optimizers/matrix/sparse/SparseLevenbergMarquardtSolver.cc
)

set(MATH_MATRIX_OPTIMIZER_HEADERS
        headers/optimizers/matrix/MatrixOptimizer.h
        headers/optimizers/matrix/GradientOptimizer.h
        headers/optimizers/matrix/NewtonOptimizer.h
        headers/optimizers/matrix/NewtonGaussSolver.h
        headers/optimizers/matrix/LevenbergMarquardtSolver.h
        headers/optimizers/matrix/sparse/SparseLevenbergMarquardtSolver.h
)

set(MATH_EIGEN_OPTIMIZER_SOURCES
        src/optimizers/eigen/StochasticGradientOptimizer.cc
        src/optimizers/eigen/AdamOptimizer.cc
)

set(MATH_EIGEN_OPTIMIZER_HEADERS
        headers/optimizers/eigen/EigenOptimizer.h
        headers/optimizers/eigen/AdamOptimizer.h
        headers/optimizers/eigen/StochasticGradientOptimizer.h
        headers/optimizers/eigen/LMForTest.h
        headers/optimizers/eigen/LMWithSparse.h
)

set(MATH_TASK_BASE_HEADERS
        headers/tasks/base/Task.h
)

set(MATH_MATRIX_TASK_HEADERS
        headers/tasks/matrix/TaskMatrix.h
        headers/tasks/matrix/TaskF.h
        headers/tasks/matrix/LSMTask.h
        headers/tasks/matrix/SparseLSMTask.h
)

set(MATH_EIGEN_TASK_HEADERS
        headers/tasks/eigen/TaskEigen.h
        headers/tasks/eigen/LSMFORLMTask.h
)

add_library(Math
        ${MATH_CORE_SOURCES}
        ${MATH_DECOMPOSITION_SOURCES}
        ${MATH_MATRIX_OPTIMIZER_SOURCES}
        ${MATH_EIGEN_OPTIMIZER_SOURCES}
)

target_sources(Math PRIVATE
        ${MATH_CORE_HEADERS}
        ${MATH_LINEAR_ALGEBRA_HEADERS}
        ${MATH_DECOMPOSITION_HEADERS}
        ${MATH_GRAPH_HEADERS}
        ${MATH_OPTIMIZER_BASE_HEADERS}
        ${MATH_MATRIX_OPTIMIZER_HEADERS}
        ${MATH_EIGEN_OPTIMIZER_HEADERS}
        ${MATH_TASK_BASE_HEADERS}
        ${MATH_MATRIX_TASK_HEADERS}
        ${MATH_EIGEN_TASK_HEADERS}
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src" PREFIX "Source Files" FILES
        ${MATH_CORE_SOURCES}
        ${MATH_DECOMPOSITION_SOURCES}
        ${MATH_MATRIX_OPTIMIZER_SOURCES}
        ${MATH_EIGEN_OPTIMIZER_SOURCES}
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/headers" PREFIX "Header Files" FILES
        ${MATH_CORE_HEADERS}
        ${MATH_LINEAR_ALGEBRA_HEADERS}
        ${MATH_DECOMPOSITION_HEADERS}
        ${MATH_GRAPH_HEADERS}
        ${MATH_OPTIMIZER_BASE_HEADERS}
        ${MATH_MATRIX_OPTIMIZER_HEADERS}
        ${MATH_EIGEN_OPTIMIZER_HEADERS}
        ${MATH_TASK_BASE_HEADERS}
        ${MATH_MATRIX_TASK_HEADERS}
        ${MATH_EIGEN_TASK_HEADERS}
)

target_include_directories(Math
    PUBLIC
        headers
        headers/core
        headers/linear_algebra
        headers/optimizers
        headers/optimizers/base
        headers/optimizers/matrix
        headers/optimizers/eigen
        headers/decomposition
        headers/decomposition/ordering
        headers/graph
        headers/tasks
        headers/tasks/base
        headers/tasks/matrix
        headers/tasks/eigen
)

target_link_libraries(Math
    PUBLIC Eigen3::Eigen
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
