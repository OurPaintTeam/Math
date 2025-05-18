cmake_minimum_required(VERSION 3.24)
# Увеличиваем максимальную длину путей к объектным файлам (меньше будет warnings)
set(CMAKE_OBJECT_PATH_MAX 300 CACHE INTERNAL "")

project(Math)

# Укажите стандарт C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Используем FetchContent для загрузки Eigen
include(FetchContent)

# Отключаем глобально сборку тестов и примеров (меньше будет warnings)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Загрузка Eigen
FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG         3.4.0
)


# Отключаем тестирование и сборку примеров Eigen (меньше будет warnings)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Загрузка Eigen, если он не был загружен
FetchContent_MakeAvailable(eigen)

# Убедитесь, что директория Eigen подключена
include_directories(${eigen_SOURCE_DIR})
# Добавьте папки с заголовочными файлами
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

if (NOT TARGET gtest)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG 03597a01ee50ed33e9dfd640b249b4be3799d395
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Сборка тестов
enable_testing()


# Added executable tests files
add_executable(MatrixTest tests/TestsMatrix.cc)
target_link_libraries(MatrixTest Math gtest gtest_main)

add_executable(QRTest tests/TestsQR.cc)
target_link_libraries(QRTest Math gtest gtest_main)

add_executable(QRPerformanceTest tests/QRPerformanceTests.cc)
target_link_libraries(QRPerformanceTest Math gtest gtest_main)

add_executable(FunctionTest tests/FunctionTest.cc)
target_link_libraries(FunctionTest Math gtest gtest_main)

add_executable(OptimizationTaskTest tests/TaskTest.cc)
target_link_libraries(OptimizationTaskTest Math gtest gtest_main)

add_executable(GradientOptimizerTest tests/GradientOptimizerTest.cc)
target_link_libraries(GradientOptimizerTest Math gtest gtest_main)

add_executable(NewtonOptimizerTest tests/NewtonGaussSolverTests.cc)
target_link_libraries(NewtonOptimizerTest Math gtest gtest_main)

add_executable(NewtonGaussSolverTests tests/NewtonGaussSolverTests.cc)
target_link_libraries(NewtonGaussSolverTests Math gtest gtest_main)

add_executable(LSMTaskTest tests/LSMTaskTest.cc)
target_link_libraries(LSMTaskTest Math gtest gtest_main)

add_executable(ErrorFunctionTest tests/ErrorFunctionsTests.cc)
target_link_libraries(ErrorFunctionTest Math gtest gtest_main)

add_executable(LMTest tests/LMTest.cc)
target_link_libraries(LMTest Math gtest gtest_main)

add_executable(LMTestWithOurMatrix tests/OurLMTest.cc)
target_link_libraries(LMTestWithOurMatrix Math gtest gtest_main)

add_executable(SimpleGraph tests/graphgtests.cc)
target_link_libraries(SimpleGraph gtest gtest_main)

# Run tests
add_test(NAME FunctionTests COMMAND FunctionTest)
add_test(NAME OptimizationTaskTest COMMAND OptimizationTaskTest)
add_test(NAME GradientOptimizerTests COMMAND GradientOptimizerTest)
add_test(NAME NewtonOptimizerTests COMMAND NewtonOptimizerTest)
add_test(NAME NewtonGaussSolverTests COMMAND NewtonGaussSolverTests)
add_test(NAME MatrixTest COMMAND MatrixTest)
add_test(NAME QRTest COMMAND QRTest)
add_test(NAME ErrorFunctionTest COMMAND ErrorFunctionTest)
add_test(NAME LMTest COMMAND LMTest)
add_test(NAME LMTestWithOurMatrix COMMAND LMTestWithOurMatrix)
add_test(NAME SimpleGraph COMMAND SimpleGraph)
