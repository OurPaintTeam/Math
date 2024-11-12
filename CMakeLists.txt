cmake_minimum_required(VERSION 3.10)
project(MinimizerOptimizer)

# Укажите стандарт C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Добавьте папки с заголовочными файлами
include_directories(headers)
include_directories(headers/Optimizers)
include_directories(headers/decomposition)

# Сборка основной библиотеки
add_library(MinimizerOptimizer src/Function.cpp src/GradientOptimizer.cpp src/ErrorFunctions.cpp src/NewtonOptimizer.cpp)

# Сборка тестов
enable_testing()

# Подключение GoogleTest с использованием FetchContent
include(FetchContent)

FetchContent_Declare(
        gtest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.11.0  # Укажите последнюю стабильную версию или конкретный тег
)

# Загрузка GoogleTest
FetchContent_MakeAvailable(gtest)

# Добавляем исполняемые файлы тестов
add_executable(FunctionTest tests/FunctionTest.cpp)
target_link_libraries(FunctionTest MinimizerOptimizer gtest gtest_main)

add_executable(OptimizationTaskTest tests/TaskTest.cpp)
target_link_libraries(OptimizationTaskTest MinimizerOptimizer gtest gtest_main)

add_executable(GradientOptimizerTest tests/GradientOptimizerTest.cpp)
target_link_libraries(GradientOptimizerTest MinimizerOptimizer gtest gtest_main)

add_executable(NewtonOptimizerTest tests/NewtonOptimizerTest.cpp)
target_link_libraries(NewtonOptimizerTest MinimizerOptimizer gtest gtest_main)


add_executable(LSMTaskTest tests/LSMTaskTest.cpp)
target_link_libraries(LSMTaskTest MinimizerOptimizer gtest gtest_main)

add_executable(MatrixTest tests/TestsMatrix.cpp)
target_link_libraries(MatrixTest MinimizerOptimizer gtest gtest_main)

add_executable(QRTest tests/TestsQR.cpp)
target_link_libraries(QRTest MinimizerOptimizer gtest gtest_main)

add_executable(ErrorFunctionTest tests/ErrorFunctionsTests.cpp)
target_link_libraries(ErrorFunctionTest MinimizerOptimizer gtest gtest_main)
# Запускаем тесты
add_test(NAME FunctionTests COMMAND FunctionTest)
add_test(NAME OptimizationTaskTest COMMAND OptimizationTaskTest)
add_test(NAME GradientOptimizerTests COMMAND GradientOptimizerTest)
add_test(NAME NewtonOptimizerTests COMMAND NewtonOptimizerTest)
add_test(NAME MatrixTest COMMAND MatrixTest)
add_test(NAME QRTest COMMAND QRTest)