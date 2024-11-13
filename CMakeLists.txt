cmake_minimum_required(VERSION 3.10)
project(MinimizerOptimizer)

# Укажите стандарт C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Используем FetchContent для загрузки Eigen
include(FetchContent)

# Загрузка Eigen
FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.3  # Выберите версию, которую хотите использовать
)

# Загрузка Eigen, если он не был загружен
FetchContent_MakeAvailable(eigen)

# Убедитесь, что директория Eigen подключена
include_directories(${eigen_SOURCE_DIR})
# Добавьте папки с заголовочными файлами
include_directories(headers)
include_directories(headers/Optimizers)
include_directories(headers/decomposition)

# Сборка основной библиотеки
add_library(MinimizerOptimizer src/LevenbergMarquardtSolver.cpp src/QR.cpp src/Function.cpp src/ErrorFunctions.cpp src/GradientOptimizer.cpp src/NewtonOptimizer.cpp)

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
add_executable(MatrixTest tests/TestsMatrix.cpp)
target_link_libraries(MatrixTest MinimizerOptimizer gtest gtest_main)

add_executable(QRTest tests/TestsQR.cpp)
target_link_libraries(QRTest MinimizerOptimizer gtest gtest_main)

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

add_executable(ErrorFunctionTest tests/ErrorFunctionsTests.cpp)
target_link_libraries(ErrorFunctionTest MinimizerOptimizer gtest gtest_main)

add_executable(LMTest tests/LMTest.cpp)
target_link_libraries(LMTest MinimizerOptimizer gtest gtest_main)

add_executable(LMTestWithOurMatrix tests/OurLMTest.cpp)
target_link_libraries(LMTestWithOurMatrix MinimizerOptimizer gtest gtest_main)
# Запускаем тесты
add_test(NAME FunctionTests COMMAND FunctionTest)
add_test(NAME OptimizationTaskTest COMMAND OptimizationTaskTest)
add_test(NAME GradientOptimizerTests COMMAND GradientOptimizerTest)
add_test(NAME NewtonOptimizerTests COMMAND NewtonOptimizerTest)
add_test(NAME MatrixTest COMMAND MatrixTest)
add_test(NAME QRTest COMMAND QRTest)
add_test(NAME ErrorFunctionTest COMMAND ErrorFunctionTest)
add_test(NAME LMTest COMMAND LMTest)
add_test(NAME LMTestWithOurMatrix COMMAND LMTestWithOurMatrix)