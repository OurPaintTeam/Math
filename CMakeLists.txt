cmake_minimum_required(VERSION 3.10)
project(MinimizerOptimizer)

# Укажите стандарт C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Добавьте папки с заголовочными файлами
include_directories(headers)
include_directories(headers/Optimizers)

# Сборка основной библиотеки
add_library(MinimizerOptimizer src/Function.cpp src/GradientOptimizer.cpp src/NewtonOptimizer.cpp)

# Сборка тестов
enable_testing()

# Подключение GoogleTest с использованием FetchContent
include(FetchContent)

FetchContent_Declare(
        gtest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.11.0 # Укажите последнюю стабильную версию или конкретный тег
)

# Загрузка GoogleTest
FetchContent_MakeAvailable(gtest)

# Добавляем исполняемый файл тестов
add_executable(FunctionTest tests/FunctionTest.cpp)

# Линкуем тесты с библиотекой и GoogleTest
target_link_libraries(FunctionTest MinimizerOptimizer gtest gtest_main)

# Запускаем тесты
add_test(NAME FunctionTests COMMAND FunctionTest)

#Настройка для тестов OptimizationTask
add_executable(OptimizationTaskTest tests/TaskTest.cpp)

# Линкуем тесты с библиотекой и GoogleTest
target_link_libraries(OptimizationTaskTest MinimizerOptimizer gtest gtest_main)

# Запускаем тесты
add_test(NAME OptimizationTaskTest COMMAND OptimizationTaskTest)

#Настройка для тестов GradientOptimizer
add_executable(GradientOptimizerTest tests/GradientOptimizerTest.cpp)

# Линкуем тесты с библиотекой и GoogleTest
target_link_libraries(GradientOptimizerTest MinimizerOptimizer gtest gtest_main)

# Запускаем тесты
add_test(NAME GradientOptimizerTests COMMAND GradientOptimizerTest)

# Тест для NewtonOptimizer
add_executable(NewtonOptimizerTest tests/NewtonOptimizerTest.cpp)

# Линкуем тесты с библиотекой и GoogleTest
target_link_libraries(NewtonOptimizerTest MinimizerOptimizer gtest gtest_main)

# Запускаем тесты
add_test(NAME NewtonOptimizerTests COMMAND NewtonOptimizerTest)

# Тест для LSMTask
add_executable(LSMTaskTest tests/LSMTaskTest.cpp)

# Линкуем тесты с библиотекой и GoogleTest
target_link_libraries(LSMTaskTest MinimizerOptimizer gtest gtest_main)

# Запускаем тесты
add_test(NAME LSMTaskTests COMMAND LSMTaskTest)
