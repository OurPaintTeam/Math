include(CTest)
enable_testing()

include(FetchContent)
if (NOT TARGET gtest)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG 03597a01ee50ed33e9dfd640b249b4be3799d395
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

option(ENABLE_MEMCHECK "Configure CTest memcheck with Valgrind" ON)
set(TEST_TIMEOUT_DEFAULT 60 CACHE STRING "Per-test timeout in seconds")

# Settings Valgrind for CTest (-T memcheck)
if (ENABLE_MEMCHECK)
  find_program(CTEST_MEMORYCHECK_COMMAND NAMES valgrind)
  set(CTEST_MEMORYCHECK_COMMAND_OPTIONS
      "--leak-check=full --error-exitcode=1"
      CACHE STRING "Valgrind options for ctest -T memcheck")
  set(MEMORYCHECK_COMMAND "${CTEST_MEMORYCHECK_COMMAND}")
  set(MEMORYCHECK_COMMAND_OPTIONS "${CTEST_MEMORYCHECK_COMMAND_OPTIONS}")
endif()

function(add_math_test name)
    add_executable(${name} ${src})
    target_link_libraries(${name} Math gtest gtest_main)
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES
      LABELS "unit"
      TIMEOUT ${TEST_TIMEOUT_DEFAULT}
    )
endfunction()

# Run tests
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS *.cc)
foreach(src ${TEST_SOURCES})
  get_filename_component(tname "${src}" NAME_WE)
  add_math_test(${tname})
endforeach()
