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

# Settings Valgrind for CTest
if (ENABLE_MEMCHECK)
    find_program(MEMORYCHECK_COMMAND NAMES valgrind)
    if (NOT MEMORYCHECK_COMMAND)
        message(WARNING "ENABLE_MEMCHECK=ON, but valgrind not found; memcheck-tests will not configure")
    else()
        set(MEMORYCHECK_COMMAND_OPTIONS
            --leak-check=full
            --error-exitcode=1
            CACHE STRING "Valgrind options for memcheck tests"
        )
    endif()
endif()

function(add_math_test name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} Math gtest gtest_main)

    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES
      LABELS "unit"
      TIMEOUT ${TEST_TIMEOUT_DEFAULT}
    )

    if (ENABLE_MEMCHECK)
        set(memcheck_name "memcheck_${name}")
        add_test(NAME ${memcheck_name}
                COMMAND ${MEMORYCHECK_COMMAND} ${MEMORYCHECK_COMMAND_OPTIONS} $<TARGET_FILE:${name}>)
        set_tests_properties(${memcheck_name} PROPERTIES
                LABELS "unit;memcheck"
                TIMEOUT ${TEST_TIMEOUT_DEFAULT}
        )
    endif()
endfunction()

# Run tests
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS *.cc)
foreach(src ${TEST_SOURCES})
  get_filename_component(tname "${src}" NAME_WE)
  add_math_test(${tname} "${src}")
endforeach()
