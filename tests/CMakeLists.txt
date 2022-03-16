cmake_minimum_required(VERSION 3.10)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# Main test runner executable
add_executable(TestRunner ${SOURCES})
target_include_directories(TestRunner PRIVATE thirdparty/doctest)
target_link_libraries(TestRunner PRIVATE ${JUST_ODE_TARGET_NAME})
install(TARGETS TestRunner DESTINATION bin)

# CTest integration
include(thirdparty/doctest/scripts/doctest.cmake)
doctest_discover_tests(TestRunner)
