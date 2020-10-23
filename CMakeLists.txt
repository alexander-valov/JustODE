cmake_minimum_required(VERSION 2.8)

cmake_policy(SET CMP0048 NEW)
project(RungeKutta VERSION 0.0.2)

enable_testing()

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/Installation")

set (CMAKE_BUILD_TYPE Debug)
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -m64 -Wall -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -Wall -pedantic")

add_executable(RungeKutta main.cpp)

target_include_directories (RungeKutta PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(TARGETS RungeKutta DESTINATION bin)

add_subdirectory(Tests)