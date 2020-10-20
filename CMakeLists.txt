cmake_minimum_required(VERSION 2.8)

project(RK45)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/Installation")

set (CMAKE_BUILD_TYPE Debug)
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -m64 -Wall -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -Wall -pedantic")

add_executable(RK45 main.cpp)

target_include_directories (RK45 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(TARGETS RK45 DESTINATION bin)