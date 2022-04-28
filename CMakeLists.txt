cmake_minimum_required(VERSION 3.10)

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/scripts/version.txt VERSION)

project(
    JustODE
    VERSION ${VERSION}
    LANGUAGES CXX
)

message("\n=======================================================")
message("Configuring JustODE")
message("-------------------------------------------------------")
message("-- Version: ${VERSION}")

# ---------------------------------------------------------------
# Check if JustODE is built as subproject or it's main project
# ---------------------------------------------------------------
set(JUST_ODE_MAIN_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(JUST_ODE_MAIN_PROJECT ON)
endif()

message("-- Is JustODE the main project: ${JUST_ODE_MAIN_PROJECT}")

# ---------------------------------------------------------------
# Define options
# ---------------------------------------------------------------
if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()

option(JUST_ODE_BUILD_TESTS "Build the unit tests" ${JUST_ODE_MAIN_PROJECT})
option(JUST_ODE_SINGLE_HEADER "Use single-header version" OFF)

message("-- Is tests be built: ${JUST_ODE_BUILD_TESTS}")

# ---------------------------------------------------------------
# Define configuration
# ---------------------------------------------------------------
set(JUST_ODE_TARGET_NAME ${PROJECT_NAME})
if (JUST_ODE_SINGLE_HEADER)
    set(JUST_ODE_INCLUDE_BUILD_DIR "${PROJECT_SOURCE_DIR}/include_single")
    message("-- Using single-header version")
else()
    set(JUST_ODE_INCLUDE_BUILD_DIR "${PROJECT_SOURCE_DIR}/include")
    message("-- Using multi-header version")
endif()

# ---------------------------------------------------------------
# Default build type
# ---------------------------------------------------------------
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message("-- Using default value of CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
else()
    message("-- Obtained CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
endif()

# ---------------------------------------------------------------
# Add INTERFACE library
# ---------------------------------------------------------------
add_library(${JUST_ODE_TARGET_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${JUST_ODE_TARGET_NAME} ALIAS ${JUST_ODE_TARGET_NAME})
target_compile_features(${JUST_ODE_TARGET_NAME} INTERFACE cxx_std_17)
target_include_directories(
    ${JUST_ODE_TARGET_NAME}
    INTERFACE
    $<BUILD_INTERFACE:${JUST_ODE_INCLUDE_BUILD_DIR}>
    $<INSTALL_INTERFACE:include>
)

# ---------------------------------------------------------------
# Buld unit tests
# ---------------------------------------------------------------
if (JUST_ODE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message("=======================================================\n")