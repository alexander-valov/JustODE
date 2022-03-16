cmake_minimum_required(VERSION 3.10)

project(
    JustODE
    VERSION 0.0.2
    LANGUAGES CXX
)

# ---------------------------------------------------------------
# Check if JustODE is built as subproject or it's main project
# ---------------------------------------------------------------
set(JUST_ODE_MAIN_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(JUST_ODE_MAIN_PROJECT ON)
endif()

# ---------------------------------------------------------------
# Define options
# ---------------------------------------------------------------
if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()

option(JUST_ODE_BUILD_TESTS "Build the unit tests" ${JUST_ODE_MAIN_PROJECT})
option(JUST_ODE_SINGLE_HEADER "Use single-header version" OFF)

# ---------------------------------------------------------------
# Define configuration
# ---------------------------------------------------------------
set(JUST_ODE_TARGET_NAME ${PROJECT_NAME})
if (JUST_ODE_SINGLE_HEADER)
    set(JUST_ODE_INCLUDE_BUILD_DIR "${PROJECT_SOURCE_DIR}/include_single")
    message("  ${JUST_ODE_TARGET_NAME}: Using single-header version")
else()
    set(JUST_ODE_INCLUDE_BUILD_DIR "${PROJECT_SOURCE_DIR}/include")
    message("  ${JUST_ODE_TARGET_NAME}: Using multi-header version")
endif()

# ---------------------------------------------------------------
# Default build type
# ---------------------------------------------------------------
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message("  ${JUST_ODE_TARGET_NAME}: Using default value of CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
else()
    message("  ${JUST_ODE_TARGET_NAME}: Obtained CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
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