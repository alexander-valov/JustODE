cmake_minimum_required(VERSION 3.10)

project(
    RungeKutta
    VERSION 0.0.2
    LANGUAGES CXX
)

# ---------------------------------------------------------------
# Check if RungeKutta is built as subproject or it's main project
# ---------------------------------------------------------------
set(RUNGE_KUTTA_MAIN_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(RUNGE_KUTTA_MAIN_PROJECT ON)
endif()

# ---------------------------------------------------------------
# Define configuration
# ---------------------------------------------------------------
set(RUNGE_KUTTA_TARGET_NAME ${PROJECT_NAME})
set(RUNGE_KUTTA_INCLUDE_BUILD_DIR "${PROJECT_SOURCE_DIR}/include")

# ---------------------------------------------------------------
# Default build type
# ---------------------------------------------------------------
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message("  ${RUNGE_KUTTA_TARGET_NAME}: Using default value of CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
else()
    message("  ${RUNGE_KUTTA_TARGET_NAME}: Obtained CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
endif()

# ---------------------------------------------------------------
# Add INTERFACE library
# ---------------------------------------------------------------
add_library(${RUNGE_KUTTA_TARGET_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${RUNGE_KUTTA_TARGET_NAME} ALIAS ${RUNGE_KUTTA_TARGET_NAME})
target_compile_features(${RUNGE_KUTTA_TARGET_NAME} INTERFACE cxx_std_17)
target_include_directories(
    ${RUNGE_KUTTA_TARGET_NAME}
    INTERFACE
    $<BUILD_INTERFACE:${RUNGE_KUTTA_INCLUDE_BUILD_DIR}>
    $<INSTALL_INTERFACE:include>
)

add_subdirectory(thirdparty/matplotplusplus)
add_subdirectory(src)