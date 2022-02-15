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
# Define configuration
# ---------------------------------------------------------------
set(JUST_ODE_TARGET_NAME ${PROJECT_NAME})
set(JUST_ODE_INCLUDE_BUILD_DIR "${PROJECT_SOURCE_DIR}/include")

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

add_subdirectory(thirdparty/matplotplusplus)
add_subdirectory(src)