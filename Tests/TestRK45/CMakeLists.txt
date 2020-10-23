cmake_minimum_required (VERSION 2.6)

# Find all *.cpp files in ${CMAKE_CURRENT_SOURCE_DIR}/src/ and put them in SOURCES variable
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# Include folder needed to compile this executable
# Compiler will search headers in this folder only during this executable build
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)  

# Compile executable from source files in SOURCES variable
add_executable(TestRK45 ${SOURCES})         

# Libraries needed by this executable
# target_link_libraries(<name> <library_to_link1> <library_to_link2> ...)
# ${CMAKE_STATIC_LIBRARY_PREFIX} - contains library prefix which may be different on different platforms
# ${CMAKE_STATIC_LIBRARY_SUFFIX} - contains library suffix which may be different on different platforms
target_link_libraries(TestRK45 gtest_main)

# Include directories connected with this library
# Compiler will search headers in this folder when other library or executable is going to link with this library
#target_include_directories(HeatEquationParallelUnitTest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Where to install this library when "make install" command issued
install (TARGETS TestRK45 DESTINATION bin/Tests)
