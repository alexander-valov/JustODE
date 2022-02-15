cmake_minimum_required(VERSION 3.10)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(RKMain ${SOURCES})
target_link_libraries(RKMain PRIVATE ${JUST_ODE_TARGET_NAME} matplot)
install(TARGETS RKMain DESTINATION bin)