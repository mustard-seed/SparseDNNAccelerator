cmake_minimum_required (VERSION 3.1)

set (PROJECT_NAME "sparseDNNAccel")

project (${PROJECT_NAME})

set (CMAKE_BUILD_TYPE "Debug")
add_subdirectory(googletest)
add_subdirectory(weight_memory_test)
add_subdirectory(prototype_pe_test)
add_subdirectory(toy_ping_pong)


