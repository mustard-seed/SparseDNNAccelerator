cmake_minimum_required (VERSION 3.1)

set (PROJECT_NAME "sparseDNNAccel")

project (${PROJECT_NAME})

set (CMAKE_BUILD_TYPE "Debug")
add_subdirectory(googletest)

#============================
# YAML-CPP
#============================
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "disable yaml tests")
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "disable yaml tools")
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "disable yaml contrib")
add_subdirectory(yaml-cpp)
add_subdirectory(common)
add_subdirectory(model_container)
add_subdirectory(accelerator_wrapper)
add_subdirectory(graph_factory)
add_subdirectory(prototype_pe_test)
add_subdirectory(flexible_bitmask_compression_test)
add_subdirectory(full_system)

