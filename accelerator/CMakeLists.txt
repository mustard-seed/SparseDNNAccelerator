cmake_minimum_required (VERSION 3.1)

set (PROJECT_NAME "sparseDNNAccel")

project (${PROJECT_NAME})

include (${PROJECT_SOURCE_DIR}/config.cmake)

add_subdirectory(googletest)

#============================
# YAML-CPP Git submodule
#============================
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "disable yaml tests")
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "disable yaml tools")
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "disable yaml contrib")
add_subdirectory(yaml-cpp)
#===========================
# CNPY Git submodule
#==========================
add_subdirectory (cnpy)
add_subdirectory(common)
add_subdirectory(model_container)
add_subdirectory(accelerator_wrapper)
add_subdirectory(graph_factory)
add_subdirectory(prototype_pe_test)
add_subdirectory(flexible_bitmask_compression_test)
add_subdirectory(full_system)
add_subdirectory(spw_pe_test)
