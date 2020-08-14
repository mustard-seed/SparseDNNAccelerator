include (${PROJECT_SOURCE_DIR}/config.cmake)

set (LIBRARY_NAME "accelerator_wrapper")
############################################
# Find Intel FPGA OpenCL SDK
#############################################
file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/common/src/*.cpp
    )

find_package (Boost 1.56 REQUIRED)

add_definitions(-DFPGA_DEVICE)
add_library (${LIBRARY_NAME} ${SOURCES})
target_link_libraries(${LIBRARY_NAME} ${Boost_LIBRARIES} gtest_main aocl_utils)
target_include_directories(${LIBRARY_NAME}
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/common/inc
        ${Boost_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/device/RTL
    )
target_compile_definitions(
    ${LIBRARY_NAME}
    PUBLIC
    ${PREPROCESSOR_DEFS}
    )

set_target_properties (${LIBRARY_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}")

add_executable(accelerator_wrapper_test ${CMAKE_CURRENT_SOURCE_DIR}/accelerator_wrapper_test.cpp)
target_link_libraries(accelerator_wrapper_test ${LIBRARY_NAME})
set_target_properties (accelerator_wrapper_test PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}")

