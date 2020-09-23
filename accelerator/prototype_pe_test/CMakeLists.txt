set (TEST_NAME "prototypePE")

set (PREPROCESSOR_DEFS
#        -DWEIGHT_MEMORY_TEST
        -DHW_DEBUG
#        -DDIRECT_COMPRESSION_SIMD
         -DPE_SYSTEM
         -DPE_PROTOTYPE_TEST
        )
add_definitions(
#   -DDIRECT_COMPRESSION_SIMD
    -DPE_SYSTEM
    -DPE_PROTOTYPE_TEST
    )

#########################
#Obtain the host source files
#########################
file(GLOB HOST_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${PROJECT_SOURCE_DIR}/common/src/floatFixedPointConversion.cpp
        ${PROJECT_SOURCE_DIR}/common/src/tensorCompression.cpp
)

###########################
#Add the include files
###########################
include_directories(
    ${PROJECT_SOURCE_DIR}/common/inc
    ${PROJECT_SOURCE_DIR}/device
    ${CMAKE_CURRENT_SOURCE_DIR}
)

##################################
#Add the RTL directories and libraries
##################################
set (rtl_directory ${PROJECT_SOURCE_DIR}/device/RTL
)

set (rtl_lib
    -l rtl_lib.aoclib)
INCLUDE_DIRECTORIES (${rtl_directory})


#######################################d
##Add extra library for the host below
######################################
#set (OpenCV_DIR /home/jamesliu/arm_opencv/Build CACHE STRING "Path of OpenCV" FORCE)
#find_package (OpenCV REQUIRED )

#include_directories(${OpenCV_INCLUDE_DIRS})
#add_definitions(-DUSE_OPENCV)

find_package (Boost 1.56 REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
#####################################
##Add unit test for the host library below
#####################################
add_definitions(-DFPGA_DEVICE)
add_executable (${TEST_NAME}_host ${HOST_SRC})
target_link_libraries(${TEST_NAME}_host ${Boost_LIBRARIES} gtest_main aocl_utils)
target_compile_definitions(
    ${TEST_NAME}_host
    PUBLIC
    ${PREPROCESSOR_DEFS}
    )

set_target_properties (${TEST_NAME}_host PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}")


####################################
##Generate the hardware targets
####################################

include (${PROJECT_SOURCE_DIR}/aoc.cmake)

# Find the kernel source files. Note, GLOB isn't the recommended way. One needs to touch the CMakesList.txt to effect changes
#file(GLOB_RECURSE kernel_files
##     ${PROJECT_SOURCE_DIR}/device/peComponents.cl
#     ${PROJECT_SOURCE_DIR}/device/sparse_pe_system.cl
#     ${CMAKE_CURRENT_SOURCE_DIR}/prototypePETest.cl
#     ${PROJECT_SOURCE_DIR}/device/device_utils.cl
#)

list (APPEND kernel_files
    ${PROJECT_SOURCE_DIR}/device/sparse_pe_system.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/prototypePETest.cl
    ${PROJECT_SOURCE_DIR}/device/device_utils.cl)

add_aoc_target (
    TARGET_NAME ${TEST_NAME}
    TARGET_TYPE EMULATION
    SOURCES_LIST ${kernel_files} 
    HEADER_DIR ${PROJECT_SOURCE_DIR}/device
    RTL_DIR ${rtl_directory}
    PREPROCESSOR_DEFS_LIST ${PREPROCESSOR_DEFS}
    RTL_LIB ${rtl_lib}
    BOARD_NAME $BOARD_NAME}
    )

add_aoc_target (
    TARGET_NAME ${TEST_NAME}
    TARGET_TYPE RTL_ONLY
    SOURCES_LIST ${kernel_files} 
    HEADER_DIR ${PROJECT_SOURCE_DIR}/device
    RTL_DIR ${rtl_directory} 
    PREPROCESSOR_DEFS_LIST ${PREPROCESSOR_DEFS}
    RTL_LIB ${rtl_lib}
    BOARD_NAME ${BOARD_NAME}
    )

add_aoc_target (
    TARGET_NAME ${TEST_NAME}
    TARGET_TYPE NORMAL_HW
    SOURCES_LIST ${kernel_files} 
    HEADER_DIR ${PROJECT_SOURCE_DIR}/device
    RTL_DIR ${rtl_directory} 
    PREPROCESSOR_DEFS_LIST ${PREPROCESSOR_DEFS}
    RTL_LIB ${rtl_lib}
    BOARD_NAME ${BOARD_NAME}
    )

add_aoc_target (
    TARGET_NAME ${TEST_NAME}
    TARGET_TYPE RELEASE_HW
    SOURCES_LIST ${kernel_files}
    HEADER_DIR ${PROJECT_SOURCE_DIR}/device
    RTL_DIR ${rtl_directory}
    PREPROCESSOR_DEFS_LIST ${PREPROCESSOR_DEFS}
    RTL_LIB ${rtl_lib}
    BOARD_NAME ${BOARD_NAME}
    )

add_aoc_target (
    TARGET_NAME ${TEST_NAME}
    TARGET_TYPE PROFILE_HW
    SOURCES_LIST ${kernel_files} 
    HEADER_DIR ${PROJECT_SOURCE_DIR}/device
    RTL_DIR ${rtl_directory} 
    PREPROCESSOR_DEFS_LIST ${PREPROCESSOR_DEFS}
    RTL_LIB ${rtl_lib}
    BOARD_NAME ${BOARD_NAME}
    )

add_aoc_target (
    TARGET_NAME ${TEST_NAME}
    TARGET_TYPE FAST_COMPILE_HW
    SOURCES_LIST ${kernel_files}
    HEADER_DIR ${PROJECT_SOURCE_DIR}/device
    RTL_DIR ${rtl_directory}
    PREPROCESSOR_DEFS_LIST ${PREPROCESSOR_DEFS}
    RTL_LIB ${rtl_lib}
    BOARD_NAME ${BOARD_NAME}
    )

#add_aoc_target (
#    TARGET_NAME ${TEST_NAME}
#    TARGET_TYPE SIMULATION
#    SOURCES_LIST ${kernel_files}
#    HEADER_DIR ${PROJECT_SOURCE_DIR}/device
#    RTL_DIR ${rtl_directory}
#    PREPROCESSOR_DEFS_LIST ${PREPROCESSOR_DEFS}
#    #RTL_LIB ${rtl_lib})
#    )




