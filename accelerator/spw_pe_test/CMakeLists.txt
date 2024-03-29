set (TEST_NAME "SpWPETest")

set (PREPROCESSOR_DEFS
#        -DWEIGHT_MEMORY_TEST
#        -DHW_DEBUG
        -DSPW_TEST
        -DSPW_SYSTEM
        )

add_definitions(
#   -DDIRECT_COMPRESSION_SIMD
    -DSPW_TEST
    -DSPW_SYSTEM
    )

############################################
#First, we must be able to find Intel FPGA OpenCL SDK
#############################################
if(DEFINED ENV{INTELFPGAOCLSDKROOT})
else()
    message(FATAL_ERROR "The environment variable INTELFPGAOCLSDKROOT isn't set.")
endif()

###########################################################
#Intel FPGA OpenCL SDK Compile and linkage configurations
###########################################################
if (BOARD_NAME MATCHES "DE10Standard")
    ##Add library directories for DE10Standard
    message (STATUS "Linking against DE10-Standard libraries")
    message (STATUS "$ENV{INTELFPGAOCLSDKROOT}/host/arm32/lib")
    message (STATUS "$ENV{INTELFPGAOCLSDKROOT}/board/de10_standard/arm32/lib")
    file (GLOB intelfpga_host_libraries
        $ENV{INTELFPGAOCLSDKROOT}/host/arm32/lib/*.so
        )
    file (GLOB intelfpga_device_libraries
        $ENV{INTELFPGAOCLSDKROOT}/board/de10_standard/arm32/lib/*.so
        )
    add_definitions(-DC5SOC)
    list (APPEND PREPROCESSOR_DEFS -DC5SOC)
elseif (BOARD_NAME MATCHES "A10REF")
    ##Add library directories for a10_ref
    add_definitions(-DARRIA10)
    list (APPEND PREPROCESSOR_DEFS -DARRIA10)
    message (STATUS "Linking against A10-Ref libraries")
elseif (BOARD_NAME MATCHES "A10PAC")
    add_definitions(-DARRIA10 -DA10PAC)
    ##Add library directories for a10_pac
    list (APPEND PREPROCESSOR_DEFS -DARRIA10)
    message (STATUS "Linking against A10-Pac libraries")
elseif (BOARD_NAME MATCHES "DE5ANET")
    add_definitions(-DARRIA10)
    ##Add library directories for de5a_net
    list (APPEND PREPROCESSOR_DEFS -DARRIA10)
    message (STATUS "Linking against DE5A-NET libraries")
else()
    message (STATUS "The board is ${BOARD_NAME}")
    message (FATAL_ERROR "Unsupported board!")
endif()

#########################
#Obtain the host source files
#########################
file(GLOB HOST_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
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
target_include_directories(${TEST_NAME}_host
    PUBLIC
    ${PROJECT_SOURCE_DIR}/device
    ${CMAKE_CURRENT_SOURCE_DIR}
    )
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
##     ${PROJECT_SOURCE_DIR}/device/simple_weight_memory_test.cl
#     ${PROJECT_SOURCE_DIR}/device/sparse_pe_system.cl
#     ${PROJECT_SOURCE_DIR}/device/device_utils.cl
#)
list (APPEND kernel_files
    ${CMAKE_CURRENT_SOURCE_DIR}/spw_pe_test_harness.cl
    ${PROJECT_SOURCE_DIR}/device/pe.cl)

list (APPEND header_dirs
    ${PROJECT_SOURCE_DIR}/device
    ${CMAKE_CURRENT_SOURCE_DIR})

add_aoc_target (
    TARGET_NAME ${TEST_NAME}
    TARGET_TYPE EMULATION
    SOURCES_LIST ${kernel_files} 
    HEADER_DIR_LIST ${header_dirs}
    RTL_DIR ${rtl_directory}
    PREPROCESSOR_DEFS_LIST ${PREPROCESSOR_DEFS}
    RTL_LIB ${rtl_lib}
    BOARD_NAME ${BOARD_NAME}
    )

add_aoc_target (
    TARGET_NAME ${TEST_NAME}
    TARGET_TYPE RTL_ONLY
    SOURCES_LIST ${kernel_files} 
    HEADER_DIR_LIST ${header_dirs}
    RTL_DIR ${rtl_directory} 
    PREPROCESSOR_DEFS_LIST ${PREPROCESSOR_DEFS}
    RTL_LIB ${rtl_lib}
    BOARD_NAME ${BOARD_NAME}
    )

add_aoc_target (
    TARGET_NAME ${TEST_NAME}
    TARGET_TYPE NORMAL_HW
    SOURCES_LIST ${kernel_files} 
    HEADER_DIR_LIST ${header_dirs}
    RTL_DIR ${rtl_directory} 
    PREPROCESSOR_DEFS_LIST ${PREPROCESSOR_DEFS}
    RTL_LIB ${rtl_lib}
    BOARD_NAME ${BOARD_NAME}
    )

add_aoc_target (
    TARGET_NAME ${TEST_NAME}
    TARGET_TYPE RELEASE_HW
    SOURCES_LIST ${kernel_files}
    HEADER_DIR_LIST ${header_dirs}
    RTL_DIR ${rtl_directory}
    PREPROCESSOR_DEFS_LIST ${PREPROCESSOR_DEFS}
    RTL_LIB ${rtl_lib}
    BOARD_NAME ${BOARD_NAME}
    )

add_aoc_target (
    TARGET_NAME ${TEST_NAME}
    TARGET_TYPE PROFILE_HW
    SOURCES_LIST ${kernel_files} 
    HEADER_DIR_LIST ${header_dirs}
    RTL_DIR ${rtl_directory} 
    PREPROCESSOR_DEFS_LIST ${PREPROCESSOR_DEFS}
    RTL_LIB ${rtl_lib}
    )

add_aoc_target (
    TARGET_NAME ${TEST_NAME}
    TARGET_TYPE FAST_COMPILE_HW
    SOURCES_LIST ${kernel_files}
    HEADER_DIR_LIST ${header_dirs}
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




