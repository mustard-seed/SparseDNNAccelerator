include (${PROJECT_SOURCE_DIR}/config.cmake)

set (LIB_NAME "graph_factory")

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
elseif (BOARD_NAME MATCHES "A10REF")
    ##Add library directories for a10_ref
    add_definitions(-DARRIA10)
    message (STATUS "Linking against A10-Ref libraries")
elseif (BOARD_NAME MATCHES "A10PAC")
    add_definitions(-DARRIA10 -DA10PAC)
    ##Add library directories for a10_pac
    message (STATUS "Linking against A10-Pac libraries")
elseif (BOARD_NAME MATCHES "DE5ANET")
    add_definitions(-DARRIA10)
    ##Add library directories for de5a_net
    message (STATUS "Linking against DE5A-NET libraries")
else()
    message (STATUS "The board is ${BOARD_NAME}")
    message (FATAL_ERROR "Unsupported board!")
endif()

#########################
#Obtain the host source files
#########################
file(GLOB LIB_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

####################################################################
#Add the library as a build target and set is include directory
####################################################################
find_package (Boost 1.56 REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

add_library(${LIB_NAME}
    ${LIB_SRC}
    )
target_include_directories(${LIB_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Boost_INCLUDE_DIRS})
target_link_libraries(${LIB_NAME}
    PUBLIC
    aocl_utils accelerator_wrapper model_container
    )

set_target_properties (${LIB_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}")

#######################################d
##Add extra library for the host below
######################################
#set (OpenCV_DIR /home/jamesliu/arm_opencv/Build CACHE STRING "Path of OpenCV" FORCE)
#find_package (OpenCV REQUIRED )

#include_directories(${OpenCV_INCLUDE_DIRS})
#add_definitions(-DUSE_OPENCV)

#####################################
##Add unit test for the host library below
#####################################
add_definitions(-DFPGA_DEVICE)
add_executable (${LIB_NAME}_test ${CMAKE_CURRENT_SOURCE_DIR}/graph_factory_test.cpp)
target_link_libraries(${LIB_NAME}_test gtest_main ${LIB_NAME})
target_include_directories(${LIB_NAME}_test
    PUBLIC
    ${PROJECT_SOURCE_DIR}/device
    )

set_target_properties (${LIB_NAME}_test PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}")




