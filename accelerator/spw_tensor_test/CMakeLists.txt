set (TEST_NAME "spw_tensor_test")

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

#########################################################
#SOURCE
#########################################################
set (SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${PROJECT_SOURCE_DIR}/common/src/floatFixedPointConversion.cpp
    ${PROJECT_SOURCE_DIR}/common/src/spwTensorCompression.cpp
    )

########################################################
# FIND THIRD PARTY LIBRARIES
########################################################
find_package (Boost 1.56 REQUIRED)

#################################
# GENERATE THE EXECUTABLE TARGET
#################################
add_executable (${TEST_NAME} ${SOURCES})
target_link_libraries(${TEST_NAME} ${Boost_LIBRARIES} gtest_main aocl_utils)
target_include_directories(${TEST_NAME}
    PUBLIC
    ${PROJECT_SOURCE_DIR}/device
    ${PROJECT_SOURCE_DIR}/common/inc
    )
set_target_properties (${TEST_NAME}
    PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}")
