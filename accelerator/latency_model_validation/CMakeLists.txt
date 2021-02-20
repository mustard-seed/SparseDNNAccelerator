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

add_executable (latency_model_validation ${CMAKE_CURRENT_SOURCE_DIR}/latency_model_validation.cpp)
target_link_libraries(latency_model_validation gtest_main graph_factory)
target_include_directories(latency_model_validation
    PUBLIC
    ${PROJECT_SOURCE_DIR}/device
    )

set_target_properties (latency_model_validation PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}")




