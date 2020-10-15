#############################################
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

#Find Boost
find_package (Boost 1.56 REQUIRED)

set (PREPROCESSOR_DEFS
#        -DWEIGHT_MEMORY_TEST
#        -DHW_DEBUG
        FLEXIBLE_BITMASK_COMPRESSION
        WEIGHT_MEMORY
        MEMORY_READER
        MEMORY_WRITER
        WEIGHT_MEMORY_INTERCONNECT
        ACTIVATION_MEMORY_INTERCONNECT
        PE_SYSTEM
        IA_MEMORY
        OA_MEMORY
        FULL_SYSTEM
        MISC_ENGINE
        )

####################################################
# AOCL Utilities library
####################################################

#Purpose: Generate build targets for the Intel OpenCL common utilities
set (AOCL_LIBRARY_NAME "aocl_utils")

#########################
#Obtain the host source files
#########################
file(GLOB AOCL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/AOCLUtilsCpp/*.cpp
)

add_library(${AOCL_LIBRARY_NAME} ${AOCL_SOURCES})

###########################
#Add the include files
###########################
target_include_directories(${AOCL_LIBRARY_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/AOCLUtilsCpp
    ${Boost_INCLUDE_DIRS}
    $ENV{INTELFPGAOCLSDKROOT}/include/kernel_headers
    $ENV{INTELFPGAOCLSDKROOT}/host/include
)

#Link the library with OpenCL libraries
if ("${intelfpga_host_libraries}" STREQUAL "")
    #Hack for non arm boards
    execute_process(
            COMMAND aocl link-config
            OUTPUT_VARIABLE aocl_link_config
        )
    #Need to remove the newline, or the make file will complain "missing separator"
    string(REGEX REPLACE "\n$" "" aocl_link_config "${aocl_link_config}")
    target_link_libraries(${AOCL_LIBRARY_NAME} ${aocl_link_config})
else()
    target_link_libraries(${AOCL_LIBRARY_NAME}
         ${intelfpga_host_libraries}
         ${intelfpga_device_libraries}
         )
endif()

target_link_libraries(${AOCL_LIBRARY_NAME}
         rt
         pthread
    )
set_target_properties (${AOCL_LIBRARY_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}")

target_compile_definitions(
    ${AOCL_LIBRARY_NAME}
    PUBLIC
    ${PREPROCESSOR_DEFS}
    )

#################################################################
# Library for tensor compression, and instruction generation
#################################################################
file(GLOB UTIL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library("data_instruction_gen" ${UTIL_SOURCES})
target_include_directories("data_instruction_gen"
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${PROJECT_SOURCE_DIR}/device
        ${Boost_INCLUDE_DIRS}
        $ENV{INTELFPGAOCLSDKROOT}/include/kernel_headers
        $ENV{INTELFPGAOCLSDKROOT}/host/include
)
set_target_properties ("data_instruction_gen" PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}")

target_compile_definitions(
    data_instruction_gen
    PUBLIC
    ${PREPROCESSOR_DEFS}
    )
