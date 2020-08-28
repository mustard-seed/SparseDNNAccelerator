include (${PROJECT_SOURCE_DIR}/config.cmake)

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

set (LIBRARY_NAME "accelerator_wrapper")
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
        ${PROJECT_SOURCE_DIR}/device
    )
target_compile_definitions(
    ${LIBRARY_NAME}
    PUBLIC
    ${PREPROCESSOR_DEFS}
    )

set_target_properties (${LIBRARY_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}")

#add_executable(accelerator_wrapper_test ${CMAKE_CURRENT_SOURCE_DIR}/accelerator_wrapper_test.cpp)
#target_link_libraries(accelerator_wrapper_test ${LIBRARY_NAME})
#set_target_properties (accelerator_wrapper_test PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}")

