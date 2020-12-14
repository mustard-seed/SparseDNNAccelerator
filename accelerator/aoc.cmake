# Function: add_aoc_target
# Purpose: Generates Intel FPGA aoc compiler targets
# TARGET_NAME: Single valued, required. Prefix assigned to the compiler target
# TARGET_TYPE: Single valued, required. Type of the aoc target. Valid options are EMULATION, RTL_ONLY, PROFILE_HW, and NORMAL_HW, and RELEASE_HW.
#              EMULATION -- The OpenCL kernels will be compiled as an x86 program with debug symbols to be executed on the CPU. 
#                           Useful for validating the correctness of the kernels.
#                           The target's name and the output aocx name end with "_aoc_emulation"
#              RTL_ONLY -- The aoc compiler will stop after generating the RTL and the early compilation report.
#                          The target's name and the output aocx name end with "_aoc_rtl_only"
#              PROFILE_HW -- The aoc compiler will add performance counters and go through place and route to generate an FPGA
#                            bitstream. The target's name and the output aocx name end with "_aoc_profile_hw"
#              NORMAL_HW -- The aoc compiler will go through place and route to generate an FPGA
#                            bitstream. The target's name and the output aocx name end with "_aoc_normal_hw"
#              RELEASE_HW -- The aoc compiler will go through place and route to generate an FPGA
#                            bitstream with high placement and routing effort. The target's name and the output aocx name end with
#                            "_aoc_normal_hw" 
#              FAST_COMPILE_HW -- The aoc compiler will go through place and route to generate an FPGA
#                            bitstream with minimal placement and routing effort. The target's name and the output aocx name end with
#                            "_aoc_fast_compile_hw"
#              SIMULATION -- The aoc compiler will generate a cycle accurate simulation library for the kernels
#                            Use this only on AOCL 19.1 or above.
#                            The target's name and the output aocx name end wi
# RTL_DIR:     Single valued, optional. Directory of the custom RTL library path. Must be supplied if RTL_LIB is specified
# BOARD_NAME     Single valued. Optional. Need to set if DE10Standard
# RTL_LIB_LIST:    List, optional. -l flags of rtl libraries. Must be supplied if RTL_DIR is specified. e.g. {-l mylib1.aoclib -l mylib2.aoclib}
# # HEADER_DIR_LIST:  List, optional. Custom header include path for the compiler to see. There is no need to pass <INTELFPGAOCLSDKROOT>/include/kernel_headers via this argument
# SOURCES_LIST: List, required. List of OpenCL kernel files (*.cl) to be compiled.
# PREPROCESSOR_DEFS_LIST: List, optional. List of preprocessor definition flags. e.g. {-DMY_MACRO1=xxxx -DMY_MACRO2=yyy}             
function (add_aoc_target)
    set (options )
    set (oneValueArgs TARGET_NAME TARGET_TYPE RTL_DIR BOARD_NAME)
    set (multiValueArgs  SOURCES_LIST HEADER_DIR_LIST PREPROCESSOR_DEFS_LIST RTL_LIB)

    cmake_parse_arguments(add_aoc_target "${options}" "${oneValueArgs}" "${multiValueArgs}" "${ARGN}" )

    list(REMOVE_DUPLICATES add_aoc_target_SOURCES_LIST)
    list(SORT add_aoc_target_SOURCES_LIST)

    list (APPEND occflags -O3 ${add_aoc_target_SOURCES_LIST})

    #Initialize the aoc compilation flags with the common elements for all compilation targets
    list (APPEND occflags -v -report -fp-relaxed
            ${add_aoc_target_PREPROCESSOR_DEFS_LIST}
            -I $ENV{INTELFPGAOCLSDKROOT}/include/kernel_headers)   

    # Add the custom header if needed to
    #if (${add_aoc_target_HEADER_DIR_LIST})
    foreach (header_dir ${add_aoc_target_HEADER_DIR_LIST})
      list (APPEND occflags -I ${header_dir} )
    endforeach()
    #endif()

    if ("${add_aoc_target_BOARD_NAME}" MATCHES "DE10Standard")
        set (FMAX -fmax=160)
    else ()
        set (FMAX -fmax=300)
        set (SEED -seed=9)
    endif() 
    
    #Check whether a valid build type is selected
    if ("${add_aoc_target_TARGET_TYPE}" STREQUAL "EMULATION")
        if ("${add_aoc_target_BOARD_NAME}" MATCHES "DE10Standard")
            message (STATUS "Board is DE10, so no emulation target is generated.")
        else()
            message (STATUS "Generating the emulation AOC target")
            set (target_name_local "${add_aoc_target_TARGET_NAME}_aoc_emulation")

                list (APPEND occflags
                        -march=emulator
                        -emulator-channel-depth-model=strict
                        -DEMULATOR
                        -g
                     )
         endif()
    elseif("${add_aoc_target_TARGET_TYPE}" STREQUAL "RTL_ONLY")
        message (STATUS "Generating the RTL_ONLY AOC target")
        set (target_name_local "${add_aoc_target_TARGET_NAME}_aoc_rtl_only")
        if ("${add_aoc_target_BOARD_NAME}" MATCHES "DE10Standard")
            list (APPEND occflags
                    -c
                    -o ${target_name_local}
                    ${FMAX}
                 )
         else()
             list (APPEND occflags
                     -rtl
     #                -o ${target_name_local}
                     ${FMAX}
                     ${SEED}
                  )
         endif()
    elseif("${add_aoc_target_TARGET_TYPE}" STREQUAL "PROFILE_HW")
        message (STATUS "Generating the PROFILE_HW AOC target")
        set (target_name_local "${add_aoc_target_TARGET_NAME}_aoc_profile_hw")
        list (APPEND occflags 
                -profile
                -high-effort
#                -o ${target_name_local}
                ${FMAX}
                ${SEED}
             )
     elseif("${add_aoc_target_TARGET_TYPE}" STREQUAL "FAST_COMPILE_HW")
         message (STATUS "Generating the FAST_COMPILE_HW AOC target")
         set (target_name_local "${add_aoc_target_TARGET_NAME}_aoc_fast_compile_hw")
         list (APPEND occflags
                 -fast-compile
#                 -o ${target_name_local}
                ${FMAX}
                ${SEED}
              )
    elseif("${add_aoc_target_TARGET_TYPE}" STREQUAL "NORMAL_HW")
        message (STATUS "Generating the NORMAL_HW AOC target")
        set (target_name_local "${add_aoc_target_TARGET_NAME}_aoc_normal_hw")
        list (APPEND occflags
#                -o ${target_name_local}
                ${FMAX}
                ${SEED}
             )
    elseif("${add_aoc_target_TARGET_TYPE}" STREQUAL "RELEASE_HW")
        message (STATUS "Generating the RELEASE_HW AOC target")
        set (target_name_local "${add_aoc_target_TARGET_NAME}_aoc_release_hw")
        list (APPEND occflags 
                -high-effort
 #               -o ${target_name_local}
                ${FMAX}
                ${SEED}
             )
    else()
        message (FATAL_ERROR "Illegal TARGET_TYPE passed to the function add_aoc_target. Valid options are EMULATION, RTL_ONLY, PROFILE_HW, 
            NORMAL_HW, and RELEASE_HW.")
    endif()

    # Disable DDR burst-interleaving of global memory on A10PAC
    if ( (NOT ("${add_aoc_target_TARGET_TYPE}" STREQUAL "EMULATION")) 
      AND ("${add_aoc_target_BOARD_NAME}" MATCHES "A10PAC") )
      list (APPEND occflags
                     -no-interleaving=default
                  )
      message (STATUS "Disabling global memory burst-interleaving for HW compilation on PACs")
    endif ()

    #Add the library for custom RTL if needed
    if ("${add_aoc_target_RTL_DIR}" STREQUAL "")
    else()
        list (APPEND occflags -I ${add_aoc_target_RTL_DIR} -L ${add_aoc_target_RTL_DIR}
                ${add_aoc_target_RTL_LIB})
    endif()

    if (("${add_aoc_target_TARGET_TYPE}" STREQUAL "EMULATION") AND ("${add_aoc_target_BOARD_NAME}" MATCHES "DE10Standard"))
    else()
        add_custom_target(${target_name_local}
            COMMAND mkdir ${CMAKE_CURRENT_BINARY_DIR}/${target_name_local} -p && cd ${CMAKE_CURRENT_BINARY_DIR}/${target_name_local} && aoc ${occflags}
        )
    endif()

    #message (STATUS ${occflags})
    
endfunction()
