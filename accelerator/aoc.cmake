# Functions used to generate the hardware targets

function (add_aoc_target)
    set (options )
    set (oneValueArgs TARGET_NAME TARGET_TYPE HEADER_DIR RTL_DIR RTL_LIB)
    set (multiValueArgs  SOURCES_LIST PREPROCESSOR_DEFS_LIST)

    cmake_parse_arguments(add_aoc_target "${options}" "${oneValueArgs}" "${multiValueArgs}" "${ARGN}" )    

    #Initialize the aoc compilation flags with the common elements for all compilation targets
    list (APPEND occflags -v -report -fp-relaxed
            ${add_aoc_target_PREPROCESSOR_DEFS_LIST}
            -I ${add_aoc_target_HEADER_DIR}
            -I $ENV{INTELFPGAOCLSDKROOT}/include/kernel_headers)       
    
    #Check whether a valid build type is selected
    if ("${add_aoc_target_TARGET_TYPE}" STREQUAL "EMULATION")
        message (STATUS "Generating the emulation AOC target")
        set (target_name_local "${add_aoc_target_TARGET_NAME}_aoc_emulation")
        list (APPEND occflags 
                -march=emulator 
                -emulator-channel-depth-model=strict
                -DEMULATOR
                -o ${target_name_local}
             )
    elseif("${add_aoc_target_TARGET_TYPE}" STREQUAL "RTL_ONLY")
        message (STATUS "Generating the RTL_ONLY AOC target")
        set (target_name_local "${add_aoc_target_TARGET_NAME}_aoc_rtl_only")
        list (APPEND occflags 
                -c
                -o ${target_name_local}
             )
    elseif("${add_aoc_target_TARGET_TYPE}" STREQUAL "PROFILE_HW")
        message (STATUS "Generating the PROFILE_HW AOC target")
        set (target_name_local "${add_aoc_target_TARGET_NAME}_aoc_profile_hw")
        list (APPEND occflags 
                -profile
                -high-effort
                -o ${target_name_local}
             ) 
    elseif("${add_aoc_target_TARGET_TYPE}" STREQUAL "NORMAL_HW")
        message (STATUS "Generating the NORMAL_HW AOC target")
        set (target_name_local "${add_aoc_target_TARGET_NAME}_aoc_normal_hw")
        list (APPEND occflags
                -o ${target_name_local}
             )
    elseif("${add_aoc_target_TARGET_TYPE}" STREQUAL "RELEASE_HW")
        message (STATUS "Generating the RELEASE_HW AOC target")
        set (target_name_local "${add_aoc_target_TARGET_NAME}_aoc_release_hw")
        list (APPEND occflags 
                -high-effort
                -o ${target_name_local}
             )
    else()
        message (FATAL_ERROR "Illegal TARGET_TYPE passed to the function add_aoc_target. Valid options are EMULATION, RTL_ONLY, PROFILE_HW, 
            NORMAL_HW, and RELEASE_HW.")
    endif()

    list(REMOVE_DUPLICATES add_aoc_target_SOURCES_LIST)
    list(SORT add_aoc_target_SOURCES_LIST)

    if ("${add_hw_emulation_target_RTL_LIB}" STREQUAL "")
    else()
        list (APPEND occflags -I ${add_hw_profile_target_RTL_DIR} -L ${add_hw_profile_target_RTL_DIR}
                -l ${add_hw_profile_target_RTL_LIB})
    endif()

    list (APPEND occflags ${add_aoc_target_SOURCES_LIST})
    
    add_custom_target(${target_name_local}
        COMMAND aoc ${occflags}
    )
    
endfunction()
