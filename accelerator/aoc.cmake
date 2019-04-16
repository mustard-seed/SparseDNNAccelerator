# Functions used to generate the hardware targets

function(add_hw_emulation_target)
    set (options )
    set (oneValueArgs TARGET_NAME HEADER_DIR RTL_DIR RTL_LIB)
    set (multiValueArgs  SOURCES_LIST PREPROCESSOR_DEFS_LIST)

    cmake_parse_arguments(add_hw_emulation_target "${options}" "${oneValueArgs}" "${multiValueArgs}" "${ARGN}" )   
    
    list(REMOVE_DUPLICATES add_hw_emulation_target_SOURCES_LIST)
    list(SORT add_hw_emulation_target_SOURCES_LIST)

    if ("${add_hw_emulation_target_PREPROCESSOR_DEFS_LIST}" STREQUAL "")
    else()
        list(REMOVE_DUPLICATES add_hw_emulation_target_PREPROCESSOR_DEFS_LIST)
    endif()

    set (target_name_local "${add_hw_emulation_target_TARGET_NAME}_hw_emu")

    list (APPEND occflags -v -report -march=emulator -emulator-channel-depth-model=strict -fp-relaxed
                -DEMULATOR
                ${add_hw_emulation_target_PREPROCESSOR_DEFS_LIST}
                -o ${target_name_local}
                -I ${add_hw_emulation_target_HEADER_DIR}
                -I $ENV{INTELFPGAOCLSDKROOT}/include/kernel_headers)

    if ("${add_hw_emulation_target_RTL_LIB}" STREQUAL "")
    else()
        list (APPEND occflags -I ${add_hw_emulation_target_RTL_DIR} -L ${add_hw_emulation_target_RTL_DIR}
                -l ${add_hw_emulation_target_RTL_LIB})
    endif()

    list (APPEND occflags ${add_hw_emulation_target_SOURCES_LIST})
    
    add_custom_target(${target_name_local}
        COMMAND aoc ${occflags}
    )
    
endfunction()

function(add_hw_report_target)

    set (options )
    set (oneValueArgs TARGET_NAME HEADER_DIR RTL_DIR RTL_LIB)
    set (multiValueArgs  SOURCES_LIST PREPROCESSOR_DEFS_LIST)

    cmake_parse_arguments(add_hw_report_target "${options}" "${oneValueArgs}" "${multiValueArgs}" "${ARGN}" )  

    list(REMOVE_DUPLICATES add_hw_report_target_SOURCES_LIST)
    list(SORT add_hw_report_target_SOURCES_LIST)

    if ("${add_hw_report_target_PREPROCESSOR_DEFS_LIST}" STREQUAL "")
    else()
        list(REMOVE_DUPLICATES add_hw_report_target_PREPROCESSOR_DEFS_LIST)
    endif()

    set (target_name_local "${add_hw_report_target_TARGET_NAME}_hw_report")
    
    list (APPEND occflags -v -report -c -fp-relaxed
                -o ${target_name_local}
                ${add_hw_report_target_PREPROCESSOR_DEFS_LIST}
                -I ${add_hw_report_target_HEADER_DIR}
                -I $ENV{INTELFPGAOCLSDKROOT}/include/kernel_headers)

    if ("${add_hw_report_target_RTL_LIB}" STREQUAL "")
    else()
        list (APPEND occflags -I ${add_hw_report_target_RTL_DIR} -L ${add_hw_report_target_RTL_DIR}
                -l ${add_hw_report_target_RTL_LIB})
    endif()

    list (APPEND occflags ${add_hw_report_target_SOURCES_LIST})
    
    add_custom_target(${target_name_local}
        COMMAND aoc ${occflags}
    )
    
endfunction()

function(add_hw_profile_target)

    set (options )
    set (oneValueArgs TARGET_NAME HEADER_DIR RTL_DIR RTL_LIB)
    set (multiValueArgs  SOURCES_LIST PREPROCESSOR_DEFS_LIST)

    cmake_parse_arguments(add_hw_profile_target "${options}" "${oneValueArgs}" "${multiValueArgs}" "${ARGN}" )  

    list(REMOVE_DUPLICATES add_hw_profile_target_SOURCES_LIST)
    list(SORT add_hw_profile_target_SOURCES_LIST)

    if ("${add_hw_profile_target_PREPROCESSOR_DEFS_LIST}" STREQUAL "")
    else()
        list(REMOVE_DUPLICATES add_hw_profile_target_PREPROCESSOR_DEFS_LIST)
    endif()

    set (target_name_local "${add_hw_profile_target_TARGET_NAME}_hw_profile")
    
    list (APPEND occflags -v -report -fp-relaxed -profile -high-effort
                -o ${target_name_local}
                ${add_hw_profile_target_PREPROCESSOR_DEFS_LIST}
                -I ${add_hw_profile_target_HEADER_DIR}
                -I $ENV{INTELFPGAOCLSDKROOT}/include/kernel_headers)

    if ("${add_hw_emulation_target_RTL_LIB}" STREQUAL "")
    else()
        list (APPEND occflags -I ${add_hw_profile_target_RTL_DIR} -L ${add_hw_profile_target_RTL_DIR}
                -l ${add_hw_profile_target_RTL_LIB})
    endif()

    list (APPEND occflags ${add_hw_profile_target_SOURCES_LIST})
    
    add_custom_target(${target_name_local}
        COMMAND aoc ${occflags}
    )
    
endfunction()

function(add_hw_release_target)

    set (options )
    set (oneValueArgs TARGET_NAME HEADER_DIR RTL_DIR RTL_LIB)
    set (multiValueArgs  SOURCES_LIST PREPROCESSOR_DEFS_LIST)

    cmake_parse_arguments(add_hw_release_target "${options}" "${oneValueArgs}" "${multiValueArgs}" "${ARGN}" )  

    list(REMOVE_DUPLICATES add_hw_release_target_SOURCES_LIST)
    list(SORT add_hw_release_target_SOURCES_LIST)

    if ("${add_hw_release_target_PREPROCESSOR_DEFS_LIST}" STREQUAL "")
    else()
        list(REMOVE_DUPLICATES add_hw_release_target_PREPROCESSOR_DEFS_LIST)
    endif()

    set (target_name_local "${add_hw_release_target_TARGET_NAME}_hw_release")
    
    list (APPEND occflags -v -report -fp-relaxed -high-effort
                -o ${target_name_local}
                ${add_hw_release_target_PREPROCESSOR_DEFS_LIST}
                -I ${add_hw_release_target_HEADER_DIR}
                -I $ENV{INTELFPGAOCLSDKROOT}/include/kernel_headers)

    if ("${add_hw_release_target_RTL_LIB}" STREQUAL "")
    else()
        list (APPEND occflags -I ${add_hw_release_target_HEADER_DIR} -L ${add_hw_release_target_RTL_DIR}
                -l ${add_hw_release_target_RTL_LIB})
    endif()

    list (APPEND occflags ${add_hw_release_target_SOURCES_LIST})
    
    add_custom_target(${target_name_local}
        COMMAND aoc ${occflags}
    )
    
endfunction()
