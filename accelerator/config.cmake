cmake_minimum_required(VERSION 2.8)

if (NOT PROJECT_NAME)
    message(WARNING "Project name is unset! This file should only be included after calling project(...)")
endif(NOT PROJECT_NAME)

if (NOT GLOBAL_CONFIG_SET)
    set(GLOBAL_CONFIG_SET TRUE)
    set(CMAKE_VERBOSE_MAKEFILE TRUE)
    set (CMAKE_CXX_STANDARD 11)
    set (CMAKE_CXX_STANDARD_REQUIRED ON)
    
###########Select CMAKE_BUILD_TYPE#############
# Possible options:
# Debug
# Release
# MinSizeRel
# RelWithDebInfo
###############################################

    set(CMAKE_BUILD_TYPE Release)
    
    set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
    if (CMAKE_BUILD_TYPE MATCHES "Debug")
        message(STATUS "DEBUG flag is enabled.")
        add_definitions(-DDEBUG)
    endif(CMAKE_BUILD_TYPE MATCHES "Debug")

    ####Optional: Profliing################
    #add_definitions(-DPROFILE)

    set(DEFENSIVE_FLAGS "-D_GLIBCXX_ASSERTIONS") # -D_FORTIFY_SOURCE=2 would enable runtime overflow checks, but requires O2
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0)
        set(DEFENSIVE_FLAGS "${DEFENSIVE_FLAGS} -fstack-clash-protection")
    endif()
    set(DEBUGGING_FLAGS "-fasynchronous-unwind-tables -grecord-gcc-switches")
    set(WARNING_FLAGS "-Wall -W -Wno-unused-parameter -Werror=format-security -Werror=implicit-function-declaration -Wl,-z,now -Wl,-z,relro") #  -Wl,-z,defs would prevent underlinking, but doesn't play nice with ROS/Catkin
    #set(OPTIMIZATION_FLAGS "-pipe -Og") # -O2 is required for -D_FORTIFY_SOURCE=2, but would obfuscate debugging
    if (CMAKE_BUILD_TYPE MATCHES "Debug")
        set(OPTIMIZATION_DEBUG_FLAGS "-pipe -Og -g")
    else(CMAKE_BUILD_TYPE MATCHES "Release")
        set(OPTIMIZATION_DEBUG_FLAGS "-pipe -O3")
    endif(CMAKE_BUILD_TYPE MATCHES "Debug")
    
    #Add -fPIC to make the compiler happy with linking gflag
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DEFENSIVE_FLAGS} ${DEBUGGING_FLAGS} ${WARNING_FLAGS} ${OPTIMIZATION_FLAGS} -fPIC -std=c++11")
endif()

if (NOT ${PROJECT_NAME}_CONFIG_SET)
    set(${PROJECT_NAME}_CONFIG_SET TRUE)
    message (STATUS "C Compiler: ${CMAKE_C_COMPILER}")
    message (STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
else(NOT ${PROJECT_NAME}_CONFIG_SET)
    message(WARNING "config.cmake was included twice for the same project - make sure that the `include(config.cmake)` call happens after calling project(...).")
endif(NOT ${PROJECT_NAME}_CONFIG_SET)
