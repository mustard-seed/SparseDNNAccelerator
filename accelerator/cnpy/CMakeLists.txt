# Adapted from from https://github.com/rogersce/cnpy.git
# Commit: 4e8810b
# CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)
#if(COMMAND cmake_policy)
#	cmake_policy(SET CMP0003 NEW)
#endif(COMMAND cmake_policy)

#project(CNPY)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#option(ENABLE_STATIC "Build static (.a) library" ON)
if (ZLIB_ROOT)
    message (STATUS "ZLIB_ROOT is set to ${ZLIB_ROOT}")
endif()
find_package(ZLIB REQUIRED)

if (BOARD_NAME MATCHES "DE10Standard")
    find_package (Boost 1.56 REQUIRED)
    add_library(boost_regex STATIC IMPORTED)
    set_target_properties(boost_regex PROPERTIES IMPORTED_LOCATION "${BOOST_ROOT}/lib/libboost_regex.a")
else()
    find_package (Boost 1.56 REQUIRED regex)
endif()

add_library(cnpy "cnpy.cpp")
target_link_libraries(cnpy ${ZLIB_LIBRARIES} boost_regex)
target_include_directories(cnpy PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ZLIB_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS})
set_target_properties (cnpy PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}")

add_executable(cnpy_example "example1.cpp")
target_link_libraries(cnpy_example cnpy)
target_include_directories(cnpy_example PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties (cnpy_example PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}")
