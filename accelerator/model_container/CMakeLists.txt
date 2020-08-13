include (${PROJECT_SOURCE_DIR}/config.cmake)

set (TEST_NAME "fullSystem")

########################################
# Declare the library
########################################
# Relevant source files
file (GLOB SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
  )

# Add the library
add_library(model_container ${SOURCES})

# Add the headers
target_include_directories(model_container PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add dependencies
target_link_libraries(model_container
    yaml-cpp
  )

#######################################d
##Add test
######################################
find_package (Boost 1.56 REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
#####################################
##Add unit test for the host library below
#####################################
set(TEST_NAME model_container_test)
add_executable (${TEST_NAME} model_container_test.cpp)
target_link_libraries(${TEST_NAME} model_container ${Boost_LIBRARIES} gtest_main)
