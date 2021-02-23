set(APP imagenet_demo)
##Find OpenCV##
find_package(OpenCV REQUIRED dnn imgproc highgui)

##Define the build target
message (STATUS ${OpenCV_DIR})
add_executable(${APP} ${CMAKE_CURRENT_SOURCE_DIR}/imagenet_demo.cpp)
target_link_libraries(${APP}
    graph_factory
    accelerator_wrapper
    ${OpenCV_LIBS}
    )
target_include_directories(${APP}
    PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/device
    )
set_target_properties (${APP} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS}")
