cmake_minimum_required(VERSION 3.1.0)

project(RealsenseExamplesHelloRealsense)

add_executable(rs-helloworld rs-helloworld.cpp)
set_property(TARGET rs-helloworld PROPERTY CXX_STANDARD_11)
target_link_libraries(rs-helloworld ${DEPENDENCIES})
set_target_properties(rs-helloworld PROPERTIES FOLDER "Examples")

install(
    TARGETS rs-helloworld 
    RUNTIME DESTINATION bin)