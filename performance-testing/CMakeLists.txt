# 'Google_test' is the subproject name
cmake_minimum_required(VERSION 3.26.4)
project(Performance-Testing LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(Performance-Testing  main.cu)
target_link_libraries(Performance-Testing PUBLIC Matrix)

#set(CMAKE_VERBOSE_MAKEFILE ON)

#set(CMAKE_CXX_STANDARD  20)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CUDA_STANDARD 20)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)


#set_target_properties(Matrix PROPERTIES LINKER_LANGUAGE CXX)
#set_target_properties(Performance-Testing PROPERTIES LINKER_LANGUAGE CXX)

add_compile_definitions(PERFORMANCE_LOGGING)

