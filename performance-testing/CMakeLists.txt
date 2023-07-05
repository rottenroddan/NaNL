# 'Google_test' is the subproject name
cmake_minimum_required(VERSION 3.25.2)
project(Performance-Testing LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(Performance-Testing  main.cu)
target_link_libraries(Performance-Testing Matrix)

set_target_properties(Performance-Testing PROPERTIES CUDA_ARCHITECTURES 80)

add_compile_definitions(PERFORMANCE_LOGGING)