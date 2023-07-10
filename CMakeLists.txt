cmake_minimum_required(VERSION 3.25.2)
project(NaNL LANGUAGES CUDA CXX)

add_executable(NaNL main.cu)

set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

target_link_libraries(NaNL PUBLIC Matrix)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(performance-testing)

#add_compile_definitions(PERFORMANCE_LOGGING)