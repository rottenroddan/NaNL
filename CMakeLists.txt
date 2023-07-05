cmake_minimum_required(VERSION 3.25.2)
project(NaNL LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
#    set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
#    set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
#endif()
#
#set(CMAKE_CUDA_FLAGS "-std=c++20")

add_executable(NaNL main.cu)

set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

#target_link_libraries(NaNL PUBLIC ${CUDA_LIBRARIES} Matrix)
target_link_libraries(NaNL PUBLIC Matrix)

add_subdirectory(src)
add_subdirectory(tests)
#add_subdirectory(performance-testing)

#add_compile_definitions(PERFORMANCE_LOGGING)