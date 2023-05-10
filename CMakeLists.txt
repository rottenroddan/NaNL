cmake_minimum_required(VERSION 3.22)
project(NaNL LANGUAGES CUDA CXX)


find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(NaNL main.cu)

target_link_libraries(NaNL PUBLIC ${CUDA_LIBRARIES} Matrix)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(performance-testing)

#add_compile_definitions(PERFORMANCE_LOGGING)
