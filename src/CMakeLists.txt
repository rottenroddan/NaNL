cmake_minimum_required(VERSION 3.26.4)
project(Matrix LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


#set(CMAKE_CXX_STANDARD  20)
#set(CMAKE_CUDA_STANDARD 20)
#
#if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
#    set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
#    set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
#endif()

#enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_library(Matrix STATIC ThreadPool/ThreadPool.cuh ThreadPool/ThreadPool.cu empty.cu)

target_link_libraries(Matrix CUDA::cudart)
target_link_libraries(Matrix CUDA::cuda_driver)
target_link_libraries(Matrix ${CUDA_LIBRARIES})


set_target_properties(Matrix PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# set(SOURCE_FILES Matrix_Impl/BaseMatrix.cu Matrix_Impl/Matrix.cu Matrix_Impl/MatrixFileLoader.cu)
set_property(TARGET Matrix PROPERTY CUDA_ARCHITECTURES 80)

#if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#    target_compile_options(pybind11 INTERFACE -fsized-deallocation)
#endif()

target_include_directories(Matrix PUBLIC ${CUDA_INCLUDE_DIRS}
        Kernels
        Matrix
        Matrix/MatrixFileLoader
        Matrix/MatrixUtility
        Matrix/Logger
        MemoryBlock
        ThreadPool )