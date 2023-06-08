cmake_minimum_required(VERSION 3.22)
project(Matrix LANGUAGES CUDA)
set(CMAKE_CUDA_STANDARD 17)

add_library(Matrix STATIC empty.cu)

enable_language(CUDA)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})


target_link_libraries(Matrix CUDA::cudart)
target_link_libraries(Matrix CUDA::cuda_driver)
target_link_libraries(Matrix ${CUDA_LIBRARIES})

set_target_properties(Matrix PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# set(SOURCE_FILES Matrix_Impl/BaseMatrix.cu Matrix_Impl/Matrix.cu Matrix_Impl/MatrixFileLoader.cu)
set_property(TARGET Matrix PROPERTY CUDA_ARCHITECTURES 80)

target_include_directories(Matrix PUBLIC Kernels Matrix Matrix/Logger MemoryBlock ThreadPool)
