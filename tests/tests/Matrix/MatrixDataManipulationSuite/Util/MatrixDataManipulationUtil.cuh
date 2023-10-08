//
// Created by steve on 9/28/2023.
//

#ifndef NANL_MATRIXDATAMANIPULATIONUTIL_CUH
#define NANL_MATRIXDATAMANIPULATIONUTIL_CUH

#include <gtest/gtest.h>
#include <Matrix.cuh>
#include <PagedMemoryBlock.cuh>
#include <PinnedMemoryBlock.cuh>
#include <DeviceMemoryBlock.cuh>
#include <TensorCoreAligned8.cuh>
#include <TensorCoreAligned16.cuh>
#include <TensorCoreAligned32.cuh>

template<class T, template<class, class> class MemoryBlock = NaNL::PagedMemoryBlock, class Alignment = NaNL::Unaligned>
static NaNL::Matrix<T , MemoryBlock, Alignment> preload_test_matrix(uint64_t rows, uint64_t cols) {
    NaNL::Matrix<T, MemoryBlock, Alignment> rtnMatrix(rows, cols);
    T itter = 0;
    for(uint64_t i = 0; i < rows; i++) {
        for(uint64_t j = 0; j < cols; j++) {
            rtnMatrix[i][j] = itter++;
        }
    }

    return rtnMatrix;
}

template<class T, template<class, class> class MemoryBlock, class Alignment>
static void validate_test_matrix(const NaNL::Matrix<T, MemoryBlock, Alignment>& matrix) {
    T itter = 0;
    for(uint64_t i = 0; i < matrix.getRows(); i++) {
        for(uint64_t j = 0; j < matrix.getCols(); j++) {
            ASSERT_EQ(matrix[i][j], itter++) << "at i: " << i << " j: " << j << std::endl;
        }
    }
}

#endif //NANL_MATRIXDATAMANIPULATIONUTIL_CUH
