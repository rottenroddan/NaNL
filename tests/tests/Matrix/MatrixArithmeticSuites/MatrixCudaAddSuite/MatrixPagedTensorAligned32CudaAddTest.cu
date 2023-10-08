//
// Created by steve on 9/29/2023.
//

#include "gtest/gtest.h"
#include <Matrix.cuh>
#include <DeviceMemoryBlock.cuh>
#include "MatrixCudaAddSuite.cuh"

TEST_F(MatrixCudaAddSuite, Should_Add_Small_Matrices_To_Correct_Values_When_Paged_TensorAligned32) {
    try {
        auto a = smallTestMatrices->getCopyOfA<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>();
        auto b = smallTestMatrices->getCopyOfB<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>();
        auto truth = smallTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

        auto startTime = std::chrono::high_resolution_clock::now();

        auto c = a.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixAddOperation::Cuda);

        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsedTimeMicro = std::chrono::duration_cast<std::chrono::microseconds>(endTime-startTime).count();
        auto elapsedTimeSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();

        std::cout << "CudaAdd time: " << std::setw(9) << std::right << elapsedTimeMicro << "μs. ("<< std::left << std::setw(7)  << elapsedTimeSeconds / 1000.0 << "s. )"<< std::endl;

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        for (unsigned int i = 0; i < c.getRows(); i++) {
            for (unsigned int j = 0; j < c.getCols(); j++) {
                ASSERT_EQ(c[i][j], truth[i][j]) << "i: " << i << " j: " << j << std::endl;
            }
        }
    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}

TEST_F(MatrixCudaAddSuite, Should_Add_Medium_Matrices_To_Correct_Values_When_Paged_TensorAligned32) {
    try {
        auto a = mediumTestMatrices->getCopyOfA<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>();
        auto b = mediumTestMatrices->getCopyOfB<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>();
        auto truth = mediumTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

        auto startTime = std::chrono::high_resolution_clock::now();

        auto c = a.add<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>(b, NaNL::MatrixAddOperation::Cuda);

        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsedTimeMicro = std::chrono::duration_cast<std::chrono::microseconds>(endTime-startTime).count();
        auto elapsedTimeSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();

        std::cout << "CudaAdd time: " << std::setw(9) << std::right << elapsedTimeMicro << "μs. ("<< std::left << std::setw(7)  << elapsedTimeSeconds / 1000.0 << "s. )"<< std::endl;

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        for (unsigned int i = 0; i < c.getRows(); i++) {
            for (unsigned int j = 0; j < c.getCols(); j++) {
                ASSERT_EQ(c[i][j], truth[i][j]) << "i: " << i << " j: " << j << std::endl;
            }
        }
    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}

TEST_F(MatrixCudaAddSuite, Should_Add_Large_Matrices_To_Correct_Values_When_Paged_TensorAligned32) {
    try {
        auto a = largeTestMatrices->getCopyOfA<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>();
        auto b = largeTestMatrices->getCopyOfB<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>();
        auto truth = largeTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

        auto startTime = std::chrono::high_resolution_clock::now();

        auto c = a.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixAddOperation::Cuda);

        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsedTimeMicro = std::chrono::duration_cast<std::chrono::microseconds>(endTime-startTime).count();
        auto elapsedTimeSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();

        std::cout << "CudaAdd time: " << std::setw(9) << std::right << elapsedTimeMicro << "μs. ("<< std::left << std::setw(7)  << elapsedTimeSeconds / 1000.0 << "s. )"<< std::endl;

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        for (unsigned int i = 0; i < c.getRows(); i++) {
            for (unsigned int j = 0; j < c.getCols(); j++) {
                ASSERT_EQ(c[i][j], truth[i][j]) << "i: " << i << " j: " << j << std::endl;
            }
        }

    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}