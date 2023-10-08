//
// Created by steve on 8/3/2023.
//

//
// Created by steve on 7/16/2023.
//

#include "gtest/gtest.h"
#include <Matrix.cuh>
#include <DeviceMemoryBlock.cuh>
#include "MatrixHostAddSuite.cuh"
#include <TensorCoreAligned32.cuh>

TEST_F(MatrixHostAddSuite, Should_Add_Small_Matrices_To_Correct_Values_When_Device_Tensor_Aligned_32) {
    try {
        auto a = smallTestMatrices->getCopyOfA<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32>().copyTo<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32, float>();
        auto b = smallTestMatrices->getCopyOfB<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32>().copyTo<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32, float>();
        auto truth = smallTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>();

        auto startTime = std::chrono::high_resolution_clock::now();

        auto c = a.add<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32>(b, NaNL::MatrixAddOperation::Host);

        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsedTimeMicro = std::chrono::duration_cast<std::chrono::microseconds>(endTime-startTime).count();
        auto elapsedTimeSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();

        std::cout << "HostAdd time: " << std::setw(9) << std::right << elapsedTimeMicro << "μs. ("<< std::left << std::fixed << std::setw(7)  << elapsedTimeSeconds / 1000.0 << "s. )"<< std::endl;

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        auto hostC = c.copyTo<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>();

        // validations must be done on host. operator[] doesn't exist on
        // Matrix<.,DeviceMemoryBlock,.>...
        for (unsigned int i = 0; i < hostC.getRows(); i++) {
            for (unsigned int j = 0; j < hostC.getCols(); j++) {
                ASSERT_EQ(hostC[i][j], truth[i][j]);
            }
        }
    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}

TEST_F(MatrixHostAddSuite, Should_Add_Medium_Matrices_To_Correct_Values_When_Device_Tensor_Aligned_32) {
    try {
        auto a = mediumTestMatrices->getCopyOfA<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32>().copyTo<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32, float>();
        auto b = mediumTestMatrices->getCopyOfB<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32>().copyTo<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32, float>();
        auto truth = mediumTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>();

        std::cout << "A: " << a.copyTo<NaNL::PagedMemoryBlock, NaNL::Unaligned>()[1326][1088] << std::endl;
        std::cout << "B: " << b.copyTo<NaNL::PagedMemoryBlock, NaNL::Unaligned>()[1326][1088] << std::endl;
        std::cout << "C: " << truth[1326][1088] << std::endl;

        auto startTime = std::chrono::high_resolution_clock::now();

        auto c = a.add<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32>(b, NaNL::MatrixAddOperation::Host);

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        auto hostC = c.copyTo<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>();

        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsedTimeMicro = std::chrono::duration_cast<std::chrono::microseconds>(endTime-startTime).count();
        auto elapsedTimeSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();

        std::cout << "HostAdd time: " << std::setw(9) << std::right << elapsedTimeMicro << "μs. ("<< std::left << std::fixed << std::setw(7)  << elapsedTimeSeconds / 1000.0 << "s. )"<< std::endl;

        // validations must be done on host. operator[] doesn't exist on
        // Matrix<.,DeviceMemoryBlock,.>...
        for (unsigned int i = 0; i < hostC.getRows(); i++) {
            for (unsigned int j = 0; j < hostC.getCols(); j++) {
                ASSERT_EQ(hostC[i][j], truth[i][j]);
            }
        }
    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}

TEST_F(MatrixHostAddSuite, Should_Add_Large_Matrices_To_Correct_Values_When_Device_Tensor_Aligned_32) {
    try {
        auto a = largeTestMatrices->getCopyOfA<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32>().copyTo<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32, float>();
        auto b = largeTestMatrices->getCopyOfB<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32>().copyTo<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32, float>();
        auto truth = largeTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>();

        auto startTime = std::chrono::high_resolution_clock::now();

        auto c = a.add<NaNL::DeviceMemoryBlock, NaNL::TensorCoreAligned32>(b);

        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsedTimeMicro = std::chrono::duration_cast<std::chrono::microseconds>(endTime-startTime).count();
        auto elapsedTimeSeconds = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();

        std::cout << "HostAdd time: " << std::setw(9) << std::right << elapsedTimeMicro << "μs. ("<< std::left << std::fixed << std::setw(7)  << elapsedTimeSeconds / 1000.0 << "s. )"<< std::endl;

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        auto hostC = c.copyTo<NaNL::PagedMemoryBlock, NaNL::TensorCoreAligned32>();

        // validations must be done on host. operator[] doesn't exist on
        // Matrix<.,DeviceMemoryBlock,.>...
        for (unsigned int i = 0; i < hostC.getRows(); i++) {
            for (unsigned int j = 0; j < hostC.getCols(); j++) {
                ASSERT_EQ(hostC[i][j], truth[i][j]);
            }
        }
    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}