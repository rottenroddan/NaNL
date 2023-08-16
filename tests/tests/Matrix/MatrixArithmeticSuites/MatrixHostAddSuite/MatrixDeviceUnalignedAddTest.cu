//
// Created by steve on 7/16/2023.
//

#include "gtest/gtest.h"
#include <Matrix.cuh>
#include <DeviceMemoryBlock.cuh>
#include "MatrixHostAddSuite.cuh"

TEST_F(MatrixHostAddSuite, Should_Add_Small_Matrices_To_Correct_Values_When_Device_UnAligned) {
    try {
        NaNL::Matrix<int, NaNL::DeviceMemoryBlock, NaNL::Unaligned> a = smallTestMatrices->getCopyOfA<NaNL::DeviceMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int, NaNL::DeviceMemoryBlock, NaNL::Unaligned> b = smallTestMatrices->getCopyOfB<NaNL::DeviceMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int, NaNL::PagedMemoryBlock, NaNL::Unaligned> truth = smallTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

        auto c = a.add<NaNL::DeviceMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixDeviceOperation::Host);

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        auto hostC = c.copyTo<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

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

TEST_F(MatrixHostAddSuite, Should_Add_Medium_Matrices_To_Correct_Values_When_Device_UnAligned) {
    try {
        NaNL::Matrix<int, NaNL::DeviceMemoryBlock, NaNL::Unaligned> a = mediumTestMatrices->getCopyOfA<NaNL::DeviceMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int, NaNL::DeviceMemoryBlock, NaNL::Unaligned> b = mediumTestMatrices->getCopyOfB<NaNL::DeviceMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int, NaNL::PagedMemoryBlock, NaNL::Unaligned> truth = mediumTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

        auto c = a.add<NaNL::DeviceMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixDeviceOperation::Host);

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        auto hostC = c.copyTo<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

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

TEST_F(MatrixHostAddSuite, Should_Add_Large_Matrices_To_Correct_Values_When_Device_UnAligned) {
    try {
        NaNL::Matrix<int, NaNL::DeviceMemoryBlock, NaNL::Unaligned> a = largeTestMatrices->getCopyOfA<NaNL::DeviceMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int, NaNL::DeviceMemoryBlock, NaNL::Unaligned> b = largeTestMatrices->getCopyOfB<NaNL::DeviceMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int, NaNL::PagedMemoryBlock, NaNL::Unaligned> truth = largeTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

        auto c = a.add<NaNL::DeviceMemoryBlock, NaNL::Unaligned>(b);

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        auto hostC = c.copyTo<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

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