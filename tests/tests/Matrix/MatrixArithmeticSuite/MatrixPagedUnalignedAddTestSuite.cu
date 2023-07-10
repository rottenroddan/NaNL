//
// Created by steve on 6/6/2023.
//

#include "gtest/gtest.h"
#include "Matrix.cuh"
#include "MatrixFileLoader/MatrixFileLoader.cuh"


class MatrixAddTest : public ::testing::Test {
private:
    class TestMatrices {
        NaNL::Matrix<int> a;
        NaNL::Matrix<int> b;
        NaNL::Matrix<int> truth;
    public:
        TestMatrices(std::string filePath) {
            MatrixFileLoader fileLoader(filePath);

            // read file into Matrices.
            a = fileLoader.readValuesIntoMatrix<int>();
            b = fileLoader.readValuesIntoMatrix<int>();
            truth = fileLoader.readValuesIntoMatrix<int>();
        }

        template<template<class, class> class rMemory, class rAlignment>
        NaNL::Matrix<int, rMemory, rAlignment> getCopyOfA() {
            return a.template copyTo<rMemory, rAlignment>();
        }

        template<template<class, class> class rMemory, class rAlignment>
        NaNL::Matrix<int, rMemory, rAlignment> getCopyOfB() {
            return b.template copyTo<rMemory, rAlignment>();
        }

        template<template<class, class> class rMemory, class rAlignment>
        NaNL::Matrix<int, rMemory, rAlignment> getCopyOfTruth() {
            return truth.template copyTo<rMemory, rAlignment>();
        }
    };
protected:
    static TestMatrices* smallTestMatrices;
    static TestMatrices* mediumTestMatrices;
    static TestMatrices* largeTestMatrices;

    static void SetUpTestSuite() {
        smallTestMatrices = new TestMatrices("data/MATRIX_ADDITION_TEST_100x100");
        mediumTestMatrices = new TestMatrices("data/MATRIX_ARITHMETIC_TEST_1327x1823");
        largeTestMatrices = new TestMatrices("data/MATRIX_ADDITION_TEST_5000x5000");
    }

    static void TearDownTestSuite() {
        delete smallTestMatrices;
        delete mediumTestMatrices;
        delete largeTestMatrices;
    }
};

MatrixAddTest::TestMatrices* MatrixAddTest::smallTestMatrices = nullptr;
MatrixAddTest::TestMatrices* MatrixAddTest::mediumTestMatrices = nullptr;
MatrixAddTest::TestMatrices* MatrixAddTest::largeTestMatrices = nullptr;

TEST_F(MatrixAddTest, Should_Add_Small_Matrices_To_Correct_Values_When_Host_UnAligned) {
    try {
        NaNL::Matrix<int> a = smallTestMatrices->getCopyOfA<NaNL::PagedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int> b = smallTestMatrices->getCopyOfB<NaNL::PagedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int> truth = smallTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

        auto c = a.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixDeviceOperation::Host);

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        for (unsigned int i = 0; i < c.getRows(); i++) {
            for (unsigned int j = 0; j < c.getCols(); j++) {
                ASSERT_EQ(c[i][j], truth[i][j]);
            }
        }
    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}

TEST_F(MatrixAddTest, Should_Add_Medium_Matrices_To_Correct_Values_When_Host_UnAligned) {
    try {
        NaNL::Matrix<int> a = mediumTestMatrices->getCopyOfA<NaNL::PagedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int> b = mediumTestMatrices->getCopyOfB<NaNL::PagedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int> truth = mediumTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

        auto c = a.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixDeviceOperation::Host);

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        for (unsigned int i = 0; i < c.getRows(); i++) {
            for (unsigned int j = 0; j < c.getCols(); j++) {
                ASSERT_EQ(c[i][j], truth[i][j]);
            }
        }
    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}

TEST_F(MatrixAddTest, Should_Add_Large_Matrices_To_Correct_Values_When_Host_UnAligned) {
    try {
        NaNL::Matrix<int> a = largeTestMatrices->getCopyOfA<NaNL::PagedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int> b = largeTestMatrices->getCopyOfB<NaNL::PagedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int> truth = largeTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

        auto c = a.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(b);

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        for (unsigned int i = 0; i < c.getRows(); i++) {
            for (unsigned int j = 0; j < c.getCols(); j++) {
                ASSERT_EQ(c[i][j], truth[i][j]);
            }
        }
    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}

/*
TEST_F(MatrixAddTest, Should_Add_Small_Matrices_To_Correct_Values_When_Pinned_UnAligned) {
    try {
        NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> a = smallTestMatrices->getCopyOfA<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> b = smallTestMatrices->getCopyOfB<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> truth = smallTestMatrices->getCopyOfTruth<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();

        auto c = a.add<NaNL::PinnedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixDeviceOperation::Host);

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        for (unsigned int i = 0; i < c.getRows(); i++) {
            for (unsigned int j = 0; j < c.getCols(); j++) {
                ASSERT_EQ(c[i][j], truth[i][j]);
            }
        }
    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}

TEST_F(MatrixAddTest, Should_Add_Medium_Matrices_To_Correct_Values_When_Pinned_UnAligned) {
    try {
        NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> a = mediumTestMatrices->getCopyOfA<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> b = mediumTestMatrices->getCopyOfB<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> truth = mediumTestMatrices->getCopyOfTruth<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();

        auto c = a.add<NaNL::PinnedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixDeviceOperation::Host);

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        for (unsigned int i = 0; i < c.getRows(); i++) {
            for (unsigned int j = 0; j < c.getCols(); j++) {
                ASSERT_EQ(c[i][j], truth[i][j]);
            }
        }
    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}

TEST_F(MatrixAddTest, Should_Add_Large_Matrices_To_Correct_Values_When_Pinned_UnAligned) {
    try {
        NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> a = largeTestMatrices->getCopyOfA<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> b = largeTestMatrices->getCopyOfB<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();
        NaNL::Matrix<int, NaNL::PinnedMemoryBlock, NaNL::Unaligned> truth = largeTestMatrices->getCopyOfTruth<NaNL::PinnedMemoryBlock, NaNL::Unaligned>();

        auto c = a.add<NaNL::PinnedMemoryBlock, NaNL::Unaligned>(b);

        ASSERT_EQ(c.getRows(), truth.getRows());
        ASSERT_EQ(c.getCols(), truth.getCols());

        for (unsigned int i = 0; i < c.getRows(); i++) {
            for (unsigned int j = 0; j < c.getCols(); j++) {
                ASSERT_EQ(c[i][j], truth[i][j]);
            }
        }
    } catch (std::exception e) {
        std::cout << e.what();
        FAIL();
    }
}*/