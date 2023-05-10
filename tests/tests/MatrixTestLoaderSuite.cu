//
// Created by steve on 12/9/2022.
//


#include <gtest/gtest.h>
#include <Matrix.cuh>
#include <MatrixFileLoader.cuh>


TEST(MatrixTestLoaderSuite, Should_Parse_Matrix_From_File_When_Provided_Small_Good_File) {
    MatrixFileLoader fileLoader("data/MATRIX_FILELOADER_TEST_SMALL");

    ASSERT_EQ(fileLoader.getTotalMatrices(), 3);

    NaNL::Matrix<int, NaNL::Device::Host> firstMatrix = fileLoader.readValuesIntoMatrix<int, NaNL::Device::Host>();

    // make sure sizes are same.
    ASSERT_EQ(firstMatrix.getRows(), 2);
    ASSERT_EQ(firstMatrix.getCols(), 2);
    ASSERT_EQ(firstMatrix.getTotalSize(), 4);

    ASSERT_EQ(firstMatrix[0][0], 10);
    ASSERT_EQ(firstMatrix[0][1], 9);
    ASSERT_EQ(firstMatrix[1][0], 8);
    ASSERT_EQ(firstMatrix[1][1], 7);

    NaNL::Matrix<int, NaNL::Device::Host> secondMatrix = fileLoader.readValuesIntoMatrix<int, NaNL::Device::Host>();

    ASSERT_EQ(secondMatrix.getRows(), 2);
    ASSERT_EQ(secondMatrix.getCols(), 2);
    ASSERT_EQ(secondMatrix.getTotalSize(), 4);

    ASSERT_EQ(secondMatrix[0][0], 6);
    ASSERT_EQ(secondMatrix[0][1], 5);
    ASSERT_EQ(secondMatrix[1][0], 4);
    ASSERT_EQ(secondMatrix[1][1], 3);

    NaNL::Matrix<int, NaNL::Device::Host> thirdMatrix = fileLoader.readValuesIntoMatrix<int, NaNL::Device::Host>();

    ASSERT_EQ(thirdMatrix.getRows(), 3);
    ASSERT_EQ(thirdMatrix.getCols(), 3);
    ASSERT_EQ(thirdMatrix.getTotalSize(), 9);

    ASSERT_EQ(thirdMatrix[0][0], 16);
    ASSERT_EQ(thirdMatrix[0][1], 14);
    ASSERT_EQ(thirdMatrix[0][2], 12);
    ASSERT_EQ(thirdMatrix[1][0], 10);
    ASSERT_EQ(thirdMatrix[1][1], 8);
    ASSERT_EQ(thirdMatrix[1][2], 6);
    ASSERT_EQ(thirdMatrix[2][0], 4);
    ASSERT_EQ(thirdMatrix[2][1], 2);
    ASSERT_EQ(thirdMatrix[2][2], 0);

}

TEST(MatrixTestLoaderSuite, Should_Parse_Matrix_From_File_When_Provided_Large_Good_File) {
    MatrixFileLoader fileLoader("data/MATRIX_FILELOADER_TEST_LARGE");

    ASSERT_EQ(fileLoader.getTotalMatrices(), 4);

    NaNL::Matrix<int, NaNL::Device::Host> firstMatrix = fileLoader.readValuesIntoMatrix<int, NaNL::Device::Host>();
    NaNL::Matrix<int, NaNL::Device::Host> secondMatrix = fileLoader.readValuesIntoMatrix<int, NaNL::Device::Host>();

    // make sure sizes are same.
    ASSERT_EQ(firstMatrix.getRows(), 1000);
    ASSERT_EQ(firstMatrix.getCols(), 1000);
    ASSERT_EQ(firstMatrix.getTotalSize(), 1000000);

    ASSERT_EQ(secondMatrix.getRows(), 1000);
    ASSERT_EQ(secondMatrix.getCols(), 1000);
    ASSERT_EQ(secondMatrix.getTotalSize(), 1000000);

    for(unsigned int i = 0; i < firstMatrix.getRows(); i++) {
        for(unsigned int j = 0; j < firstMatrix.getCols(); j++) {
            ASSERT_EQ(firstMatrix.get(i,j), secondMatrix.get(i,j));
        }
    }

    NaNL::Matrix<int, NaNL::Device::Host> thirdMatrix = fileLoader.readValuesIntoMatrix<int, NaNL::Device::Host>();
    NaNL::Matrix<int, NaNL::Device::Host> fourthMatrix = fileLoader.readValuesIntoMatrix<int, NaNL::Device::Host>();

    // make sure sizes are same.
    ASSERT_EQ(thirdMatrix.getRows(), 240);
    ASSERT_EQ(thirdMatrix.getCols(), 1800);
    ASSERT_EQ(thirdMatrix.getTotalSize(), 432000);

    ASSERT_EQ(fourthMatrix.getRows(), 240);
    ASSERT_EQ(fourthMatrix.getCols(), 1800);
    ASSERT_EQ(fourthMatrix.getTotalSize(), 432000);

    for(unsigned int i = 0; i < thirdMatrix.getRows(); i++) {
        for(unsigned int j = 0; j < thirdMatrix.getCols(); j++) {
            ASSERT_EQ(thirdMatrix.get(i,j), fourthMatrix.get(i,j));
        }
    }
}