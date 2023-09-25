//
// Created by steve on 8/11/2023.
//

#include "MatrixMultiCudaAddSuite.cuh"

TestMatrices* MatrixMultiCudaAddSuite::smallTestMatrices = nullptr;
TestMatrices* MatrixMultiCudaAddSuite::mediumTestMatrices = nullptr;
TestMatrices* MatrixMultiCudaAddSuite::largeTestMatrices = nullptr;

void MatrixMultiCudaAddSuite::SetUpTestSuite() {
    smallTestMatrices = new TestMatrices("data/bin/MATRIX_ADDITION_TEST_100x100.bin");
    mediumTestMatrices = new TestMatrices("data/bin/MATRIX_ADDITION_TEST_1327x1823.bin");
    largeTestMatrices = new TestMatrices("data/bin/MATRIX_ADDITION_TEST_5000x5000.bin");
}

void MatrixMultiCudaAddSuite::TearDownTestSuite() {
    delete smallTestMatrices;
    delete mediumTestMatrices;
    delete largeTestMatrices;
}