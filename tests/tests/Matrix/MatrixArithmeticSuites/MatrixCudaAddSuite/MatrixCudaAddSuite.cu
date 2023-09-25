//
// Created by steve on 8/11/2023.
//

#include "MatrixCudaAddSuite.cuh"

TestMatrices* MatrixCudaAddSuite::smallTestMatrices = nullptr;
TestMatrices* MatrixCudaAddSuite::mediumTestMatrices = nullptr;
TestMatrices* MatrixCudaAddSuite::largeTestMatrices = nullptr;

void MatrixCudaAddSuite::SetUpTestSuite() {
    smallTestMatrices = new TestMatrices("data/bin/MATRIX_ADDITION_TEST_100x100.bin");
    mediumTestMatrices = new TestMatrices("data/bin/MATRIX_ADDITION_TEST_1327x1823.bin");
    largeTestMatrices = new TestMatrices("data/bin/MATRIX_ADDITION_TEST_5000x5000.bin");
}

void MatrixCudaAddSuite::TearDownTestSuite() {
    delete smallTestMatrices;
    delete mediumTestMatrices;
    delete largeTestMatrices;
}