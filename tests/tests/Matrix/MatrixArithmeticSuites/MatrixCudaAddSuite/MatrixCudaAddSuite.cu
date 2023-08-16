//
// Created by steve on 8/11/2023.
//

#include "MatrixCudaAddSuite.cuh"

TestMatrices* MatrixCudaAddSuite::smallTestMatrices = nullptr;
TestMatrices* MatrixCudaAddSuite::mediumTestMatrices = nullptr;
TestMatrices* MatrixCudaAddSuite::largeTestMatrices = nullptr;

void MatrixCudaAddSuite::SetUpTestSuite() {
    smallTestMatrices = new TestMatrices("data/MATRIX_ADDITION_TEST_100x100");
    mediumTestMatrices = new TestMatrices("data/MATRIX_ARITHMETIC_TEST_1327x1823");
    largeTestMatrices = new TestMatrices("data/MATRIX_ADDITION_TEST_5000x5000");
}

void MatrixCudaAddSuite::TearDownTestSuite() {
    delete smallTestMatrices;
    delete mediumTestMatrices;
    delete largeTestMatrices;
}