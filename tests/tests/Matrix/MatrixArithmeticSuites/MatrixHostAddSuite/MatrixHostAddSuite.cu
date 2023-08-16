//
// Created by steve on 7/16/2023.
//

#include "MatrixHostAddSuite.cuh"

TestMatrices* MatrixHostAddSuite::smallTestMatrices = nullptr;
TestMatrices* MatrixHostAddSuite::mediumTestMatrices = nullptr;
TestMatrices* MatrixHostAddSuite::largeTestMatrices = nullptr;

void MatrixHostAddSuite::SetUpTestSuite() {
    smallTestMatrices = new TestMatrices("data/MATRIX_ADDITION_TEST_100x100");
    mediumTestMatrices = new TestMatrices("data/MATRIX_ARITHMETIC_TEST_1327x1823");
    largeTestMatrices = new TestMatrices("data/MATRIX_ADDITION_TEST_5000x5000");
}

void MatrixHostAddSuite::TearDownTestSuite() {
    delete smallTestMatrices;
    delete mediumTestMatrices;
    delete largeTestMatrices;
}