//
// Created by steve on 7/16/2023.
//

#include "MatrixAddSuite.cuh"

TestMatrices* MatrixAddSuite::smallTestMatrices = nullptr;
TestMatrices* MatrixAddSuite::mediumTestMatrices = nullptr;
TestMatrices* MatrixAddSuite::largeTestMatrices = nullptr;

void MatrixAddSuite::SetUpTestSuite() {
    smallTestMatrices = new TestMatrices("data/MATRIX_ADDITION_TEST_100x100");
    mediumTestMatrices = new TestMatrices("data/MATRIX_ARITHMETIC_TEST_1327x1823");
    largeTestMatrices = new TestMatrices("data/MATRIX_ADDITION_TEST_5000x5000");
}

void MatrixAddSuite::TearDownTestSuite() {
    delete smallTestMatrices;
    delete mediumTestMatrices;
    delete largeTestMatrices;
}