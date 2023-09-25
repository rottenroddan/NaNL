//
// Created by steve on 7/16/2023.
//

#include "MatrixHostAddSuite.cuh"

TestMatrices* MatrixHostAddSuite::smallTestMatrices = nullptr;
TestMatrices* MatrixHostAddSuite::mediumTestMatrices = nullptr;
TestMatrices* MatrixHostAddSuite::largeTestMatrices = nullptr;

void MatrixHostAddSuite::SetUpTestSuite() {
    smallTestMatrices = new TestMatrices("data/bin/MATRIX_ADDITION_TEST_100x100.bin");
    mediumTestMatrices = new TestMatrices("data/bin/MATRIX_ADDITION_TEST_1327x1823.bin");
    largeTestMatrices = new TestMatrices("data/bin/MATRIX_ADDITION_TEST_5000x5000.bin");
}

void MatrixHostAddSuite::TearDownTestSuite() {
    delete smallTestMatrices;
    delete mediumTestMatrices;
    delete largeTestMatrices;
}