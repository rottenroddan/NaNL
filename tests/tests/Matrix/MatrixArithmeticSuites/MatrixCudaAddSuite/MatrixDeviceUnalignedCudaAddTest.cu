//
// Created by steve on 9/29/2023.
//

TEST_F(MatrixCudaAddSuite, Should_Add_Small_Matrices_To_Correct_Values_When_Device_UnAligned) {
    try {
        auto a = smallTestMatrices->getCopyOfA<NaNL::DeviceMemoryBlock, NaNL::Unaligned>();
        auto b = smallTestMatrices->getCopyOfB<NaNL::DeviceMemoryBlock, NaNL::Unaligned>();
        auto truth = smallTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

        auto startTime = std::chrono::high_resolution_clock::now();

        auto c = a.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixAddOperation::Cuda);

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

TEST_F(MatrixCudaAddSuite, Should_Add_Medium_Matrices_To_Correct_Values_When_Device_UnAligned) {
    try {
        auto a = mediumTestMatrices->getCopyOfA<NaNL::DeviceMemoryBlock, NaNL::Unaligned>();
        auto b = mediumTestMatrices->getCopyOfB<NaNL::DeviceMemoryBlock, NaNL::Unaligned>();
        auto truth = mediumTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

        auto startTime = std::chrono::high_resolution_clock::now();

        auto c = a.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixAddOperation::Cuda);

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

TEST_F(MatrixCudaAddSuite, Should_Add_Large_Matrices_To_Correct_Values_When_Device_UnAligned) {
    try {
        auto a = largeTestMatrices->getCopyOfA<NaNL::DeviceMemoryBlock, NaNL::Unaligned>();
        auto b = largeTestMatrices->getCopyOfB<NaNL::DeviceMemoryBlock, NaNL::Unaligned>();
        auto truth = largeTestMatrices->getCopyOfTruth<NaNL::PagedMemoryBlock, NaNL::Unaligned>();

        auto startTime = std::chrono::high_resolution_clock::now();

        auto c = a.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(b, NaNL::MatrixAddOperation::Cuda);


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