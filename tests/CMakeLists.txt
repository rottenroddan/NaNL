# 'Google_test' is the subproject name
cmake_minimum_required(VERSION 3.26.4)
project(Tests LANGUAGES CUDA CXX)

# 'lib' is the folder with Google Test sources
add_subdirectory(lib/googletest-main)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# Copies the files from the data folder and inserts them into the current binary directory.
# Replaces existing ones by deleting them first.
#file(GLOB_RECURSE MY_FILES ${CMAKE_CURRENT_BINARY_DIR}/data/*)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data/ )

file(GLOB BASE_MATRIX_TEST_SRC CONFIGURE_DEPENDS "tests/Matrix/*/*TestSuite.cu")


set(MATRIX_DATA_MANIP_TESTS "tests/Matrix/MatrixDataManipulationSuite/MatrixPagedCopySuite/MatrixPagedCopyTest.cu")
set(MATRIX_DATA_MANIP_SOURCE ${MATRIX_DATA_MANIP_TESTS})

#file(GLOB MATRIX_ARITHMETIC_TEST_SRC CONFIGURE_DEPENDS "tests/Matrix/MatrixArithmeticSuites/*/*Test.cu")


#list(FILTER MATRIX_ARITHMETIC_SUITES_TEST_SRC INCLUDE REGEX "tests/Matrix/MatrixArithmeticSuites/*/.*Test\\.cu$")

# file(GLOB BASE_MATRIX_TEST_SRC CONFIGURE_DEPENDS "tests/Matrix/BaseMatrixTestSuite/*.cu")
# file(GLOB MATRIX_CONSTRUCTOR_TEST_SRC CONFIGURE_DEPENDS "tests/Matrix/MatrixConstructorSuite/*.cu")
# file(GLOB ARITHMETIC_TEST_SRC CONFIGURE_DEPENDS "tests/Matrix/MatrixArithmeticSuite/*.cu")

# 'Google_Tests_run' is the target name
# 'test1.cpp tests2.cpp' are source files with tests
add_executable(Tests
        ${BASE_MATRIX_TEST_SRC}
        tests/Matrix/MatrixArithmeticSuites/MatrixArithmeticRunner/MatrixArithmeticRunner.cu
        tests/Matrix/MatrixDataManipulationSuite/MatrixPagedCopySuite/MatrixPagedCopySuite.cu
        tests/Matrix/MatrixDataManipulationSuite/MatrixPagedCopySuite/MatrixPagedCopyTest.cu
        tests/Matrix/MatrixDataManipulationSuite/MatrixPinnedCopySuite/MatrixPinnedCopySuite.cu
        tests/Matrix/MatrixDataManipulationSuite/MatrixPinnedCopySuite/MatrixPinnedCopyTest.cu
        tests/Matrix/MatrixDataManipulationSuite/MatrixDeviceMoveSuite/MatrixDeviceMoveSuite.cu
        tests/Matrix/MatrixDataManipulationSuite/MatrixDeviceMoveSuite/MatrixDeviceMoveTest.cu
        tests/MatrixTestLoaderSuite.cu
        tests/ThreadPoolTestSuite.cu)

target_link_libraries(Tests Matrix)
target_link_libraries(Tests gtest gtest_main)

set(CMAKE_CXX_STANDARD  20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Move files from test/data to BINARIES/test/data
include_directories(${CMAKE_CURRENT_BINARY_DIR}/data/)
add_custom_command(
        TARGET Tests PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/data/
        ${PROJECT_BINARY_DIR}/data/
)