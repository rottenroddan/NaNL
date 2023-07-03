# 'Google_test' is the subproject name
project(Tests LANGUAGES CUDA CXX)

# 'lib' is the folder with Google Test sources
add_subdirectory(lib/googletest-main)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# Copies the files from the data folder and inserts them into the current binary directory.
# Replaces existing ones by deleting them first.
#file(GLOB_RECURSE MY_FILES ${CMAKE_CURRENT_BINARY_DIR}/data/*)
#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data/ )


file(GLOB BASE_MATRIX_TEST_SRC CONFIGURE_DEPENDS "tests/Matrix/*/*.cu")

# file(GLOB BASE_MATRIX_TEST_SRC CONFIGURE_DEPENDS "tests/Matrix/BaseMatrixTestSuite/*.cu")
# file(GLOB MATRIX_CONSTRUCTOR_TEST_SRC CONFIGURE_DEPENDS "tests/Matrix/MatrixConstructorSuite/*.cu")
# file(GLOB ARITHMETIC_TEST_SRC CONFIGURE_DEPENDS "tests/Matrix/MatrixArithmeticSuite/*.cu")

# 'Google_Tests_run' is the target name
# 'test1.cpp tests2.cpp' are source files with tests
add_executable(Tests
        ${BASE_MATRIX_TEST_SRC}
        tests/MatrixTestLoaderSuite.cu
        tests/ThreadPoolTestSuite.cu)

target_link_libraries(Tests Matrix)
target_link_libraries(Tests gtest gtest_main)

# Move files from test/data to BINARIES/test/data
include_directories(${CMAKE_CURRENT_BINARY_DIR}/data/)
add_custom_command(
        TARGET Tests PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/data/
        ${PROJECT_BINARY_DIR}/data/
)