# 'Google_test' is the subproject name
project(Tests LANGUAGES CUDA CXX)

# 'lib' is the folder with Google Test sources
add_subdirectory(lib/googletest-main)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# Copies the files from the data folder and inserts them into the current binary directory.
# Replaces existing ones by deleting them first.
file(GLOB_RECURSE MY_FILES ${CMAKE_CURRENT_BINARY_DIR}/data/*)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data/ )
include_directories(${CMAKE_CURRENT_BINARY_DIR}/data/)

# 'Google_Tests_run' is the target name
# 'test1.cpp tests2.cpp' are source files with tests
add_executable(Tests tests/BaseMatrixConstructorTestSuite.cu
        tests/BaseMatrixTestSuite.cu
        tests/MatrixTestLoaderSuite.cu
        tests/MatrixAddSubtractTestSuite.cu
        tests/ThreadPoolTestSuite.cu)

target_link_libraries(Tests Matrix)
target_link_libraries(Tests gtest gtest_main)

#add_test()