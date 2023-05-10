//
// Created by steve on 5/6/2023.
//

#include <iostream>
#include <Matrix.cuh>
#include <ThreadPool.cuh>

//void testAdd_host_32768_32768() {
//    PERFORMANCE_LOGGING_BEGIN
//
//    NaNL::Matrix<double, NaNL::Device::Host> a(32768, 32768);
//    NaNL::Matrix<double, NaNL::Device::Host> b(32768, 32768);
//
//    a.add(b);
//
//    PERFORMANCE_LOGGING_END
//    PERFORMANCE_LOGGING_LOG
//}
//
//void testAdd_cuda_32768_32768() {
//    PERFORMANCE_LOGGING_BEGIN
//
//    NaNL::Matrix<double, NaNL::Device::Cuda> a(32768, 32768);
//    NaNL::Matrix<double, NaNL::Device::Cuda> b(32768, 32768);
//
//    a.add(b);
//
//    PERFORMANCE_LOGGING_END
//    PERFORMANCE_LOGGING_LOG
//}

void testAdd_host_16384_16384() {
    PERFORMANCE_LOGGING_BEGIN

    NaNL::Matrix<double, NaNL::Device::Host> a(16384,16384);
    NaNL::Matrix<double, NaNL::Device::Host> b(16384,16384);

    a.add(b);

    PERFORMANCE_LOGGING_END
    PERFORMANCE_LOGGING_LOG
}

void testAdd_cuda_16384_16384() {
    PERFORMANCE_LOGGING_BEGIN

    NaNL::Matrix<double, NaNL::Device::Cuda> a(16384,16384);
    NaNL::Matrix<double, NaNL::Device::Cuda> b(16384,16384);

    a.add(b);

    PERFORMANCE_LOGGING_END
    PERFORMANCE_LOGGING_LOG
}

void testAdd_host_8192_8192() {
    PERFORMANCE_LOGGING_BEGIN

    NaNL::Matrix<double, NaNL::Device::Host> a(8192,8192);
    NaNL::Matrix<double, NaNL::Device::Host> b(8192,8192);

    a.add(b);

    PERFORMANCE_LOGGING_END
    PERFORMANCE_LOGGING_LOG
}

void testAdd_cuda_8192_8192() {
    PERFORMANCE_LOGGING_BEGIN

    NaNL::Matrix<double, NaNL::Device::Cuda> a(8192,8192);
    NaNL::Matrix<double, NaNL::Device::Cuda> b(8192,8192);

    a.add(b);

    PERFORMANCE_LOGGING_END
    PERFORMANCE_LOGGING_LOG
}

void testAdd_host_4096_4096() {
    PERFORMANCE_LOGGING_BEGIN

    NaNL::Matrix<double, NaNL::Device::Host> a(4096,4096);
    NaNL::Matrix<double, NaNL::Device::Host> b(4096,4096);

    a.add(b);

    PERFORMANCE_LOGGING_END
    PERFORMANCE_LOGGING_LOG
}

void testAdd_cuda_4096_4096() {
    PERFORMANCE_LOGGING_BEGIN

    NaNL::Matrix<double, NaNL::Device::Cuda> a(4096,4096);
    NaNL::Matrix<double, NaNL::Device::Cuda> b(4096,4096);

    a.add(b);

    PERFORMANCE_LOGGING_END
    PERFORMANCE_LOGGING_LOG
}

void testAdd_host_2048_2048() {
    PERFORMANCE_LOGGING_BEGIN

    NaNL::Matrix<double, NaNL::Device::Host> a(2048,2048);
    NaNL::Matrix<double, NaNL::Device::Host> b(2048,2048);

    a.add(b);

    PERFORMANCE_LOGGING_END
    PERFORMANCE_LOGGING_LOG
}

void testAdd_cuda_2048_2048() {
    PERFORMANCE_LOGGING_BEGIN

    NaNL::Matrix<double, NaNL::Device::Cuda> a(2048,2048);
    NaNL::Matrix<double, NaNL::Device::Cuda> b(2048,2048);

    a.add(b);

    PERFORMANCE_LOGGING_END
    PERFORMANCE_LOGGING_LOG
}

void testAdd_host_1024_1024() {
    PERFORMANCE_LOGGING_BEGIN

    NaNL::Matrix<double, NaNL::Device::Host> a(1024,1024);
    NaNL::Matrix<double, NaNL::Device::Host> b(1024,1024);

    a.add(b);

    PERFORMANCE_LOGGING_END
    PERFORMANCE_LOGGING_LOG
}

void testAdd_cuda_1024_1024() {
    PERFORMANCE_LOGGING_BEGIN

    NaNL::Matrix<double, NaNL::Device::Cuda> a(1024,1024);
    NaNL::Matrix<double, NaNL::Device::Cuda> b(1024,1024);

    a.add(b);

    PERFORMANCE_LOGGING_END
    PERFORMANCE_LOGGING_LOG
}

void testAdd_host_512_512() {
    PERFORMANCE_LOGGING_BEGIN

    NaNL::Matrix<double, NaNL::Device::Host> a(512,512);
    NaNL::Matrix<double, NaNL::Device::Host> b(512,512);

    a.add(b);

    PERFORMANCE_LOGGING_END
    PERFORMANCE_LOGGING_LOG
}

void testAdd_cuda_512_512() {
    PERFORMANCE_LOGGING_BEGIN

    NaNL::Matrix<double, NaNL::Device::Cuda> a(512,512);
    NaNL::Matrix<double, NaNL::Device::Cuda> b(512,512);

    a.add(b);

    PERFORMANCE_LOGGING_END
    PERFORMANCE_LOGGING_LOG
}

void test_all_add() {
//    testAdd_host_32768_32768();
//    testAdd_cuda_32768_32768();

    testAdd_host_16384_16384();
    testAdd_cuda_16384_16384();
    testAdd_cuda_16384_16384();

    testAdd_host_8192_8192();
    testAdd_cuda_8192_8192();

    testAdd_host_4096_4096();
    testAdd_cuda_4096_4096();

    testAdd_host_2048_2048();
    testAdd_cuda_2048_2048();

    testAdd_host_1024_1024();
    testAdd_cuda_1024_1024();

    testAdd_host_512_512();
    testAdd_cuda_512_512();
}