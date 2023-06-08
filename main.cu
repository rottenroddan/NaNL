#include <chrono>
#include <iostream>
#include <Matrix.cuh>
#include "MatrixFileLoader/MatrixFileLoader.cuh"

/*
template <typename T>
void testShit() {
    T* _dev_a;
    T* a = new T[10000];

    cudaMalloc((void**)&_dev_a, 10000 * sizeof(T));


    cudaError_t err = cudaMemcpy(_dev_a, a, 10000 * sizeof(T), cudaMemcpyHostToDevice);
    std::cout << cudaGetErrorName(err) << " : " << cudaGetErrorString(err) << std::endl;
}*/


#define ADD_SIZE 20000

//void host() {
//    typedef std::chrono::high_resolution_clock Clock;
//
//    NaNL::Matrix<unsigned long long, NaNL::Device::Host> x(ADD_SIZE,ADD_SIZE);
//    NaNL::Matrix<unsigned long long, NaNL::Device::Host> y(ADD_SIZE,ADD_SIZE);
//
//    auto hostTimerStart = Clock::now();
//    x.add(y);
//    auto hostTimerEnd = Clock::now();
//
//    std::cout << "PagedUnalligned add time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(hostTimerEnd - hostTimerStart).count() << " nanoseconds." << std::endl;
//
//}
//
//void cuda() {
//    typedef std::chrono::high_resolution_clock Clock;
//
//    NaNL::Matrix<unsigned long long, NaNL::Device::Cuda> xx(ADD_SIZE,ADD_SIZE);
//    NaNL::Matrix<unsigned long long, NaNL::Device::Cuda> yy(ADD_SIZE,ADD_SIZE);
//
//    auto cudaTimerStart = Clock::now();
//    xx.add(yy);
//    auto cudaTimerEnd = Clock::now();
//
//    std::cout << "Cuda add time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(cudaTimerEnd - cudaTimerStart).count() << " nanoseconds." << std::endl;
//}

int main() {

    unsigned long long MAX_PINNED_SIZE = 15.1*1024l*1024l * (unsigned long long)1024;
    std::cout << MAX_PINNED_SIZE << " bytes." <<std::endl;
    int* host_arr;

    cudaError_t x = cudaMallocHost((void**)&host_arr, MAX_PINNED_SIZE);
    std::cout << cudaGetErrorString(x) << std::endl;

    /*
    NaNL::Matrix<int, NaNL::Device::Cuda> cudaWarmUp(1000, 1000);
    cudaWarmUp.add(cudaWarmUp);

    std::cout << host_arr[0] << std::endl;
    std::cout << host_arr[1] << std::endl;
    std::cout << host_arr[2] << std::endl;
    std::cout << host_arr[3] << std::endl;
    std::cout << host_arr[4] << std::endl;
    std::cout << host_arr[5] << std::endl;

    host();
    host();
    host();
    cuda();
    cuda();
    cuda();*/

    return 0;
}
