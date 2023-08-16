//
// Created by steve on 6/10/2023.
//

#include "DeviceMemoryBlock.cuh"

namespace NaNL {
    template<class T, typename Alignment>
    DeviceMemoryBlock<T, Alignment>::DeviceMemoryBlock(uint64_t rows, uint64_t cols) :
    Internal::BaseMemoryBlock<T, Alignment>(rows, cols, Internal::MemoryTypes::CudaDevice)
    {
        T *_deviceArr;
        gpuErrchk(cudaMalloc((void **) &_deviceArr, this->actualRows * this->actualCols * sizeof(T)));
        this->_matrix = std::unique_ptr<T[], void (*)(T*)>(_deviceArr, _freeDeviceMemory);

        // set cuda device
        cudaPointerAttributes attribute;
        gpuErrchk(cudaPointerGetAttributes(&attribute, _deviceArr));
        this->setCudaDevice(attribute.device);
    }
} // NaNL