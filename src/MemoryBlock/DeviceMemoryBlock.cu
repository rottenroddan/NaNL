//
// Created by steve on 6/10/2023.
//

#include "DeviceMemoryBlock.cuh"

namespace NaNL {
    template<class T>
    NaNL::DeviceMemoryBlock<T>::DeviceMemoryBlock(uint64_t totalSize) :
    Internal::BaseMemoryBlock<T>(Internal::MemoryTypes::CudaDevice)
    {
        T *_deviceArr;
        gpuErrchk(cudaMalloc((void **) &_deviceArr, totalSize * sizeof(T)));
        this->_matrix = std::unique_ptr<T[], void (*)(T*)>(_deviceArr, _freeDeviceMemory);

        // set cuda device
        cudaPointerAttributes attribute;
        gpuErrchk(cudaPointerGetAttributes(&attribute, _deviceArr));
        this->setCudaDevice(attribute.device);
    }
} // NaNL