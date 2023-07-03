//
// Created by steve on 6/10/2023.
//

#include "BaseMemoryBlock.cuh"

namespace NaNL {

    namespace Internal {

        template<typename T>
        BaseMemoryBlock<T>::BaseMemoryBlock() : BaseMemoryBlock(NaNL::Internal::MemoryTypes::Host) { ; }

        template<typename T>
        BaseMemoryBlock<T>::BaseMemoryBlock(MemoryTypes type) : _matrix(nullptr, nullptr) , memoryType(type) { ; }

        template<typename T>
        MemoryTypes BaseMemoryBlock<T>::getMemoryType() const {
            return this->memoryType;
        }

        template<typename T>
        inline int64_t BaseMemoryBlock<T>::getCudaDevice() const {
            return this->cudaDevice;
        }

        template<typename T>
        inline void BaseMemoryBlock<T>::setCudaDevice(int64_t device) {
            this->cudaDevice = device;
        }

        template<typename T>
        inline T* BaseMemoryBlock<T>::getMatrix() const {
            return this->_matrix.get();
        }

        template<typename T>
        bool BaseMemoryBlock<T>::isDeleted() const {
            return this->deleted;
        }

        template<typename T>
        void BaseMemoryBlock<T>::setDeleteFlag(bool deleted) {
            this->deleted = deleted;
        }
    }

} // NaNL