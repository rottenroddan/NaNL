//
// Created by steve on 6/10/2023.
//

#include "BaseMemoryBlock.cuh"

namespace NaNL {

    namespace Internal {

        template<typename T, typename Alignment>
        BaseMemoryBlock<T, Alignment>::BaseMemoryBlock(uint64_t rows, uint64_t cols) : BaseMemoryBlock(rows, cols, NaNL::Internal::MemoryTypes::Host) { ; }

        template<typename T, typename Alignment>
        BaseMemoryBlock<T, Alignment>::BaseMemoryBlock(uint64_t rows, uint64_t cols, MemoryTypes type) : _matrix(nullptr, nullptr),
        memoryType(type),
        Alignment(rows, cols) { ; }

        template<typename T, typename Alignment>
        MemoryTypes BaseMemoryBlock<T, Alignment>::getMemoryType() const {
            return this->memoryType;
        }

        template<typename T, typename Alignment>
        inline int64_t BaseMemoryBlock<T, Alignment>::getCudaDevice() const {
            return this->cudaDevice;
        }

        template<typename T, typename Alignment>
        inline void BaseMemoryBlock<T, Alignment>::setCudaDevice(int64_t device) {
            this->cudaDevice = device;
        }

        template<typename T, typename Alignment>
        inline T* BaseMemoryBlock<T, Alignment>::getMatrix() const {
            return this->_matrix.get();
        }

        template<typename T, typename Alignment>
        bool BaseMemoryBlock<T, Alignment>::isDeleted() const {
            return this->deleted;
        }

        template<typename T, typename Alignment>
        void BaseMemoryBlock<T, Alignment>::setDeleteFlag(bool deleted) {
            this->deleted = deleted;
        }
    }

} // NaNL