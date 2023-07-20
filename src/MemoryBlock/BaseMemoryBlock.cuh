//
// Created by steve on 6/10/2023.
//

#ifndef NANL_BASEMEMORYBLOCK_CUH
#define NANL_BASEMEMORYBLOCK_CUH

namespace NaNL {

    namespace Internal {

        enum class MemoryTypes {
            Host, CudaPinned, CudaDevice
        };

        template<typename T, typename Alignment>
        class BaseMemoryBlock : public Alignment {
        private:
            MemoryTypes memoryType;
            int64_t cudaDevice = -1;
        protected:
            bool deleted = false;
            std::unique_ptr<T[], void(*)(T*)> _matrix;

            inline void setDeleteFlag(bool deleted);
        public:
            inline BaseMemoryBlock(uint64_t rows, uint64_t cols);

            explicit inline BaseMemoryBlock(uint64_t rows, uint64_t cols, MemoryTypes type);

            inline MemoryTypes getMemoryType() const;

            inline int64_t getCudaDevice() const;

            inline void setCudaDevice(int64_t);

            inline T* getMatrix() const;

            inline bool isDeleted() const;
        };
    }
} // NaNL

#include "BaseMemoryBlock.cu"

#endif //NANL_BASEMEMORYBLOCK_CUH
