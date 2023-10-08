//
// Created by steve on 9/20/2023.
//

#ifndef NANL_MATRIXINBINARYFILELOADER_CUH
#define NANL_MATRIXINBINARYFILELOADER_CUH

#include "../../BaseMatrix/BaseMatrix.cuh"
#include "../Matrix.cuh"

namespace NaNL {
    class MatrixInBinaryFileLoader {
    private:
        unsigned long totalMatrices = 0;
        unsigned long currentMatrix = 0;
        bool closed;
        FILE* file;
    public:
        inline MatrixInBinaryFileLoader() = delete;
        explicit inline MatrixInBinaryFileLoader(const std::string& file);

        inline unsigned long getTotalMatrices();
        inline unsigned long getCurrentMatrix();

        template<class T, template<class, class> class Memory = NaNL::PagedMemoryBlock,
                class Alignment = NaNL::Unaligned>
        inline Matrix<T, Memory, Alignment> read();

        inline ~MatrixInBinaryFileLoader();

        inline void close();
    };

} // NaNL

#include "MatrixInBinaryFileLoader.cu"

#endif //NANL_MATRIXINBINARYFILELOADER_CUH
