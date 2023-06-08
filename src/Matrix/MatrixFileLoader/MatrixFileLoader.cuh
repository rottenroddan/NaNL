//
// Created by steve on 12/17/2022.
//


#ifndef NANL_MATRIXFILELOADER_CUH
#define NANL_MATRIXFILELOADER_CUH

#include "../../BaseMatrix/BaseMatrix.cuh"
#include "Matrix.cuh"

#include <deque>
#include <fstream>
#include <sstream>
#include <string>

class MatrixFileLoader {
private:
    unsigned long totalMatrices = 0;
    unsigned long currentMatrix = 0;
    std::ifstream file;
public:
    explicit inline MatrixFileLoader(const std::string& file);

    inline unsigned long getTotalMatrices();
    inline unsigned long getCurrentMatrix();

    template<class T, template<typename> class Memory = NaNL::PagedMemoryBlock,
            template<class, template<typename> class> class Alignment = NaNL::Unaligned>
    inline NaNL::Matrix<T, Memory, Alignment> readValuesIntoMatrix();

    inline ~MatrixFileLoader();
};

#include "MatrixFileLoader.cu"

#endif //NANL_MATRIXFILELOADER_CUH
