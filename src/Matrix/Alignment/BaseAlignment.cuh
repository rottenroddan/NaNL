//
// Created by steve on 5/26/2023.
//

#ifndef NANL_BASEALIGNMENT_CUH
#define NANL_BASEALIGNMENT_CUH

namespace NaNL {
    template<class T, template<typename> class Memory>
    class BaseAlignment : protected Memory<T> {
    protected:
        BaseAlignment(uint64_t rows, uint64_t cols);

        uint64_t rows{}, actualRows{};
        uint64_t cols{}, actualCols{};
        uint64_t totalSize{}, actualTotalSize{};
    public:
        virtual inline uint64_t getRows() = 0;

        virtual inline uint64_t getActualRows() = 0;

        virtual inline uint64_t getCols() = 0;

        virtual inline uint64_t getActualCols() = 0;

        virtual inline uint64_t getTotalSize() = 0;

        virtual inline uint64_t getActualTotalSize() = 0;
    };
}



#include "BaseAlignment.cu"

#endif //NANL_BASEALIGNMENT_CUH
