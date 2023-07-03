//
// Created by steve on 5/26/2023.
//

#ifndef NANL_BASEALIGNMENT_CUH
#define NANL_BASEALIGNMENT_CUH

namespace NaNL {
    template<class T, template<typename> class Memory>
    class BaseAlignment : public Memory<T> {
    protected:
        inline BaseAlignment(uint64_t rows, uint64_t cols);

        uint64_t rows{}, actualRows{};
        uint64_t cols{}, actualCols{};
        uint64_t totalSize{}, actualTotalSize{};
    public:
        virtual inline uint64_t getRows() const = 0;

        virtual inline uint64_t getActualRows() const = 0;

        virtual inline uint64_t getCols() const = 0;

        virtual inline uint64_t getActualCols() const = 0;

        virtual inline uint64_t getTotalSize() const = 0;

        virtual inline uint64_t getActualTotalSize() const = 0;

        virtual inline void align(uint64_t rows, uint64_t cols) = 0;
    };
}



#include "BaseAlignment.cu"

#endif //NANL_BASEALIGNMENT_CUH
