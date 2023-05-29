//
// Created by steve on 2/26/2023.
//
#include <iostream>
#include <memory>
#include <PagedMemoryBlock.cuh>
#include <Alignment/Unaligned.cuh>
#include <type_traits>

template<class T, template<typename> class Memory = NaNL::PagedMemoryBlock,
        template<class, template<typename > class >class Alignment = NaNL::Unaligned>
class BaseMatrix : protected Alignment<T,Memory> {
public:
    static_assert(std::is_base_of<NaNL::BaseAlignment<T, Memory> ,Alignment<T, Memory>>::value, "Template argument 'Alignment' must inherit from BaseAlignment class." );

    BaseMatrix(uint64_t rows, uint64_t cols) : Alignment<T, Memory>(rows, cols)  {

    }

};




int main() {

    //NaNL::Unaligned<int, NaNL::PagedMemoryBlock> un(100, 100);

    BaseMatrix<int> a(100 , 100);

    return 0;
}