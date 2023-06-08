//
// Created by steve on 2/26/2023.
//
#include <iostream>
#include <memory>
#include <PagedMemoryBlock.cuh>
#include <PinnedMemoryBlock.cuh>
#include <../Alignment/Unaligned.cuh>
#include "../src/BaseMatrix/BaseMatrix.cuh"
#include <Matrix.cuh>
#include <type_traits>

//template<class T, template<typename> class Memory = NaNL::PagedMemoryBlock,
//        template<class, template<typename > class >class Alignment = NaNL::Unaligned>
//class BaseMatrix : protected Alignment<T,Memory> {
//public:
//    static_assert(std::is_base_of<NaNL::BaseAlignment<T, Memory> ,Alignment<T, Memory>>::value, "Template argument 'Alignment' must inherit from BaseAlignment class." );
//
//    BaseMatrix(uint64_t rows, uint64_t cols) : Alignment<T, Memory>(rows, cols)  {
//
//    }
//
//};

class B;

class A {
private:
    friend class B;
    void privateMethod() {
        std::cout << "Private" << std::endl;
    }
public:
    void sendHer();
};

class B {
public:
    static void hello() {
        A obj;
        obj.privateMethod();
    }
};

void A::sendHer() {
    B::hello();
}


int main() {

    //NaNL::Matrix<int> x(100, 100);

    A obj;
    obj.sendHer();

    NaNL::Matrix<int> p(100, 100);

    auto x = p.add<NaNL::PagedMemoryBlock, NaNL::Unaligned>(p);

    std::cout << x.getTotalSize() << std::endl;

    //NaNL::Unaligned<int, NaNL::PagedMemoryBlock> un(100, 100);

    //BaseMatrix<int> a(100 , 100);

    return 0;
}