//
// Created by steve on 12/12/2022.
//

#ifndef NANL_BASEMATRIX_CUH
#define NANL_BASEMATRIX_CUH

#ifndef INT_MAX
#define INT_MAX 2147483647
#endif

#include "../Alignment/Unaligned.cuh"
#include "../MemoryBlock/PagedMemoryBlock.cuh"
#include "../MemoryBlock/PinnedMemoryBlock.cuh"
#include "../MemoryBlock/DeviceMemoryBlock.cuh"
#include "../Logger/Logger.cuh"
#include "../MemoryBlock/Deleters.cu"

#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <thread>
#include <concepts>
#include <shared_mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace NaNL {

    template<class T, template<class, class> class Memory = NaNL::PagedMemoryBlock,
            class Alignment = NaNL::Unaligned>
    class BaseMatrix : public Memory<T, Alignment> {
    protected:

        /**
         * Private constructor, derived class must pass number of rows, columns and a function
         * for freeing memory that is allocated.
         * @param numberOfRows Total rows.
         * @param numberOfCols Total columns.
         * @param deleter Function reference to deleter function.
         */
        inline BaseMatrix(uint64_t rows, uint64_t cols);

        /**
         *
         * @param copyOfMatrix
         */
        //inline virtual void _copy(const BaseMatrix &copyOfMatrix) = 0;

    public:
        //static_assert(std::is_base_of<NaNL::BaseAlignment<T, Memory>, Alignment<T, Memory>>::value,
                      //"Template argument 'Alignment' must inherit from BaseAlignment class.");

        /**
         * Matrix exception for when the matrix array is accessed out of it's bounds.
         */
        class MatrixIndexIsOutOfBounds : public std::exception {
        private:
            std::string _msg;
        public:
            inline const char *what() const noexcept override;
        };

        /**
         * Matrix exception for when the matrix operation on two(or more) Matrices are incompatible
         * wth one another.
         */
        class MatrixIsInvalidShape : public std::exception {
        private:
            std::string _msg;
        public:
            inline const char *what() const noexcept override;
        };

        /**
         * Copies the paramater matrix into this matrix. Anything stored
         * in this pointer will be deleted.
         * @param copyOfMatrix
         */
        inline void copy(const BaseMatrix &copyOfMatrix);



        /**
         *
         * @param moveMatrix
         */
        inline BaseMatrix<T, Memory, Alignment>& operator=(BaseMatrix<T, Memory, Alignment> &&moveMatrix) noexcept;

        /**
         * Compares this matrix to the referenced BaseMatrix. If both rows and
         * columns match up, then returns true. Else false.
         * @param b Reference to other matrix to check if shape is the same.
         * @return True if both Matrices are same shape. Else false.
         */
        inline bool validateMatricesAreSameShape(const BaseMatrix<T, Memory, Alignment> &b) const;

        //BaseMatrix<T>& operator=(const BaseMatrix<T> &rhs);


        //virtual void subtract() = 0;
        //virtual void dot() = 0;

    };

}
/**
//    template<typename T>
//    class BaseMatrix {
//    private:
//        std::unique_ptr<T[], void(*)(T*)> matrix;
//        unsigned long rows;
//        unsigned long cols;
//        unsigned long totalSize;
//
//        /**
//         * Private constructor, derived class must pass number of rows, columns and a function
//         * for freeing memory that is allocated.
//         * @param numberOfRows Total rows.
//         * @param numberOfCols Total columns.
//         * @param deleter Function reference to deleter function.
//         */
//        inline BaseMatrix(unsigned long numberOfRows, unsigned long numberOfCols, void(*deleter)(T*));
//
//        /**
//         * Private method that set's this->rows/cols fields based on the below params to that.
//         * Also calculates the total size of the array needed to house the desired matrix.
//         * @param numberOfRows Total rows.
//         * @param numberOfCols Total columns.
//         */
//        inline void _setDimensions(unsigned long numberOfRows, unsigned long numberOfCols);
//
//        /**
//         *
//         * @param copyOfMatrix
//         */
//        //inline virtual void _copy(const BaseMatrix &copyOfMatrix) = 0;
//
//        friend class Matrix<T,Device::Host>;
//        friend class Matrix<T,Device::Cuda>;
//    public:
//        /**
//         * Matrix exception for when the matrix array is accessed out of it's bounds.
//         */
//        class MatrixIndexIsOutOfBounds : public std::exception {
//        private:
//            std::string _msg;
//        public:
//            const char* what() const noexcept override;
//        };
//
//        /**
//         * Matrix exception for when the matrix operation on two(or more) Matrices are incompatible
//         * wth one another.
//         */
//        class MatrixIsInvalidShape : public std::exception {
//        private:
//            std::string _msg;
//        public:
//            const char* what() const noexcept override;
//        };
//
//        /**
//         * @return The totalSize of the matrix.
//         */
//        inline unsigned long getTotalSize() const;
//
//        /**
//         * @return The total amount of rows.
//         */
//        inline unsigned long getRows() const;
//
//        /**
//         * @return The total amount of cols.
//         */
//        inline unsigned long getCols() const;
//
//        /**
//         * Copies the paramater matrix into this matrix. Anything stored
//         * in this pointer will be deleted.
//         * @param copyOfMatrix
//         */
//        inline void copy(const BaseMatrix& copyOfMatrix);
//
//        /**
//         * Returns a pointer to the startTimepoint of the row based on the index provided.
//         * Then you can call a second [] and treat this as your column index.
//         * Like such: mxn matrix(M) -> M[3][10] (4th row, 11th column)
//         * @param i Index of the startTimepoint of the row you want to access.
//         * @return Pointer of type 'T' of the row at the provided index.
//         */
//        inline T* operator[](unsigned long i) noexcept;
//
//        /**
//         * Returns the value associated at the ith and jth position.
//         * @param i Index of the row.
//         * @param j Index of the col.
//         * @return
//         */
//        inline T get(unsigned i, unsigned j) const;
//
//        /**
//         * Compares this matrix to the referenced BaseMatrix. If both rows and
//         * columns match up, then returns true. Else false.
//         * @param b Reference to other matrix to check if shape is the same.
//         * @return True if both Matrices are same shape. Else false.
//         */
//        inline bool validateMatricesAreSameShape(const BaseMatrix<T> &b) const;
//
//        //BaseMatrix<T>& operator=(const BaseMatrix<T> &rhs);
//
//        inline virtual void add(const BaseMatrix &b) = 0;
//        //virtual void subtract() = 0;
//        //virtual void dot() = 0;
//    };
//}
#include "BaseMatrix.cu"


#endif //NANL_BASEMATRIX_CUH
