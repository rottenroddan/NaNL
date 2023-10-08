//
// Created by steve on 7/16/2023.
//

#include "TestMatrices.cuh"

TestMatrices::TestMatrices(std::string filePath) : fileLoader(filePath),
    a(fileLoader.read<int , NaNL::PagedMemoryBlock, NaNL::Unaligned>()),
    b(fileLoader.read<int, NaNL::PagedMemoryBlock, NaNL::Unaligned>()),
    truth(fileLoader.read<int, NaNL::PagedMemoryBlock, NaNL::Unaligned>())
{

}

template<template<class, class> class rMemory, class rAlignment>
NaNL::Matrix<int, rMemory, rAlignment> TestMatrices::getCopyOfA() {
    return a.template copyTo<rMemory, rAlignment>();
}

template<template<class, class> class rMemory, class rAlignment>
NaNL::Matrix<int, rMemory, rAlignment> TestMatrices::getCopyOfB() {
    return b.template copyTo<rMemory, rAlignment>();
}

template<template<class, class> class rMemory, class rAlignment>
NaNL::Matrix<int, rMemory, rAlignment> TestMatrices::getCopyOfTruth() {
    return truth.template copyTo<rMemory, rAlignment>();
}