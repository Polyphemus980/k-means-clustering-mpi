#include <iostream>
#include <cuda_runtime.h>

#include "k_means_data.cuh"

int main(int argc, char **argv)
{
    constexpr size_t DIM = 3;
    std::string inputPath = argc == 1 ? "input/test_input.txt" : argv[1];
    KMeansData::KMeansData<DIM> h_kMeansData{inputPath};
    auto firstX = KMeansData::KMeansData<DIM>::getPointCoord(h_kMeansData, 0, 0);
    std::cout << "X of first point: " << firstX << "\n";
    auto secondZ = KMeansData::KMeansData<DIM>::getPointCoord(h_kMeansData, 1, 2);
    std::cout << "Z of second point: " << secondZ << "\n";
    return 0;
}