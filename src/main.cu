#include <iostream>
#include <cuda_runtime.h>

#include "k_means_data_cpu.cuh"

int main(int argc, char **argv)
{
    constexpr size_t DIM = 2;
    std::string inputPath = argc == 1 ? "input/test_input.txt" : argv[1];
    KMeansDataCPU::KMeansData<DIM> h_kMeansData{inputPath};
    return 0;
}