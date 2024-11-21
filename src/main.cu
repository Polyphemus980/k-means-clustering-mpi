#include <iostream>
#include <cuda_runtime.h>

#include "k_means_data.cuh"

int main(int argc, char **argv)
{
    constexpr size_t DIM = 3;
    std::string inputPath = argc == 1 ? "input/test_input.txt" : argv[1];
    KMeansData::KMeansData<DIM> h_kMeansData{inputPath};

    // Points
    auto points = h_kMeansData.getValues();
    std::cout << "Number of points: " << points.size() / DIM << "\n";
    auto firstX = h_kMeansData.getPointCoord(0, 0);
    std::cout << "X of first point: " << firstX << "\n";
    auto secondZ = h_kMeansData.getPointCoord(1, 2);
    std::cout << "Z of second point: " << secondZ << "\n";

    // Clusters
    auto clusters = h_kMeansData.getClustersValues();
    std::cout << "Number of clusters: " << clusters.size() / DIM << "\n";
    auto firstClusterY = h_kMeansData.getClusterCoord(0, 1);
    std::cout << "Y of first cluster: " << firstClusterY << "\n";
    return 0;
}