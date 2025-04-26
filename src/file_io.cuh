#ifndef FILE_IO_H
#define FILE_IO_H

#include <cstdlib>
#include <exception>

#include "utils.cuh"
#include "k_means_data.cuh"
#include "cpu_timer.cuh"

namespace FileIO
{
    Utils::Parameters LoadParamsFromTextFile(FILE *file);
    Utils::Parameters LoadParamsFromBinFile(FILE *file);
    KMeansData::KMeansData LoadFromTextFile(FILE *file, size_t pointsCount, size_t clustersCount, size_t DIM);
    KMeansData::KMeansData LoadFromBinFile(FILE *file, size_t pointsCount, size_t clustersCount, size_t DIM);
    void SaveResultToTextFile(const char *outputPath, Utils::ClusteringResult results, size_t clustersCount, size_t DIM);
} // FileIO

#endif // FILE_IO_H