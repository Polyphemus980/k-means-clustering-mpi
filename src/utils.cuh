#ifndef UTILS_H
#define UTILS_H

#include <thrust/host_vector.h>

namespace Utils
{
    enum class AlgorithmMode
    {
        CPU,
        GPU_FIRST,
        GPU_SECOND
    };

    enum class InputFileType
    {
        TEXT,
        BINARY
    };

    struct ProgramArgs
    {
        AlgorithmMode algorithmMode;
        InputFileType inputFileType;
        const char *inputFilePath;
        const char *outputFilePath;
    };

    struct Parameters
    {
        size_t pointsCount;
        size_t clustersCount;
        size_t dimensions;
    };

    struct ClusteringResult
    {
        thrust::host_vector<float> clustersValues;
        thrust::host_vector<size_t> membership;
    };

    void usage(const char *s);
} // Utils

#endif // UTILS_H