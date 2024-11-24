#ifndef UTILS_H
#define UTILS_H

#include <iostream>

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

    void usage(const char *s);
} // Utils

#endif // UTILS_H