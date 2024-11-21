#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>
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

    typedef struct
    {
        AlgorithmMode algorithmMode;
        InputFileType inputFileType;
        const char *inputFilePath;
        const char *outputFilePath;
    } ProgramArgs;

    typedef struct
    {
        size_t pointsCount;
        size_t clustersCount;
        size_t dimensions;
    } Parameters;

    void usage(const char *s);
    Parameters loadParamsFromTextFile(FILE *file);
} // Utils

#endif // UTILS_H