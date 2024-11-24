#include <cstdio>

#include "file_io.cuh"

namespace FileIO
{

    Utils::Parameters LoadParamsFromTextFile(FILE *file)
    {
        Utils::Parameters p{};
        if (fscanf(file, "%zu %zu %zu", &p.pointsCount, &p.dimensions, &p.clustersCount) != 3)
        {
            throw std::runtime_error("Invalid txt file format");
        }
        return p;
    }

    Utils::Parameters LoadParamsFromBinFile(FILE *file)
    {
        Utils::Parameters p{};
        if (fread(&p.pointsCount, sizeof(int), 1, file) != 1)
        {
            throw std::runtime_error("Invalid binary file format");
        }
        if (fread(&p.dimensions, sizeof(int), 1, file) != 1)
        {
            throw std::runtime_error("Invalid binary file format");
        }
        if (fread(&p.clustersCount, sizeof(int), 1, file) != 1)
        {
            throw std::runtime_error("Invalid binary file format");
        }
        return p;
    }
} // FileIO