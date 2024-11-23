#include <cstdio>

#include "file_io.cuh"

namespace FileIO
{

    Utils::Parameters loadParamsFromTextFile(FILE *file)
    {
        Utils::Parameters p{};
        if (fscanf(file, "%zu %zu %zu", &p.pointsCount, &p.dimensions, &p.clustersCount) != 3)
        {
            throw std::runtime_error("Invalid txt file format");
        }
        return p;
    }

} // FileIO