#include <cstdio>

#include "file_io.cuh"

namespace FileIO
{

    Utils::Parameters loadParamsFromTextFile(FILE *file)
    {
        Utils::Parameters p{};
        fscanf(file, "%zu %zu %zu", &p.pointsCount, &p.dimensions, &p.clustersCount);
        return p;
    }

} // FileIO