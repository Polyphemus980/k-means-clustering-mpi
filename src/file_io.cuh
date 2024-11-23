#ifndef FILE_IO_H
#define FILE_IO_H

#include <cstdlib>
#include "utils.cuh"
#include "k_means_data.cuh"

namespace FileIO
{
    Utils::Parameters loadParamsFromTextFile(FILE *file);

    // We assume that first line of file is already read - we are only reading points
    // This function take ownership of the `file` and must close it
    template <size_t DIM>
    KMeansData::KMeansData<DIM> LoadFromTextFile(FILE *file, size_t pointsCount, size_t clustersCount)
    {
        if (!file)
        {
            throw std::invalid_argument("Error: invalid file pointer");
        }

        thrust::host_vector<float> values(pointsCount * DIM);
        thrust::host_vector<float> clustersValues(clustersCount * DIM);
        for (size_t i = 0; i < pointsCount; i++)
        {
            for (size_t j = 0; j < DIM; j++)
            {
                fscanf(file, "%f", &values[j * pointsCount + i]);
                if (i < clustersCount)
                {
                    clustersValues[j * clustersCount + i] = values[j * pointsCount + i];
                }
            }
        }

        fclose(file);
        KMeansData::KMeansData<DIM> data{pointsCount, clustersCount, values, clustersValues};
        return data;
    }

} // FileIO

#endif // FILE_IO_H