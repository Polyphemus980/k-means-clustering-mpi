#ifndef FILE_IO_H
#define FILE_IO_H

#include <cstdlib>
#include <exception>

#include "utils.cuh"
#include "k_means_data.cuh"
#include "k_means_clustering_cpu.cuh"

namespace FileIO
{
    Utils::Parameters LoadParamsFromTextFile(FILE *file);
    Utils::Parameters LoadParamsFromBinFile(FILE *file);

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
            for (size_t d = 0; d < DIM; d++)
            {
                if (fscanf(file, "%f", &values[d * pointsCount + i]) != 1)
                {
                    throw std::runtime_error("Invalid txt file format");
                }
                if (i < clustersCount)
                {
                    clustersValues[d * clustersCount + i] = values[d * pointsCount + i];
                }
            }
        }

        fclose(file);
        KMeansData::KMeansData<DIM> data{pointsCount, clustersCount, values, clustersValues};
        return data;
    }

    template <size_t DIM>
    KMeansData::KMeansData<DIM> LoadFromBinFile(FILE *file, size_t pointsCount, size_t clustersCount)
    {
        if (!file)
        {
            throw std::invalid_argument("Error: invalid file pointer");
        }

        thrust::host_vector<float> values(pointsCount * DIM);
        thrust::host_vector<float> clustersValues(clustersCount * DIM);

        float *buffer = (float *)malloc(sizeof(float) * values.size());
        if (buffer == nullptr)
        {
            throw std::runtime_error("Cannot malloc buffer");
        }
        if (fread(buffer, sizeof(float), values.size(), file) != values.size())
        {
            throw std::runtime_error("Invalid binary file format");
        }

        for (size_t i = 0; i < pointsCount; i++)
        {
            for (size_t d = 0; d < DIM; d++)
            {

                values[d * pointsCount + i] = buffer[i * DIM + d];
                if (i < clustersCount)
                {
                    clustersValues[d * clustersCount + i] = values[d * pointsCount + i];
                }
            }
        }

        fclose(file);
        KMeansData::KMeansData<DIM> data{pointsCount, clustersCount, values, clustersValues};
        return data;
    }

    template <size_t DIM>
    void SaveCPUResultToTextFile(const char *outputPath, KMeansClusteringCPU::CpuClusteringResult results, size_t clustersCount)
    {
        FILE *file = fopen(outputPath, "w");
        if (!file)
        {
            throw std::invalid_argument("Error: cannot open output file");
        }

        for (size_t j = 0; j < clustersCount; j++)
        {
            for (size_t d = 0; d < DIM; d++)
            {
                auto value = KMeansData::Helpers<DIM>::GetCoord(results.clustersValues, clustersCount, j, d);
                if (d == DIM - 1)
                {
                    fprintf(file, "%f", value);
                }
                else
                {
                    fprintf(file, "%f ", value);
                }
            }
            fprintf(file, "\n");
        }

        for (size_t i = 0; i < results.membership.size(); i++)
        {
            fprintf(file, "%zu\n", results.membership[i]);
        }

        fclose(file);
    }

} // FileIO

#endif // FILE_IO_H