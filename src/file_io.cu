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

    // We assume that first line of file is already read - we are only reading points
    // This function take ownership of the `file` and must close it
    KMeansData::KMeansData LoadFromTextFile(FILE *file, size_t pointsCount, size_t clustersCount, size_t DIM)
    {
        CpuTimer::Timer cpuTimer;
        if (!file)
        {
            throw std::invalid_argument("Error: invalid file pointer");
        }

        thrust::host_vector<float> values(pointsCount * DIM);
        thrust::host_vector<float> clustersValues(clustersCount * DIM);
        printf("[START] Load from text file\n");
        cpuTimer.start();
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
        cpuTimer.end();
        cpuTimer.printResult("Load from text file");

        KMeansData::KMeansData data{pointsCount, clustersCount, DIM, values, clustersValues};
        return data;
    }

    KMeansData::KMeansData LoadFromBinFile(FILE *file, size_t pointsCount, size_t clustersCount, size_t DIM)
    {
        CpuTimer::Timer cpuTimer;

        if (!file)
        {
            throw std::invalid_argument("Error: invalid file pointer");
        }

        thrust::host_vector<float> values(pointsCount * DIM);
        thrust::host_vector<float> clustersValues(clustersCount * DIM);

        printf("[START] Load from binary file\n");
        cpuTimer.start();
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
        cpuTimer.end();
        cpuTimer.printResult("Load from binary file");

        fclose(file);
        KMeansData::KMeansData data{pointsCount, clustersCount, DIM, values, clustersValues};
        return data;
    }

    void SaveResultToTextFile(const char *outputPath, Utils::ClusteringResult results, size_t clustersCount, size_t DIM)
    {
        CpuTimer::Timer cpuTimer;

        FILE *file = fopen(outputPath, "w");
        if (!file)
        {
            throw std::invalid_argument("Error: cannot open output file");
        }

        printf("[START] Save results to file\n");
        cpuTimer.start();
        for (size_t j = 0; j < clustersCount; j++)
        {
            for (size_t d = 0; d < DIM; d++)
            {
                auto value = KMeansData::Helpers::GetCoord(results.clustersValues, clustersCount, j, d);
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
        cpuTimer.end();
        cpuTimer.printResult("Save results to file");
    }

} // FileIO