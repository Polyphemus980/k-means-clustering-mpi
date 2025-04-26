#include "k_means_data.cuh"

namespace KMeansData
{
    KMeansDataGPU KMeansData::transformToGPURepresentation() const
    {
        CpuTimer::Timer cpuTimer;
        float *d_pointsValues = nullptr;
        float *d_clustersValues = nullptr;

        try
        {
            CHECK_CUDA(cudaMalloc(&d_pointsValues, sizeof(float) * _values.size()));
            CHECK_CUDA(cudaMalloc(&d_clustersValues, sizeof(float) * _clustersValues.size()));

            printf("[START] Copy data from CPU to GPU\n");
            cpuTimer.start();

            CHECK_CUDA(cudaMemcpy(d_pointsValues, _values.data(), sizeof(float) * _values.size(), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_clustersValues, _clustersValues.data(), sizeof(float) * _clustersValues.size(), cudaMemcpyHostToDevice));

            cpuTimer.end();
            cpuTimer.printResult("Copy data from CPU to GPU");
        }
        catch (const std::runtime_error &e)
        {
            fprintf(stderr, "[ERROR]: %s", e.what());
            // We only want to deallocate in case of error - otherwise it will be deallocated by the caller of this function
            if (d_pointsValues != nullptr)
            {
                cudaFree(d_pointsValues);
            }
            if (d_clustersValues != nullptr)
            {
                cudaFree(d_clustersValues);
            }
            throw e;
        }

        return KMeansDataGPU{
            .pointsCount = _pointsCount,
            .clustersCount = _clustersCount,
            .DIM = _DIM,
            .d_pointsValues = d_pointsValues,
            .d_clustersValues = d_clustersValues,
        };
    }

} // KMeansData