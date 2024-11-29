#ifndef K_MEANS_DATA
#define K_MEANS_DATA

#include <cstdio>
#include <thrust/host_vector.h>
#include <string>
#include <sstream>
#include <fstream>
#include <exception>

#include "cpu_timer.cuh"
#include "utils.cuh"

namespace KMeansData
{
    struct KMeansDataGPU
    {
        size_t pointsCount;
        size_t clustersCount;
        float *d_pointsValues;
        float *d_clustersValues;
    };

    class Helpers
    {
    public:
        __inline__ static float GetCoord(const thrust::host_vector<float> &container, size_t elementsCount, size_t elementIndex, size_t coordIndex)
        {
            return container[coordIndex * elementsCount + elementIndex];
        }

        __inline__ __device__ static float GetCoord(const float *d_container, size_t elementsCount, size_t elementIndex, size_t coordIndex)
        {
            return d_container[coordIndex * elementsCount + elementIndex];
        }
    };

    template <size_t DIM>
    class KMeansData
    {
    private:
        size_t _pointsCount;
        size_t _clustersCount;
        // This is vector of all coords combinec
        // For example for DIM = 3 and 2 points its:
        // [x1 x2 y1 y2 z1 z2]
        thrust::host_vector<float> _values;
        // The same idea is for storing clusters
        thrust::host_vector<float> _clustersValues;

    public:
        KMeansData<DIM>() {}

        KMeansData<DIM>(size_t pointsCount, size_t clustersCount, thrust::host_vector<float> values, thrust::host_vector<float> clustersValues) : _pointsCount(pointsCount), _clustersCount(clustersCount), _values(values), _clustersValues(clustersValues)
        {
        }

        const thrust::host_vector<float> &getValues() const
        {
            return this->_values;
        }

        size_t getPointsCount() const
        {
            return this->_pointsCount;
        }

        size_t getClustersCount() const
        {
            return this->_clustersCount;
        }

        const thrust::host_vector<float> &getClustersValues() const
        {
            return this->_clustersValues;
        }

        __inline__ float getPointCoord(size_t pointIndex, size_t coordIndex) const
        {
            return Helpers::GetCoord(this->_values, this->_pointsCount, pointIndex, coordIndex);
        }

        __inline__ float getClusterCoord(size_t clusterIndex, size_t coordIndex) const
        {
            return Helpers::GetCoord(this->_clustersValues, this->_clustersCount, clusterIndex, coordIndex);
        }

        KMeansDataGPU transformToGPURepresentation() const
        {
            CpuTimer::Timer cpuTimer;
            float *d_pointsValues;
            float *d_clustersValues;
            CHECK_CUDA(cudaMalloc(&d_pointsValues, sizeof(float) * _values.size()));
            CHECK_CUDA(cudaMalloc(&d_clustersValues, sizeof(float) * _clustersValues.size()));

            cpuTimer.start();

            CHECK_CUDA(cudaMemcpy(d_pointsValues, _values.data(), sizeof(float) * _values.size(), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_clustersValues, _clustersValues.data(), sizeof(float) * _clustersValues.size(), cudaMemcpyHostToDevice));

            cpuTimer.end();
            cpuTimer.printResult("Copy data from CPU to GPU");

            return KMeansDataGPU{
                .pointsCount = _pointsCount,
                .clustersCount = _clustersCount,
                .d_pointsValues = d_pointsValues,
                .d_clustersValues = d_clustersValues,
            };
        }
    };
} // KMeansData

#endif // K_MEANS_DATA