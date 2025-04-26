#ifndef K_MEANS_DATA
#define K_MEANS_DATA

#include <cstdio>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
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
        size_t DIM;
        float *d_pointsValues;
        float *d_clustersValues;
    };

    struct KMeansDataGPUThrust
    {
        size_t pointsCount;
        size_t clustersCount;
        thrust::device_vector<float> pointsValues;
        thrust::device_vector<float> clustersValues;
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

    class KMeansData
    {
    private:
        size_t _pointsCount;
        size_t _clustersCount;
        size_t _DIM;
        // This is vector of all coords combinec
        // For example for DIM = 3 and 2 points its:
        // [x1 x2 y1 y2 z1 z2]
        thrust::host_vector<float> _values;
        // The same idea is for storing clusters
        thrust::host_vector<float> _clustersValues;

    public:
        KMeansData() {}

        KMeansData(size_t pointsCount, size_t clustersCount, size_t DIM, thrust::host_vector<float> values, thrust::host_vector<float> clustersValues) : _pointsCount(pointsCount), _clustersCount(clustersCount), _DIM(DIM), _values(values), _clustersValues(clustersValues)
        {
        }

        __inline__ const thrust::host_vector<float> &getValues() const
        {
            return this->_values;
        }

        __inline__ size_t getPointsCount() const
        {
            return this->_pointsCount;
        }

        __inline__ size_t getClustersCount() const
        {
            return this->_clustersCount;
        }

        __inline__ size_t getDIM() const
        {
            return this->_DIM;
        }

        __inline__ const thrust::host_vector<float> &getClustersValues() const
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

        KMeansDataGPU transformToGPURepresentation() const;
    };
} // KMeansData

#endif // K_MEANS_DATA