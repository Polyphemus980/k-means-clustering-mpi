#ifndef K_MEANS_DATA
#define K_MEANS_DATA

#include <cstdio>
#include <thrust/host_vector.h>
#include <string>
#include <sstream>
#include <fstream>
#include <exception>

namespace KMeansData
{

    template <size_t DIM>
    class Helpers
    {
    public:
        static float __inline_hint__ GetCoord(const thrust::host_vector<float> &container, size_t elementsCount, size_t elementIndex, size_t coordIndex)
        {
            return container[coordIndex * elementsCount + elementIndex];
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

        float __inline_hint__ getPointCoord(size_t pointIndex, size_t coordIndex) const
        {
            return Helpers<DIM>::GetCoord(this->_values, this->_pointsCount, pointIndex, coordIndex);
        }

        float __inline_hint__ getClusterCoord(size_t clusterIndex, size_t coordIndex) const
        {
            return Helpers<DIM>::GetCoord(this->_clustersValues, this->_clustersCount, clusterIndex, coordIndex);
        }
    };
} // KMeansData

#endif // K_MEANS_DATA