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

        KMeansData<DIM>(size_t pointsCount, size_t clustersCount, thrust::host_vector<float> values, thrust::host_vector<float> clustersValues) : _pointsCount(pointsCount), _clustersCount(clustersCount), _values(values), _clustersValues(clustersValues)
        {
        }

    public:
        // We assume that first line of file is already read - we are only reading points
        // This function take ownership of the `file` and must close it
        static KMeansData<DIM> LoadFromTextFile(FILE *file, size_t pointsCount, size_t clustersCount)
        {
            if (!file)
            {
                throw std::invalid_argument("Error: invalid file pointer");
            }

            thrust::host_vector<float> values{};
            values.resize(pointsCount * DIM);
            thrust::host_vector<float> clustersValues{};
            clustersValues.resize(clustersCount * DIM);
            for (size_t i = 0; i < pointsCount; i++)
            {
                for (size_t j = 0; j < DIM; j++)
                {
                    fscanf(file, "%f", &values[j * pointsCount + i]);
                    if (i < clustersCount)
                    {
                        clustersValues[j * pointsCount + i] = values[j * pointsCount + i];
                    }
                }
            }

            fclose(file);
            KMeansData<DIM> data{pointsCount, clustersCount, values, clustersValues};
            return data;
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
            return this->_values[coordIndex * this->_pointsCount + pointIndex];
        }

        float __inline_hint__ getClusterCoord(size_t clusterIndex, size_t coordIndex) const
        {
            return this->_clustersValues[coordIndex * this->_pointsCount + clusterIndex];
        }
    };
} // KMeansData

#endif // K_MEANS_DATA