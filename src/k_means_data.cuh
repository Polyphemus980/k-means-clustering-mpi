#ifndef K_MEANS_DATA
#define K_MEANS_DATA

#include <thrust/host_vector.h>
#include <filesystem>
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

    public:
        KMeansData<DIM>(const std::filesystem::path &filePath)
        {
            std::ifstream file{filePath};

            if (!file)
            {
                throw std::runtime_error{"Error while reading input from file: file not opened"};
            }

            std::string line;
            // load first line with N and k
            if (!std::getline(file, line))
            {
                throw std::runtime_error("Error input file: empty file");
            }
            std::istringstream iss{line};
            if (!(iss >> this->_pointsCount >> this->_clustersCount))
            {
                throw std::runtime_error("Invalid first line, expected \"<POINTS_COUNT> <CLUSTERS_COUNT>\"");
            }

            this->_values.resize(this->_pointsCount * DIM);
            this->_clustersValues.resize(this->_clustersCount * DIM);
            for (size_t i = 0; i < this->_pointsCount; i++)
            {
                if (!std::getline(file, line))
                {
                    throw std::runtime_error{"Invalid input file: too few points provided"};
                }
                std::istringstream iss{line};
                for (size_t j = 0; j < DIM; j++)
                {
                    iss >> this->_values[j * this->_pointsCount + i];
                    if (i < this->_clustersCount)
                    {
                        this->_clustersValues[j * this->_pointsCount + i] = this->_values[j * this->_pointsCount + i];
                    }
                }
            }
        }

        const thrust::host_vector<float> &getValues() const
        {
            return this->_values;
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