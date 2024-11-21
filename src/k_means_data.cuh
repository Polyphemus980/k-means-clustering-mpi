#ifndef K_MEANS_DATA
#define K_MEANS_DATA

#include <array>
#include <vector>
#include <filesystem>
#include <string>
#include <sstream>
#include <fstream>
#include <exception>

namespace KMeansData
{

    // TODO: remove it, just put std::vecttor<float> _clustersValues in KMeansData
    template <size_t DIM>
    class Cluster
    {
    private:
        std::array<float, DIM> _coords;

    public:
        Cluster<DIM>()
        {
            this->_coords.fill(0.0);
        }

        Cluster<DIM>(const std::array<float, DIM> &coords)
        {
            this->_coords = coords;
        }
    };

    template <size_t DIM>
    class KMeansData
    {
    private:
        size_t _points_count;
        size_t _clusters_count;
        // This is vector of all coords combinec
        // For example for DIM = 3 and 2 points its:
        // [x1 x2 y1 y2 z1 z2]
        // TODO: maybe use thrust::host_vector
        std::vector<float> _values;

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
            if (!(iss >> this->_points_count >> this->_clusters_count))
            {
                throw std::runtime_error("Invalid first line, expected \"<POINTS_COUNT> <CLUSTERS_COUNT>\"");
            }

            // TODO: here we should also load initial clusters
            this->_values.resize(this->_points_count * DIM);
            for (size_t i = 0; i < this->_points_count; i++)
            {
                if (!std::getline(file, line))
                {
                    throw std::runtime_error{"Invalid input file: too few points provided"};
                }
                std::istringstream iss{line};
                for (size_t j = 0; j < DIM; j++)
                {
                    iss >> this->_values[j * this->_points_count + i];
                }
            }
        }

        const std::vector<float> &getValues() const
        {
            return this->_values;
        }

        const float *getRawValues() const
        {
            return this->_values.data();
        }

        static float __inline_hint__ __host__ getPointCoord(const KMeansData<DIM> &data, size_t pointIndex, size_t coordIndex)
        {
            return data._values[coordIndex * data._points_count + pointIndex];
        }

        static float __inline_hint__ __device__ getPointCoord(float *data, size_t pointsCount, size_t pointIndex, size_t coordIndex)
        {
            return data[coordIndex * pointsCount + pointIndex];
        }
    };
} // KMeansData

#endif // K_MEANS_DATA