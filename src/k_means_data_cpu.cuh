#ifndef K_MEANS_DATA
#define K_MEANS_DATA

#include <array>
#include <vector>
#include <filesystem>
#include <string>
#include <sstream>
#include <fstream>
#include <exception>

// Should be used on CPU only
namespace KMeansDataCPU
{

    template <size_t DIM>
    class Point
    {
    private:
        std::array<float, DIM> _coords;

    public:
        Point<DIM>(const std::string &fileLine)
        {
            this->_coords = std::array<float, DIM>{};
            std::istringstream iss{fileLine};
            for (size_t i = 0; i < DIM; i++)
            {
                iss >> _coords[i];
            }
        }
    };

    template <size_t DIM>
    class KMeansData
    {
    private:
        size_t _points_count;
        size_t _clusters_count;
        std::vector<Point<DIM>> _points;

    public:
        KMeansData<DIM>(const std::filesystem::path &filePath)
        {
            std::ifstream file{filePath};

            if (!file)
            {
                throw std::runtime_error{"Error while reading input from file"};
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

            this->_points.reserve(this->_points_count);
            for (size_t i = 0; i < this->_points_count; i++)
            {
                if (!std::getline(file, line))
                {
                    throw std::runtime_error{"Invalid input file: too few points provided"};
                }
                this->_points.push_back(Point<DIM>{line});
            }
        }
    };
} // KMeansDataCPU

#endif