#ifndef K_MEANS_CLUSTERING_CPU
#define K_MEANS_CLUSTERING_CPU

#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/fill.h>

#include "k_means_data.cuh"
#include "consts.cuh"

namespace KMeansClusteringCPU
{
    typedef struct
    {
        thrust::host_vector<float> clustersValues;
        thrust::host_vector<size_t> membership;
    } CpuClusteringResult;

    template <size_t DIM>
    float calculatePointClusterDinstance(const KMeansData::KMeansData<DIM> &h_kMeansData, size_t pointIndex, thrust::host_vector<float> &clusters, size_t clusterIndex)
    {
        float distance = 0;
        for (size_t d = 0; d < DIM; d++)
        {
            float diff = h_kMeansData.getPointCoord(pointIndex, d) - KMeansData::Helpers<DIM>::GetCoord(clusters, h_kMeansData.getClustersCount(), clusterIndex, d);
            distance += diff * diff;
        }
        return sqrt(distance);
    }

    template <size_t DIM>
    size_t findNearestCluster(const KMeansData::KMeansData<DIM> &h_kMeansData, size_t pointIndex, thrust::host_vector<float> &clusters)
    {
        auto points_values = h_kMeansData.getValues();
        size_t minIndex = 0;
        float minDistance = calculatePointClusterDinstance(h_kMeansData, pointIndex, clusters, 0);
        for (size_t j = 1; j < h_kMeansData.getClustersCount(); j++)
        {
            float distance = calculatePointClusterDinstance(h_kMeansData, pointIndex, clusters, j);
            if (distance < minDistance)
            {
                minDistance = distance;
                minIndex = j;
            }
        }
        return minIndex;
    }

    template <size_t DIM>
    void updateNewCluster(const KMeansData::KMeansData<DIM> &h_kMeansData, size_t pointIndex, size_t clusterIndex, thrust::host_vector<float> &newClusters, thrust::host_vector<size_t> &newClustersSize)
    {
        for (size_t d = 0; d < DIM; d++)
        {
            newClusters[d * h_kMeansData.getClustersCount() + clusterIndex] += h_kMeansData.getPointCoord(pointIndex, d);
        }
        newClustersSize[clusterIndex]++;
    }

    template <size_t DIM>
    void updateCluster(size_t clustersCount, thrust::host_vector<float> &clusters, thrust::host_vector<float> &newClusters, thrust::host_vector<size_t> &newClustersSize, size_t clusterIndex)
    {
        for (size_t d = 0; d < DIM; d++)
        {
            for (size_t j = 0; j < clustersCount; j++)
            {
                clusters[d * clustersCount + j] = KMeansData::Helpers<DIM>::GetCoord(newClusters, clustersCount, j, d) / newClustersSize[j];
            }
        }
    }

    template <size_t DIM>
    CpuClusteringResult kMeanClustering(const KMeansData::KMeansData<DIM> &h_kMeansData)
    {
        // we init this vector with n elements, each one with value k (meaning they aren't in any cluster)
        thrust::host_vector<size_t> membership{h_kMeansData.getPointsCount(), h_kMeansData.getClustersCount()};
        bool has_change = false;
        thrust::host_vector<float> clusters{h_kMeansData.getClustersValues()};
        thrust::host_vector<float> newClusters(h_kMeansData.getClustersCount() * DIM, 0);
        thrust::host_vector<size_t> newClustersSize(h_kMeansData.getClustersCount(), 0);
        for (size_t k = 0; k < Consts::MAX_ITERATION; k++)
        {
            has_change = false;
            thrust::fill(newClusters.begin(), newClusters.end(), 0);
            thrust::fill(newClustersSize.begin(), newClustersSize.end(), 0);
            for (size_t i = 0; i < h_kMeansData.getPointsCount(); i++)
            {
                size_t nearestClusterIndex = findNearestCluster<DIM>(h_kMeansData, i, clusters);
                if (membership[i] != nearestClusterIndex)
                {
                    membership[i] = nearestClusterIndex;
                    has_change = true;
                }
                updateNewCluster<DIM>(h_kMeansData, i, membership[i], newClusters, newClustersSize);
            }
            for (size_t j = 0; j < h_kMeansData.getClustersCount(); j++)
            {
                updateCluster<DIM>(h_kMeansData.getClustersCount(), clusters, newClusters, newClustersSize, j);
            }
            if (!has_change)
                break;
        }
        return CpuClusteringResult{
            .clustersValues = clusters,
            .membership = membership};
    }
} // KMeansClusteringCPU

#endif // K_MEANS_CLUSTERING_CPU