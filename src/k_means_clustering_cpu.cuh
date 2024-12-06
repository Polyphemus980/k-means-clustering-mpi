#ifndef K_MEANS_CLUSTERING_CPU
#define K_MEANS_CLUSTERING_CPU

#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/fill.h>

#include "k_means_data.cuh"
#include "consts.cuh"
#include "utils.cuh"

namespace KMeansClusteringCPU
{

    template <size_t DIM>
    float calculatePointClusterDinstance(const thrust::host_vector<float> &points, size_t pointsCount, size_t pointIndex, const thrust::host_vector<float> &clusters, size_t clustersCount, size_t clusterIndex)
    {
        float distance = 0;
        for (size_t d = 0; d < DIM; d++)
        {
            float diff = KMeansData::Helpers::GetCoord(points, pointsCount, pointIndex, d) - KMeansData::Helpers::GetCoord(clusters, clustersCount, clusterIndex, d);
            distance += diff * diff;
        }
        return distance;
    }

    template <size_t DIM>
    size_t findNearestCluster(const thrust::host_vector<float> &points, size_t pointsCount, size_t pointIndex, const thrust::host_vector<float> &clusters, size_t clustersCount)
    {
        size_t minIndex = 0;
        float minDistance = calculatePointClusterDinstance<DIM>(points, pointsCount, pointIndex, clusters, clustersCount, 0);
        for (size_t j = 1; j < clustersCount; j++)
        {
            float distance = calculatePointClusterDinstance<DIM>(points, pointsCount, pointIndex, clusters, clustersCount, j);
            if (distance < minDistance)
            {
                minDistance = distance;
                minIndex = j;
            }
        }
        return minIndex;
    }

    template <size_t DIM>
    void updateNewCluster(const thrust::host_vector<float> &points, size_t pointsCount, size_t pointIndex, thrust::host_vector<float> &newClusters, thrust::host_vector<size_t> &newClustersSize, size_t clustersCount, size_t clusterIndex)
    {
        for (size_t d = 0; d < DIM; d++)
        {
            newClusters[d * clustersCount + clusterIndex] += KMeansData::Helpers::GetCoord(points, pointsCount, pointIndex, d);
        }
        newClustersSize[clusterIndex]++;
    }

    template <size_t DIM>
    void updateCluster(thrust::host_vector<float> &clusters, const thrust::host_vector<float> &newClusters, const thrust::host_vector<size_t> &newClustersSize, size_t clustersCount, size_t clusterIndex)
    {
        for (size_t d = 0; d < DIM; d++)
        {
            for (size_t j = 0; j < clustersCount; j++)
            {
                clusters[d * clustersCount + j] = KMeansData::Helpers::GetCoord(newClusters, clustersCount, j, d) / newClustersSize[j];
            }
        }
    }

    template <size_t DIM>
    Utils::ClusteringResult kMeanClustering(const KMeansData::KMeansData<DIM> &h_kMeansData)
    {
        CpuTimer::Timer cpuTimer;
        printf("[START] K-means clustering (main algorithm)\n");
        cpuTimer.start();

        auto points = h_kMeansData.getValues();
        auto pointsCount = h_kMeansData.getPointsCount();
        auto clustersCount = h_kMeansData.getClustersCount();
        // we init this vector with n elements, each one with value k (meaning they aren't in any cluster)
        thrust::host_vector<size_t> membership(h_kMeansData.getPointsCount(), h_kMeansData.getClustersCount());
        size_t changeCount;
        thrust::host_vector<float> clusters{h_kMeansData.getClustersValues()};
        thrust::host_vector<float> newClusters(h_kMeansData.getClustersCount() * DIM, 0);
        thrust::host_vector<size_t> newClustersSize(h_kMeansData.getClustersCount(), 0);
        for (size_t k = 0; k < Consts::MAX_ITERATION; k++)
        {
            changeCount = 0;
            thrust::fill(newClusters.begin(), newClusters.end(), 0);
            thrust::fill(newClustersSize.begin(), newClustersSize.end(), 0);
            for (size_t i = 0; i < h_kMeansData.getPointsCount(); i++)
            {
                size_t nearestClusterIndex = findNearestCluster<DIM>(points, pointsCount, i, clusters, clustersCount);
                if (membership[i] != nearestClusterIndex)
                {
                    membership[i] = nearestClusterIndex;
                    changeCount++;
                }
                updateNewCluster<DIM>(points, pointsCount, i, newClusters, newClustersSize, clustersCount, membership[i]);
            }
            printf("[INFO] Iteration: %ld, changed points: %ld\n", k, changeCount);
            if (changeCount == 0)
            {
                break;
            }
            for (size_t j = 0; j < clustersCount; j++)
            {
                updateCluster<DIM>(clusters, newClusters, newClustersSize, clustersCount, j);
            }
        }

        cpuTimer.end();
        cpuTimer.printResult("K-means clustering (main algorithm)");

        return Utils::ClusteringResult{
            .clustersValues = clusters,
            .membership = membership};
    }
} // KMeansClusteringCPU

#endif // K_MEANS_CLUSTERING_CPU