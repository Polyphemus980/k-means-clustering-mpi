#ifndef K_MEANS_CLUSTERING_CPU
#define K_MEANS_CLUSTERING_CPU

#include <thrust/host_vector.h>
#include <thrust/fill.h>

#include "k_means_data.cuh"

namespace KMeansClusteringCPU
{
    template <size_t DIM>
    void kMeanClustering(const KMeansData::KMeansData<DIM> &h_kMeansData, const float treshold)
    {
        // we init this vector with n elements, each one with value k (meaning they aren't in any cluster)
        thrust::host_vector<size_t> membership{h_kMeansData.getPointsCount(), h_kMeansData.getClustersCount()};
        size_t changedCount{};
        thrust::host_vector<float> clusters{h_kMeansData.getClustersValues()};
        thrust::host_vector<float> newClusters{h_kMeansData * DIM};
        thrust::host_vector<size_t> newClustersSize{h_kMeansData};
        while (true /*TODO: what should be here?*/)
        {
            changedCount = 0;
            thrust::fill(newClusters.begin(), newClusters.end, 0);
            thrust::fill(newClustersSize.begin(), newClustersSize.end, 0);
            for (size_t i = 0; i < h_kMeansData.getPointsCount(); i++)
            {
                size_t nearestClusterIndex = findNearestCluster<DIM>(h_kMeansData, i, clusters);
                if (membership[i] != nearestClusterIndex)
                {
                    membership[i] = nearestClusterIndex;
                    changedCount++;
                }
                updateNewCluster<DIM>(h_kMeansData, i, membership[i], newClusters, newClustersSize);
            }
            for (size_t j = 0; j < h_kMeansData.getClustersCount(); j++)
            {
                updateCluster<DIM>(clusters, newClusters, newClustersSize, j);
            }
        }
        // TODO: it should return some output here
    }

    template <size_t DIM>
    size_t findNearestCluster(const KMeansData::KMeansData<DIM> &h_kMeansData, size_t point_index, thrust::host_vector<float> &clusters)
    {
        // TODO:
    }

    template <size_t DIM>
    void updateNewCluster(const KMeansData::KMeansData<DIM> &h_kMeansData, size_t point_index, size_t cluster_index, thrust::host_vector<float> &newClusters, thrust::host_vector<size_t> &newClustersSize)
    {
        // TODO:
    }

    template <size_t DIM>
    void updateCluster(thrust::host_vector<float> &clusters, thrust::host_vector<float> &newClusters, thrust::host_vector<size_t> &newClustersSize, size_t clusterIndex)
    {
        // TODO:
    }
} // KMeansClusteringCPU

#endif // K_MEANS_CLUSTERING_CPU