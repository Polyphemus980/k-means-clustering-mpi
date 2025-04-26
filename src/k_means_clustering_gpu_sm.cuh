#ifndef K_MEANS_CLUSTERING_GPU_SM
#define K_MEANS_CLUSTERING_GPU_SM

#include <thrust/host_vector.h>
#include <cstdint>

#include "k_means_data.cuh"
#include "utils.cuh"
#include "consts.cuh"
#include "cpu_timer.cuh"
#include "gpu_timer.cuh"

namespace KMeansClusteringGPUSM
{
    __device__ float pointToClusterDistanceSquared(KMeansData::KMeansDataGPU d_data, size_t pointIndex, size_t clusterIndex);

    __device__ size_t findNearestCluster(KMeansData::KMeansDataGPU d_data, size_t pointIndex);

    __global__ void calculateMembershipAndNewClusters(KMeansData::KMeansDataGPU d_data, float *d_newClusters, uint32_t *d_newClustersMembershipCount, size_t *d_memberships, int *d_shouldContinue);

    __global__ void accumulateNewClustersMemerships(KMeansData::KMeansDataGPU d_data, size_t *d_clustersMembershipCount, uint32_t *d_newClustersMembershipCount, size_t previousBlocksCount);

    __global__ void updateClusters(KMeansData::KMeansDataGPU d_data, size_t *d_clustersMembershipCount, float *d_newClusters, size_t previousBlocksCount);

    Utils::ClusteringResult kMeansClustering(KMeansData::KMeansDataGPU d_data);
} // KMeansClusteringGPUSM

#endif // K_MEANS_CLUSTERING_GPU_SM