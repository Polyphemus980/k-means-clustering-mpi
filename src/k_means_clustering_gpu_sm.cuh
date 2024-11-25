#ifndef K_MEANS_CLUSTERING_GPU_SM
#define K_MEANS_CLUSTERING_GPU_SM

#include <thrust/host_vector.h>

#include "k_means_data.cuh"
#include "utils.cuh"
#include "consts.cuh"

namespace KMeansClusteringGPUSM
{
    template <size_t DIM>
    __device__ size_t findNearestCluster(KMeansData::KMeansDataGPU &d_data, size_t pointIndex)
    {
        // TODO:
    }

    template <size_t DIM>
    __global__ void calculateMembershipAndNewClusters(KMeansData::KMeansDataGPU &d_data, float *d_newClusters, float *d_newClustersMembershipCount, size_t *d_memberships, bool *hasAnyChanged)
    {
        auto threadId = blockDim.x * blockIdx.x + threadIdx.x;

        extern __shared__ char sharedMemory[];
        float *s_clusters = reinterpret_cast<float *>(sharedMemory);
        size_t *s_clustersMembershipCount = reinterpret_cast<size_t *>(&s_clusters[d_data.clustersCount * dim]);
        bool *s_hasChanged = reinterpret_cast<bool *>(&s_clustersMembershipCount[d_data.clustersCount * dim]);

        if (threadId == 0)
        {
            s_hasChanged[0] = false;
        }

        if (threadId < d_data.clustersCount * dim)
        {
            s_clusters[threadId] = d_data.d_clustersValues[threadId];
        }
        if (threadId < d_data.pointsCount)
        {
            s_clustersMembershipCount[threadId] = 0;
        }

        // Ensure shared memory is properly initialized
        __syncthreads();

        if (threadId < d_data.pointsCount)
        {
            auto nearestClusterIndex = findNearestCluster<DIM>(d_data, threadId);
            for (size_t d = 0; d < DIM; d++)
            {
                atomicAdd(&s_clusters[d * d_data.clustersCount + nearestClusterIndex], KMeansData::Helpers<DIM>::GetCoord(d_data.d_pointsValues, d_data.pointsCount, threadId, d));
                atomicAdd(s_clustersMembershipCount[nearestClusterIndex], 1);
            }
            auto previousClusterIndex = d_memberships[threadId];
            if (previousClusterIndex != nearestClusterIndex)
            {
                atomicOr(&s_hasChanged[0], true);
                d_memberships[threadId] = nearestClusterIndex;
            }
        }

        // Finish all calculation made on shared memory
        __syncthreads();

        if (threadId < d_data.clustersCount * dim)
        {
            if (threadId == 0)
            {
                atomicOr(hasAnyChanged, s_hasChanged[0]);
            }
            d_newClusters[blockIdx.x * d_data.clustersCount + threadId] = s_clusters[threadId];
            d_newClustersMembershipCount[blockIdx.x * d_data.clustersCount + threadId] = s_clustersMembershipCount[threadId];
        }
    }

    template <size_t DIM>
    Utils::ClusteringResult kMeansClustering(KMeansData::KMeansDataGPU &d_data)
    {
        const uint blocksCount = ceil(d_data.pointsCount * 1.0 / Consts::THREADS_PER_BLOCK);

        // TODO: check cuda errors
        size_t *d_memberships;
        float *d_newClusters;
        size_t *d_newClustersMembershipCount;
        // We have array for each block
        cudaMalloc(&d_memberships, sizeof(size_t) * d_data.pointsCount);
        cudaMalloc(&d_newClusters, sizeof(float) * d_data.pointsCount * DIM * blocksCount);
        cudaMalloc(&d_newClustersMembershipCount, sizeof(float) * d_data.pointsCount * DIM * blocksCount);
        // We initialize the array that membership[i] = size_t::MAX
        cudaMemset(d_memberships, 0xFF, sizeof(size_t) * d_data.pointsCount);

        for (size_t k = 0; k < Consts::MAX_ITERATION; k++)
        {
            bool hasAnyChanged = false;
            cudaMemset(d_newClusters, 0, sizeof(float) * d_data.pointsCount * DIM * blocksCount);
            cudaMemset(d_newClustersMembershipCount, 0, sizeof(float) * d_data.pointsCount * DIM * blocksCount);
            calculateMembershipAndNewClusters<DIM><<<blocksCount, Consts::THREADS_PER_BLOCK>>>(d_data, d_newClusters, d_newClustersMembershipCount, d_memberships, &hasAnyChanged);
            // TODO: calculate new clusters based on updated
            if (!hasAnyChanged)
            {
                break;
            }
        }

        thrust::host_vector<float> clustersValues(d_data.clustersCount * DIM);
        thrust::host_vector<size_t> membership(d_data.pointsCount);
        cudaMemcpy(clustersValues.data(), d_data.d_clustersValues, sizeof(float) * clustersValues.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(membership.data(), d_memberships, sizeof(size_t) * d_data.pointsCount, cudaMemcpyDeviceToHost);

        cudaFree(d_memberships);
        cudaFree(d_newClusters);
        cudaFree(d_newClustersMembershipCount);

        return Utils::ClusteringResult{
            .membership = membership,
            .clustersValues = clustersValues,
        };
    }
} // KMeansClusteringGPUSM

#endif // K_MEANS_CLUSTERING_GPU_SM