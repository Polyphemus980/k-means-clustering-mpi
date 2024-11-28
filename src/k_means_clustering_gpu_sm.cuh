#ifndef K_MEANS_CLUSTERING_GPU_SM
#define K_MEANS_CLUSTERING_GPU_SM

#include <thrust/host_vector.h>
#include <cmath>
#include <cstdint>

#include "k_means_data.cuh"
#include "utils.cuh"
#include "consts.cuh"

namespace KMeansClusteringGPUSM
{
    template <size_t DIM>
    __device__ float pointToClusterDistance(KMeansData::KMeansDataGPU d_data, size_t pointIndex, size_t clusterIndex)
    {
        float distance = 0;
        for (size_t d = 0; d < DIM; d++)
        {
            float diff = KMeansData::Helpers<DIM>::GetCoord(d_data.d_pointsValues, d_data.pointsCount, pointIndex, d) - KMeansData::Helpers<DIM>::GetCoord(d_data.d_clustersValues, d_data.clustersCount, clusterIndex, d);

            distance += diff * diff;
        }
        return sqrt(distance);
    }

    template <size_t DIM>
    __device__ size_t findNearestCluster(KMeansData::KMeansDataGPU d_data, size_t pointIndex)
    {
        float minDist = pointToClusterDistance<DIM>(d_data, pointIndex, 0);
        size_t minDistIndex = 0;
        for (size_t j = 1; j < d_data.clustersCount; j++)
        {
            float dist = pointToClusterDistance<DIM>(d_data, pointIndex, j);
            if (dist < minDist)
            {
                minDist = dist;
                minDistIndex = j;
            }
        }
        return minDistIndex;
    }

    // Function for finding new membership for each point
    // Each thread should be responsible for single point
    template <size_t DIM>
    __global__ void calculateMembershipAndNewClusters(KMeansData::KMeansDataGPU d_data, float *d_newClusters, uint32_t *d_newClustersMembershipCount, size_t *d_memberships, int *d_shouldContinue)
    {
        auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
        auto localThreadId = threadIdx.x;

        extern __shared__ int sharedMemory[];
        float *s_clusters = (float *)sharedMemory;
        uint32_t *s_clustersMembershipCount = (uint32_t *)&s_clusters[d_data.clustersCount * DIM];
        int *s_shouldContinue = (int *)&s_clustersMembershipCount[d_data.clustersCount];

        // Initialize shared memory in each block
        if (localThreadId == 0)
        {
            s_shouldContinue[0] = 0;
        }
        if (localThreadId < d_data.clustersCount * DIM)
        {
            s_clusters[localThreadId] = 0;
        }
        if (localThreadId < d_data.clustersCount)
        {
            s_clustersMembershipCount[localThreadId] = 0;
        }

        // Ensure shared memory is properly initialized
        __syncthreads();

        // For each point find its nearest cluster, update membership table and save results in shared memory
        if (threadId < d_data.pointsCount)
        {
            auto nearestClusterIndex = findNearestCluster<DIM>(d_data, threadId);
            for (size_t d = 0; d < DIM; d++)
            {
                atomicAdd(&s_clusters[d * d_data.clustersCount + nearestClusterIndex], KMeansData::Helpers<DIM>::GetCoord(d_data.d_pointsValues, d_data.pointsCount, threadId, d));
            }
            atomicAdd(&s_clustersMembershipCount[nearestClusterIndex], 1);
            auto previousClusterIndex = d_memberships[threadId];
            if (previousClusterIndex != nearestClusterIndex)
            {
                atomicOr(&s_shouldContinue[0], 1);
                d_memberships[threadId] = nearestClusterIndex;
            }
        }

        // Finish all calculation made on shared memory
        __syncthreads();

        // Copy results from shared memory to global memory
        if (localThreadId == 0)
        {
            d_shouldContinue[blockIdx.x] = s_shouldContinue[0];
        }

        if (localThreadId < d_data.clustersCount * DIM)
        {
            d_newClusters[blockIdx.x * d_data.clustersCount * DIM + localThreadId] = s_clusters[localThreadId];
        }

        if (localThreadId < d_data.clustersCount)
        {
            d_newClustersMembershipCount[blockIdx.x * d_data.clustersCount + localThreadId] = s_clustersMembershipCount[localThreadId];
        }
    }

    // Function for updating clusters based on new membership
    // There should be thread spawned for every cluster for every dimension, so CLUSTERS_COUNT * DIM total
    template <size_t DIM>
    __global__ void updateClusters(KMeansData::KMeansDataGPU d_data, size_t *d_clustersMembershipCount, float *d_newClusters, uint32_t *d_newClustersMembershipCount, size_t previousBlocksCount)
    {
        auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
        if (threadId < d_data.clustersCount)
        {
            d_clustersMembershipCount[threadId] = 0;
            // For each cluster we calculate how many points belong to it accumulating results from all blocks
            for (size_t b = 0; b < previousBlocksCount; b++)
            {
                d_clustersMembershipCount[threadId] += d_newClustersMembershipCount[d_data.clustersCount * b + threadId];
            }
        }

        // We can call it here, because we know that this kernel will always be launched in only one block
        __syncthreads();

        if (threadId < d_data.clustersCount * DIM)
        {
            d_data.d_clustersValues[threadId] = 0;
            // For each cluster dimension we accumulate results from all blocks
            for (size_t b = 0; b < previousBlocksCount; b++)
            {
                d_data.d_clustersValues[threadId] += d_newClusters[d_data.clustersCount * DIM * b + threadId];
            }
            // Can we somehow remove this `%` operation? its probably slow
            size_t clusterId = threadId % d_data.clustersCount;
            // We divide by number of points in cluster to get mean
            d_data.d_clustersValues[threadId] /= d_clustersMembershipCount[clusterId];
        }
    }

    template <size_t DIM>
    Utils::ClusteringResult kMeansClustering(KMeansData::KMeansDataGPU d_data)
    {
        // PointsCount is always greater than dim * clustersCount * newClustersBlockCount (~ 20 * 20 * 1000 = 400 000 << 1 000 000 )
        const uint32_t newClustersBlocksCount = ceil(d_data.pointsCount * 1.0 / Consts::THREADS_PER_BLOCK);
        const size_t newClustersSharedMemorySize = d_data.clustersCount * DIM * sizeof(float) + d_data.clustersCount * sizeof(uint32_t) + sizeof(int);

        // We want to have clustersCount * DIM threads
        // We know in worst case scenario it's 20 * 20 = 400 < 1024, so it's always gonna fit in one block
        const uint32_t updateClustersBlocksCount = 1;

        // Check if device has enough memory for our shared memory size
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        if (newClustersSharedMemorySize > prop.sharedMemPerBlock)
        {
            throw std::runtime_error("Required shared memory exceeds device limits");
        }

        // GPU allocations
        size_t *d_memberships;
        CHECK_CUDA(cudaMalloc(&d_memberships, sizeof(size_t) * d_data.pointsCount));
        // We initialize the array that membership[i] = size_t::MAX
        CHECK_CUDA(cudaMemset(d_memberships, 0xFF, sizeof(size_t) * d_data.pointsCount));

        size_t *d_clustersMembershipCount;
        CHECK_CUDA(cudaMalloc(&d_clustersMembershipCount, sizeof(size_t) * d_data.clustersCount));

        float *d_newClusters;
        // We have separate clustersValues for each block
        CHECK_CUDA(cudaMalloc(&d_newClusters, sizeof(float) * d_data.clustersCount * DIM * newClustersBlocksCount));

        uint32_t *d_newClustersMembershipCount;
        // We have separate clustersCount for each block
        CHECK_CUDA(cudaMalloc(&d_newClustersMembershipCount, sizeof(uint32_t) * d_data.clustersCount * newClustersBlocksCount));

        int *d_shouldContinue;
        CHECK_CUDA(cudaMalloc(&d_shouldContinue, sizeof(int) * newClustersBlocksCount));

        // CPU allocation
        int *shouldContinue = (int *)malloc(sizeof(int) * newClustersBlocksCount);
        if (shouldContinue == nullptr)
        {
            throw std::runtime_error("Cannot allocate memory");
        }

        // We don't need to call cudaDeviceSynchronzie because we use single device and we don't use cuda streams
        for (size_t k = 0; k < Consts::MAX_ITERATION; k++)
        {
            // Call to first kernel
            calculateMembershipAndNewClusters<DIM><<<newClustersBlocksCount, Consts::THREADS_PER_BLOCK, newClustersSharedMemorySize>>>(d_data, d_newClusters, d_newClustersMembershipCount, d_memberships, d_shouldContinue);
            CHECK_CUDA(cudaGetLastError());

            // If all blocks return false than we know that no change was made and we can break from loop
            CHECK_CUDA(cudaMemcpy(shouldContinue, d_shouldContinue, sizeof(int) * newClustersBlocksCount, cudaMemcpyDeviceToHost));
            bool totalShouldContinue = false;
            for (size_t b = 0; b < newClustersBlocksCount; b++)
            {
                if (shouldContinue[b] != 0)
                {
                    totalShouldContinue = true;
                    break;
                }
            }
            if (!totalShouldContinue)
            {
                break;
            }

            // Call to second kernel
            updateClusters<DIM><<<updateClustersBlocksCount, Consts::THREADS_PER_BLOCK>>>(d_data, d_clustersMembershipCount, d_newClusters, d_newClustersMembershipCount, newClustersBlocksCount);
            CHECK_CUDA(cudaGetLastError());
        }

        // Prepare memory for storing results on CPU side
        thrust::host_vector<float> clustersValues(d_data.clustersCount * DIM);
        thrust::host_vector<size_t> membership(d_data.pointsCount);

        // Wait for GPU to finish calculations
        CHECK_CUDA(cudaDeviceSynchronize());

        // Copy result from GPU to CPU
        CHECK_CUDA(cudaMemcpy(clustersValues.data(), d_data.d_clustersValues, sizeof(float) * clustersValues.size(), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(membership.data(), d_memberships, sizeof(size_t) * d_data.pointsCount, cudaMemcpyDeviceToHost));

        // GPU deallocations
        cudaFree(d_memberships);
        cudaFree(d_clustersMembershipCount);
        cudaFree(d_newClusters);
        cudaFree(d_newClustersMembershipCount);
        cudaFree(d_shouldContinue);
        cudaFree(d_data.d_pointsValues);
        cudaFree(d_data.d_clustersValues);

        // CPU deallocation
        free(shouldContinue);

        return Utils::ClusteringResult{
            .clustersValues = clustersValues,
            .membership = membership,
        };
    }
} // KMeansClusteringGPUSM

#endif // K_MEANS_CLUSTERING_GPU_SM