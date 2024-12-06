#ifndef K_MEANS_CLUSTERING_GPU_THRUST
#define K_MEANS_CLUSTERING_GPU_THRUST

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

#include "utils.cuh"
#include "k_means_data.cuh"
#include "consts.cuh"
#include "cpu_timer.cuh"

namespace KMeansClusteringGPUThrust
{
    template <size_t DIM>
    __device__ float pointToClusterDistanceSquared(float *d_pointsValues, size_t pointsCount, float *d_clustersValues, size_t clustersCount, size_t pointIndex, size_t clusterIndex)
    {
        float distance = 0;
        for (size_t d = 0; d < DIM; d++)
        {
            float diff = KMeansData::Helpers::GetCoord(d_pointsValues, pointsCount, pointIndex, d) - KMeansData::Helpers::GetCoord(d_clustersValues, clustersCount, clusterIndex, d);

            distance += diff * diff;
        }
        return distance;
    }

    template <size_t DIM>
    __device__ size_t findNearestCluster(float *d_pointsValues, size_t pointsCount, float *d_clustersValues, size_t clustersCount, size_t pointIndex)
    {
        float minDist = pointToClusterDistanceSquared<DIM>(d_pointsValues, pointsCount, d_clustersValues, clustersCount, pointIndex, 0);
        size_t minDistIndex = 0;
        for (size_t j = 1; j < clustersCount; j++)
        {
            float distSquared = pointToClusterDistanceSquared<DIM>(d_pointsValues, pointsCount, d_clustersValues, clustersCount, pointIndex, j);

            if (distSquared < minDist)
            {
                minDist = distSquared;
                minDistIndex = j;
            }
        }
        return minDistIndex;
    }

    // Function for finding new membership for each point
    // Each thread should be responsible for single point
    template <size_t DIM>
    __global__ void calculateMemberships(float *d_pointsValues, size_t pointsCount, float *d_clustersValues, size_t clustersCount, size_t *d_memberships, int *d_shouldContinue)
    {
        auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
        auto localThreadId = threadIdx.x;

        __shared__ int s_shouldContinue[1];

        // Initialize shared memory in each block
        if (localThreadId == 0)
        {
            s_shouldContinue[0] = 0;
        }

        // Ensure shared memory is properly initialized
        __syncthreads();

        // For each point find its nearest cluster, update membership table and save results in shared memory
        if (threadId < pointsCount)
        {
            auto nearestClusterIndex = findNearestCluster<DIM>(d_pointsValues, pointsCount, d_clustersValues, clustersCount, threadId);
            auto previousClusterIndex = d_memberships[threadId];
            if (previousClusterIndex != nearestClusterIndex)
            {
                atomicAdd(&s_shouldContinue[0], 1);
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
    }

    template <size_t DIM>
    void updateClusters(KMeansData::KMeansDataGPUThrust &data, const thrust::device_vector<size_t> &memberships)
    {
        // How many points are assigned to each cluster
        thrust::device_vector<size_t> clustersMembershipsCount(data.clustersCount);

        // We don't want to sort and change original memberships - we must work on copy
        thrust::device_vector<size_t> membershipsCopy(memberships);

        // Calculate how many points are asigned to each cluster
        thrust::sort(thrust::device, membershipsCopy.begin(), membershipsCopy.end());
        thrust::reduce_by_key(
            thrust::device,
            membershipsCopy.begin(),
            membershipsCopy.end(),
            // Iterator that always returns 1 - ideal for our case of counting elements with same key
            thrust::constant_iterator<size_t>(1),
            // We don't care about keys after reduce, we only want to have values - so we can discard keys instead
            // of using some dummy vector for it
            thrust::make_discard_iterator(),
            clustersMembershipsCount.begin());

        // clustersMembershipsCount repeated DIM times
        thrust::device_vector<size_t> clustersMembershipsCountAllDims(data.clustersCount * DIM);

        // Sums of coords of points assigned to clusters
        // e.g. first element - sum of first coord of all points assigned to first clusters
        thrust::device_vector<float> clustersSumsInAllDimensions(data.clustersCount * DIM);

        // Helper vectors used for storing pointsValues in d-th dimension, which will be sorted by their membership
        thrust::device_vector<float> pointsValuesInDimension(data.pointsCount);

        // Calculate sums of coordinates for each cluster (separately for each dimension) + copy membershipCount for dimension
        for (size_t d = 0; d < DIM; d++)
        {
            // Copy memberships count for d-th dimension (done here so we only have one loop)
            thrust::copy(clustersMembershipsCount.begin(), clustersMembershipsCount.end(), clustersMembershipsCountAllDims.begin() + d * data.clustersCount);

            // Restore to original state
            thrust::copy(memberships.begin(), memberships.end(), membershipsCopy.begin());

            auto dimensionStart = data.pointsValues.begin() + d * data.pointsCount;
            auto dimensionEnd = dimensionStart + data.pointsCount;
            thrust::copy(dimensionStart, dimensionEnd, pointsValuesInDimension.begin());

            // Calculate sum of all points (d-th dimension) assigned to each cluster
            thrust::sort_by_key(
                thrust::device,
                membershipsCopy.begin(),
                membershipsCopy.end(),
                pointsValuesInDimension.begin());
            thrust::reduce_by_key(
                thrust::device,
                membershipsCopy.begin(),
                membershipsCopy.end(),
                pointsValuesInDimension.begin(),
                // Same as before - we don't care about ouput keys and can safely discard them
                thrust::make_discard_iterator(),
                clustersSumsInAllDimensions.begin() + d * data.clustersCount);
        }

        // Calculate mean for each cluster in every dimension
        thrust::transform(
            thrust::device,
            clustersSumsInAllDimensions.begin(),
            clustersSumsInAllDimensions.end(),
            thrust::make_transform_iterator(clustersMembershipsCountAllDims.begin(), [] __host__ __device__(size_t count)
                                            { return count > 0 ? 1.0f / count : 0.0f; }),
            data.clustersValues.begin(),
            thrust::multiplies<float>());
    }

    template <size_t DIM>
    Utils::ClusteringResult kMeansClustering(KMeansData::KMeansDataGPUThrust data)
    {
        CpuTimer::Timer cpuTimer;

        // We initialize memberships array with clustersCount, so that in first step each point doesn't have any cluster asssigned
        thrust::device_vector<size_t> memberships(data.pointsCount, data.clustersCount);

        const uint32_t calculateMembershipsBlocksCount = ceil(data.pointsCount * 1.0 / Consts::THREADS_PER_BLOCK);
        thrust::device_vector<int> shouldContinue(calculateMembershipsBlocksCount, 0);

        int *h_shouldContinue = nullptr;

        bool isError = false;
        std::runtime_error error("placeholder");

        try
        {
            h_shouldContinue = (int *)malloc(sizeof(int) * calculateMembershipsBlocksCount);

            if (h_shouldContinue == nullptr)
            {
                throw std::runtime_error("Cannot allocate memory");
            }

            printf("[START] K-means clustering (main algorithm)\n");
            cpuTimer.start();

            auto d_memberships = thrust::raw_pointer_cast(memberships.data());
            auto d_shouldContinue = thrust::raw_pointer_cast(shouldContinue.data());
            auto d_pointsValues = thrust::raw_pointer_cast(data.pointsValues.data());
            auto d_clustersValues = thrust::raw_pointer_cast(data.clustersValues.data());

            for (size_t k = 0; k < Consts::MAX_ITERATION; k++)
            {
                // Calculate new membership
                calculateMemberships<DIM><<<calculateMembershipsBlocksCount, Consts::THREADS_PER_BLOCK>>>(d_pointsValues, data.pointsCount, d_clustersValues, data.clustersCount, d_memberships, d_shouldContinue);
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaDeviceSynchronize());

                // If all blocks return false than we know that no change was made and we can break from loop
                CHECK_CUDA(cudaMemcpy(h_shouldContinue, d_shouldContinue, sizeof(int) * calculateMembershipsBlocksCount, cudaMemcpyDeviceToHost));
                size_t totalShouldContinue = 0;
                for (size_t b = 0; b < calculateMembershipsBlocksCount; b++)
                {
                    totalShouldContinue += h_shouldContinue[b];
                }
                printf("[INFO] Iteration: %ld, changed points: %ld\n", k, totalShouldContinue);
                if (totalShouldContinue == 0)
                {
                    break;
                }

                updateClusters<DIM>(data, memberships);
            }
            cpuTimer.end();
            cpuTimer.printResult("K-means clustering (main algorithm)");
        }
        catch (const std::runtime_error &e)
        {
            fprintf(stderr, "[ERROR]: %s", e.what());
            isError = true;
            error = e;
            goto ERROR_HANDLING;
        }

    ERROR_HANDLING:
        if (h_shouldContinue != nullptr)
        {
            free(h_shouldContinue);
        }

        if (isError)
        {
            throw error;
        }

        thrust::host_vector<float> clustersValues(data.clustersValues);
        thrust::host_vector<size_t> hostMemberships(memberships);

        return Utils::ClusteringResult{
            .clustersValues = clustersValues,
            .membership = hostMemberships,
        };
    }

} // KMeansClusteringGPUThrust

#endif // K_MEANS_CLUSTERING_GPU_THRUST