#ifndef K_MEANS_CLUSTERING_GPU_THRUST
#define K_MEANS_CLUSTERING_GPU_THRUST

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

#include "utils.cuh"
#include "k_means_data.cuh"
#include "consts.cuh"
#include "cpu_timer.cuh"

namespace KMeansClusteringGPUThrust
{
    template <size_t DIM>
    __device__ float pointToClusterDistanceSquared(const float *d_pointsValues, size_t pointsCount, size_t pointIndex, const float *d_clustersValues, size_t clustersCount, size_t clusterIndex)
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
    struct FindClusterFunctor
    {
        const float *d_pointsValues;
        const float *d_clustersValues;
        size_t *d_memberships;
        size_t pointsCount;
        size_t clustersCount;

        __host__ __device__ FindClusterFunctor(
            const float *pointsValues,
            const float *clustersValues,
            size_t *memberships,
            size_t pointsCount,
            size_t clustersCount)
            : d_pointsValues(pointsValues),
              d_clustersValues(clustersValues),
              d_memberships(memberships),
              pointsCount(pointsCount),
              clustersCount(clustersCount) {}

        __host__ __device__ size_t operator()(size_t pointIndex) const
        {
            float minDistance = pointToClusterDistanceSquared<DIM>(d_pointsValues, pointsCount, pointIndex, d_clustersValues, clustersCount, 0);
            size_t minDistanceIndex = 0;

            for (size_t j = 1; j < clustersCount; j++)
            {
                float dist = pointToClusterDistanceSquared<DIM>(d_pointsValues, pointsCount, pointIndex, d_clustersValues, clustersCount, j);
                if (dist < minDistance)
                {
                    minDistance = dist;
                    minDistanceIndex = j;
                }
            }

            size_t previousCluster = d_memberships[pointIndex];
            d_memberships[pointIndex] = minDistanceIndex;

            return (minDistanceIndex != previousCluster ? 1 : 0);
        }
    };

    // Returns number of points that changed their membership
    template <size_t DIM>
    size_t findClustersForPoints(const KMeansData::KMeansDataGPUThrust &data, thrust::device_vector<size_t> &memberships)
    {
        const float *d_pointsValues = thrust::raw_pointer_cast(data.pointsValues.data());
        const float *d_clustersValues = thrust::raw_pointer_cast(data.clustersValues.data());
        size_t *d_memberships = thrust::raw_pointer_cast(memberships.data());

        FindClusterFunctor<DIM> findClusterFunctor(
            d_pointsValues,
            d_clustersValues,
            d_memberships,
            data.pointsCount,
            data.clustersCount);

        size_t changedPointsCount = thrust::transform_reduce(
            thrust::device,
            thrust::counting_iterator<size_t>(0),
            thrust::counting_iterator<size_t>(data.pointsCount),
            findClusterFunctor,
            0,
            thrust::plus<size_t>());

        return changedPointsCount;
    }

    template <size_t DIM>
    void updateClusters(KMeansData::KMeansDataGPUThrust &data, const thrust::device_vector<size_t> &memberships)
    {
        thrust::device_vector<size_t> clustersMembershipsCount(data.clustersCount);

        // We don't care about this - we just pass it because thrust functions need it
        thrust::device_vector<float> outputKeys(data.pointsCount);

        thrust::device_vector<size_t> membershipsCopy(memberships);

        // Calculate how many points are asigned to each cluster
        thrust::sort(thrust::device, membershipsCopy.begin(), membershipsCopy.end());
        thrust::reduce_by_key(
            thrust::device,
            membershipsCopy.begin(),
            membershipsCopy.end(),
            thrust::constant_iterator<size_t>(1),
            outputKeys.begin(),
            clustersMembershipsCount.begin());

        // Calculate new clusters coords (separately for each dimension)
        for (size_t d = 0; d < DIM; d++)
        {
            thrust::device_vector<size_t> membershipsInnerCopy(memberships);

            auto dimensionStart = data.pointsValues.begin() + d * data.pointsCount;
            auto dimensionEnd = dimensionStart + data.pointsCount;
            thrust::device_vector<float> pointsValuesInDimension(data.pointsCount);
            thrust::copy(dimensionStart, dimensionEnd, pointsValuesInDimension.begin());

            thrust::device_vector<float> clustersSumsInDimension(data.pointsCount);

            // Calculate sum of all points (d dimension) assigned to each cluster
            thrust::sort_by_key(
                thrust::device,
                membershipsInnerCopy.begin(),
                membershipsInnerCopy.end(),
                pointsValuesInDimension.begin());
            thrust::reduce_by_key(
                thrust::device,
                membershipsInnerCopy.begin(),
                membershipsInnerCopy.end(),
                pointsValuesInDimension.begin(),
                outputKeys.begin(),
                clustersSumsInDimension.begin());

            // Calculate means
            auto clustersDimensionStart = data.clustersValues.begin() + d * data.clustersCount;
            thrust::transform(
                thrust::device,
                clustersSumsInDimension.begin(),
                clustersSumsInDimension.begin() + data.clustersCount,
                thrust::make_transform_iterator(clustersMembershipsCount.begin(), [] __host__ __device__(size_t count)
                                                { return count > 0 ? 1.0f / count : 0.0f; }),
                clustersDimensionStart,
                thrust::multiplies<float>());
        }
    }

    template <size_t DIM>
    Utils::ClusteringResult kMeansClustering(KMeansData::KMeansDataGPUThrust data)
    {
        CpuTimer::Timer cpuTimer;

        // We initialize memberships array with POINTS COUNT, so that in first step each point doesn't have any cluster asssigned
        thrust::device_vector<size_t> memberships(data.pointsCount, data.pointsCount);

        printf("[START] K-means clustering (main algorithm)\n");
        cpuTimer.start();
        for (size_t k = 0; k < Consts::MAX_ITERATION; k++)
        {
            size_t changedPointsCount = findClustersForPoints<DIM>(data, memberships);
            printf("[INFO] Iteration: %ld, changed points: %ld\n", k, changedPointsCount);
            if (changedPointsCount == 0)
            {
                break;
            }
            updateClusters<DIM>(data, memberships);
        }
        cpuTimer.end();
        cpuTimer.printResult("K-means clustering (main algorithm)");

        thrust::host_vector<float> clustersValues(data.clustersValues);
        thrust::host_vector<size_t> hostMemberships(memberships);

        return Utils::ClusteringResult{
            .clustersValues = clustersValues,
            .membership = hostMemberships,
        };
    }

} // KMeansClusteringGPUThrust

#endif // K_MEANS_CLUSTERING_GPU_THRUST