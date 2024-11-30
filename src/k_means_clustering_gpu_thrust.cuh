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

    // Returns number of points that changed their membership
    template <size_t DIM>
    size_t findClustersForPoints(const KMeansData::KMeansDataGPUThrust &data, thrust::device_vector<size_t> &memberships)
    {
        const float *d_pointsValues = thrust::raw_pointer_cast(data.pointsValues.data());
        const float *d_clustersValues = thrust::raw_pointer_cast(data.clustersValues.data());
        size_t *d_memberships = thrust::raw_pointer_cast(memberships.data());

        size_t changedPointsCount = thrust::transform_reduce(
            thrust::counting_iterator<size_t>(0),
            thrust::counting_iterator<size_t>(data.pointsCount),
            [=] __host__ __device__(size_t pointIndex)
            {
                float minDistance = pointToClusterDistanceSquared<DIM>(d_pointsValues, data.pointsCount, pointIndex, d_clustersValues, data.clustersCount, 0);
                size_t minDistanceIndex = 0;
                for (size_t j = 1; j < data.clustersCount; j++)
                {
                    float dist = pointToClusterDistanceSquared<DIM>(d_pointsValues, data.pointsCount, pointIndex, d_clustersValues, data.clustersCount, j);
                    if (dist < minDistance)
                    {
                        minDistance = dist;
                        minDistanceIndex = j;
                    }
                }
                size_t previousCluster = d_memberships[pointIndex];
                d_memberships[pointIndex] = minDistanceIndex;
                return (minDistanceIndex != previousCluster ? 1 : 0);
            },
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
        thrust::sort(membershipsCopy.begin(), membershipsCopy.end());
        thrust::reduce_by_key(
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

            // // Calculate sum of all points (d dimension) assigned to each cluster
            thrust::sort_by_key(
                membershipsInnerCopy.begin(),
                membershipsInnerCopy.end(),
                pointsValuesInDimension.begin());
            thrust::reduce_by_key(
                membershipsInnerCopy.begin(),
                membershipsInnerCopy.end(),
                pointsValuesInDimension.begin(),
                outputKeys.begin(),
                clustersSumsInDimension.begin());

            // // Calculate means
            // FIXME: there is illegal memory access error in this code
            // auto clustersDimensionStart = data.clustersValues.begin() + d * data.clustersCount;
            // thrust::transform(
            //     clustersSumsInDimension.begin(),
            //     clustersSumsInDimension.end(),
            //     thrust::make_transform_iterator(clustersMembershipsCount.begin(), [] __host__ __device__(size_t count)
            //                                     { return count > 0 ? 1.0f / count : 0.0f; }),
            //     clustersDimensionStart,
            //     thrust::multiplies<float>());
        }
    }

    template <size_t DIM>
    Utils::ClusteringResult kMeansClustering(KMeansData::KMeansDataGPUThrust data)
    {
        // We initialize memberships array with POINTS COUNT, so that in first step each point doesn't have any cluster asssigned
        thrust::device_vector<size_t> memberships(data.pointsCount, data.pointsCount);

        for (size_t k = 0; k < Consts::MAX_ITERATION; k++)
        {
            size_t changedPointsCount = findClustersForPoints<DIM>(data, memberships);
            if (changedPointsCount == 0)
            {
                break;
            }
            updateClusters<DIM>(data, memberships);
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