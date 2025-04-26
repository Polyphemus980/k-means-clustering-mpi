#include "k_means_clustering_gpu_sm.cuh"

namespace KMeansClusteringGPUSM
{
    __device__ float pointToClusterDistanceSquared(KMeansData::KMeansDataGPU d_data, size_t pointIndex, size_t clusterIndex)
    {
        float distance = 0;
        for (size_t d = 0; d < d_data.DIM; d++)
        {
            float diff = KMeansData::Helpers::GetCoord(d_data.d_pointsValues, d_data.pointsCount, pointIndex, d) - KMeansData::Helpers::GetCoord(d_data.d_clustersValues, d_data.clustersCount, clusterIndex, d);

            distance += diff * diff;
        }
        return distance;
    }

    __device__ size_t findNearestCluster(KMeansData::KMeansDataGPU d_data, size_t pointIndex)
    {
        float minDist = pointToClusterDistanceSquared(d_data, pointIndex, 0);
        size_t minDistIndex = 0;
        for (size_t j = 1; j < d_data.clustersCount; j++)
        {
            float distSquared = pointToClusterDistanceSquared(d_data, pointIndex, j);

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
    __global__ void calculateMembershipAndNewClusters(KMeansData::KMeansDataGPU d_data, float *d_newClusters, uint32_t *d_newClustersMembershipCount, size_t *d_memberships, int *d_shouldContinue)
    {
        auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
        auto localThreadId = threadIdx.x;

        extern __shared__ int sharedMemory[];
        float *s_clusters = (float *)sharedMemory;
        uint32_t *s_clustersMembershipCount = (uint32_t *)&s_clusters[d_data.clustersCount * d_data.DIM];
        int *s_shouldContinue = (int *)&s_clustersMembershipCount[d_data.clustersCount];

        // Initialize shared memory in each block
        if (localThreadId == 0)
        {
            s_shouldContinue[0] = 0;
        }
        if (localThreadId < d_data.clustersCount * d_data.DIM)
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
            auto nearestClusterIndex = findNearestCluster(d_data, threadId);
            for (size_t d = 0; d < d_data.DIM; d++)
            {
                atomicAdd(&s_clusters[d * d_data.clustersCount + nearestClusterIndex], KMeansData::Helpers::GetCoord(d_data.d_pointsValues, d_data.pointsCount, threadId, d));
            }
            atomicAdd(&s_clustersMembershipCount[nearestClusterIndex], 1);
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

        if (localThreadId < d_data.clustersCount * d_data.DIM)
        {
            d_newClusters[blockIdx.x * d_data.clustersCount * d_data.DIM + localThreadId] = s_clusters[localThreadId];
        }

        if (localThreadId < d_data.clustersCount)
        {
            d_newClustersMembershipCount[blockIdx.x * d_data.clustersCount + localThreadId] = s_clustersMembershipCount[localThreadId];
        }
    }

    // Function for accumulating clusters memberships count
    // There should be thread for every cluster
    // We know it always will be run in single block
    __global__ void accumulateNewClustersMemerships(KMeansData::KMeansDataGPU d_data, size_t *d_clustersMembershipCount, uint32_t *d_newClustersMembershipCount, size_t previousBlocksCount)
    {
        auto threadId = threadIdx.x;
        d_clustersMembershipCount[threadId] = 0;
        // For each cluster we calculate how many points belong to it accumulating results from all blocks
        for (size_t b = 0; b < previousBlocksCount; b++)
        {
            d_clustersMembershipCount[threadId] += d_newClustersMembershipCount[d_data.clustersCount * b + threadId];
        }
    }

    // Function for updating clusters based on new membership
    // There should be thread spawned for every cluster for every dimension, so CLUSTERS_COUNT * DIM total
    // We know it always will be run in single block
    __global__ void updateClusters(KMeansData::KMeansDataGPU d_data, size_t *d_clustersMembershipCount, float *d_newClusters, size_t previousBlocksCount)
    {
        auto threadId = threadIdx.x;
        d_data.d_clustersValues[threadId] = 0;
        // For each cluster dimension we accumulate results from all blocks
        for (size_t b = 0; b < previousBlocksCount; b++)
        {
            d_data.d_clustersValues[threadId] += d_newClusters[d_data.clustersCount * d_data.DIM * b + threadId];
        }
        size_t clusterId = threadId % d_data.clustersCount;
        // We divide by number of points in cluster to get mean
        d_data.d_clustersValues[threadId] /= d_clustersMembershipCount[clusterId];
    }

    Utils::ClusteringResult kMeansClustering(KMeansData::KMeansDataGPU d_data)
    {
        CpuTimer::Timer cpuTimer;
        GpuTimer::Timer gpuTimer;

        // PointsCount is always greater than dim * clustersCount * newClustersBlockCount (~ 20 * 20 * 1000 = 400 000 << 1 000 000 )
        const uint32_t newClustersBlocksCount = ceil(d_data.pointsCount * 1.0 / Consts::THREADS_PER_BLOCK);
        const size_t newClustersSharedMemorySize = d_data.clustersCount * d_data.DIM * sizeof(float) + d_data.clustersCount * sizeof(uint32_t) + sizeof(int);

        // We want to have clustersCount threads
        // We know in worse case scenario it's 20 threads < 1024
        const uint32_t accumulateNewClustersMemershipsBlocksCount = 1;

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

        size_t *d_memberships = nullptr;
        size_t *d_clustersMembershipCount = nullptr;
        float *d_newClusters = nullptr;
        uint32_t *d_newClustersMembershipCount = nullptr;
        int *d_shouldContinue = nullptr;
        int *shouldContinue = nullptr;

        // Prepare memory for storing results on CPU side
        thrust::host_vector<float> clustersValues(d_data.clustersCount * d_data.DIM);
        thrust::host_vector<size_t> membership(d_data.pointsCount);

        bool isError = false;
        std::runtime_error error("placeholder");

        try
        {
            // GPU allocations
            CHECK_CUDA(cudaMalloc(&d_memberships, sizeof(size_t) * d_data.pointsCount));
            // We initialize the array that membership[i] = size_t::MAX
            CHECK_CUDA(cudaMemset(d_memberships, 0xFF, sizeof(size_t) * d_data.pointsCount));

            CHECK_CUDA(cudaMalloc(&d_clustersMembershipCount, sizeof(size_t) * d_data.clustersCount));

            // We have separate clustersValues for each block
            CHECK_CUDA(cudaMalloc(&d_newClusters, sizeof(float) * d_data.clustersCount * d_data.DIM * newClustersBlocksCount));

            // We have separate clustersCount for each block
            CHECK_CUDA(cudaMalloc(&d_newClustersMembershipCount, sizeof(uint32_t) * d_data.clustersCount * newClustersBlocksCount));

            CHECK_CUDA(cudaMalloc(&d_shouldContinue, sizeof(int) * newClustersBlocksCount));

            // CPU allocation
            shouldContinue = (int *)malloc(sizeof(int) * newClustersBlocksCount);

            if (shouldContinue == nullptr)
            {
                throw std::runtime_error("Cannot allocate memory");
            }

            printf("[START] K-means clustering (main algorithm)\n");
            gpuTimer.start();
            // We don't need to call cudaDeviceSynchronzie because we use single device and we don't use cuda streams
            for (size_t k = 0; k < Consts::MAX_ITERATION; k++)
            {
                // Calculate new membership
                calculateMembershipAndNewClusters<<<newClustersBlocksCount, Consts::THREADS_PER_BLOCK, newClustersSharedMemorySize>>>(d_data, d_newClusters, d_newClustersMembershipCount, d_memberships, d_shouldContinue);
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaDeviceSynchronize());

                // If all blocks return false than we know that no change was made and we can break from loop
                CHECK_CUDA(cudaMemcpy(shouldContinue, d_shouldContinue, sizeof(int) * newClustersBlocksCount, cudaMemcpyDeviceToHost));
                size_t totalShouldContinue = 0;
                for (size_t b = 0; b < newClustersBlocksCount; b++)
                {
                    totalShouldContinue += shouldContinue[b];
                }
                printf("[INFO] Iteration: %ld, changed points: %ld\n", k, totalShouldContinue);
                if (totalShouldContinue == 0)
                {
                    break;
                }

                // Accumulate counts from all blocks from previous kernel
                accumulateNewClustersMemerships<<<accumulateNewClustersMemershipsBlocksCount, d_data.clustersCount>>>(d_data, d_clustersMembershipCount, d_newClustersMembershipCount, newClustersBlocksCount);
                CHECK_CUDA(cudaGetLastError());

                // Calculate new clusters
                updateClusters<<<updateClustersBlocksCount, d_data.clustersCount * d_data.DIM>>>(d_data, d_clustersMembershipCount, d_newClusters, newClustersBlocksCount);
                CHECK_CUDA(cudaGetLastError());
            }
            gpuTimer.end();
            gpuTimer.printResult("K-means clustering (main algorithm)");

            // Wait for GPU to finish calculations
            CHECK_CUDA(cudaDeviceSynchronize());

            // Copy result from GPU to CPU
            printf("[START] Copy data from GPU to CPU\n");
            cpuTimer.start();
            CHECK_CUDA(cudaMemcpy(clustersValues.data(), d_data.d_clustersValues, sizeof(float) * clustersValues.size(), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(membership.data(), d_memberships, sizeof(size_t) * d_data.pointsCount, cudaMemcpyDeviceToHost));
            cpuTimer.end();
            cpuTimer.printResult("Copy data from GPU to CPU");
        }
        catch (const std::runtime_error &e)
        {
            fprintf(stderr, "[ERROR]: %s", e.what());
            isError = true;
            error = e;
            goto ERROR_HANDLING;
        }

    ERROR_HANDLING:
        // GPU deallocations
        if (d_memberships != nullptr)
        {
            cudaFree(d_memberships);
        }
        if (d_clustersMembershipCount != nullptr)
        {
            cudaFree(d_clustersMembershipCount);
        }
        if (d_newClusters != nullptr)
        {
            cudaFree(d_newClusters);
        }
        if (d_newClustersMembershipCount != nullptr)
        {
            cudaFree(d_newClustersMembershipCount);
        }
        if (d_shouldContinue != nullptr)
        {
            cudaFree(d_shouldContinue);
        }
        if (d_data.d_pointsValues != nullptr)
        {
            cudaFree(d_data.d_pointsValues);
        }
        if (d_data.d_clustersValues != nullptr)
        {
            cudaFree(d_data.d_clustersValues);
        }

        // CPU deallocation
        if (shouldContinue != nullptr)
        {
            free(shouldContinue);
        }

        if (isError)
        {
            throw error;
        }

        return Utils::ClusteringResult{
            .clustersValues = clustersValues,
            .membership = membership,
        };
    }

    Utils::ClusteringResult kMeansClusteringMPI(const KMeansData::KMeansData &h_kMeansData)
    {
        CpuTimer::Timer cpuTimer;
        GpuTimer::Timer gpuTimer;

        // Initialize MPI environment
        int rank, size;
        MPI_Init(NULL, NULL);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        printf("[INFO] Process %d of %d started\n", rank, size);

        // Get data dimensions
        size_t pointsCount = h_kMeansData.getPointsCount();
        size_t clustersCount = h_kMeansData.getClustersCount();
        size_t DIM = h_kMeansData.getDIM();

        // Calculate how many points each process will handle
        size_t pointsPerProcess = pointsCount / size;
        size_t remainingPoints = pointsCount % size;

        // Calculate start and end indices for this process
        size_t startPointIdx = rank * pointsPerProcess + (rank < remainingPoints ? rank : remainingPoints);
        size_t localPointsCount = pointsPerProcess + (rank < remainingPoints ? 1 : 0);

        printf("[INFO] Process %d handles points %zu to %zu\n", rank, startPointIdx, startPointIdx + localPointsCount - 1);

        // Prepare to distribute data
        thrust::host_vector<float> localPointsValues(localPointsCount * DIM);

        // On root process, prepare send counts and displacements
        thrust::host_vector<int> sendcounts(size);
        thrust::host_vector<int> displs(size);
        if (rank == 0)
        {
            int displacement = 0;
            for (int i = 0; i < size; i++)
            {
                size_t pointsForProcess = pointsPerProcess + (i < remainingPoints ? 1 : 0);
                sendcounts[i] = pointsForProcess * DIM;
                displs[i] = displacement;
                displacement += pointsForProcess * DIM;
            }
        }

        // Distribute point data - only rank 0 provides the source data
        MPI_Scatterv(
            rank == 0 ? h_kMeansData.getValues().data() : nullptr,
            rank == 0 ? sendcounts.data() : nullptr,
            rank == 0 ? displs.data() : nullptr,
            MPI_FLOAT,
            localPointsValues.data(),
            localPointsCount * DIM,
            MPI_FLOAT,
            0, MPI_COMM_WORLD);

        // For cluster centers, broadcast to all processes
        thrust::host_vector<float> clustersValues;
        if (rank == 0)
        {
            // Use the initial cluster values from the host data
            clustersValues = h_kMeansData.getClustersValues();
        }
        else
        {
            // Other processes allocate memory to receive the data
            clustersValues.resize(clustersCount * DIM);
        }

        // Broadcast initial cluster centers to all processes
        MPI_Bcast(clustersValues.data(), clustersCount * DIM, MPI_FLOAT, 0, MPI_COMM_WORLD);

        // Set up GPU data
        KMeansData::KMeansDataGPU d_data;
        d_data.DIM = DIM;
        d_data.clustersCount = clustersCount;
        d_data.pointsCount = localPointsCount;

        // GPU allocations - now only allocate and transfer the local subset
        CHECK_CUDA(cudaMalloc(&d_data.d_pointsValues, sizeof(float) * localPointsCount * DIM));
        CHECK_CUDA(cudaMemcpy(d_data.d_pointsValues, localPointsValues.data(),
                              sizeof(float) * localPointsCount * DIM, cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMalloc(&d_data.d_clustersValues, sizeof(float) * clustersCount * DIM));
        CHECK_CUDA(cudaMemcpy(d_data.d_clustersValues, clustersValues.data(),
                              sizeof(float) * clustersCount * DIM, cudaMemcpyHostToDevice));

        // Other GPU memory allocations
        const uint32_t newClustersBlocksCount = ceil(localPointsCount * 1.0 / Consts::THREADS_PER_BLOCK);
        const size_t newClustersSharedMemorySize = clustersCount * DIM * sizeof(float) +
                                                   clustersCount * sizeof(uint32_t) + sizeof(int);

        // Resource allocation
        size_t *d_memberships = nullptr;
        size_t *d_clustersMembershipCount = nullptr;
        float *d_newClusters = nullptr;
        uint32_t *d_newClustersMembershipCount = nullptr;
        int *d_shouldContinue = nullptr;
        int *shouldContinue = nullptr;

        bool isError = false;
        std::runtime_error error("placeholder");

        try
        {
            // Allocate device memory
            CHECK_CUDA(cudaMalloc(&d_memberships, sizeof(size_t) * localPointsCount));
            CHECK_CUDA(cudaMemset(d_memberships, 0xFF, sizeof(size_t) * localPointsCount));

            CHECK_CUDA(cudaMalloc(&d_clustersMembershipCount, sizeof(size_t) * clustersCount));
            CHECK_CUDA(cudaMalloc(&d_newClusters, sizeof(float) * clustersCount * DIM * newClustersBlocksCount));
            CHECK_CUDA(cudaMalloc(&d_newClustersMembershipCount, sizeof(uint32_t) * clustersCount * newClustersBlocksCount));
            CHECK_CUDA(cudaMalloc(&d_shouldContinue, sizeof(int) * newClustersBlocksCount));

            // CPU buffers for communication
            shouldContinue = (int *)malloc(sizeof(int) * newClustersBlocksCount);
            thrust::host_vector<size_t> localMembership(localPointsCount);
            thrust::host_vector<float> localClusterSums(clustersCount * DIM, 0.0f);
            thrust::host_vector<size_t> localClusterCounts(clustersCount, 0);
            thrust::host_vector<float> globalClusterSums(clustersCount * DIM, 0.0f);
            thrust::host_vector<size_t> globalClusterCounts(clustersCount, 0);

            if (shouldContinue == nullptr)
            {
                throw std::runtime_error("Cannot allocate memory");
            }

            printf("[START] Process %d: K-means clustering (main algorithm)\n", rank);
            gpuTimer.start();

            bool continueIterating = true;
            int globalContinue = 1;

            for (size_t k = 0; k < Consts::MAX_ITERATION && continueIterating; k++)
            {
                // Calculate new membership
                calculateMembershipAndNewClusters<<<newClustersBlocksCount, Consts::THREADS_PER_BLOCK, newClustersSharedMemorySize>>>(
                    d_data, d_newClusters, d_newClustersMembershipCount, d_memberships, d_shouldContinue);
                CHECK_CUDA(cudaGetLastError());
                CHECK_CUDA(cudaDeviceSynchronize());

                // Check if points changed clusters
                CHECK_CUDA(cudaMemcpy(shouldContinue, d_shouldContinue, sizeof(int) * newClustersBlocksCount, cudaMemcpyDeviceToHost));
                int localShouldContinue = 0;
                for (size_t b = 0; b < newClustersBlocksCount; b++)
                {
                    localShouldContinue += shouldContinue[b];
                }

                // Gather local cluster information
                accumulateNewClustersMemerships<<<1, clustersCount>>>(
                    d_data, d_clustersMembershipCount, d_newClustersMembershipCount, newClustersBlocksCount);
                CHECK_CUDA(cudaGetLastError());

                // Extract local cluster sums - gather raw sums before calculating new centers
                // We need raw sums for MPI reduction
                size_t *h_clustersMembershipCount = localClusterCounts.data();
                CHECK_CUDA(cudaMemcpy(h_clustersMembershipCount, d_clustersMembershipCount,
                                      sizeof(size_t) * clustersCount, cudaMemcpyDeviceToHost));

                // Get local cluster sums from GPU
                float *h_localClusterSums = localClusterSums.data();
                for (size_t c = 0; c < clustersCount; c++)
                {
                    for (size_t d = 0; d < DIM; d++)
                    {
                        float sum = 0.0f;
                        for (size_t b = 0; b < newClustersBlocksCount; b++)
                        {
                            CHECK_CUDA(cudaMemcpy(&sum, &d_newClusters[b * clustersCount * DIM + d * clustersCount + c],
                                                  sizeof(float), cudaMemcpyDeviceToHost));
                            h_localClusterSums[c * DIM + d] += sum;
                        }
                    }
                }

                // Use MPI to combine results across all processes
                MPI_Allreduce(&localShouldContinue, &globalContinue, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(localClusterSums.data(), globalClusterSums.data(), clustersCount * DIM,
                              MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(localClusterCounts.data(), globalClusterCounts.data(), clustersCount,
                              MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

                // Calculate new cluster centers based on global data
                for (size_t c = 0; c < clustersCount; c++)
                {
                    if (globalClusterCounts[c] > 0)
                    {
                        for (size_t d = 0; d < DIM; d++)
                        {
                            clustersValues[c * DIM + d] = globalClusterSums[c * DIM + d] / globalClusterCounts[c];
                        }
                    }
                }

                // Update clusters on all GPUs with the new centroids
                CHECK_CUDA(cudaMemcpy(d_data.d_clustersValues, clustersValues.data(),
                                      sizeof(float) * clustersCount * DIM, cudaMemcpyHostToDevice));

                // Check if we should continue iterating
                continueIterating = (globalContinue > 0);

                // Reset local sums and counts for next iteration
                std::fill(localClusterSums.begin(), localClusterSums.end(), 0.0f);
                std::fill(localClusterCounts.begin(), localClusterCounts.end(), 0);

                printf("[INFO] Process %d: Iteration %ld, global changed points: %d\n", rank, k, globalContinue);
            }

            gpuTimer.end();
            if (rank == 0)
            {
                gpuTimer.printResult("K-means clustering (main algorithm)");
            }

            // Gather final memberships
            CHECK_CUDA(cudaMemcpy(localMembership.data(), d_memberships,
                                  sizeof(size_t) * localPointsCount, cudaMemcpyDeviceToHost));

            // Prepare to gather all memberships to rank 0
            thrust::host_vector<size_t> globalMembership;
            thrust::host_vector<int> recvCounts;
            thrust::host_vector<int> displacements;

            if (rank == 0)
            {
                globalMembership.resize(pointsCount);
                recvCounts.resize(size);
                displacements.resize(size);

                int displacement = 0;
                for (int i = 0; i < size; i++)
                {
                    int localSize = pointsPerProcess + (i < remainingPoints ? 1 : 0);
                    recvCounts[i] = localSize;
                    displacements[i] = displacement;
                    displacement += localSize;
                }
            }

            // Gather all memberships to rank 0
            MPI_Gatherv(localMembership.data(), localPointsCount, MPI_UNSIGNED_LONG,
                        rank == 0 ? globalMembership.data() : nullptr,
                        rank == 0 ? recvCounts.data() : nullptr,
                        rank == 0 ? displacements.data() : nullptr,
                        MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

            // Create result
            Utils::ClusteringResult result;
            if (rank == 0)
            {
                result.clustersValues = clustersValues;
                result.membership = globalMembership;
            }

            // Cleanup resources
            if (d_memberships)
                cudaFree(d_memberships);
            if (d_clustersMembershipCount)
                cudaFree(d_clustersMembershipCount);
            if (d_newClusters)
                cudaFree(d_newClusters);
            if (d_newClustersMembershipCount)
                cudaFree(d_newClustersMembershipCount);
            if (d_shouldContinue)
                cudaFree(d_shouldContinue);
            if (d_data.d_pointsValues)
                cudaFree(d_data.d_pointsValues);
            if (d_data.d_clustersValues)
                cudaFree(d_data.d_clustersValues);

            // CPU cleanup
            if (shouldContinue)
                free(shouldContinue);

            // Finalize MPI before returning
            MPI_Finalize();

            return result;
        }
        catch (const std::runtime_error &e)
        {
            fprintf(stderr, "[ERROR] Process %d: %s\n", rank, e.what());
            isError = true;
            error = e;

            // Cleanup resources
            if (d_memberships)
                cudaFree(d_memberships);
            if (d_clustersMembershipCount)
                cudaFree(d_clustersMembershipCount);
            if (d_newClusters)
                cudaFree(d_newClusters);
            if (d_newClustersMembershipCount)
                cudaFree(d_newClustersMembershipCount);
            if (d_shouldContinue)
                cudaFree(d_shouldContinue);
            if (d_data.d_pointsValues)
                cudaFree(d_data.d_pointsValues);
            if (d_data.d_clustersValues)
                cudaFree(d_data.d_clustersValues);

            // CPU cleanup
            if (shouldContinue)
                free(shouldContinue);

            // Finalize MPI
            MPI_Finalize();

            throw error;
        }
    }

} // KMeansClusteringGPUSM
