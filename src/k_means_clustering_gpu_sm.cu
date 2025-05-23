#include "k_means_clustering_gpu_sm.cuh"
#include "mpi.h"
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

    __global__ void calculateMembershipMPI(KMeansData::KMeansDataGPU d_data, int *d_memberships, int *d_shouldContinue)
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
        if (threadId < d_data.pointsCount)
        {
            auto nearestClusterIndex = findNearestCluster(d_data, threadId);
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

    __global__ void calculateNewClustersMPI(int clustersCount, int DIM, float *d_pointsValues, int pointsCount, float *d_newClusters, uint32_t *d_newClustersMembershipCount, int *d_memberships)
    {
        auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
        auto localThreadId = threadIdx.x;

        extern __shared__ int sharedMemory[];
        float *s_clusters = (float *)sharedMemory;
        uint32_t *s_clustersMembershipCount = (uint32_t *)&s_clusters[clustersCount * DIM];

        // Initialize shared memory in each block
        if (localThreadId < clustersCount * DIM)
        {
            s_clusters[localThreadId] = 0.0f;
        }

        if (localThreadId < clustersCount)
        {
            s_clustersMembershipCount[localThreadId] = 0.0f;
        }

        // Ensure shared memory is properly initialized
        __syncthreads();

        // For each point find its nearest cluster, update membership table and save results in shared memory
        if (threadId < pointsCount)
        {
            auto clusterIndex = d_memberships[threadId];
            for (size_t d = 0; d < DIM; d++)
            {
                atomicAdd(&s_clusters[d * clustersCount + clusterIndex], KMeansData::Helpers::GetCoord(d_pointsValues, pointsCount, threadId, d));
            }
            atomicAdd(&s_clustersMembershipCount[clusterIndex], 1);
        }

        // Finish all calculation made on shared memory
        __syncthreads();

        // Copy results from shared memory to global memory
        if (localThreadId < clustersCount * DIM)
        {
            d_newClusters[blockIdx.x * clustersCount * DIM + localThreadId] = s_clusters[localThreadId];
        }

        if (localThreadId < clustersCount)
        {
            d_newClustersMembershipCount[blockIdx.x * clustersCount + localThreadId] = s_clustersMembershipCount[localThreadId];
        }
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
            CHECK_CUDA(cudaMalloc((void **)&d_memberships, sizeof(size_t) * d_data.pointsCount));
            // We initialize the array that membership[i] = size_t::MAX
            CHECK_CUDA(cudaMemset(d_memberships, 0xFF, sizeof(size_t) * d_data.pointsCount));

            CHECK_CUDA(cudaMalloc((void **)&d_clustersMembershipCount, sizeof(size_t) * d_data.clustersCount));

            // We have separate clustersValues for each block
            CHECK_CUDA(cudaMalloc((void **)&d_newClusters, sizeof(float) * d_data.clustersCount * d_data.DIM * newClustersBlocksCount));

            // We have separate clustersCount for each block
            CHECK_CUDA(cudaMalloc((void **)&d_newClustersMembershipCount, sizeof(uint32_t) * d_data.clustersCount * newClustersBlocksCount));

            CHECK_CUDA(cudaMalloc((void **)&d_shouldContinue, sizeof(int) * newClustersBlocksCount));

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

    Utils::ClusteringResult kMeansClusteringMPI(const KMeansData::KMeansData &h_kMeansData, int size)
    {
        CpuTimer::Timer cpuTimer;
        GpuTimer::Timer gpuTimer;

        int pointsCount = h_kMeansData.getPointsCount();
        int clustersCount = h_kMeansData.getClustersCount();
        int DIM = h_kMeansData.getDIM();

        // Calculate how many points each process will handle
        int pointsPerProcess = pointsCount / size;
        int remainingPoints = pointsCount % size;

        printf("Size: %d \n", size);

        for (int i = 1; i < size; i++)
        {
            printf("[INFO] Sending dim to process %d \n", i);
            MPI_Send(&DIM, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        int *localPointCounts = (int *)malloc(sizeof(int) * size);
        int *startPointIndices = (int *)malloc(sizeof(int) * size);

        printf("pointsPerProcess %d \n", pointsPerProcess);
        startPointIndices[0] = 0;
        localPointCounts[0] = pointsPerProcess + (0 < remainingPoints ? 1 : 0);

        for (int i = 1; i < size; i++)
        {
            startPointIndices[i] = i * pointsPerProcess + std::min(i, remainingPoints);
            localPointCounts[i] = pointsPerProcess + (i < remainingPoints ? 1 : 0);
            printf("[INFO] Process %d handles %d points\n",
                   i, localPointCounts[i]);
            MPI_Send(&localPointCounts[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        for (int i = 1; i < size; i++)
        {
            printf("[INFO] Sending clustersCount to process %d \n", i);
            MPI_Send(&clustersCount, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        const float *pointValues = h_kMeansData.getValues().data();
        const float *clusterValues = h_kMeansData.getClustersValues().data();

        for (int i = 1; i < size; i++)
        {
            printf("[INFO] Sending points to process %d \n", i);
            MPI_Send(pointValues + startPointIndices[i] * DIM, localPointCounts[i] * DIM, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            printf("[INFO] Sending points to process %d \n", i);
            MPI_Send(clusterValues, clustersCount * DIM, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }

        float *h_clusterValues = (float *)malloc(sizeof(float) * clustersCount * DIM);

        int *h_totalMemberships = (int *)malloc(sizeof(int) * pointsCount);
        if (h_totalMemberships == nullptr)
        {
            printf("[Rank %d][ERROR] Failed to allocate memory for h_memberships\n", 0);
        }

        int *d_memberships = nullptr;
        int *d_shouldContinue = nullptr;
        int *d_totalMemberships = nullptr;

        float *d_newClustersValues = nullptr;
        uint32_t *d_newClustersMembershipCounts = nullptr;

        float *d_pointsValues = nullptr;
        float *d_clustersValues = nullptr;
        size_t *d_clustersMembershipCount = nullptr;

        printf("local points count 0 %d \n", localPointCounts[0]);
        const uint32_t blocksCount = ceil(localPointCounts[0] * 1.0 / Consts::THREADS_PER_BLOCK);
        printf("blocks count rank 0: %d", blocksCount);
        CHECK_CUDA(cudaMalloc((void **)&d_memberships, sizeof(int) * localPointCounts[0]));
        CHECK_CUDA(cudaMalloc((void **)&d_shouldContinue, sizeof(int) * blocksCount));
        CHECK_CUDA(cudaMalloc((void **)&d_totalMemberships, sizeof(int) * pointsCount));
        CHECK_CUDA(cudaMalloc((void **)&d_newClustersMembershipCounts, sizeof(uint32_t) * clustersCount));
        CHECK_CUDA(cudaMalloc((void **)&d_clustersMembershipCount, sizeof(size_t) * clustersCount));

        CHECK_CUDA(cudaMalloc((void **)&d_pointsValues, sizeof(float) * pointsCount * DIM));
        CHECK_CUDA(cudaMemcpy(d_pointsValues, pointValues, DIM * pointsCount * sizeof(float), cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMalloc((void **)&d_clustersValues, sizeof(float) * clustersCount * DIM));

        CHECK_CUDA(cudaMalloc((void **)&d_newClustersValues, sizeof(float) * clustersCount * DIM));

        CHECK_CUDA(cudaMemcpy(d_clustersValues, clusterValues, DIM * clustersCount * sizeof(float), cudaMemcpyHostToDevice));

        int *shouldContinue = nullptr;
        shouldContinue = (int *)malloc(sizeof(int) * blocksCount);

        const uint32_t newClustersBlocksCount = ceil(pointsCount * 1.0 / Consts::THREADS_PER_BLOCK);
        const size_t newClustersSharedMemorySize = clustersCount * DIM * sizeof(float) + clustersCount * sizeof(uint32_t);

        // Start of the loop - wait for ranks to calculate membership and then calculate the new centroids
        for (int i = 0; i < Consts::MAX_ITERATION; i++)
        {
            // Do my own work
            int totalShouldContinue = newMembershipsAndShouldContinue(DIM, localPointCounts[0], d_pointsValues, clustersCount, d_clustersValues, d_memberships, d_shouldContinue, shouldContinue, blocksCount);

            // Receive if memberships changed
            for (int j = 1; j < size; j++)
            {
                int f;
                MPI_Recv(&f, 1, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                totalShouldContinue += f;
            }

            printf("[INFO] Iteration: %d total changed: %d \n", i, totalShouldContinue);
            // Send info if the process continues - membership change sum == 0 || is the last iteration , break otherwise
            if (totalShouldContinue == 0 || i == Consts::MAX_ITERATION - 1)
            {
                int l = 0;
                for (int j = 1; j < size; j++)
                {
                    MPI_Send(&l, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
                }
                break;
            }
            else
            {
                int l = 1;
                for (int j = 1; j < size; j++)
                {
                    MPI_Send(&l, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
                }
            }
            // If process continues receive new memberships

            CHECK_CUDA(cudaMemcpy(h_totalMemberships, d_memberships, localPointCounts[0] * sizeof(int), cudaMemcpyDeviceToHost));

            for (int j = 1; j < size; j++)
            {
                MPI_Recv(h_totalMemberships + startPointIndices[j], localPointCounts[j], MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            CHECK_CUDA(cudaMemcpy(d_totalMemberships, h_totalMemberships, pointsCount * sizeof(int), cudaMemcpyHostToDevice));

            // Calculate and send new centroids
            KMeansData::KMeansDataGPU d_data = {
                .pointsCount = pointsCount,
                .clustersCount = clustersCount,
                .DIM = DIM,
                .d_pointsValues = d_pointsValues,
                .d_clustersValues = d_clustersValues,
            };
            calculateNewClustersMPI<<<newClustersBlocksCount, Consts::THREADS_PER_BLOCK, newClustersSharedMemorySize>>>(clustersCount, DIM, d_pointsValues, pointsCount, d_newClustersValues, d_newClustersMembershipCounts, d_totalMemberships);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            // Accumulate clusters counts from each block
            accumulateNewClustersMemerships<<<1, d_data.clustersCount>>>(d_data, d_clustersMembershipCount, d_newClustersMembershipCounts, newClustersBlocksCount);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            // Calculate new clusters
            updateClusters<<<1, d_data.clustersCount * d_data.DIM>>>(d_data, d_clustersMembershipCount, d_newClustersValues, newClustersBlocksCount);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaMemcpy(h_clusterValues, d_clustersValues, clustersCount * DIM * sizeof(float), cudaMemcpyDeviceToHost));
            for (int j = 1; j < size; j++)
            {
                MPI_Send(h_clusterValues, clustersCount * DIM, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
            }
        }

        return Utils::ClusteringResult{
            .clustersValues = thrust::host_vector<float>(h_clusterValues, h_clusterValues + clustersCount * DIM),
            .membership = thrust::host_vector<size_t>(),
        };
    }

    void kMeansClusteringMPIAdditionalRank(int rank, int size)
    {
        // Load data from main rank
        printf("my rank is: %d\n", rank);
        int DIM;
        MPI_Recv(&DIM, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Rank %d][INFO] dim: %d\n", rank, DIM);
        int pointsCount;
        MPI_Recv(&pointsCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Rank %d][INFO] pointsCount: %d\n", rank, pointsCount);
        int clustersCount;
        MPI_Recv(&clustersCount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[Rank %d][INFO] clustersCount: %d\n", rank, clustersCount);

        float *h_pointsValues = (float *)malloc(sizeof(float) * DIM * pointsCount);
        if (h_pointsValues == NULL)
        {
            printf("[Rank %d][ERROR] Failed to allocate memory for points\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Recv(h_pointsValues, DIM * pointsCount, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        float *h_clustersValues = (float *)malloc(sizeof(float) * DIM * clustersCount);
        if (h_clustersValues == NULL)
        {
            printf("[Rank %d][ERROR] Failed to allocate memory for points\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("[INFO] Received points \n");
        int iterate = 1;

        // Setup
        const uint32_t blocksCount = ceil(pointsCount * 1.0 / Consts::THREADS_PER_BLOCK);
        printf("[RANK %d] blocksCOunt: %d\n", rank, blocksCount);

        int *h_memberships = (int *)malloc(sizeof(int) * pointsCount);
        if (h_memberships == nullptr)
        {
            printf("[Rank %d][ERROR] Failed to allocate memory for h_memberships\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int *d_memberships = nullptr;
        int *d_shouldContinue = nullptr;

        float *d_pointsValues = nullptr;
        float *d_clustersValues = nullptr;

        CHECK_CUDA(cudaMalloc((void **)&d_memberships, sizeof(int) * pointsCount));
        CHECK_CUDA(cudaMemset(d_memberships, -1, sizeof(int) * pointsCount));
        CHECK_CUDA(cudaMalloc((void **)&d_shouldContinue, sizeof(int) * blocksCount));

        CHECK_CUDA(cudaMalloc((void **)&d_pointsValues, sizeof(float) * pointsCount * DIM));
        CHECK_CUDA(cudaMemcpy(d_pointsValues, h_pointsValues, DIM * pointsCount * sizeof(float), cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMalloc((void **)&d_clustersValues, sizeof(float) * clustersCount * DIM));

        int *shouldContinue = nullptr;
        shouldContinue = (int *)malloc(sizeof(int) * blocksCount);
        if (shouldContinue == nullptr)
        {
            printf("[Rank %d][ERROR] Failed to allocate memory for shouldContinue\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        while (iterate)
        {
            // Receive new clusters values
            MPI_Recv(h_clustersValues, DIM * clustersCount, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CHECK_CUDA(cudaMemcpy(d_clustersValues, h_clustersValues, DIM * clustersCount * sizeof(float), cudaMemcpyHostToDevice));

            // Calculate new memberships and if should continue
            int totalShouldContinue = newMembershipsAndShouldContinue(DIM, pointsCount, d_pointsValues, clustersCount, d_clustersValues, d_memberships, d_shouldContinue, shouldContinue, blocksCount);

            // Send if any membership changed
            MPI_Send(&totalShouldContinue, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

            // Receive if should still iterate
            MPI_Recv(&iterate, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("[Rank %d][INFO] iterate: %d\n", rank, iterate);
            if (!iterate)
            {
                break;
            }

            // Send new memberships
            CHECK_CUDA(cudaMemcpy(h_memberships, d_memberships, pointsCount * sizeof(int), cudaMemcpyDeviceToHost));
            MPI_Send(h_memberships, pointsCount, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }

        // // Cleanup
        free(h_pointsValues);
        free(h_clustersValues);
        free(shouldContinue);
        cudaFree(d_memberships);
        cudaFree(d_pointsValues);
        cudaFree(d_clustersValues);
        cudaFree(d_shouldContinue);
    }

    // Returns number of points that changed their membership
    int newMembershipsAndShouldContinue(int DIM, int pointsCount, float *d_pointsValues, int clustersCount, float *d_clustersValues, int *d_memberships, int *d_shouldContinue, int *shouldContinue, int blocksCount)
    {
        // Calculate new memberships
        KMeansData::KMeansDataGPU d_data = {
            .pointsCount = pointsCount,
            .clustersCount = clustersCount,
            .DIM = DIM,
            .d_pointsValues = d_pointsValues,
            .d_clustersValues = d_clustersValues,
        };

        calculateMembershipMPI<<<blocksCount, Consts::THREADS_PER_BLOCK>>>(d_data, d_memberships, d_shouldContinue);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // Check if any membership changed
        CHECK_CUDA(cudaMemcpy(shouldContinue, d_shouldContinue, sizeof(int) * blocksCount, cudaMemcpyDeviceToHost));
        int totalShouldContinue = 0;
        for (size_t b = 0; b < blocksCount; b++)
        {
            totalShouldContinue += shouldContinue[b];
        }
        return totalShouldContinue;
    }
} // KMeansClusteringGPUSM
