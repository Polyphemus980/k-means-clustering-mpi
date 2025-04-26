#include <cstdio>
#include <cuda_runtime.h>
#include <exception>
#include <cstring>
#include "mpi.h"

#include "utils.cuh"
#include "file_io.cuh"
#include "k_means_data.cuh"
#include "k_means_clustering_gpu_sm.cuh"

// This function is an actual entry point
// We assume that at this point `inputFile` is changed in a way that
// Only N, DIM and K was read from it
void start(FILE *inputFile, size_t pointsCount, size_t clustersCount, Utils::ProgramArgs &programArgs, size_t DIM, int size)
{
    KMeansData::KMeansData h_kMeansData;
    switch (programArgs.inputFileType)
    {
    case Utils::InputFileType::TEXT:
        h_kMeansData = FileIO::LoadFromTextFile(inputFile, pointsCount, clustersCount, DIM);
        break;
    case Utils::InputFileType::BINARY:
        h_kMeansData = FileIO::LoadFromBinFile(inputFile, pointsCount, clustersCount, DIM);
        break;
    default:
        throw std::runtime_error("UNREACHABLE");
        break;
    }

    Utils::ClusteringResult result;
    switch (programArgs.algorithmMode)
    {
    case Utils::AlgorithmMode::GPU_FIRST:
        // result = KMeansClusteringGPUSM::kMeansClustering(h_kMeansData.transformToGPURepresentation());
        result = KMeansClusteringGPUSM::kMeansClusteringMPI(h_kMeansData, size);
        break;
    default:
        throw std::runtime_error("UNREACHABLE");
    }

    FileIO::SaveResultToTextFile(programArgs.outputFilePath, result, h_kMeansData.getClustersCount(), DIM);
}

void mainRank(int argc, char **argv, int size)
{
    if (argc != 5)
    {
        fprintf(stderr, "Invalid arguments count\n");
        Utils::usage(argv[0]);
    }

    Utils::InputFileType inputFileType{};
    if (strcmp(argv[1], "txt") == 0)
    {
        inputFileType = Utils::InputFileType::TEXT;
        printf("[INFO] Data format: text\n");
    }
    else if (strcmp(argv[1], "bin") == 0)
    {
        inputFileType = Utils::InputFileType::BINARY;
        printf("[INFO] Data format: binary\n");
    }
    else
    {
        fprintf(stderr, "Invalid file type\n");

        Utils::usage(argv[0]);
    }

    Utils::AlgorithmMode algorithmMode{};
    if (strcmp(argv[2], "cpu") == 0)
    {
        algorithmMode = Utils::AlgorithmMode::CPU;
        printf("[INFO] Algorithm: CPU\n");
    }
    else if (strcmp(argv[2], "gpu1") == 0)
    {
        algorithmMode = Utils::AlgorithmMode::GPU_FIRST;
        printf("[INFO] Algorithm: GPU1 (using custom kernels and shared memory)\n");
    }
    else if (strcmp(argv[2], "gpu2") == 0)
    {
        algorithmMode = Utils::AlgorithmMode::GPU_SECOND;
        printf("[INFO] Algorithm: GPU2 (using thrust)\n");
    }
    else
    {
        fprintf(stderr, "Invalid algorithm mode\n");
        Utils::usage(argv[0]);
    }

    Utils::ProgramArgs args{
        .algorithmMode = algorithmMode,
        .inputFileType = inputFileType,
        .inputFilePath = argv[3],
        .outputFilePath = argv[4]};

    Utils::Parameters parameters{};
    FILE *inputFile;

    switch (args.inputFileType)
    {
    case Utils::InputFileType::TEXT:
        inputFile = fopen(args.inputFilePath, "r");
        if (inputFile == nullptr)
        {
            throw std::runtime_error("Cannot open input file");
        }
        parameters = FileIO::LoadParamsFromTextFile(inputFile);
        break;
    case Utils::InputFileType::BINARY:
        inputFile = fopen(args.inputFilePath, "rb");
        if (inputFile == nullptr)
        {
            throw std::runtime_error("Cannot open input file");
        }
        parameters = FileIO::LoadParamsFromBinFile(inputFile);
        break;
    default:
        throw std::runtime_error("UNREACHABLE");
        break;
    }

    printf("[INFO] Points: %zu, clusters: %zu, dimensions: %zu\n", parameters.pointsCount, parameters.clustersCount, parameters.dimensions);

    start(inputFile, parameters.pointsCount, parameters.clustersCount, args, parameters.dimensions, size);
}

int main(int argc, char **argv)
{
    // Initialize MPI environment
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("[INFO] Process %d of %d started\n", rank, size);

    if (rank == 0)
    {
        mainRank(argc, argv, size);
    }
    else
    {
        KMeansClusteringGPUSM::kMeansClusteringMPIAdditionalRank(rank, size);
    }

    return 0;
}
