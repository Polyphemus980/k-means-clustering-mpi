#include <cstdio>
#include <cuda_runtime.h>
#include <exception>
#include <cstring>

#include "utils.cuh"
#include "file_io.cuh"
#include "k_means_data.cuh"
#include "k_means_clustering_cpu.cuh"
#include "k_means_clustering_gpu_sm.cuh"
#include "k_means_clustering_gpu_thrust.cuh"

// This function is an actual entry point
// We assume that at this point `inputFile` is changed in a way that
// Only N, DIM and K was read from it
template <size_t DIM>
void start(FILE *inputFile, size_t pointsCount, size_t clustersCount, Utils::ProgramArgs &programArgs)
{
    KMeansData::KMeansData<DIM> h_kMeansData;
    switch (programArgs.inputFileType)
    {
    case Utils::InputFileType::TEXT:
        h_kMeansData = FileIO::LoadFromTextFile<DIM>(inputFile, pointsCount, clustersCount);
        break;
    case Utils::InputFileType::BINARY:
        h_kMeansData = FileIO::LoadFromBinFile<DIM>(inputFile, pointsCount, clustersCount);
        break;
    default:
        throw std::runtime_error("UNREACHABLE");
        break;
    }

    Utils::ClusteringResult result;
    switch (programArgs.algorithmMode)
    {
    case Utils::AlgorithmMode::CPU:
        result = KMeansClusteringCPU::kMeanClustering<DIM>(h_kMeansData);
        break;
    case Utils::AlgorithmMode::GPU_FIRST:
        result = KMeansClusteringGPUSM::kMeansClustering<DIM>(h_kMeansData.transformToGPURepresentation());
        break;
    case Utils::AlgorithmMode::GPU_SECOND:
        result = KMeansClusteringGPUThrust::kMeansClustering<DIM>(h_kMeansData.transformToGPUThrustRepresentation());
        break;
    default:
        throw std::runtime_error("UNREACHABLE");
    }

    FileIO::SaveResultToTextFile<DIM>(programArgs.outputFilePath, result, h_kMeansData.getClustersCount());
}

int main(int argc, char **argv)
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

    switch (parameters.dimensions)
    {
    case 1:
        start<1>(inputFile, parameters.pointsCount, parameters.clustersCount, args);
        break;
    case 2:
        start<2>(inputFile, parameters.pointsCount, parameters.clustersCount, args);
        break;
    case 3:
        start<3>(inputFile, parameters.pointsCount, parameters.clustersCount, args);
        break;
    case 4:
        start<4>(inputFile, parameters.pointsCount, parameters.clustersCount, args);
        break;
    case 5:
        start<5>(inputFile, parameters.pointsCount, parameters.clustersCount, args);
        break;
    case 6:
        start<6>(inputFile, parameters.pointsCount, parameters.clustersCount, args);
        break;
    case 7:
        start<7>(inputFile, parameters.pointsCount, parameters.clustersCount, args);
        break;
    case 8:
        start<8>(inputFile, parameters.pointsCount, parameters.clustersCount, args);
        break;
    case 9:
        start<9>(inputFile, parameters.pointsCount, parameters.clustersCount, args);
        break;
    case 10:
        start<10>(inputFile, parameters.pointsCount, parameters.clustersCount, args);
        break;
    case 11:
        start<11>(inputFile, parameters.pointsCount, parameters.clustersCount, args);
        break;
    case 12:
        start<12>(inputFile, parameters.pointsCount, parameters.clustersCount, args);
        break;
    case 13:
        start<13>(inputFile, parameters.pointsCount, parameters.clustersCount, args);
        break;
    case 14:
        start<14>(inputFile, parameters.pointsCount, parameters.clustersCount, args);
        break;
    case 15:
        start<15>(inputFile, parameters.pointsCount, parameters.clustersCount, args);
        break;
    case 16:
        start<16>(inputFile, parameters.pointsCount, parameters.clustersCount, args);
        break;
    case 17:
        start<17>(inputFile, parameters.pointsCount, parameters.clustersCount, args);
        break;
    case 18:
        start<18>(inputFile, parameters.pointsCount, parameters.clustersCount, args);
        break;
    case 19:
        start<19>(inputFile, parameters.pointsCount, parameters.clustersCount, args);
        break;
    case 20:
        start<20>(inputFile, parameters.pointsCount, parameters.clustersCount, args);
        break;
    default:
        throw std::runtime_error("Unsupported dimension");
        break;
    }

    return 0;
}
