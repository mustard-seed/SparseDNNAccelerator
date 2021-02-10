#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string> //for std::to_string
#include <unistd.h> //usleep
#include <random>
#include <bitset> //Print friendly binary number
#include <memory> //smart pointers
#include <chrono>

#include "gtest/gtest.h"
#include "boost/align/aligned_allocator.hpp"

#include "aocl_utils_cpp.hpp"
#include "params.hpp"
#include "spw_pe_test_types.hpp"
#include "device_structures.hpp"

#define EMULATE
#define MAX_BUFFER_SIZE_BYTES 65536
#define SEED 27
#if defined(EMULATE)
#define AOCX_FILE_NAME "c5_mac8bitx4_c_model.aocx"
#else
#define AOCX_FILE_NAME "spw_pe_test_harness.aocx"
#endif

typedef
std::vector<t_test_weight_host_block, boost::alignment::aligned_allocator<t_test_weight_host_block, aocl_utils_cpp::AOCL_ALIGNMENT>>
t_aligned_test_weight_host_block;

typedef
std::vector<t_test_activation_host_block, boost::alignment::aligned_allocator<t_test_activation_host_block, aocl_utils_cpp::AOCL_ALIGNMENT>>
t_aligned_test_activation_host_block;

static bool absCompare(signed char a, signed char b)
{
    return (std::abs(a) < std::abs(b));
}

/*!
 * \brief getLInfNorm
 * \details Compute the L_inf norm (max absolute value) of a given vector
 * \param _group
 * \return The L_inf norm of the vector
 */
signed char getLInfNorm(
        const std::vector<signed char>& _group
        );

/*!
 * \brief compressSpWVector
 * \details Compress a flattened char vector using balanced-block sparsity
 * The length of the vector must be divisible by
 * (_numPruneRangeinParallel * _pruneRangeSizeInCluster * _clusterSize)
 * \param _inputVector
 * \param _clusterSize
 * \param _pruneRangeSizeInCluster
 * \param _numPruneRangeInParallel
 * \param _numNZClustersPerPruneRange
 * \return Vector of the compressed values
 */
t_aligned_test_weight_host_block compressSpWVector(
            std::vector <signed char> & _inputVector,
            int _clusterSize,
            int _pruneRangeSizeInCluster,
            int _numPruneRangeInParallel,
            int _numNZClustersPerPruneRange
        );

std::vector <signed char> generateSpWVector(
            int _clusterSize,
            int _pruneRangeSizeInCluster,
            int _numPruneRange,
            int _numNZClustersPerPruneRange
        );

std::vector <signed char> generateActivationVector(
            int _clusterSize,
            int _pruneRangeSizeInCluster,
            int _numPruneRange
        );

t_aligned_test_activation_host_block packActivationVector(
            std::vector <signed char> & _vecActivation
        );

class spwTestFixture : public ::testing::Test
{
protected:
    std::string binaryFile;
    cl::Program program;
    cl::Platform clPlatform;
    cl::Context clContext;
    cl::Device clDevice;

    cl::CommandQueue clCQActivationFeeder;
    cl::CommandQueue clCQActivationDrainer;
    cl::CommandQueue clCQWeightFeeder;
    cl::CommandQueue clCQWeightDrainer;
    cl::CommandQueue clCQOutputDrainer;

    cl::Buffer bufferActivationInput;
    cl::Buffer bufferActivationOutput;
    cl::Buffer bufferWeightInput;
    cl::Buffer bufferWeightOutput;
    cl::Buffer bufferBiasInput;
    cl::Buffer bufferBiasOutput;
    cl::Buffer bufferPEOutput;

    cl::Kernel kernelActivationFeeder;
    cl::Kernel kernelActivationDrainer;
    cl::Kernel kernelFilterFeeder;
    cl::Kernel kernelFilterDrainer;
    cl::Kernel kernelResultDrainer;

    void SetUp() override;

    /*!
     * \brief launch
     * \details Wrapper of the tests
     * \param _numCompressionWindows.
     *  Controls how long the test vector is.
     *  Length of the test vector =
     *  _numCompressionWindows * PE_SIMD_SIZE * PRUNE_RANGE_IN_CLUSTER * CLUSTER_SIZE
     * \param _numNZClustersInRange
     *  Controls the sparsity. Number of non-zero clusters in each pruning range.
     * \param _bias The bias
     * \param _numRepeat Number of time to repeat a test
     */
    void launch (
            int _numCompressionWindows,
            int _numNZClustersInRange,
            t_bias _bias,
            int _numRepeat
            );
};


TEST_F(spwTestFixture, test0)
{
    int numCompressionWindows = 4;
    int numNZClustersInRange = 1;
    t_bias bias = 0x2;
    int numRepeat = 1;
//    int numCompressionWindows = 1;
//    int numNZClustersInRange = 1;
//    t_bias bias = 0x02;
//    int numRepeat = 1;

    launch(
            numCompressionWindows,
            numNZClustersInRange,
            bias,
            numRepeat
          );
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}

signed char getLInfNorm(
        const std::vector<signed char>& _group
        )
{
    auto iter = std::max_element(_group.begin(), _group.end(), absCompare);
    return std::abs(*iter);
}

t_aligned_test_weight_host_block compressSpWVector(
            std::vector <signed char> & _inputVector,
            int _clusterSize,
            int _pruneRangeSizeInCluster,
            int _numPruneRangeInParallel,
            int _numNZClustersPerPruneRange
        )
{
    int sizeCompressionWindow = _clusterSize * _pruneRangeSizeInCluster * _numPruneRangeInParallel;
    if (_inputVector.size() % sizeCompressionWindow != 0)
    {
        std::perror("[compressSpWVector] Length of the uncompressed vector is not divisible by the compression window size.");
        throw;
    }

    int numCompressionWindow = _inputVector.size() / sizeCompressionWindow;
    int numWeightBlock = numCompressionWindow * _numNZClustersPerPruneRange;
    t_aligned_test_weight_host_block compressedVector;
    compressedVector.resize( numWeightBlock );

    for (int iCompressionWindow=0; iCompressionWindow<numCompressionWindow; iCompressionWindow++)
    {
        for (int iPruneRange=0; iPruneRange<_numPruneRangeInParallel; iPruneRange++)
        {
            /*
             * Collect the L_inf norms of the clusters inside the same pruning range
             */
            std::vector<signed char> lInfNorms(_pruneRangeSizeInCluster, 0);
            for (int iCluster=0; iCluster<_pruneRangeSizeInCluster; iCluster++)
            {
                auto begin = _inputVector.begin()
                        + iCompressionWindow * sizeCompressionWindow
                        + iPruneRange * _pruneRangeSizeInCluster * _clusterSize
                        + iCluster * _clusterSize;
                auto end = begin + _clusterSize;
                std::vector<signed char> vecCluster(begin, end);
                signed char norm = getLInfNorm(vecCluster);
                lInfNorms.at(iCluster) = norm;
            }

            /*
             * Rank the cluster indices according to the Linf norms
            */
            //Generate a list of indices
            std::vector<int> indices(_pruneRangeSizeInCluster, 0);
            //Use iota to populate the indices, starting from 0
            std::iota(indices.begin(), indices.end(), 0);
            //Use the lambda expression to sort the indices based on the Linf norm non-ascending order
            std::stable_sort(indices.begin(), indices.end(),
                             [&lInfNorms](int i1, int i2) {return lInfNorms.at(i1) > lInfNorms.at(i2);});

            /*
             * Place the values and masks of the top ranked clusters into the compressed blocks
            */
            for (int iTopCluster=0; iTopCluster<_numNZClustersPerPruneRange; iTopCluster++)
            {
                int idxWeightBlock = iCompressionWindow * _numNZClustersPerPruneRange + iTopCluster;
                int idxClusterInDenseVector = indices.at(iTopCluster);
                //Copy the cluster values inside each cluster
                for (int v=0; v<_clusterSize; v++)
                {
                    int idxDenseVector =
                        iCompressionWindow * sizeCompressionWindow
                            + iPruneRange * _pruneRangeSizeInCluster * _clusterSize
                            + idxClusterInDenseVector * _clusterSize
                            + v;
                    compressedVector.at(idxWeightBlock).values[iPruneRange*_clusterSize + v]
                            = _inputVector.at(idxDenseVector);
                }

                //Set the index of the activation cluster that preserved weight cluster should occupy
                int idxPositionArray = iPruneRange / 2;
                if (iPruneRange % 2 == 0)
                {
                    //Clear the lower 4 bits
                    compressedVector.at(idxWeightBlock).indices[idxPositionArray]
                            &= 0x0F0;
                    //Setthe lower 4 bits
                    compressedVector.at(idxWeightBlock).indices[idxPositionArray]
                            |= idxClusterInDenseVector & CHAR_TO_SPW_INDEX_MASK;
                }
                else
                {
                    //Clear the higher 4 bits
                    compressedVector.at(idxWeightBlock).indices[idxPositionArray]
                            &= 0x00F;
                    //Set the higher  4 bits
                    compressedVector.at(idxWeightBlock).indices[idxPositionArray]
                            |= (idxClusterInDenseVector & CHAR_TO_SPW_INDEX_MASK) << 0x04;
                }
            } //for. iTopCluster.  Place the values and masks of the top ranked clusters into the compressed blocks
        } //for. iPruneRange.
    } //for. iCompressionWindow

    return compressedVector;
}

std::vector <signed char> generateSpWVector(
            int _clusterSize,
            int _pruneRangeSizeInCluster,
            int _numPruneRange,
            int _numNZClustersPerPruneRange
        )
{
    if (_numNZClustersPerPruneRange > _pruneRangeSizeInCluster)
    {
        std::perror("[generateSpWVector] _numNZClustersPerPruneRange should not exceed _pruneRangeSizeInCluster.");
        throw;
    }
    int vecLength = _clusterSize * _pruneRangeSizeInCluster * _numPruneRange;
    std::vector<signed char> vecSpW(vecLength, 0x0);
    std::uniform_int_distribution<int> distribution(-3,3);
    for (int iPruneRange=0; iPruneRange<_numPruneRange; iPruneRange++)
    {
        std::mt19937 ran(SEED);
        //Randomly select up to _numNZClustersPerPruneRange clusters to set to non-zero
        std::vector<int> idxClusters (_pruneRangeSizeInCluster, 0x0);
        std::iota(idxClusters.begin(), idxClusters.end(), 0);
        std::shuffle(idxClusters.begin(), idxClusters.end(), ran);
        for (int i=0; i<_numNZClustersPerPruneRange; i++)
        {
            int idx = idxClusters.at(i);
            for (int c=0; c<_clusterSize; c++)
            {
                int vecIdx = iPruneRange*_pruneRangeSizeInCluster*_clusterSize + idx*_clusterSize + c;
                vecSpW.at(vecIdx) = (signed char) distribution(ran);
            }
        }
    }
    return vecSpW;
}

std::vector <signed char> generateActivationVector(
            int _clusterSize,
            int _pruneRangeSizeInCluster,
            int _numPruneRange
        )
{
    int vecLength = _clusterSize * _pruneRangeSizeInCluster * _numPruneRange;
    std::vector<signed char> vecActivation(vecLength, 0x0);
    std::mt19937 ran(SEED);
    std::uniform_int_distribution<int> distribution(-2,2);
    for (int i=0; i<vecLength; i++)
    {
        vecActivation.at(i) = (signed char) distribution(ran);

    }
    return vecActivation;
}

t_aligned_test_activation_host_block packActivationVector(
            std::vector <signed char> & _vecActivation
        )
{
    int denseVecLength = _vecActivation.size();
    int packedLength = 1 + (denseVecLength - 1) / (PE_SIMD_SIZE * PRUNE_RANGE_IN_CLUSTER * CLUSTER_SIZE);
    int windowSize = PE_SIMD_SIZE * PRUNE_RANGE_IN_CLUSTER * CLUSTER_SIZE;
    t_aligned_test_activation_host_block vecActivation;
    vecActivation.resize(packedLength);

    for (unsigned int i=0; i<vecActivation.size(); i++)
    {
        for (unsigned int v=0; v<windowSize; v++)
        {
            int denseIdx = i*windowSize + v;
            if (denseIdx < denseVecLength)
            {
                vecActivation.at(i).values[v] = _vecActivation.at(denseIdx);
            }
            else
            {
                vecActivation.at(i).values[v] = 0;
            }
        }
    }

    return vecActivation;
}

void spwTestFixture::SetUp()
{
    binaryFile = AOCX_FILE_NAME;
#if defined(EMULATE)
    clPlatform = aocl_utils_cpp::findPlatform("Intel(R) FPGA Emulation Platform for OpenCL(TM)");
#else
    clPlatform = aocl_utils_cpp::findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
#endif
    cl_int status = CL_SUCCESS;
    int step = 0;

    std::vector<cl::Device> devices;
    status = clPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    aocl_utils_cpp::checkError(status, "Failed to query the devices");
    clDevice = devices[0];
    clContext = cl::Context({devices[0]}
                            ,NULL
                            ,&aocl_utils_cpp::oclContextCallback
                            ,NULL
                            ,&status);
    aocl_utils_cpp::checkError(status, "Failed to create context");

    //Parse the binary and invoke the kernel
    program = aocl_utils_cpp::createProgramFromBinary(
                clContext,
                binaryFile.c_str(),
                {clDevice}
                );
    status = program.build({clDevice});
    aocl_utils_cpp::checkError(status, "Failed to build program");

    //Initialze the kernels
    {
        typedef struct {
            cl::Kernel& kernel;
            std::string name;
        } t_kernel_name;

        std::vector<t_kernel_name> vecKernelNames {
          {.kernel=kernelActivationFeeder, .name="kernelActivationFeeder"},
          {.kernel=kernelActivationDrainer, .name="kernelActivationDrainer"},
          {.kernel=kernelFilterFeeder, .name="kernelFilterFeeder"},
          {.kernel=kernelFilterDrainer, .name="kernelFilterDrainer"},
          {.kernel=kernelResultDrainer, .name="kernelResultDrainer"}
        };

        for (auto & inst : vecKernelNames)
        {
            inst.kernel = cl::Kernel(program, inst.name.c_str(), &status);
            aocl_utils_cpp::checkError(status, ("Failed to create the "+inst.name+"!").c_str());
        }
    }

    //Allocate the buffers
    {
        typedef struct {
            cl::Buffer& buffer;
            cl_mem_flags memFlags;
            size_t size;
            std::string name;
        } t_buffer;

        std::vector<t_buffer> vecBuffers {
          {.buffer=bufferActivationInput, .memFlags=CL_MEM_READ_ONLY, .size=MAX_BUFFER_SIZE_BYTES,  "activation input buffer"},
          {.buffer=bufferActivationOutput, .memFlags=CL_MEM_WRITE_ONLY, .size=MAX_BUFFER_SIZE_BYTES,  "activation output buffer"},
          {.buffer=bufferWeightInput, .memFlags=CL_MEM_READ_ONLY, .size=MAX_BUFFER_SIZE_BYTES,  "weight input buffer"},
          {.buffer=bufferWeightOutput, .memFlags=CL_MEM_WRITE_ONLY, .size=MAX_BUFFER_SIZE_BYTES,  "weight output buffer"},
          {.buffer=bufferBiasInput, .memFlags=CL_MEM_READ_ONLY, .size=4,  "bias input buffer"},
          {.buffer=bufferBiasOutput, .memFlags=CL_MEM_WRITE_ONLY, .size=4,  "bias output buffer"},
          {.buffer=bufferPEOutput, .memFlags=CL_MEM_WRITE_ONLY, .size=MAX_BUFFER_SIZE_BYTES,  "pe output buffer"}
        };

        for (auto & inst : vecBuffers)
        {
            inst.buffer = cl::Buffer (
                            clContext,
                            inst.memFlags,
                            inst.size,
                            NULL,
                            &status
                        );
            aocl_utils_cpp::checkError(status, ("Failed to create the "+inst.name+"!").c_str());
        }
    }

    //Allocate the command queues
    {
        typedef struct {
            cl::CommandQueue& cq;
            std::string name;
        } t_cq;

        std::vector<t_cq> veCQs {
          {.cq=clCQActivationFeeder, .name="CQActivationFeeder"},
          {.cq=clCQActivationDrainer, .name="CQActivationDrainer"},
          {.cq=clCQWeightFeeder, .name="CQWeightFeeder"},
          {.cq=clCQWeightDrainer, .name="CQWeightDrainer"},
          {.cq=clCQOutputDrainer, .name="CQResultDrainer"}
        };

        for (auto & inst : veCQs)
        {
            inst.cq = cl::CommandQueue(clContext, clDevice, CL_QUEUE_PROFILING_ENABLE, &status);
            aocl_utils_cpp::checkError(status, ("Failed to create the "+inst.name+"!").c_str());
        }
    }
}

void spwTestFixture::launch(
        int _numCompressionWindows,
        int _numNZClustersInRange,
        t_bias _bias,
        int _numRepeat)
{
    //TODO: Print a summary about the test
    int clusterSize = CLUSTER_SIZE;
    int numParallelPruneRanges = PE_SIMD_SIZE;
    int numClustersInPruneRange = PRUNE_RANGE_IN_CLUSTER;
    int vectorLength = CLUSTER_SIZE * PE_SIMD_SIZE * PRUNE_RANGE_IN_CLUSTER * _numCompressionWindows;
    int numPruneRanges = _numCompressionWindows * PE_SIMD_SIZE;
    std::cout <<"Launching a SpW vector test."<<std::endl;
    std::cout <<"Cluster size: "<<clusterSize<<std::endl;
    std::cout <<"Number of prune ranges processed in parallel: "<<numParallelPruneRanges<<std::endl;
    std::cout <<"Size of a prune range in clusters: "<<numClustersInPruneRange<<std::endl;
    std::cout <<"Number of NZ clusters per pruning range: "<<_numNZClustersInRange<<std::endl;
    std::cout <<"Number of compression windows: "<<_numCompressionWindows<<std::endl;
    std::cout <<"Vector length: "<<vectorLength<<std::endl;
    std::cout <<"Num. of repeats: "<<_numRepeat<<std::endl;

    cl_int status = CL_SUCCESS;

    int step = 1;
    std::cout <<"Step "<<step++<<": Generate the SpW vector."<<std::endl;
    std::vector<signed char> vecSpW = generateSpWVector(
                    clusterSize,
                    numClustersInPruneRange,
                    numPruneRanges,
                    _numNZClustersInRange
                );

    std::cout <<"Step "<<step++<<": Compress the SpW vector."<<std::endl;
    t_aligned_test_weight_host_block vecCompressedSpW =
            compressSpWVector(
                    vecSpW,
                    clusterSize,
                    numClustersInPruneRange,
                    numParallelPruneRanges,
                    _numNZClustersInRange
                );

    std::cout <<"Step "<<step++<<": Generate the activation vector."<<std::endl;
    std::vector<signed char> vecActivations =
            generateActivationVector(
                    clusterSize,
                    numClustersInPruneRange,
                    numPruneRanges
                );

    std::cout <<"Step "<<step++<<": Copy the activation vector into the device layout."<<std::endl;
    t_aligned_test_activation_host_block vecPackedActivation =
                packActivationVector(
                    vecActivations
                );

    std::cout <<"Step "<<step++<<": Fill the input buffers."<<std::endl;
    //Transfer the input activations
    {
        auto numElements = vecPackedActivation.size();
        auto sizePerElement = sizeof(typeof(vecPackedActivation.at(0)));
        auto sizeOfVector = numElements * sizePerElement;
        status = clCQActivationFeeder.enqueueWriteBuffer(
                        bufferActivationInput,
                        CL_TRUE, //blocking write
                        0, //offset
                        sizeOfVector, //size of the transfer in bytes
                        vecPackedActivation.data(), //data pointers
                        NULL
                    );
        aocl_utils_cpp::checkError(status, "Failed to write bufferActivationInput.");
    }

    //Transfer the input weights
    {
        auto numElements = vecCompressedSpW.size();
        auto sizePerElement = sizeof(typeof(vecCompressedSpW.at(0)));
        auto sizeOfVector = numElements * sizePerElement;
        status = clCQWeightFeeder.enqueueWriteBuffer(
                        bufferWeightInput,
                        CL_TRUE, //blocking write
                        0, //offset
                        sizeOfVector, //size of the transfer in bytes
                        vecCompressedSpW.data(), //data pointers
                        NULL
                    );
        aocl_utils_cpp::checkError(status, "Failed to write bufferWeightInput.");
    }

    //Transfer the bias
    {
        status = clCQWeightFeeder.enqueueWriteBuffer(
                        bufferBiasInput,
                        CL_TRUE, //blocking write
                        0, //offset
                        sizeof(t_bias), //size of the transfer in bytes
                        &_bias, //data pointers
                        NULL
                    );
        aocl_utils_cpp::checkError(status, "Failed to write bufferBiasInput.");
    }

    std::cout <<"Step "<<step++<<": Set the kernel arguments."<<std::endl;
    kernelActivationFeeder.setArg(0, bufferActivationInput);
    kernelActivationFeeder.setArg(1, (cl_uint) vecPackedActivation.size());

    kernelActivationDrainer.setArg(0, bufferActivationOutput);
    kernelActivationDrainer.setArg(1, (cl_uint) vecPackedActivation.size());

    kernelFilterFeeder.setArg(0, bufferWeightInput);
    kernelFilterFeeder.setArg(1, bufferBiasInput);
    kernelFilterFeeder.setArg(2, (cl_uint) vecCompressedSpW.size());
    kernelFilterFeeder.setArg(3, (cl_uint) _numNZClustersInRange);

    kernelFilterDrainer.setArg(0, bufferWeightOutput);
    kernelFilterDrainer.setArg(1, bufferBiasOutput);
    kernelFilterDrainer.setArg(2, (cl_uint) vecCompressedSpW.size());

    kernelResultDrainer.setArg(0, bufferPEOutput);

    std::cout <<"Step "<<step++<<": Launch the kernels."<<std::endl;
    for (int n=0; n<_numRepeat; n++)
    {
        status = clCQActivationFeeder.enqueueTask(kernelActivationFeeder);
        aocl_utils_cpp::checkError(status, "Failed to launch kernelActivationFeeder");
        status = clCQActivationDrainer.enqueueTask(kernelActivationDrainer);
        aocl_utils_cpp::checkError(status, "Failed to launch kernelActivationDrainer");
        status = clCQWeightFeeder.enqueueTask(kernelFilterFeeder);
        aocl_utils_cpp::checkError(status, "Failed to launch kernelFilterFeeder");
        status = clCQWeightDrainer.enqueueTask(kernelFilterDrainer);
        aocl_utils_cpp::checkError(status, "Failed to launch kernelFilterDrainer");
        status = clCQOutputDrainer.enqueueTask(kernelResultDrainer);
        aocl_utils_cpp::checkError(status, "Failed to launch kernelResultDrainer");

        clCQOutputDrainer.finish();
        clCQActivationDrainer.finish();
        clCQWeightDrainer.finish();
    }

    std::cout <<"Step "<<step++<<": Drain the output buffers."<<std::endl;
    t_aligned_test_activation_host_block vecCompressedActivationOutput;
    vecCompressedActivationOutput.resize(vecPackedActivation.size());
    //Drain the activation
    status = clCQActivationDrainer.enqueueReadBuffer(
                    bufferActivationOutput,
                    CL_TRUE,
                    0,
                    sizeof(typeof(vecCompressedActivationOutput.at(0))) * vecCompressedActivationOutput.size(),
                    vecCompressedActivationOutput.data()
                );
    aocl_utils_cpp::checkError(status, "Failed to read the activation buffer");

    t_aligned_test_weight_host_block vecCompressedSpWOutput;
    vecCompressedSpWOutput.resize(vecCompressedSpW.size());

    //Drain the weights
    status = clCQWeightDrainer.enqueueReadBuffer(
                    bufferWeightOutput,
                    CL_TRUE,
                    0,
                    sizeof(typeof(vecCompressedSpWOutput.at(0))) * vecCompressedSpWOutput.size(),
                    vecCompressedSpWOutput.data()
                );
    aocl_utils_cpp::checkError(status, "Failed to read the weight buffer");

    //Drain the bias
    t_bias biasOutput;
    status = clCQWeightDrainer.enqueueReadBuffer(
                bufferBiasOutput,
                CL_TRUE,
                0,
                sizeof(t_bias),
                &biasOutput
            );
    aocl_utils_cpp::checkError(status, "Failed to read the bias buffer");


    //Drain the wide outputs
    t_wide_psum resultsOutput;
    status = clCQOutputDrainer.enqueueReadBuffer(
                bufferPEOutput,
                CL_TRUE,
                0,
                sizeof(t_wide_psum),
                &resultsOutput
            );
    aocl_utils_cpp::checkError(status, "Failed to read the psum output");

    std::cout <<"Step "<<step++<<": Compare the PE output."<<std::endl;
    //Calculate the expected output
    unsigned int expectedPsum = (unsigned int) _bias;
    for (int i=0; i<vecSpW.size(); i++)
    {
        signed char weight = vecSpW.at(i);
        signed char activation = vecActivations.at(i);
        expectedPsum += ((signed int) weight) * ((signed int) activation);
    }
    //Compare with the actual output
    for (int i=0; i<PE_ROWS_PER_GROUP; i++)
    {
        signed actualPSum = resultsOutput.psums[i];
        EXPECT_TRUE(actualPSum == expectedPsum)
                <<"Actual output at row "<<i<<" disagrees with the expected output."<<std::endl
                <<"Actual output: "<<actualPSum<<". Expected output: "<<expectedPsum<<std::endl;
    }

    std::cout <<"Step "<<step++<<": Compare the compressed SpW blocks."<<std::endl;
    for (int i=0; i<vecCompressedSpW.size(); i++)
    {
        t_test_weight_host_block expectedWeightBlock = vecCompressedSpW.at(i);
        t_test_weight_host_block actualWeightBlock = vecCompressedSpWOutput.at(i);
        //Compare the values
        for (int j=0; j< (PE_SIMD_SIZE * CLUSTER_SIZE); j++)
        {
            signed char expectedValue = expectedWeightBlock.values[j];
            signed char actualValue = actualWeightBlock.values[j];
            EXPECT_TRUE(expectedValue == actualValue)
                    <<"Weight disagreement at [block, position]: ["<<i<<" "<<j<<"]."<<std::endl
                   <<"Actual value: "<<actualValue<<". Expected value: "<<expectedValue<<std::endl;
        }

        //Compare the indices
        for (int j=0; j< INDEX_CHAR_ARRAY_SIZE; j++)
        {
            unsigned char expectedBitmask = expectedWeightBlock.indices[j];
            unsigned char actualBitmask = actualWeightBlock.indices[j];
            EXPECT_TRUE(expectedBitmask == actualBitmask)
                    <<"Weight bitmask disagreement at [block, position]: ["<<i<<" "<<j<<"]."<<std::endl
                   <<"Actual bitmask: "<<actualBitmask<<". Expected bitmask: "<<expectedBitmask<<std::endl;
        }
    }

    std::cout <<"Step "<<step++<<": Compare the activation blocks."<<std::endl;
    for (int i=0; i<vecPackedActivation.size(); i++)
    {
        t_test_activation_host_block expectedActivationBlock = vecPackedActivation.at(i);
        t_test_activation_host_block actualActivationBlock = vecCompressedActivationOutput.at(i);

        //Compare the values in the blocks
        for (int j=0; j< (PE_SIMD_SIZE * PRUNE_RANGE_IN_CLUSTER * CLUSTER_SIZE); j++)
        {
            signed char expectedValue = expectedActivationBlock.values[j];
            signed char actualValue = actualActivationBlock.values[j];
            EXPECT_TRUE(expectedValue == actualValue)
                    <<"Activation disagreement at [block, position]: ["<<i<<" "<<j<<"]."<<std::endl
                   <<"Actual value: "<<actualValue<<". Expected value: "<<expectedValue<<std::endl;
        }
    }
}
