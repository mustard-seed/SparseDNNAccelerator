#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "aocl_utils_cpp.hpp"
#include "params.hpp"
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string> //for std::to_string
#include <unistd.h> //usleep
#include <random>
#include <bitset> //Print friendly binary number

#include "gtest/gtest.h"
#include "boost/align/aligned_allocator.hpp"

#include "floatFixedPointConversion.hpp"
#include "tensorCompression.hpp"
#include "vectorType.hpp"

#define W0 0.33333333
#define W1 0.33333333
#define W2 0.33333333
#define K_SIZE 3
#define MAX_DRAM_BYTE_WEIGHT_INPUT 134217728
#define MAX_DRAM_BYTE_WEIGHT_OUTPUT 32768
#define MAX_DRAM_BYTE_STREAMER_ADDRESS 32768

typedef
std::vector<cl_float, boost::alignment::aligned_allocator<cl_float, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_ushort>
t_aligned_float_vector;

typedef struct {
    unsigned int sequenceId;
    unsigned char targetFilterRow;
} t_filter_coordinates;


class testFixture : public ::testing::Test {
protected:
    std::string binaryFile;
    cl::Program program;
    cl::Platform clPlatform;
    cl::Context clContext;
    cl::Device clDevice;

    cl::CommandQueue clCQFilterWriter;
    cl::CommandQueue clCQTensorChecker;

    cl::Kernel kernelFilterWriter;
    cl::Kernel kernelTensorChecker;

    cl::Buffer bufferWeightWideInput;
    cl::Buffer bufferTransferAddressInput;
    cl::Buffer bufferWeightNarrowOutput;

    //t_aligned_dram_block_vector inputWeightWideVector;
    //t_aligned_streamblock_address_vector inputTransferAddressVector;
    //t_aligned_transfer_block_vector outputTransferBlockVector;


    void SetUp() override
    {
        //std::cout<<"Type in the aocx file name: ";
        //std::cin >> binaryFile;
#ifdef C5SOC
        binaryFile = "simpleWeightMemory_aoc_release_hw.aocx";
#else
        binaryFile = "simpleWeightMemory_aoc_emulation.aocx";
#endif

        //Setup and platform and the context
        cl_int status = CL_SUCCESS;
        clPlatform = aocl_utils_cpp::findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
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

        kernelFilterWriter = cl::Kernel(program, "kernelFilterWriter", &status);
        aocl_utils_cpp::checkError(status, "Failed to create the filter writer kernel!");

        kernelTensorChecker = cl::Kernel(program, "kernelTensorChecker", &status);
        aocl_utils_cpp::checkError(status, "Failed to created the kernelTensorChecker kernel!");

        clCQFilterWriter = cl::CommandQueue(
                    clContext,
                    clDevice,
                    CL_QUEUE_PROFILING_ENABLE,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the command queue clCQFilterWriter!");

        clCQTensorChecker = cl::CommandQueue(
                    clContext,
                    clDevice,
                    CL_QUEUE_PROFILING_ENABLE,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the command queue clCQTensorChecker!");

        cl_ulong maxBufferSizeByte = clDevice.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE> (&status);
        aocl_utils_cpp::checkError(status, "Failed to query the maximum buffer size in bytes!");

        cl_ulong weightWideInputSize = maxBufferSizeByte < MAX_DRAM_BYTE_WEIGHT_INPUT ? maxBufferSizeByte : MAX_DRAM_BYTE_WEIGHT_INPUT;
        std::cout <<"Setting the weightWideInput buffer. Size: "<<weightWideInputSize<<" bytes."<<std::endl;
        bufferWeightWideInput = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        weightWideInputSize,
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferWeightWideInput!");

        cl_ulong weightNarrowOutputSize = maxBufferSizeByte < MAX_DRAM_BYTE_WEIGHT_OUTPUT ? maxBufferSizeByte : MAX_DRAM_BYTE_WEIGHT_OUTPUT;
        std::cout <<"Setting the weightOutput buffer. Size: "<<weightNarrowOutputSize<<" bytes."<<std::endl;
        bufferWeightNarrowOutput = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        weightNarrowOutputSize,
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferWeightNarrowOutput!");

        cl_ulong addressInputSize = maxBufferSizeByte < MAX_DRAM_BYTE_STREAMER_ADDRESS ? maxBufferSizeByte : MAX_DRAM_BYTE_STREAMER_ADDRESS;
        std::cout <<"Setting the transfer address address buffer. Size: "<<addressInputSize<<" bytes."<<std::endl;
        bufferTransferAddressInput = cl::Buffer (
                    clContext,
                    CL_MEM_READ_ONLY,
                    addressInputSize,
                    NULL,
                    &status
                );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferTransferAddressInput!");


        std::cout <<"AOCL setup compelete"<<std::endl;

        //Need to setup numInstructions, idx, and idy separately
    }

    void launch (
            flexibleDirectCompressedTensor & compressedFilters,
            t_aligned_transfer_block_vector & outputWeightVector,

            unsigned short outputWidth, //Q
            unsigned char sizeOutputWidthTile, //TQ
            //unsigned short numOutputWidthTile, // ceil (Q / TQ)
            unsigned char sizeOutputWidthTile0, //Special case: TQ for the tiles on the left boundary
            unsigned char maxPeCols0,

            unsigned short outputHeight, //P
            unsigned char sizeOutputHeightTile, //TP
            //unsigned char sizeOutputHeightTile, // ceil (P / TP)
            unsigned char sizeOutputHeightTile0, //Special case: for the left corner

            unsigned short numFiltersInGroup, // G
            unsigned short numPeRows, //F
            //unsigned short numFoldInGroup // ceil (G / F))

            //kernel checker arguments
            unsigned char targetPeRow,
            unsigned short targetFilter,
            unsigned int sequenceID
            )
    {
        cl_int status;
\
        auto sizeTransferBlockElement = sizeof(typeof(compressedFilters.valueVector.at(0)));
        auto numTransferBlocks = compressedFilters.valueVector.size();
        auto valueVectorSizeBytes = sizeTransferBlockElement * numTransferBlocks;

        std::cout <<"Transferring the weights"<<std::endl;
        cl::Event weightTransferEvent;
        status = clCQFilterWriter.enqueueWriteBuffer(bufferWeightWideInput,
                                             CL_TRUE,
                                             0,
                                             valueVectorSizeBytes,
                                             compressedFilters.valueVector.data(),
                                             NULL,
                                             &weightTransferEvent);
        aocl_utils_cpp::checkError(status, "Failed to write the weight input vector");
        std::cout <<"Transferred "<<valueVectorSizeBytes<<" bytes in to bufferWeightWideInput"<<std::endl;
        std::cout <<"Each t_transfer_block takes up "<<sizeTransferBlockElement<<" bytes."<<std::endl;
        std::cout <<"Number of transfer blocks transmitted: "<<numTransferBlocks<<std::endl;
        std::cout <<"External memory stride in terms of transfer blocks: "<<compressedFilters.externalMemoryAddressStride<<std::endl;

        std::cout <<"Transferring the stream addresses"<<std::endl;
        status = clCQFilterWriter.enqueueWriteBuffer(bufferTransferAddressInput,
                                             CL_TRUE,
                                             0,
                                             sizeof(typeof(compressedFilters.streamBlockAddressVector.at(0))) * compressedFilters.streamBlockAddressVector.size(),
                                             compressedFilters.streamBlockAddressVector.data(),
                                             NULL);
        aocl_utils_cpp::checkError(status, "Failed to write the stream block address vector");

        std::cout <<"Setting kernel arguments"<<std::endl;
        //Setup the buffer arguments and number of transfer for the test interface
        kernelFilterWriter.setArg(0, bufferWeightWideInput); //pDramWeights
        kernelFilterWriter.setArg(1, bufferTransferAddressInput); //pStreamBlockAddress
        kernelFilterWriter.setArg(2, compressedFilters.externalMemoryAddressStride); //strideExternalMemory

        kernelFilterWriter.setArg(3, (cl_ushort) outputWidth); //outputWidth
        kernelFilterWriter.setArg(4, (cl_uchar) sizeOutputWidthTile); //TQ
        unsigned char numOutputWidthTile = (outputWidth < ((unsigned short) sizeOutputWidthTile0)) ?
                    1 : (unsigned char) (std::ceil( ((float) (outputWidth - sizeOutputWidthTile0)) / (float) (sizeOutputWidthTile) )) + 1;
        kernelFilterWriter.setArg(5 , (cl_uchar) numOutputWidthTile); //ceil (Q/TQ)
        kernelFilterWriter.setArg(6 , (cl_uchar) sizeOutputWidthTile0); //TQ0;
        kernelFilterWriter.setArg(7, (cl_uchar) maxPeCols0); //A0

        kernelFilterWriter.setArg(8, (cl_ushort) outputHeight); //outputHeight
        kernelFilterWriter.setArg(9, (cl_uchar) sizeOutputHeightTile ); //TP
        unsigned char numOutputHeightTile = (outputHeight < ((unsigned short) sizeOutputHeightTile0)) ?
                    1 : (unsigned char) (std::ceil( ((float) (outputHeight - sizeOutputHeightTile0)) / (float) (sizeOutputHeightTile) )) + 1;
        kernelFilterWriter.setArg(10, (cl_uchar) numOutputHeightTile ); // ceil (P/TP)
        kernelFilterWriter.setArg(11, (cl_uchar) sizeOutputHeightTile0); //TP0

        unsigned int L = (unsigned int) compressedFilters.num3DTensors;
        kernelFilterWriter.setArg(12, (cl_uint) L); //L

        kernelFilterWriter.setArg(13, (cl_ushort) (L / ((unsigned int) numFiltersInGroup)) ); // L / G
        kernelFilterWriter.setArg(14, (cl_ushort) numFiltersInGroup); //G
        kernelFilterWriter.setArg(15, (cl_ushort) std::ceil( ((float) (numFiltersInGroup)) / ((float) (numPeRows)) ) ); //ceil (G/F)

        kernelTensorChecker.setArg(0, bufferWeightNarrowOutput);
        kernelTensorChecker.setArg(1, (cl_uchar) targetPeRow);
        kernelTensorChecker.setArg(2, (cl_uint) sequenceID);

        clCQFilterWriter.finish();
        std::vector<cl::Event> elist;
        elist.push_back(weightTransferEvent);
        cl::Event kernelEvent;
        status = clCQTensorChecker.enqueueTask(kernelTensorChecker, &elist, &kernelEvent);
        aocl_utils_cpp::checkError(status, "Failed to launch the tensor checker kernel!");

        std::cout <<"Launch!"<<std::endl;
        status = clCQFilterWriter.enqueueTask(kernelFilterWriter, &elist);
        aocl_utils_cpp::checkError(status, "Failed to launch the filter writer kernel!");

        //Retrieve data
        clCQTensorChecker.finish();
        //clCQFilterWriter.finish();

        auto size = compressedFilters.streamBlockAddressVector.at(targetFilter);
        outputWeightVector.resize(size);

        status = clCQTensorChecker.enqueueReadBuffer(
                    bufferWeightNarrowOutput,
                    CL_TRUE,
                    0,
                    sizeof(typeof(outputWeightVector.at(0))) * size,
                    outputWeightVector.data()
                    );

        aocl_utils_cpp::checkError(status, "Failed to read the tensor checker data back!");

        cl_ulong weightTransferStartTime = weightTransferEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong weightTransferEndTime = weightTransferEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        cl_double weightTransferRunTime = (cl_double)((weightTransferEndTime - weightTransferStartTime) * (cl_double)(1e-3));

        cl_ulong kernelStartTime = kernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong kernelEndTime = kernelEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        cl_double kernelRunTime = (cl_double)((kernelEndTime - kernelStartTime) * (cl_double)(1e-3));

        std::cout <<"Weight transfer time (us): "<<weightTransferRunTime<<std::endl;
        std::cout <<"kernelTensorChecker run time (us): "<<kernelRunTime<<std::endl;
    } //launch

    void checkTensor (
            flexibleDirectCompressedTensor & compTensor,
            t_aligned_transfer_block_vector & outputVector,
            unsigned short targetFilter
            )
    {
        int numTransferBlocksToCheck =
                compTensor.streamBlockAddressVector.at(targetFilter);
        int indexInCompTensor =
                ((int) compTensor.externalMemoryAddressStride) * ((int) targetFilter);
        for (int i=0; i<numTransferBlocksToCheck; i++, indexInCompTensor++)
        {
            t_transfer_block transmittedBlock = outputVector.at(i);
            t_transfer_block goldenBlock = compTensor.valueVector.at(indexInCompTensor);
            for (int j=0; j<TRANSFER_SIZE; j++)
            {
                for (int k=0; k<CLUSTER_SIZE; k++)
                {
                    char transmittedValue = transmittedBlock.values[j].cluster_values[k];
                    char goldenValue = goldenBlock.values[j].cluster_values[k];
                    EXPECT_TRUE(transmittedValue == goldenValue)
                        <<"iTransferBlock, jCluster, kValue, transmittedValue, goldenValue "
                        <<i<<" "<<j<<" "<<k<<" "<<(int) transmittedValue<<" "<<(int) goldenValue<<std::endl;
                }
            }
        }
    } //checkTensor

};

/*!
 * \brief calculateStreamerRowID
 * \details Cacluate the filter streamer row id of a given filter.
 * \param groupSize  Number of filters inside a convolution group
 * \param numPeRows Number of PE rows
 * \param targetFilter  Index of the targetfilter
 * \return
 */
unsigned char calculateStreamerRowId (unsigned short groupSize, unsigned short numPeRows, unsigned short targetFilter);

/*!
 * \brief calculateFilterCoordinates
 * \details Calculates the sequence id of the target filter
 * \param outputWidth
 * \param sizeOutputWidthTile
 * \param numOutputWidthTile
 * \param sizeOutputWidthTile0
 * \param outputHeight
 * \param numOutputHeightTile
 * \param sizeOutputHeightTile
 * \param sizeOutputHeightTile0
 * \param numFiltersInKernel
 * \param numGroups
 * \param numFiltersInGroup
 * \param numFoldInGroup
 * \param numPeRows
 * \param targetOutputP
 * \param targetOutputQ
 * \param targetPeRow
 * \return
 */
t_filter_coordinates calculateFilterCoordinates(unsigned short outputWidth, //ceil (Q / TQ)
        unsigned char numOutputWidthTile, //Q
        unsigned char sizeOutputWidthTile, // TQ
        unsigned char sizeOutputWidthTile0, //Special case: TQ for the tiles on the left boundary
        unsigned char maxPeCols0,

        unsigned short outputHeight, //P
        unsigned char numOutputHeightTile, //ceil (P / TP)
        unsigned char sizeOutputHeightTile, // TP
        unsigned char sizeOutputHeightTile0, //L

        unsigned short numGroups, // L / G
        unsigned short numFiltersInGroup, // G
        unsigned short numFoldInGroup, // ceil (G / F)

        unsigned short numPeRows,
        unsigned short numPeCols,
        unsigned short targetOutputP, //target output height
        unsigned short targetOutputQ, //target output width
        unsigned short targetFilter //target filter
        );

t_aligned_float_vector initialize_vector(unsigned seed,
                       float bernProb,
                       int numTensors,
                       int height,
                       int width,
                       int channel,
                       float min,
                       float max
                                     ) {
    std::mt19937 generator(seed);
    std::bernoulli_distribution bernDistribution(bernProb);
    std::uniform_real_distribution<float> uniDistribution(min, max);
    int numElements = numTensors * height * width * channel;
    t_aligned_float_vector vector(numElements, 0.0f);

    for (unsigned i=0; i<numElements; i++) {
        float val = bernDistribution(generator) ?
                    uniDistribution(generator) : 0.0f;
        vector.at(i) = val;
    }
    return vector;
}

std::vector<fixedPointNumber> initialize_deterministic_vector(
            int numTensors,
            int height,
            int width,
            int channel,
            char fractionWidth,
            char integerWidth
        )
{
        int numElementsPerFilter = height * width * channel;
        std::vector<fixedPointNumber> fpVector;
        fpVector.resize(numElementsPerFilter*numTensors);

        unsigned q=0;
        for (unsigned i = 0; i<numTensors; i++)
        {
            char val = -128;
            for (unsigned int j=0;j<numElementsPerFilter;j++, q++)
            {
                fpVector.at(q) = fixedPointNumber((char)val, fractionWidth, integerWidth);
                val = (val==127) ? -128 : val + 1;
            }
        }

        return fpVector;
}

//#define PLAY
#ifdef PLAY
TEST_F (testFixture, play) {
    //Fixed point convertion arguments and sparsity control
    char fracWidth = 4, intWidth = 3;
    float bernProb = 1.0;
    int seed = 1999;
    float min = 1.0;
    float max = 3.0;

    //Kernel arugments
    //int numFilters = 128;
    int numFilters = 16;
    int kernelHeight = 3;
    int kernelWidth = 3;
    unsigned short tileSizeKernelWidth = 32; //not needed, to be removed after refactoring the compression code
    int inputFeaturMapChannels = 3;
    int numElements = numFilters * kernelHeight * kernelWidth * inputFeaturMapChannels;

    unsigned short maxScalarIndexInChannelGroup = inputFeaturMapChannels - 1;
    unsigned  char maxClusterIndexInCompressionBlock = COMPRESSION_WINDOW_SIZE - 1;
    unsigned char maxClusterIndexInTransferBlock = TRANSFER_SIZE - 1;
    unsigned char maxScalarIndexInCluster = CLUSTER_SIZE - 1;

    //Output tensor arguments
    unsigned short outputWidth = 16;
    unsigned char sizeOutputWidthTile = 2;
    unsigned char sizeOutputWidthTile0 = 1;
    unsigned char numOutputWidthTile = (outputWidth < ((unsigned short) sizeOutputWidthTile0)) ?
                1 : (unsigned char) (std::ceil( ((float) (outputWidth - sizeOutputWidthTile0)) / (float) (sizeOutputWidthTile) )) + 1;
    unsigned short outputHeight = 16;
    unsigned char sizeOutputHeightTile = 2;
    unsigned char sizeOutputHeightTile0 = 1;
    unsigned char numOutputHeightTile = (outputHeight < ((unsigned short) sizeOutputHeightTile0)) ?
                1 : (unsigned char) (std::ceil( ((float) (outputHeight - sizeOutputHeightTile0)) / (float) (sizeOutputHeightTile) )) + 1;

    unsigned short numFiltersInGroup = numFilters;
    unsigned char numPeRows = PE_ROWS;
    unsigned char numPeCols = PE_COLS;

    //Target arguments
    unsigned short targetOutputP = 15;
    unsigned short targetOutputQ = 15;
    unsigned short targetFilter = 15;

    t_aligned_float_vector floatVector = initialize_vector(
                seed,
                bernProb,
                numFilters,
                kernelHeight,
                kernelWidth,
                inputFeaturMapChannels,
                min,
                max
                );

    std::vector<fixedPointNumber> fpVector;
    fpVector.resize(numElements);

    for (int i=0; i<numElements; i++) {
        fixedPointNumber fpValue(floatVector.at(i), fracWidth, intWidth);
        fpVector.at(i) = fpValue;
    }

//    std::vector<fixedPointNumber> fpVector = initialize_deterministic_vector(
//                   numFilters,
//                   kernelHeight,
//                   kernelWidth,
//                   inputFeaturMapChannels,
//                   4,
//                   3
//                );

    std::cout <<"Start to compress the kernel tensor"<<std::endl;
    flexibleDirectCompressedTensor compTensor(
                fpVector,
                numFilters,
                inputFeaturMapChannels,
                kernelWidth,
                kernelHeight,
                tileSizeKernelWidth, //not needed
                maxScalarIndexInChannelGroup,
                maxClusterIndexInCompressionBlock,
                maxClusterIndexInTransferBlock,
                maxScalarIndexInCluster,
                true //isKernel
                );

    std::cout <<"Calculate the filter row and sequence id to intercept"<<std::endl;
    unsigned short numGroups = numFilters / ((unsigned int) numFiltersInGroup);
    unsigned short numFoldsInGroup = (unsigned short) std::ceil( ((float) (numFiltersInGroup)) / ((float) (numPeRows)) );

    t_filter_coordinates targetCoordinates =
            calculateFilterCoordinates(
                    outputWidth,
                    numOutputWidthTile,
                    sizeOutputWidthTile,
                    sizeOutputWidthTile0,
                    numPeCols,

                    outputHeight,
                    numOutputHeightTile,
                    sizeOutputHeightTile,
                    sizeOutputHeightTile0,

                    numGroups,
                    numFiltersInGroup,
                    numFoldsInGroup,

                    numPeRows,
                    numPeCols,
                    targetOutputP,
                    targetOutputQ,
                    targetFilter
                );

    //launch
    std::cout <<"Launching the kernels"<<std::endl;
    t_aligned_transfer_block_vector outputVector;
    launch (
        compTensor,
        outputVector,

        outputWidth,
        sizeOutputWidthTile,
        sizeOutputWidthTile0,
        numPeCols,

        outputHeight,
        sizeOutputHeightTile,
        sizeOutputHeightTile0,

        numFiltersInGroup,
        numPeRows,

        targetCoordinates.targetFilterRow,
        targetFilter,
        targetCoordinates.sequenceId
       );

    //Check the results
    std::cout <<"Checking results"<<std::endl;
    checkTensor(compTensor, outputVector, targetFilter);

}
#else
TEST_F (testFixture, fullyConnectedSmall) {
    //Fixed point convertion arguments and sparsity control
    char fracWidth = 4, intWidth = 3;
    float bernProb = 1.0;
    int seed = 1256;
    float min = 1.0;
    float max = 1.01;

    //Kernel arugments
    int numFilters = 16;
    int kernelHeight = 1;
    int kernelWidth = 1;
    unsigned short tileSizeKernelWidth = 32; //not needed, to be removed after refactoring the compression code
    int inputFeaturMapChannels = 16;
    int numElements = numFilters * kernelHeight * kernelWidth * inputFeaturMapChannels;

    unsigned short maxScalarIndexInChannelGroup = inputFeaturMapChannels - 1;
    unsigned  char maxClusterIndexInCompressionBlock = COMPRESSION_WINDOW_SIZE - 1;
    unsigned char maxClusterIndexInTransferBlock = TRANSFER_SIZE - 1;
    unsigned char maxScalarIndexInCluster = CLUSTER_SIZE - 1;

    //Output tensor arguments
    unsigned short outputWidth = 1;
    unsigned char sizeOutputWidthTile = 32;
    unsigned char sizeOutputWidthTile0 = 31;
    unsigned char numOutputWidthTile = (outputWidth < ((unsigned short) sizeOutputWidthTile0)) ?
                1 : (unsigned char) (std::ceil( ((float) (outputWidth - sizeOutputWidthTile0)) / (float) (sizeOutputWidthTile) )) + 1;
    unsigned short outputHeight = 1;
    unsigned char sizeOutputHeightTile = 32;
    unsigned char sizeOutputHeightTile0 = 31;
    unsigned char numOutputHeightTile = (outputHeight < ((unsigned short) sizeOutputHeightTile0)) ?
                1 : (unsigned char) (std::ceil( ((float) (outputHeight - sizeOutputHeightTile0)) / (float) (sizeOutputHeightTile) )) + 1;

    unsigned short numFiltersInGroup = numFilters;
    unsigned char numPeRows = PE_ROWS;
    unsigned char numPeCols = PE_COLS;

    //Target arguments
#ifdef C5SOC
    unsigned short targetOutputP = 0;
    unsigned short targetOutputQ = 0;
    unsigned short targetFilter = numFilters - 1;
#else
    unsigned short targetOutputP = 0;
    unsigned short targetOutputQ = 0;
    unsigned short targetFilter = 15;
#endif

    t_aligned_float_vector floatVector = initialize_vector(
                seed,
                bernProb,
                numFilters,
                kernelHeight,
                kernelWidth,
                inputFeaturMapChannels,
                min,
                max
                );

    std::vector<fixedPointNumber> fpVector;
    fpVector.resize(numElements);

    for (int i=0; i<numElements; i++) {
        fixedPointNumber fpValue(floatVector.at(i), fracWidth, intWidth);
        fpVector.at(i) = fpValue;
    }

    std::cout <<"Start to compress the kernel tensor"<<std::endl;
    flexibleDirectCompressedTensor compTensor(
                fpVector,
                numFilters,
                inputFeaturMapChannels,
                kernelWidth,
                kernelHeight,
                tileSizeKernelWidth, //not needed
                maxScalarIndexInChannelGroup,
                maxClusterIndexInCompressionBlock,
                maxClusterIndexInTransferBlock,
                maxScalarIndexInCluster,
                true //isKernel
                );

    std::cout <<"Calculate the filter row and sequence id to intercept"<<std::endl;
    unsigned short numGroups = numFilters / ((unsigned int) numFiltersInGroup);
    unsigned short numFoldsInGroup = (unsigned short) std::ceil( ((float) (numFiltersInGroup)) / ((float) (numPeRows)) );

    t_filter_coordinates targetCoordinates =
            calculateFilterCoordinates(
                    outputWidth,
                    numOutputWidthTile,
                    sizeOutputWidthTile,
                    sizeOutputWidthTile0,
                    numPeCols,

                    outputHeight,
                    numOutputHeightTile,
                    sizeOutputHeightTile,
                    sizeOutputHeightTile0,

                    numGroups,
                    numFiltersInGroup,
                    numFoldsInGroup,

                    numPeRows,
                    numPeCols,
                    targetOutputP,
                    targetOutputQ,
                    targetFilter
                );

    //launch
    std::cout <<"Launching the kernels"<<std::endl;
    t_aligned_transfer_block_vector outputVector;
    launch (
        compTensor,
        outputVector,

        outputWidth,
        sizeOutputWidthTile,
        sizeOutputWidthTile0,
        numPeCols,

        outputHeight,
        sizeOutputHeightTile,
        sizeOutputHeightTile0,

        numFiltersInGroup,
        numPeRows,

        targetCoordinates.targetFilterRow,
        targetFilter,
        targetCoordinates.sequenceId
       );

    //Check the results
    std::cout <<"Checking results"<<std::endl;
    checkTensor(compTensor, outputVector, targetFilter);

}

TEST_F (testFixture, convSmall3by3) {
    //Fixed point convertion arguments and sparsity control
    char fracWidth = 4, intWidth = 3;
    float bernProb = 0.1;
    int seed = 1256;
    float min = -2.0;
    float max = 2.0;

    //Kernel arugments
    int numFilters = 4;
    int kernelHeight = 3;
    int kernelWidth = 3;
    unsigned short tileSizeKernelWidth = 32; //not needed, to be removed after refactoring the compression code
    int inputFeaturMapChannels = 4;
    int numElements = numFilters * kernelHeight * kernelWidth * inputFeaturMapChannels;

    unsigned short maxScalarIndexInChannelGroup = inputFeaturMapChannels - 1;
    unsigned  char maxClusterIndexInCompressionBlock = COMPRESSION_WINDOW_SIZE - 1;
    unsigned char maxClusterIndexInTransferBlock = TRANSFER_SIZE - 1;
    unsigned char maxScalarIndexInCluster = CLUSTER_SIZE - 1;

    //Output tensor arguments
    unsigned short outputWidth = 48;
    unsigned char sizeOutputWidthTile = 11;
    unsigned char sizeOutputWidthTile0 = 5;
    unsigned char numOutputWidthTile = (outputWidth < ((unsigned short) sizeOutputWidthTile0)) ?
                1 : (unsigned char) (std::ceil( ((float) (outputWidth - sizeOutputWidthTile0)) / (float) (sizeOutputWidthTile) )) + 1;
    unsigned short outputHeight = 48;
    unsigned char sizeOutputHeightTile = 11;
    unsigned char sizeOutputHeightTile0 = 5;
    unsigned char numOutputHeightTile = (outputHeight < ((unsigned short) sizeOutputHeightTile0)) ?
                1 : (unsigned char) (std::ceil( ((float) (outputHeight - sizeOutputHeightTile0)) / (float) (sizeOutputHeightTile) )) + 1;

    unsigned short numFiltersInGroup = numFilters;
    unsigned char numPeRows = PE_ROWS;
    unsigned char numPeCols = PE_COLS;

    //Target arguments
#ifdef C5SOC
    unsigned short targetOutputP = outputHeight - 1;
    unsigned short targetOutputQ = outputWidth - 1 ;
    unsigned short targetFilter = numFilters - 1;
#else
    unsigned short targetOutputP = 11;
    unsigned short targetOutputQ = 7;
    unsigned short targetFilter = 3;
#endif

    t_aligned_float_vector floatVector = initialize_vector(
                seed,
                bernProb,
                numFilters,
                kernelHeight,
                kernelWidth,
                inputFeaturMapChannels,
                min,
                max
                );

    std::vector<fixedPointNumber> fpVector;
    fpVector.resize(numElements);

    for (int i=0; i<numElements; i++) {
        fixedPointNumber fpValue(floatVector.at(i), fracWidth, intWidth);
        fpVector.at(i) = fpValue;
    }

    std::cout <<"Start to compress the kernel tensor"<<std::endl;
    flexibleDirectCompressedTensor compTensor(
                fpVector,
                numFilters,
                inputFeaturMapChannels,
                kernelWidth,
                kernelHeight,
                tileSizeKernelWidth, //not needed
                maxScalarIndexInChannelGroup,
                maxClusterIndexInCompressionBlock,
                maxClusterIndexInTransferBlock,
                maxScalarIndexInCluster,
                true //isKernel
                );

    std::cout <<"Calculate the filter row and sequence id to intercept"<<std::endl;
    unsigned short numGroups = numFilters / ((unsigned int) numFiltersInGroup);
    unsigned short numFoldsInGroup = (unsigned short) std::ceil( ((float) (numFiltersInGroup)) / ((float) (numPeRows)) );

    t_filter_coordinates targetCoordinates =
            calculateFilterCoordinates(
                    outputWidth,
                    numOutputWidthTile,
                    sizeOutputWidthTile,
                    sizeOutputWidthTile0,
                    numPeCols,

                    outputHeight,
                    numOutputHeightTile,
                    sizeOutputHeightTile,
                    sizeOutputHeightTile0,

                    numGroups,
                    numFiltersInGroup,
                    numFoldsInGroup,

                    numPeRows,
                    numPeCols,
                    targetOutputP,
                    targetOutputQ,
                    targetFilter
                );

    //launch
    std::cout <<"Launching the kernels"<<std::endl;
    t_aligned_transfer_block_vector outputVector;
    launch (
        compTensor,
        outputVector,

        outputWidth,
        sizeOutputWidthTile,
        sizeOutputWidthTile0,
        numPeCols,

        outputHeight,
        sizeOutputHeightTile,
        sizeOutputHeightTile0,

        numFiltersInGroup,
        numPeRows,

        targetCoordinates.targetFilterRow,
        targetFilter,
        targetCoordinates.sequenceId
       );

    //Check the results
    std::cout <<"Checking results"<<std::endl;
    checkTensor(compTensor, outputVector, targetFilter);

}

//Delay the large tests on FPGA
#ifdef C5SOC
TEST_F (testFixture, fullyConnectedLargeSparse) {
    //Fixed point convertion arguments and sparsity control
    char fracWidth = 4, intWidth = 3;
    float bernProb = 0.01;
    int seed = 1256;
    float min = -3.0;
    float max = 3.0;

    //Kernel arugments
    int numFilters = 1024;
    int kernelHeight = 1;
    int kernelWidth = 1;
    unsigned short tileSizeKernelWidth = 32; //not needed, to be removed after refactoring the compression code
    int inputFeaturMapChannels = 4096;
    int numElements = numFilters * kernelHeight * kernelWidth * inputFeaturMapChannels;

    unsigned short maxScalarIndexInChannelGroup = inputFeaturMapChannels - 1;
    unsigned  char maxClusterIndexInCompressionBlock = COMPRESSION_WINDOW_SIZE - 1;
    unsigned char maxClusterIndexInTransferBlock = TRANSFER_SIZE - 1;
    unsigned char maxScalarIndexInCluster = CLUSTER_SIZE - 1;

    //Output tensor arguments
    unsigned short outputWidth = 1;
    unsigned char sizeOutputWidthTile = 32;
    unsigned char sizeOutputWidthTile0 = 31;
    unsigned char numOutputWidthTile = (outputWidth < ((unsigned short) sizeOutputWidthTile0)) ?
                1 : (unsigned char) (std::ceil( ((float) (outputWidth - sizeOutputWidthTile0)) / (float) (sizeOutputWidthTile) )) + 1;
    unsigned short outputHeight = 1;
    unsigned char sizeOutputHeightTile = 32;
    unsigned char sizeOutputHeightTile0 = 31;
    unsigned char numOutputHeightTile = (outputHeight < ((unsigned short) sizeOutputHeightTile0)) ?
                1 : (unsigned char) (std::ceil( ((float) (outputHeight - sizeOutputHeightTile0)) / (float) (sizeOutputHeightTile) )) + 1;

    unsigned short numFiltersInGroup = numFilters;
    unsigned char numPeRows = PE_ROWS;
    unsigned char numPeCols = PE_COLS;

    //Target arguments
    unsigned short targetOutputP = 0;
    unsigned short targetOutputQ = 0;
    unsigned short targetFilter = 1023;

    t_aligned_float_vector floatVector = initialize_vector(
                seed,
                bernProb,
                numFilters,
                kernelHeight,
                kernelWidth,
                inputFeaturMapChannels,
                min,
                max
                );

    std::vector<fixedPointNumber> fpVector;
    fpVector.resize(numElements);

    for (int i=0; i<numElements; i++) {
        fixedPointNumber fpValue(floatVector.at(i), fracWidth, intWidth);
        fpVector.at(i) = fpValue;
    }

    std::cout <<"Start to compress the kernel tensor"<<std::endl;
    flexibleDirectCompressedTensor compTensor(
                fpVector,
                numFilters,
                inputFeaturMapChannels,
                kernelWidth,
                kernelHeight,
                tileSizeKernelWidth, //not needed
                maxScalarIndexInChannelGroup,
                maxClusterIndexInCompressionBlock,
                maxClusterIndexInTransferBlock,
                maxScalarIndexInCluster,
                true //isKernel
                );

    std::cout <<"Calculate the filter row and sequence id to intercept"<<std::endl;
    unsigned short numGroups = numFilters / ((unsigned int) numFiltersInGroup);
    unsigned short numFoldsInGroup = (unsigned short) std::ceil( ((float) (numFiltersInGroup)) / ((float) (numPeRows)) );

    t_filter_coordinates targetCoordinates =
            calculateFilterCoordinates(
                    outputWidth,
                    numOutputWidthTile,
                    sizeOutputWidthTile,
                    sizeOutputWidthTile0,
                    numPeCols,

                    outputHeight,
                    numOutputHeightTile,
                    sizeOutputHeightTile,
                    sizeOutputHeightTile0,

                    numGroups,
                    numFiltersInGroup,
                    numFoldsInGroup,

                    numPeRows,
                    numPeCols,
                    targetOutputP,
                    targetOutputQ,
                    targetFilter
                );

    //launch
    std::cout <<"Launching the kernels"<<std::endl;
    t_aligned_transfer_block_vector outputVector;
    launch (
        compTensor,
        outputVector,

        outputWidth,
        sizeOutputWidthTile,
        sizeOutputWidthTile0,
        numPeCols,

        outputHeight,
        sizeOutputHeightTile,
        sizeOutputHeightTile0,

        numFiltersInGroup,
        numPeRows,

        targetCoordinates.targetFilterRow,
        targetFilter,
        targetCoordinates.sequenceId
       );

    //Check the results
    std::cout <<"Checking results"<<std::endl;
    checkTensor(compTensor, outputVector, targetFilter);

}

TEST_F (testFixture, fullyConnectedLargeDense) {
    //Fixed point convertion arguments and sparsity control
    char fracWidth = 4, intWidth = 3;
    float bernProb = 1.0;
    int seed = 1256;
    float min = -3.0;
    float max = 3.0;

    //Kernel arugments
    int numFilters = 1024;
    int kernelHeight = 1;
    int kernelWidth = 1;
    unsigned short tileSizeKernelWidth = 32; //not needed, to be removed after refactoring the compression code
    int inputFeaturMapChannels = 4096;
    int numElements = numFilters * kernelHeight * kernelWidth * inputFeaturMapChannels;

    unsigned short maxScalarIndexInChannelGroup = inputFeaturMapChannels - 1;
    unsigned  char maxClusterIndexInCompressionBlock = COMPRESSION_WINDOW_SIZE - 1;
    unsigned char maxClusterIndexInTransferBlock = TRANSFER_SIZE - 1;
    unsigned char maxScalarIndexInCluster = CLUSTER_SIZE - 1;

    //Output tensor arguments
    unsigned short outputWidth = 1;
    unsigned char sizeOutputWidthTile = 32;
    unsigned char sizeOutputWidthTile0 = 31;
    unsigned char numOutputWidthTile = (outputWidth < ((unsigned short) sizeOutputWidthTile0)) ?
                1 : (unsigned char) (std::ceil( ((float) (outputWidth - sizeOutputWidthTile0)) / (float) (sizeOutputWidthTile) )) + 1;
    unsigned short outputHeight = 1;
    unsigned char sizeOutputHeightTile = 32;
    unsigned char sizeOutputHeightTile0 = 31;
    unsigned char numOutputHeightTile = (outputHeight < ((unsigned short) sizeOutputHeightTile0)) ?
                1 : (unsigned char) (std::ceil( ((float) (outputHeight - sizeOutputHeightTile0)) / (float) (sizeOutputHeightTile) )) + 1;

    unsigned short numFiltersInGroup = numFilters;
    unsigned char numPeRows = PE_ROWS;
    unsigned char numPeCols = PE_COLS;

    //Target arguments
    unsigned short targetOutputP = 0;
    unsigned short targetOutputQ = 0;
    unsigned short targetFilter = 1023;

    t_aligned_float_vector floatVector = initialize_vector(
                seed,
                bernProb,
                numFilters,
                kernelHeight,
                kernelWidth,
                inputFeaturMapChannels,
                min,
                max
                );

    std::vector<fixedPointNumber> fpVector;
    fpVector.resize(numElements);

    for (int i=0; i<numElements; i++) {
        fixedPointNumber fpValue(floatVector.at(i), fracWidth, intWidth);
        fpVector.at(i) = fpValue;
    }

    std::cout <<"Start to compress the kernel tensor"<<std::endl;
    flexibleDirectCompressedTensor compTensor(
                fpVector,
                numFilters,
                inputFeaturMapChannels,
                kernelWidth,
                kernelHeight,
                tileSizeKernelWidth, //not needed
                maxScalarIndexInChannelGroup,
                maxClusterIndexInCompressionBlock,
                maxClusterIndexInTransferBlock,
                maxScalarIndexInCluster,
                true //isKernel
                );

    std::cout <<"Calculate the filter row and sequence id to intercept"<<std::endl;
    unsigned short numGroups = numFilters / ((unsigned int) numFiltersInGroup);
    unsigned short numFoldsInGroup = (unsigned short) std::ceil( ((float) (numFiltersInGroup)) / ((float) (numPeRows)) );

    t_filter_coordinates targetCoordinates =
            calculateFilterCoordinates(
                    outputWidth,
                    numOutputWidthTile,
                    sizeOutputWidthTile,
                    sizeOutputWidthTile0,
                    numPeCols,

                    outputHeight,
                    numOutputHeightTile,
                    sizeOutputHeightTile,
                    sizeOutputHeightTile0,

                    numGroups,
                    numFiltersInGroup,
                    numFoldsInGroup,

                    numPeRows,
                    numPeCols,
                    targetOutputP,
                    targetOutputQ,
                    targetFilter
                );

    //launch
    std::cout <<"Launching the kernels"<<std::endl;
    t_aligned_transfer_block_vector outputVector;
    launch (
        compTensor,
        outputVector,

        outputWidth,
        sizeOutputWidthTile,
        sizeOutputWidthTile0,
        numPeCols,

        outputHeight,
        sizeOutputHeightTile,
        sizeOutputHeightTile0,

        numFiltersInGroup,
        numPeRows,

        targetCoordinates.targetFilterRow,
        targetFilter,
        targetCoordinates.sequenceId
       );

    //Check the results
    std::cout <<"Checking results"<<std::endl;
    checkTensor(compTensor, outputVector, targetFilter);

}

TEST_F (testFixture, convLarge3by3Sparse) {
    //Fixed point convertion arguments and sparsity control
    char fracWidth = 4, intWidth = 3;
    float bernProb = 0.1;
    int seed = 1999;
    float min = -3.0;
    float max = 3.0;

    //Kernel arugments
    int numFilters = 512;
    int kernelHeight = 3;
    int kernelWidth = 3;
    unsigned short tileSizeKernelWidth = 32; //not needed, to be removed after refactoring the compression code
    int inputFeaturMapChannels = 512;
    int numElements = numFilters * kernelHeight * kernelWidth * inputFeaturMapChannels;

    unsigned short maxScalarIndexInChannelGroup = inputFeaturMapChannels - 1;
    unsigned  char maxClusterIndexInCompressionBlock = COMPRESSION_WINDOW_SIZE - 1;
    unsigned char maxClusterIndexInTransferBlock = TRANSFER_SIZE - 1;
    unsigned char maxScalarIndexInCluster = CLUSTER_SIZE - 1;

    //Output tensor arguments
    unsigned short outputWidth = 227;
    unsigned char sizeOutputWidthTile = 32;
    unsigned char sizeOutputWidthTile0 = 31;
    unsigned char numOutputWidthTile = (outputWidth < ((unsigned short) sizeOutputWidthTile0)) ?
                1 : (unsigned char) (std::ceil( ((float) (outputWidth - sizeOutputWidthTile0)) / (float) (sizeOutputWidthTile) )) + 1;
    unsigned short outputHeight = 227;
    unsigned char sizeOutputHeightTile = 32;
    unsigned char sizeOutputHeightTile0 = 31;
    unsigned char numOutputHeightTile = (outputHeight < ((unsigned short) sizeOutputHeightTile0)) ?
                1 : (unsigned char) (std::ceil( ((float) (outputHeight - sizeOutputHeightTile0)) / (float) (sizeOutputHeightTile) )) + 1;

    unsigned short numFiltersInGroup = numFilters;
    unsigned char numPeRows = PE_ROWS;
    unsigned char numPeCols = PE_COLS;

    //Target arguments
    unsigned short targetOutputP = 226;
    unsigned short targetOutputQ = 226;
    unsigned short targetFilter = 511;

    t_aligned_float_vector floatVector = initialize_vector(
                seed,
                bernProb,
                numFilters,
                kernelHeight,
                kernelWidth,
                inputFeaturMapChannels,
                min,
                max
                );

    std::vector<fixedPointNumber> fpVector;
    fpVector.resize(numElements);

    for (int i=0; i<numElements; i++) {
        fixedPointNumber fpValue(floatVector.at(i), fracWidth, intWidth);
        fpVector.at(i) = fpValue;
    }

    std::cout <<"Start to compress the kernel tensor"<<std::endl;
    flexibleDirectCompressedTensor compTensor(
                fpVector,
                numFilters,
                inputFeaturMapChannels,
                kernelWidth,
                kernelHeight,
                tileSizeKernelWidth, //not needed
                maxScalarIndexInChannelGroup,
                maxClusterIndexInCompressionBlock,
                maxClusterIndexInTransferBlock,
                maxScalarIndexInCluster,
                true //isKernel
                );

    std::cout <<"Calculate the filter row and sequence id to intercept"<<std::endl;
    unsigned short numGroups = numFilters / ((unsigned int) numFiltersInGroup);
    unsigned short numFoldsInGroup = (unsigned short) std::ceil( ((float) (numFiltersInGroup)) / ((float) (numPeRows)) );

    t_filter_coordinates targetCoordinates =
            calculateFilterCoordinates(
                    outputWidth,
                    numOutputWidthTile,
                    sizeOutputWidthTile,
                    sizeOutputWidthTile0,
                    numPeCols,

                    outputHeight,
                    numOutputHeightTile,
                    sizeOutputHeightTile,
                    sizeOutputHeightTile0,

                    numGroups,
                    numFiltersInGroup,
                    numFoldsInGroup,

                    numPeRows,
                    numPeCols,
                    targetOutputP,
                    targetOutputQ,
                    targetFilter
                );

    //launch
    std::cout <<"Launching the kernels"<<std::endl;
    t_aligned_transfer_block_vector outputVector;
    launch (
        compTensor,
        outputVector,

        outputWidth,
        sizeOutputWidthTile,
        sizeOutputWidthTile0,
        numPeCols,

        outputHeight,
        sizeOutputHeightTile,
        sizeOutputHeightTile0,

        numFiltersInGroup,
        numPeRows,

        targetCoordinates.targetFilterRow,
        targetFilter,
        targetCoordinates.sequenceId
       );

    //Check the results
    std::cout <<"Checking results"<<std::endl;
    checkTensor(compTensor, outputVector, targetFilter);

}

TEST_F (testFixture, convLarge3by3Dense) {
    //Fixed point convertion arguments and sparsity control
    char fracWidth = 4, intWidth = 3;
    float bernProb = 1.0;
    int seed = 1999;
    float min = 1.0;
    float max = 4.0;

    //Kernel arugments
    int numFilters = 512;
    int kernelHeight = 3;
    int kernelWidth = 3;
    unsigned short tileSizeKernelWidth = 32; //not needed, to be removed after refactoring the compression code
    int inputFeaturMapChannels = 512;
    int numElements = numFilters * kernelHeight * kernelWidth * inputFeaturMapChannels;

    unsigned short maxScalarIndexInChannelGroup = inputFeaturMapChannels - 1;
    unsigned  char maxClusterIndexInCompressionBlock = COMPRESSION_WINDOW_SIZE - 1;
    unsigned char maxClusterIndexInTransferBlock = TRANSFER_SIZE - 1;
    unsigned char maxScalarIndexInCluster = CLUSTER_SIZE - 1;

    //Output tensor arguments
    unsigned short outputWidth = 227;
    unsigned char sizeOutputWidthTile = 32;
    unsigned char sizeOutputWidthTile0 = 31;
    unsigned char numOutputWidthTile = (outputWidth < ((unsigned short) sizeOutputWidthTile0)) ?
                1 : (unsigned char) (std::ceil( ((float) (outputWidth - sizeOutputWidthTile0)) / (float) (sizeOutputWidthTile) )) + 1;
    unsigned short outputHeight = 227;
    unsigned char sizeOutputHeightTile = 32;
    unsigned char sizeOutputHeightTile0 = 31;
    unsigned char numOutputHeightTile = (outputHeight < ((unsigned short) sizeOutputHeightTile0)) ?
                1 : (unsigned char) (std::ceil( ((float) (outputHeight - sizeOutputHeightTile0)) / (float) (sizeOutputHeightTile) )) + 1;

    unsigned short numFiltersInGroup = numFilters;
    unsigned char numPeRows = PE_ROWS;
    unsigned char numPeCols = PE_COLS;

    //Target arguments
    unsigned short targetOutputP = 226;
    unsigned short targetOutputQ = 226;
    unsigned short targetFilter = 511;

    t_aligned_float_vector floatVector = initialize_vector(
                seed,
                bernProb,
                numFilters,
                kernelHeight,
                kernelWidth,
                inputFeaturMapChannels,
                min,
                max
                );

    std::vector<fixedPointNumber> fpVector;
    fpVector.resize(numElements);

    for (int i=0; i<numElements; i++) {
        fixedPointNumber fpValue(floatVector.at(i), fracWidth, intWidth);
        fpVector.at(i) = fpValue;
    }

    std::cout <<"Start to compress the kernel tensor"<<std::endl;
    flexibleDirectCompressedTensor compTensor(
                fpVector,
                numFilters,
                inputFeaturMapChannels,
                kernelWidth,
                kernelHeight,
                tileSizeKernelWidth, //not needed
                maxScalarIndexInChannelGroup,
                maxClusterIndexInCompressionBlock,
                maxClusterIndexInTransferBlock,
                maxScalarIndexInCluster,
                true //isKernel
                );

    std::cout <<"Calculate the filter row and sequence id to intercept"<<std::endl;
    unsigned short numGroups = numFilters / ((unsigned int) numFiltersInGroup);
    unsigned short numFoldsInGroup = (unsigned short) std::ceil( ((float) (numFiltersInGroup)) / ((float) (numPeRows)) );

    t_filter_coordinates targetCoordinates =
            calculateFilterCoordinates(
                    outputWidth,
                    numOutputWidthTile,
                    sizeOutputWidthTile,
                    sizeOutputWidthTile0,
                    numPeCols,

                    outputHeight,
                    numOutputHeightTile,
                    sizeOutputHeightTile,
                    sizeOutputHeightTile0,

                    numGroups,
                    numFiltersInGroup,
                    numFoldsInGroup,

                    numPeRows,
                    numPeCols,
                    targetOutputP,
                    targetOutputQ,
                    targetFilter
                );

    //launch
    std::cout <<"Launching the kernels"<<std::endl;
    t_aligned_transfer_block_vector outputVector;
    launch (
        compTensor,
        outputVector,

        outputWidth,
        sizeOutputWidthTile,
        sizeOutputWidthTile0,
        numPeCols,

        outputHeight,
        sizeOutputHeightTile,
        sizeOutputHeightTile0,

        numFiltersInGroup,
        numPeRows,

        targetCoordinates.targetFilterRow,
        targetFilter,
        targetCoordinates.sequenceId
       );

    //Check the results
    std::cout <<"Checking results"<<std::endl;
    checkTensor(compTensor, outputVector, targetFilter);

}
#endif
#endif

int main(int argc, char* argv[]) {

    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}

unsigned char calculateStreamerRowId (unsigned short groupSize, unsigned short numPeRows, unsigned short targetFilter)
{
    return (unsigned char)((targetFilter % groupSize) % numPeRows);
}

t_filter_coordinates calculateFilterCoordinates(
        unsigned short outputWidth, //Q
        unsigned char numOutputWidthTile, //ceil (Q / TQ)
        unsigned char sizeOutputWidthTile, //TQ
        unsigned char sizeOutputWidthTile0, //Special case: TQ for the tiles on the left boundary
        unsigned char maxPeCols0, //maximum number of PE cols for the first tile along width

        unsigned short outputHeight, //P
        unsigned char numOutputHeightTile, //ceil (P / TP)
        unsigned char sizeOutputHeightTile, // TP
        unsigned char sizeOutputHeightTile0, //Special case: for the left corner

        //unsigned int numFiltersInKernel, //L

        unsigned short numGroups, // L / G
        unsigned short numFiltersInGroup, // G
        unsigned short numFoldInGroup, // ceil (G / F)

        unsigned short numPeRows,
        unsigned short numPeCols,
        unsigned short targetOutputP, //target output height
        unsigned short targetOutputQ, //target output width
        unsigned short targetFilter //target filter
        )
{
    unsigned int countSequence = 0;
    unsigned short countPCoveredByTP = 0;
    t_filter_coordinates result = {0, 0};

    unsigned char targetPeRow = calculateStreamerRowId(numFiltersInGroup, numPeRows, targetFilter);

    for (unsigned short tp=0; tp<numOutputHeightTile; tp++)
    {
        unsigned short sizeOutputHeightTileTemp = (countPCoveredByTP==0) ? sizeOutputHeightTile0 : sizeOutputHeightTile;
        unsigned short maxSizeOutputHeightTile = (sizeOutputHeightTileTemp < (outputHeight - countPCoveredByTP))
                    ? sizeOutputHeightTileTemp : (outputHeight - countPCoveredByTP);
        unsigned short countQCoveredByTQ = 0;
        for (unsigned short tq=0; tq<numOutputWidthTile; tq++)
        {
            unsigned short sizeOutputWidthTileTemp = (countQCoveredByTQ==0) ? sizeOutputWidthTile0 : sizeOutputWidthTile;
            unsigned short maxSizeOutputWidthTile = (sizeOutputWidthTileTemp < (outputWidth - countQCoveredByTQ))
                    ? sizeOutputWidthTileTemp : (outputWidth - countQCoveredByTQ);
            unsigned char maxPeCols = (countQCoveredByTQ == 0) ? maxPeCols0 : numPeCols;

            unsigned short filterCountCoveredAcrossGroup = 0;
            for (unsigned short gl=0; gl<numGroups; gl++)
            {
                unsigned short filterCountCoveredInGroup = filterCountCoveredAcrossGroup;
                for (unsigned short gf=0; gf<numFoldInGroup; gf++)
                {
                    unsigned short maxF = numPeRows < (numFiltersInGroup - gf*numPeRows)
                            ? numPeRows :  (numFiltersInGroup - gf*numPeRows);
                    unsigned short pCount = countPCoveredByTP;
                    for (unsigned short p=0; p<maxSizeOutputHeightTile; p++)
                    {
                        unsigned short qCountCoveredByA = countQCoveredByTQ;
                        unsigned short pqxA = 0;
                        while (pqxA < maxSizeOutputWidthTile)
                        {
                            unsigned short maxA = maxPeCols < (maxSizeOutputWidthTile - pqxA)
                                ? maxPeCols : (maxSizeOutputWidthTile - pqxA);
                            unsigned short filterCount = filterCountCoveredInGroup;
                            for (unsigned char f=0; f<maxF; f++)
                            {
                                unsigned short qCount = qCountCoveredByA;
                                for (unsigned short a=0; a<maxA; a++)
                                {
                                    if ((qCount == targetOutputQ) && (pCount == targetOutputP) && (targetFilter == filterCount))
                                    {
                                        result.sequenceId = countSequence;
                                        result.targetFilterRow = targetPeRow;
                                        return result;
                                    }
                                    qCount++;
                                } //a
                                if (f == targetPeRow)
                                {
                                    countSequence++;
                                }

                                filterCount++;
                            } //f
                            pqxA += maxA;
                            qCountCoveredByA += maxA;
                        } //pqxA
                        pCount++;
                    } //p
                    filterCountCoveredInGroup += maxF;
                } //gf
                filterCountCoveredAcrossGroup += numFiltersInGroup;
            } //gl
            countQCoveredByTQ += maxSizeOutputWidthTile;
        } //tq
        countPCoveredByTP += maxSizeOutputHeightTile;
    } //tp

    return result;
}
