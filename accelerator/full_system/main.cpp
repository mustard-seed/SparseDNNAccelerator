#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/cl.hpp"
#include "AOCLUtilsCpp/aocl_utils_cpp.hpp"
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

/*Limits on the buffer sizes
 * Assume the biggest test convolves a 32x32x64 tensor with a 128*32*32*64 tensor
 * Add a safety factor of 2
 * */
#define MAX_DRAM_BYTE_INPUT_ACTIVATION 131072
#define MAX_DRAM_BYTE_INPUT_ACTIVATION_SB_COUNT 8192
#define MAX_DRAM_BYTE_INPUT_WEIGHT 16777216
#define MAX_DRAM_BYTE_INPUT_WEIGHT_SB_COUNT 2048
#define MAX_DRAM_BYTE_INPUT_BIAS 2048
#define MAX_DRAM_BYTE_OUTPUT_ACTIVATION 262144
#define MAX_DRAM_BYTE_OUTPUT_ACTIVATION_SB_COUNT 8192

#define FRAC_WIDTH 4
#define INT_WIDTH 3
#define OUTPUT_INT_WIDTH 3

typedef
std::vector<cl_float, boost::alignment::aligned_allocator<cl_float, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_ushort>
t_aligned_float_vector;

typedef
std::vector<cl_short, boost::alignment::aligned_allocator<cl_short, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_short>
t_aligned_short_vector;

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

    //Command queues
    cl::CommandQueue clCQMemoryReader;
    cl::CommandQueue clCQOutputWriter;
    cl::CommandQueue clCQIATileController;
    cl::CommandQueue clCQOATileController;

    //The kernels
    cl::Kernel kernelMemoryReader;
    cl::Kernel kernelOutputWriter;
    cl::Kernel kernelIATileController;
    cl::Kernel KernelOATileController;

    //Buffer members associated with the memory reader kernel
    cl::Buffer bufferMemoryReaderWideWeights;
    cl::Buffer bufferMemoryReaderWeightSBCount;
    cl::Buffer bufferMemoryReaderWideInput;
    cl::Buffer bufferMemoryReaderInputSBCount;
    cl::Buffer bufferMemoryReaderBias;

    //Buffer members associated withthe output writer kernel
    cl::Buffer bufferMemoryWriterWideOutput;
    cl::Buffer bufferMemoryWriterOutputSBCount;

    void SetUp() override
    {
#ifdef C5SOC
        binaryFile = "fullSystem_aoc_release_hw.aocx";
#else
        binaryFile = "fullSystem_aoc_emulation.aocx";
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

        //Instantiate the host-side kernel objects
        kernelMemoryReader = cl::Kernel(program, "kernelMemoryReader", &status);
        aocl_utils_cpp::checkError(status, "Failed to create the memory reader kernel!");

        kernelOutputWriter = cl::Kernel(program, "kernelOutputWriter", &status);
        aocl_utils_cpp::checkError(status, "Failed to create kernelOutputWriter!");

        kernelIATileController = cl::Kernel(program, "kernelIATileController", &status);
        aocl_utils_cpp::checkError(status, "Failed to create kernelIATileController!");

        KernelOATileController = cl::Kernel(program, "kernelOATileController", &status);
        aocl_utils_cpp::checkError(status, "Failed to create kernelOATileController!");

        //Instantiate the command queues
        clCQMemoryReader = cl::CommandQueue(
                    clContext,
                    clDevice,
                    CL_QUEUE_PROFILING_ENABLE,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the command queue clCQMemoryReader!");

        clCQOutputWriter = cl::CommandQueue(
                    clContext,
                    clDevice,
                    CL_QUEUE_PROFILING_ENABLE,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the command queue clCQOutputWriter!");

        clCQIATileController = cl::CommandQueue(
                    clContext,
                    clDevice,
                    CL_QUEUE_PROFILING_ENABLE,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the command queue clCQIATileController!");

        clCQOATileController = cl::CommandQueue(
                    clContext,
                    clDevice,
                    CL_QUEUE_PROFILING_ENABLE,
                    &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the command queue clCQOATileController!");

        //Instantiate the buffers
        cl_ulong maxBufferSizeByte = clDevice.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE> (&status);
        aocl_utils_cpp::checkError(status, "Failed to query the maximum buffer size in bytes!");

        cl_ulong inputWeightBufferSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_WEIGHT ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_WEIGHT;
        std::cout <<"Setting the bufferMemoryReaderWideWeights buffer. Size: "<<inputWeightBufferSize<<" bytes."<<std::endl;
        bufferMemoryReaderWideWeights = cl::Buffer (
                        clContext,
                        CL_MEM_HOST_WRITE_ONLY|CL_MEM_READ_ONLY,
                        inputWeightBufferSize,
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferMemoryReaderWideWeights!");

        cl_ulong inputWeightSBSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_WEIGHT_SB_COUNT ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_WEIGHT_SB_COUNT;
        std::cout <<"Setting the bufferMemoryReaderWeightSBCount buffer. Size: "<<inputWeightSBSize<<" bytes."<<std::endl;
        bufferMemoryReaderWeightSBCount = cl::Buffer (
                        clContext,
                        CL_MEM_HOST_WRITE_ONLY|CL_MEM_READ_ONLY,
                        inputWeightSBSize,
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferMemoryReaderWeightSBCount!");

        cl_ulong inputActivationSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_ACTIVATION ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_ACTIVATION;
        std::cout <<"Setting the bufferMemoryReaderWideInput buffer. Size: "<<inputActivationSize<<" bytes."<<std::endl;
        bufferMemoryReaderWideInput = cl::Buffer (
                        clContext,
                        CL_MEM_HOST_WRITE_ONLY|CL_MEM_READ_ONLY,
                        inputActivationSize,
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferMemoryReaderWideInput!");

        cl_ulong inputActivationSBSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_ACTIVATION_SB_COUNT ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_ACTIVATION_SB_COUNT;
        std::cout <<"Setting the bufferMemoryReaderInputSBCount buffer. Size: "<<inputActivationSBSize<<" bytes."<<std::endl;
        bufferMemoryReaderInputSBCount = cl::Buffer (
                        clContext,
                        CL_MEM_HOST_WRITE_ONLY|CL_MEM_READ_ONLY,
                        inputActivationSBSize,
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferMemoryReaderInputSBCount!");

        cl_ulong inputBiasSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_BIAS ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_BIAS;
        std::cout <<"Setting the bufferMemoryReaderBias buffer. Size: "<<inputBiasSize<<" bytes."<<std::endl;
        bufferMemoryReaderBias = cl::Buffer (
                        clContext,
                        CL_MEM_HOST_WRITE_ONLY|CL_MEM_READ_ONLY,
                        inputBiasSize,
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferMemoryReaderBias!");

        cl_ulong outputActivationSize = maxBufferSizeByte < MAX_DRAM_BYTE_OUTPUT_ACTIVATION ? maxBufferSizeByte : MAX_DRAM_BYTE_OUTPUT_ACTIVATION;
        std::cout <<"Setting the bufferMemoryWriterWideOutput buffer. Size: "<<outputActivationSize<<" bytes."<<std::endl;
        bufferMemoryWriterWideOutput = cl::Buffer (
                        clContext,
                        CL_MEM_HOST_READ_ONLY|CL_MEM_WRITE_ONLY,
                        outputActivationSize,
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferMemoryWriterWideOutput!");

        cl_ulong outputActivtionSBSize = maxBufferSizeByte < MAX_DRAM_BYTE_OUTPUT_ACTIVATION_SB_COUNT ? maxBufferSizeByte : MAX_DRAM_BYTE_OUTPUT_ACTIVATION_SB_COUNT;
        std::cout <<"Setting the bufferMemoryWriterOutputSBCount buffer. Size: "<<outputActivtionSBSize<<" bytes."<<std::endl;
        bufferMemoryWriterOutputSBCount = cl::Buffer (
                        clContext,
                        CL_MEM_HOST_READ_ONLY|CL_MEM_WRITE_ONLY,
                        outputActivtionSBSize,
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferMemoryWriterOutputSBCount!");

        std::cout <<"AOCL setup compelete"<<std::endl;
    }

    /*!
     * \brief generateInputTensor
     * \details Generate a tensor of size inputHeight*inputWidth*numInputChannel.
     *          Along the width, every 1 strip out of widthBlockSize strips are filled with 1s. The rest are zeros
     * \param inputWidth
     * \param inputHeight
     * \param numInputChannel
     * \param widthBlockSize
     * \return The tensor generated
     */
    std::vector<fixedPointNumber> generateInputTensor (
                unsigned char inputWidth,
                unsigned char inputHeight,
                unsigned char numInputChannel,
                unsigned char widthBlockSize //1 out of withBlockSize strips along the width are filled with 1.3, the rest a zeros
            )
    {
        unsigned int numElements = inputWidth*inputHeight*numInputChannel;
        fixedPointNumber fpZero(0.0f, FRAC_WIDTH, INT_WIDTH);

        //Initialize the tensor and fill it with zero
        std::vector<fixedPointNumber> tensor (numElements, fpZero);

        //Set designated stripes to be filled with ones
        for (unsigned char iterHeight=0; iterHeight<inputHeight; iterHeight++)
        {
            for (unsigned char iterWidth=0; iterWidth<inputWidth; iterWidth += widthBlockSize)
            {
                unsigned int index = (iterHeight*inputWidth + iterWidth) * numInputChannel;
                for (unsigned char iterChannel=0; iterChannel<numInputChannel; iterChannel++)
                {
                    tensor.at(index++) = fixedPointNumber(1.3f, FRAC_WIDTH, INT_WIDTH);
                }
            }
        }
        return tensor;
    }

    /*!
     * \brief generateWeights
     * \details Generate a tensor of weights, following the OC * W * H * IC layout.
     *          W=H=kernelSize, IC=numInputChannel, OC=2*numInputChannel
     *          Filters 2k for k in [0, IC-1] are the identity filters for input channel k
     *          Filters 2k+1 for k in [0, IC-1] are filled with ones
     * \param kernelSize
     * \param numInputChannel
     * \return
     */
    std::vector<fixedPointNumber> generateWeights (
                unsigned char kernelSize,
                unsigned char numInputChannel
            )
    {
        int numWeightsInOneChannel = kernelSize*kernelSize;
        int numWeightsInOneFilter = numWeightsInOneChannel*numInputChannel;
        int numWeights = numWeightsInOneFilter*2*numInputChannel;
        int numOC = 2*numInputChannel;
        fixedPointNumber fpZero(0.0f, FRAC_WIDTH, INT_WIDTH);
        fixedPointNumber fpOne (1.0f, FRAC_WIDTH, INT_WIDTH);
        //Initialize the weight tensor
        std::vector<fixedPointNumber> wTensor (numWeights, fpZero);

        //Setting the all ones filters
        for (int iOC=1; iOC<numOC; iOC += 2)
        {
            int index = iOC * numWeightsInOneFilter;
            for (int i=0; i<numWeightsInOneFilter; i++)
            {
                wTensor.at(index++) = fpOne;
            }
        }

        //Setting the identify filters
        for (int iOC=0, iIC=0; iOC<numOC; iOC += 2, iIC++)
        {
            int index = iOC * numWeightsInOneFilter + iIC;
            for (int i=0; i<numWeightsInOneChannel; i++)
            {
                wTensor.at(index) = fpOne;
                index += numInputChannel;
            }
        }

        return wTensor;
    }

    void launch (
            unsigned char inputWidth,
            unsigned char inputHeight,
            unsigned char numInputChannel,
            unsigned char widthBlockSize, //1 out of withBlockSize strips along the width are filled with ones, the rest a zeros
            bool flagCompressOutput
            )
    {
        unsigned char kernelSize = 3;
        /* First, generate the dense, fixed point tensors
         * */
        std::cout <<"Preparing the test tensors."<<std::endl;
        std::vector<fixedPointNumber> inputTensorDense = generateInputTensor(inputWidth, inputHeight, numInputChannel, widthBlockSize);
        std::vector<fixedPointNumber> inputWeightDense = generateWeights(kernelSize, numInputChannel);
        t_aligned_short_vector biasVector (2*numInputChannel, 0x0);

        /* 2. Compress the test tensors
         * */
        std::cout <<"Compressing the test tensors."<<std::endl;
        unsigned short maxScalarIndexInChannelGroup = numInputChannel - 1;
        unsigned short maxClusterIndexInCompressionBlock = COMPRESSION_VEC_SIZE*TRANSFER_SIZE-1;
        unsigned short maxClusterIndexInTransferBlock = TRANSFER_SIZE-1;
        unsigned short maxScalarIndexInCluster = CLUSTER_SIZE-1;

        flexibleDirectCompressedTensor compressedInput (
                        inputTensorDense,
                        1, //_num3DTensors
                        numInputChannel,
                        inputWidth,
                        inputHeight,
                        maxScalarIndexInChannelGroup,
                        maxClusterIndexInCompressionBlock,
                        maxClusterIndexInTransferBlock,
                        maxScalarIndexInCluster,
                        false //isKernel
                    );

        flexibleDirectCompressedTensor compressedWeights (
                        inputWeightDense,
                        2*numInputChannel, //_num3DTensors
                        numInputChannel,
                        kernelSize, //width
                        kernelSize, //height
                        maxScalarIndexInChannelGroup,
                        maxClusterIndexInCompressionBlock,
                        maxClusterIndexInTransferBlock,
                        maxScalarIndexInCluster,
                        true //isKernel
                    );

        /* 3. Transfer buffer content
        */
        cl_int status;

        std::vector<cl::Event> launchDependencyList;

        //Transfer the input
        std::cout <<"Transfer the input activations "<<std::endl;
        {
            cl::Event event;
            auto numTransferBlocks = compressedInput.valueVector.size();
            auto sizeTransferBlockElement = sizeof(typeof(compressedInput.valueVector.at(0)));
            auto valueVectorSizeBytes = sizeTransferBlockElement * numTransferBlocks;

            std::cout <<"Transfering "<<valueVectorSizeBytes<<" bytes in to bufferMemoryReaderWideInput"<<std::endl;

            status = clCQMemoryReader.enqueueWriteBuffer(bufferMemoryReaderWideInput, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 valueVectorSizeBytes, //size
                                                 compressedInput.valueVector.data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the input activation vector");
            launchDependencyList.push_back(event);
        }

        //Transfer the input transfer block count
        std::cout <<"Transfer the input transfer block count "<<std::endl;
        {
            cl::Event event;
            auto numBlocks = compressedInput.streamBlockAddressVector.size();
            auto sizePerElement = sizeof(typeof(compressedInput.streamBlockAddressVector.at(0)));
            auto transferSizeBytes = numBlocks * sizePerElement;

            std::cout <<"Transfering "<<transferSizeBytes<<" bytes in to bufferMemoryReaderInputSBCount"<<std::endl;

            status = clCQMemoryReader.enqueueWriteBuffer(bufferMemoryReaderInputSBCount, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 transferSizeBytes, //size
                                                 compressedInput.streamBlockAddressVector.data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the input activation transfer block count vector");
            launchDependencyList.push_back(event);
        }

        //Transfer the weight
        std::cout <<"Transfer the weight "<<std::endl;
        {
            cl::Event event;
            auto numBlocks = compressedWeights.valueVector.size();
            auto sizePerElement = sizeof(typeof(compressedWeights.valueVector.at(0)));
            auto transferSizeBytes = numBlocks * sizePerElement;

            std::cout <<"Transfering "<<transferSizeBytes<<" bytes in to bufferMemoryReaderWideWeights"<<std::endl;

            status = clCQMemoryReader.enqueueWriteBuffer(bufferMemoryReaderWideWeights, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 transferSizeBytes, //size
                                                 compressedWeights.valueVector.data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the weight buffer.");
            launchDependencyList.push_back(event);
        }

        //Transfer the weight transfer block count
        std::cout <<"Transfer the weight block count "<<std::endl;
        {
            cl::Event event;
            auto numBlocks = compressedWeights.streamBlockAddressVector.size();
            auto sizePerElement = sizeof(typeof(compressedWeights.streamBlockAddressVector.at(0)));
            auto transferSizeBytes = numBlocks * sizePerElement;

            std::cout <<"Transfering "<<transferSizeBytes<<" bytes in to bufferMemoryReaderWeightSBCount"<<std::endl;

            status = clCQMemoryReader.enqueueWriteBuffer(bufferMemoryReaderWeightSBCount, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 transferSizeBytes, //size
                                                 compressedWeights.streamBlockAddressVector.data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the weight transfer block count vector");
            launchDependencyList.push_back(event);
        }

        //Transfer the bias vector
        std::cout <<"Transfer the bias vector"<<std::endl;
        {
            cl::Event event;
            auto numBlocks = biasVector.size();
            auto sizePerElement = sizeof(typeof(biasVector.at(0)));
            auto transferSizeBytes = numBlocks * sizePerElement;

            std::cout <<"Transfering "<<transferSizeBytes<<" bytes in to bufferMemoryReaderBias"<<std::endl;

            status = clCQMemoryReader.enqueueWriteBuffer(bufferMemoryReaderBias, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 transferSizeBytes, //size
                                                 biasVector.data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the bias vector");
            launchDependencyList.push_back(event);
        }

        std::cout <<"Compute the kernel arguments that can be computed from problem setup"<<std::endl;

        /*
         * Calculate derived parameters
         * */
        cl_ushort outputWidth = (cl_uchar) inputWidth;
        cl_uchar sizeOutputTileWidthPerColumnFull = 16;
        cl_ushort sizeOutputTileWidthFull = ((cl_uchar) sizeOutputTileWidthPerColumnFull) * PE_COLS;
        cl_uchar sizeOutputTileWidthPerColumnPartial = outputWidth - (outputWidth / sizeOutputTileWidthFull) * sizeOutputTileWidthFull;
        cl_short sizeOutputTileWidthPartial = sizeOutputTileWidthPerColumnPartial;
        cl_uchar numPartialColumns = 1;
        cl_uchar numOutputWidthTile = 1 + (outputWidth-1) / sizeOutputTileWidthFull;
        cl_uchar numOutputWidthFullTile = outputWidth / sizeOutputTileWidthFull;
        cl_ushort sizeInputTileWidthFull = sizeOutputTileWidthFull-1+kernelSize; //Stride is 1;
        cl_ushort sizeInputTileWidthPartial = sizeOutputTileWidthPartial-1+kernelSize; //Stride is 1;
        cl_uchar sizeInputTileWidthPerColumnFull = sizeOutputTileWidthPerColumnFull - 1 + kernelSize;
        cl_uchar sizeInputTileWidthPerColumnPartial = sizeOutputTileWidthPerColumnPartial - 1 + kernelSize;
        cl_ushort strideInputTileWidthFull = sizeOutputTileWidthFull;
        cl_ushort strideInputTileWidthPartial = sizeOutputTileWidthPartial;

        cl_ushort outputHeight = (cl_uchar) inputHeight;
        cl_uchar sizeOutputHeightTileFull = 16;
        cl_uchar sizeOutputHeightTilePartial = outputHeight - (outputHeight / sizeOutputHeightTileFull) * sizeOutputHeightTileFull;
        cl_uchar numOutputHightTile = 1 + (outputHeight-1) / sizeOutputHeightTileFull;
        cl_uchar numOutputHeightFullTile = outputHeight / sizeOutputHeightTileFull;
        cl_ushort sizeInputTileHeightFull = sizeOutputHeightTileFull - 1 + kernelSize;
        cl_ushort sizeInputTileHeightPartial = sizeOutputHeightTilePartial - 1 + kernelSize;
        cl_ushort strideInputTileHeightFull = sizeOutputHeightTileFull;
        cl_ushort strideInputTileHeightPartial = sizeOutputHeightTilePartial;

        cl_ushort numOutputTiles = numOutputHightTile * numOutputWidthTile;


        std::cout <<"Setting kernel arguments"<<std::endl;


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

//#define PLAY
TEST_F (testFixture, play) {


}


int main(int argc, char* argv[]) {

    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
