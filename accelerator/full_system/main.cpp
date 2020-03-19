#define CL_TARGET_OPENCL_VERSION 200

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
#include <memory> //smart pointers

#include <chrono>

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

#define WEIGHT_SEED 1234
#define INPUT_SEED   7653

//#define PROFILE
//#define PLAY
//#define SPARSE_LEVEL_TEST
#define TEST_TYPE TEST
#define REPEAT 1
#define EMULATE

#if defined(C5SOC) //Hack for ARMv7, otherwise chrono won't work
__asm__(".symver _ZNSt6chrono3_V212system_clock3nowEv,_ZNSt6chrono12system_clock3nowEv@GLIBCXX_3.4.11");
#endif

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

typedef enum {TEST, FULL, ZERO, DENSE, SPARSE} e_tensor_type;


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
#if defined(SPARSE_SYSTEM)
    cl::Buffer bufferMemoryReaderWeightSBCount;
#endif
    cl::Buffer bufferMemoryReaderWideInput;
#if defined(SPARSE_SYSTEM)
    cl::Buffer bufferMemoryReaderInputSBCount;
#endif
    cl::Buffer bufferMemoryReaderBias;

    //Buffer members associated withthe output writer kernel
    cl::Buffer bufferMemoryWriterWideOutput;
#if defined(SPARSE_SYSTEM)
    cl::Buffer bufferMemoryWriterOutputSBCount;
#endif

    void SetUp() override
    {
       cl_int status = CL_SUCCESS;
#ifdef C5SOC
        binaryFile = "device_utils.aocx";
        clPlatform = aocl_utils_cpp::findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
#else
        binaryFile = "smallBuffer.aocx";
#if defined(EMULATE)
        clPlatform = aocl_utils_cpp::findPlatform("Intel(R) FPGA Emulation Platform for OpenCL(TM)");
#else
        clPlatform = aocl_utils_cpp::findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
#endif
#endif

        //Setup and platform and the context
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
                        CL_MEM_READ_ONLY,
                        inputWeightBufferSize,
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferMemoryReaderWideWeights!");

#if defined(SPARSE_SYSTEM)
        cl_ulong inputWeightSBSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_WEIGHT_SB_COUNT ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_WEIGHT_SB_COUNT;
        std::cout <<"Setting the bufferMemoryReaderWeightSBCount buffer. Size: "<<inputWeightSBSize<<" bytes."<<std::endl;
        bufferMemoryReaderWeightSBCount = cl::Buffer (
                        clContext,
                        CL_MEM_READ_ONLY,
                        inputWeightSBSize,
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferMemoryReaderWeightSBCount!");
#endif
        cl_ulong inputActivationSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_ACTIVATION ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_ACTIVATION;
        std::cout <<"Setting the bufferMemoryReaderWideInput buffer. Size: "<<inputActivationSize<<" bytes."<<std::endl;
        bufferMemoryReaderWideInput = cl::Buffer (
                        clContext,
                        CL_MEM_READ_ONLY,
                        inputActivationSize,
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferMemoryReaderWideInput!");
#if defined(SPARSE_SYSTEM)
        cl_ulong inputActivationSBSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_ACTIVATION_SB_COUNT ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_ACTIVATION_SB_COUNT;
        std::cout <<"Setting the bufferMemoryReaderInputSBCount buffer. Size: "<<inputActivationSBSize<<" bytes."<<std::endl;
        bufferMemoryReaderInputSBCount = cl::Buffer (
                        clContext,
                        CL_MEM_READ_ONLY,
                        inputActivationSBSize,
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferMemoryReaderInputSBCount!");
#endif
        cl_ulong inputBiasSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_BIAS ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_BIAS;
        std::cout <<"Setting the bufferMemoryReaderBias buffer. Size: "<<inputBiasSize<<" bytes."<<std::endl;
        bufferMemoryReaderBias = cl::Buffer (
                        clContext,
                        CL_MEM_READ_ONLY,
                        inputBiasSize,
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferMemoryReaderBias!");

        cl_ulong outputActivationSize = maxBufferSizeByte < MAX_DRAM_BYTE_OUTPUT_ACTIVATION ? maxBufferSizeByte : MAX_DRAM_BYTE_OUTPUT_ACTIVATION;
        std::cout <<"Setting the bufferMemoryWriterWideOutput buffer. Size: "<<outputActivationSize<<" bytes."<<std::endl;
        bufferMemoryWriterWideOutput = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        outputActivationSize,
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferMemoryWriterWideOutput!");
#if defined(SPARSE_SYSTEM)
        cl_ulong outputActivtionSBSize = maxBufferSizeByte < MAX_DRAM_BYTE_OUTPUT_ACTIVATION_SB_COUNT ? maxBufferSizeByte : MAX_DRAM_BYTE_OUTPUT_ACTIVATION_SB_COUNT;
        std::cout <<"Setting the bufferMemoryWriterOutputSBCount buffer. Size: "<<outputActivtionSBSize<<" bytes."<<std::endl;
        bufferMemoryWriterOutputSBCount = cl::Buffer (
                        clContext,
                        CL_MEM_WRITE_ONLY,
                        outputActivtionSBSize,
                        NULL,
                        &status
                    );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer bufferMemoryWriterOutputSBCount!");
#endif
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
    //TODO: think of better test case?
    std::vector<fixedPointNumber> generateInputTensor (
                unsigned char inputWidth,
                unsigned char inputHeight,
                unsigned char numInputChannel,
                unsigned char widthBlockSize, //1 out of withBlockSize strips along the width are filled with non-zero number
                e_tensor_type eTensorType,
                float denseProb = 1.0f
            )
    {
        unsigned int numElements = inputWidth*inputHeight*numInputChannel;
        fixedPointNumber fpZero(0.0f, FRAC_WIDTH, INT_WIDTH);
        fixedPointNumber positiveNumber(1.5f, FRAC_WIDTH, INT_WIDTH);
        fixedPointNumber negativeNumber(-1.5f, FRAC_WIDTH, INT_WIDTH);

        //Initialize the tensor and fill it with zero
        std::vector<fixedPointNumber> tensor (numElements, fpZero);

        switch (eTensorType) {
        case TEST:
            {
                //Flag for writing positive or negative number
                bool writePositive = true;

                //Set designated stripes to be filled with ones
                for (unsigned char iterHeight=0; iterHeight<inputHeight; iterHeight++)
                {
                    for (unsigned char iterWidth=0; iterWidth<inputWidth; iterWidth += widthBlockSize)
                    {
                        unsigned int index = (iterHeight*inputWidth + iterWidth) * numInputChannel;
                        for (unsigned char iterChannel=0; iterChannel<numInputChannel; iterChannel++)
                        {
                            tensor.at(index++) = writePositive ? positiveNumber : negativeNumber;
                        }
                        //Invert the flag
                        writePositive = !writePositive;
                    }
                }
            }
            break;
        case FULL:
            {
                for (int i=0; i<numElements; i++)
                {
                    tensor.at(i) = positiveNumber;
                }
            }
            break;
        case SPARSE:
            {
                std::mt19937 generator(WEIGHT_SEED);
                std::bernoulli_distribution bernDistribution(denseProb);
                bool writePositive = true;
                for (unsigned char iterHeight=0; iterHeight<inputHeight; iterHeight++)
                {
                    for (unsigned char iterWidth=0; iterWidth<inputWidth; iterWidth ++)
                    {
                        unsigned int index = (iterHeight*inputWidth + iterWidth) * numInputChannel;
                        for (unsigned char iterChannel=0; iterChannel<numInputChannel; iterChannel++)
                        {
                            if (bernDistribution(generator))
                            {
                                tensor.at(index++) = writePositive ? positiveNumber : negativeNumber;
                            }
                        }
                        //Invert the flag
                        writePositive = !writePositive;
                    }
                }
            }
            break;
        default:
            break;
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
                unsigned char numInputChannel,
                e_tensor_type eTensorType,
                float denseProb = 1.0f
            )
    {
        assert(kernelSize%2==1);
        int numWeightsInOneChannel = kernelSize*kernelSize;
        int numWeightsInOneFilter = numWeightsInOneChannel*numInputChannel;
        int numWeights = numWeightsInOneFilter*2*numInputChannel;
        int numOC = 2*numInputChannel;
        fixedPointNumber fpZero(0.0f, FRAC_WIDTH, INT_WIDTH);
        fixedPointNumber fpOne (1.0f, FRAC_WIDTH, INT_WIDTH);
        fixedPointNumber fpNegative (-1.0f, FRAC_WIDTH, INT_WIDTH);
        //Initialize the weight tensor
        std::vector<fixedPointNumber> wTensor (numWeights, fpZero);

        switch (eTensorType) {
        case TEST:
            {
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
                    int index = iOC * numWeightsInOneFilter + numInputChannel*((kernelSize / 2)*kernelSize+kernelSize/2) + iIC;
                    //for (int i=0; i<numWeightsInOneChannel; i++)
                    //{
                        wTensor.at(index) = fpOne;
                        //index += numInputChannel;
                    //}
                }
            }
            break;
        case FULL:
            {
                for (int i=0; i<numWeights; i++)
                {
                    wTensor.at(i) = fpOne;
                }
            }
            break;
        case SPARSE:
            {
                std::mt19937 generator(INPUT_SEED);
                std::bernoulli_distribution bernDistribution(denseProb);
                bool writePositive = true;
                for (auto &val : wTensor)
                {
                    if (bernDistribution(generator) == true)
                    {
                        val = writePositive ? fpOne : fpNegative;
                    }
                    writePositive = !writePositive;
                }
            }
            break;
        default:
            break;
        }

        return wTensor;
    }

    void launch (
            unsigned char _inputWidth,
            unsigned char _inputHeight,
            unsigned char _numInputChannel,
            unsigned char _widthBlockSize, //1 out of widthBlockSize strips along the width are filled with ones, the rest a zeros
            unsigned char _sizeOutputTileWidthPerColFull,
            unsigned char _sizeOutputTileHeight,
            bool _flagEnableRelu,
            e_tensor_type eTensorType,
            float _bias = 2.0f,
            bool _flagCompressionOutput = true,
            float denseProb = 1.0f
            )
    {
        /* Fixed parameters
         * */
        //bool _flagCompressionOutput = true;
        cl_uchar kernelSize = 3;
        cl_uchar stride = 1;
        /* First, generate the dense, fixed point tensors
         * */
        assert(_numInputChannel <= 127);
        std::cout <<"1. Preparing the test tensors. Type: "<<eTensorType<<std::endl;
        std::vector<fixedPointNumber> inputTensorDense = generateInputTensor(_inputWidth, _inputHeight, _numInputChannel, _widthBlockSize, eTensorType, denseProb);
        std::vector<fixedPointNumber> inputWeightDense = generateWeights((unsigned char) kernelSize, _numInputChannel, eTensorType, denseProb);
        t_accumulator fixedBias = (t_accumulator) (round(_bias * (float) (1 << (FRAC_WIDTH + FRAC_WIDTH)) ));
        t_aligned_short_vector biasVector (2*_numInputChannel, fixedBias);

        /* 2. Compress the test tensors
         * */
        std::cout <<"2. Preparing the test tensors. Compression flag is: "<<eTensorType<<std::endl;
        std::cout <<"Input width:  "<<(unsigned int) _inputWidth<<std::endl
                  <<"Input height: "<<(unsigned int) _inputHeight<<std::endl
                  <<"Input channels: "<<(unsigned int) _numInputChannel<<std::endl
                  <<"Output channels: "<<(unsigned int)(2 * _numInputChannel)<<std::endl;
        if (eTensorType == SPARSE)
        {
               std::cout <<"Sparsity level is "<<(1.0f - denseProb)<<std::endl;
        }
        unsigned short maxScalarIndexInChannelGroup = _numInputChannel - 1;
        unsigned short maxClusterIndexInCompressionBlock = COMPRESSION_WINDOW_SIZE-1;
        unsigned short maxClusterIndexInTransferBlock = TRANSFER_SIZE-1;
        unsigned short maxScalarIndexInCluster = CLUSTER_SIZE-1;

        std::unique_ptr<AlignedTensor> pInput, pWeights, pOutput;

#if defined(SPARSE_SYSTEM)
                pInput.reset(new FlexibleDirectCompressedTensor(
                                inputTensorDense,
                                1, //_num3DTensors
                                _numInputChannel,
                                _inputWidth,
                                _inputHeight,
                                maxScalarIndexInChannelGroup,
                                maxClusterIndexInCompressionBlock,
                                maxClusterIndexInTransferBlock,
                                maxScalarIndexInCluster,
                                false //isKernel
                            ) );
                pWeights.reset(new FlexibleDirectCompressedTensor (
                            inputWeightDense,
                            2*_numInputChannel, //_num3DTensors
                            _numInputChannel,
                            (unsigned char) kernelSize, //width
                            (unsigned char) kernelSize, //height
                            maxScalarIndexInChannelGroup,
                            maxClusterIndexInCompressionBlock,
                            maxClusterIndexInTransferBlock,
                            maxScalarIndexInCluster,
                            true //isKernel
                        ) );
                if (_flagCompressionOutput == true)
                {
                    pOutput.reset(new FlexibleDirectCompressedTensor(
                                1, //_num3DTensors
                                2*_numInputChannel, //_channel
                                _inputWidth, //_width
                                _inputHeight, //_height
                                2*_numInputChannel-1, //_maxScalarIndexInChannelGroup
                                maxClusterIndexInCompressionBlock,
                                maxClusterIndexInTransferBlock,
                                maxScalarIndexInCluster,
                                false //isKernel
                                ) );
                }
                else
                {
                    pOutput.reset( new AlignedTensor(
                                1, //_num3DTensors
                                2*_numInputChannel, //_channel
                                _inputWidth, //_width
                                _inputHeight, //_height
                                2*_numInputChannel-1, //_maxScalarIndexInChannelGroup
                                maxClusterIndexInTransferBlock,
                                maxScalarIndexInCluster,
                                false //isKernel
                                ));
                }

#else
                pInput.reset( new AlignedTensor(
                                inputTensorDense,
                                1, //_num3DTensors
                                _numInputChannel,
                                _inputWidth,
                                _inputHeight,
                                maxScalarIndexInChannelGroup,
                                maxClusterIndexInTransferBlock,
                                maxScalarIndexInCluster,
                                false //isKernel
                            ) );
                pWeights.reset( new AlignedTensor (
                            inputWeightDense,
                            2*_numInputChannel, //_num3DTensors
                            _numInputChannel,
                            (unsigned char) kernelSize, //width
                            (unsigned char) kernelSize, //height
                            maxScalarIndexInChannelGroup,
                            maxClusterIndexInTransferBlock,
                            maxScalarIndexInCluster,
                            true //isKernel
                        ));
                pOutput.reset( new AlignedTensor(
                            1, //_num3DTensors
                            2*_numInputChannel, //_channel
                            _inputWidth, //_width
                            _inputHeight, //_height
                            2*_numInputChannel-1, //_maxScalarIndexInChannelGroup
                            maxClusterIndexInTransferBlock,
                            maxScalarIndexInCluster,
                            false //isKernel
                            ));
#endif
        std::cout <<"3. Compute the kernel arguments that can be computed from problem setup"<<std::endl;

        /*
         * 3. Calculate derived parameters
         * */
        cl_uint strideExternalMemoryWeights = pWeights->getExternalMemoryAddressStride();
        cl_uint strideExternalMemoryIA = pInput->getExternalMemoryAddressStride();
        cl_ushort strideStripIACache = strideExternalMemoryIA / WIDE_SIZE;
        cl_uint strideExternalMemoryOA = (pOutput->getExternalMemoryAddressStride()) >> WIDE_SIZE_OFFSET;

        cl_ushort outputWidth = (cl_ushort) _inputWidth;
        cl_uchar sizeOutputTileWidthPerColumnFull = (cl_ushort) _sizeOutputTileWidthPerColFull;
        cl_ushort sizeOutputTileWidthFull = ((cl_ushort) sizeOutputTileWidthPerColumnFull) * PE_COLS;
        cl_uchar sizeOutputTileWidthPerColumnPartial = outputWidth - (outputWidth / sizeOutputTileWidthFull) * sizeOutputTileWidthFull;
        cl_short sizeOutputTileWidthPartial = sizeOutputTileWidthPerColumnPartial; //Lump all the remaining one to 1 column
        cl_uchar numPartialColumns = 1;
        cl_uchar numOutputWidthTile = 1 + (outputWidth-1) / sizeOutputTileWidthFull;
        cl_uchar numOutputWidthFullTile = outputWidth / sizeOutputTileWidthFull;
        cl_ushort sizeInputTileWidthFull = sizeOutputTileWidthFull-1+kernelSize; //Stride is 1;
        cl_ushort sizeInputTileWidthPartial = sizeOutputTileWidthPartial-1+kernelSize; //Stride is 1;
        cl_uchar sizeInputTileWidthPerColumnFull = sizeOutputTileWidthPerColumnFull - 1 + kernelSize;
        cl_uchar sizeInputTileWidthPerColumnPartial = sizeOutputTileWidthPerColumnPartial - 1 + kernelSize;
        cl_ushort strideInputTileWidthFull = sizeOutputTileWidthFull;
        cl_ushort strideInputTileWidthPartial = sizeOutputTileWidthPartial;

        cl_ushort outputHeight = (cl_ushort) _inputHeight;
        cl_uchar sizeOutputHeightTileFull = (cl_ushort) _sizeOutputTileHeight;
        cl_uchar sizeOutputHeightTilePartial = outputHeight - (outputHeight / sizeOutputHeightTileFull) * sizeOutputHeightTileFull;
        cl_uchar numOutputHeightTile = 1 + (outputHeight-1) / sizeOutputHeightTileFull;
        cl_uchar numOutputHeightFullTile = outputHeight / sizeOutputHeightTileFull;
        cl_ushort sizeInputTileHeightFull = sizeOutputHeightTileFull - 1 + kernelSize;
        cl_ushort sizeInputTileHeightPartial = sizeOutputHeightTilePartial - 1 + kernelSize;
        cl_ushort strideInputTileHeightFull = sizeOutputHeightTileFull;
        cl_ushort strideInputTileHeightPartial = sizeOutputHeightTilePartial;

        cl_ushort numOutputTiles = numOutputHeightTile * numOutputWidthTile;

        cl_uchar horizontalBorderPadding = 1;
        cl_uchar verticalBorderPadding = 1;
        cl_uchar horizontalStridedPaddingShift = 0;
        cl_uchar horizontalStridedPaddingRemainderMask = 0x0;
        cl_uchar verticalStridedPaddingShift = 0;
        cl_uchar verticalStridedPaddingRemainderMask = 0x0;

        cl_ushort numFiltersInKernel = 2* _numInputChannel;
        cl_char numGroups = 1; //Don't change, even though this is very inefficient and does not really change grouped convolution
        cl_ushort numFiltersInGroup = numFiltersInKernel;
        cl_ushort numFilterFoldsInGroup = 1 + (numFiltersInGroup-1) / PE_ROWS;
        cl_ushort numFullFilterFoldsInGroup = numFiltersInGroup / PE_ROWS;
        cl_uchar numActiveRowsPartialFold = numFiltersInGroup % PE_ROWS;
        cl_ushort numCompressionWindowsInputGroup = 1 + (_numInputChannel-1) / COMPRESSION_WINDOW_SIZE / CLUSTER_SIZE;

        cl_ushort numInputGroupxTiles = numOutputTiles;
        cl_ushort numOutputGroupxTiles = numOutputTiles;

        cl_uchar enableRelu = _flagEnableRelu ? 0X1 : 0X0;
        cl_uchar enableSparsification = _flagCompressionOutput ? 0X1 : 0X0;

        //Only used for the dense case
        cl_ushort numTBCountPerFilter = (cl_ushort) ( (1 + (_numInputChannel-1) / (TRANSFER_SIZE * CLUSTER_SIZE))*kernelSize*kernelSize);
        cl_ushort numTBCountPerIAStrip = (cl_ushort) ( 1 + (_numInputChannel-1) / (TRANSFER_SIZE * CLUSTER_SIZE) );
        cl_ushort numWideCountPerOAStrip = (cl_ushort) ( 1 + (numFiltersInKernel-1) / (TRANSFER_SIZE * CLUSTER_SIZE * WIDE_SIZE) );

        std::cout <<"4. Setting kernel arguments for the input reader."<<std::endl;
        {
            //volatile __global t_dram_block* restrict pDramWeights
            kernelMemoryReader.setArg(0, bufferMemoryReaderWideWeights);

            //Pointer to filter transfer block count
            //volatile __global t_streamblock_address* restrict pFilterStreamBlockAddress,
#if defined(SPARSE_SYSTEM)
            kernelMemoryReader.setArg(1, bufferMemoryReaderWeightSBCount);
#else
            kernelMemoryReader.setArg(1, numTBCountPerFilter);
#endif
            //Pointer to input activations
            //volatile __global t_dram_block* restrict pInputActivation,
            kernelMemoryReader.setArg(2, bufferMemoryReaderWideInput);

            //Pointer to input activation transfer block count
            //volatile __global t_streamblock_address* restrict pIAStreamBlockAddress,
#if defined(SPARSE_SYSTEM)
            kernelMemoryReader.setArg(3, bufferMemoryReaderInputSBCount);
#else
            kernelMemoryReader.setArg(3, numTBCountPerIAStrip);
#endif

            //Pointer to bias
            //volatile __global t_accumulator* restrict pBias,
            kernelMemoryReader.setArg(4, bufferMemoryReaderBias);

            //Distance between the start of successive X-Y strip/filters in DRAM in terms of transfer blocks
            //unsigned int strideExternalMemoryWeights,
            kernelMemoryReader.setArg(5, strideExternalMemoryWeights);

            //unsigned int strideExternalMemoryIA,
            kernelMemoryReader.setArg(6, strideExternalMemoryIA);

            /*
            Output width tiling parameters
            */
            // unsigned short outputWidth,
            kernelMemoryReader.setArg(7, outputWidth);

            //unsigned char sizeOutputTileWidthPerColumnFull, //TQ_A
            kernelMemoryReader.setArg(8, sizeOutputTileWidthPerColumnFull);

            //unsigned short sizeOutputTileWidthFull, //TQ_A * PE_COLS
            kernelMemoryReader.setArg(9, sizeOutputTileWidthFull);

            //unsigned char sizeOutputTileWidthPerColumnPartial, //Output tile width per column for the final few columns
            kernelMemoryReader.setArg(10, sizeOutputTileWidthPerColumnPartial);

            //unsigned short sizeOutputTileWidthPartial, //partialTQ_A * APartial
            kernelMemoryReader.setArg(11, sizeOutputTileWidthPartial);

            //unsigned char numPartialColumns, //APartial
            kernelMemoryReader.setArg(12, numPartialColumns);

            //unsigned char numOutputWidthTile, //ceil (Q / (TQ_A * PE_COLS))
            kernelMemoryReader.setArg(13, numOutputWidthTile);

            //unsigned char numOutputWidthFullTile, // floor (Q / (TQ_A * PE_COLS))
            kernelMemoryReader.setArg(14, numOutputWidthFullTile);

            //unsigned short sizeInputTileWidthFull, // (sizeOutputTileWidthFull - 1)*stride + kernelSize
            kernelMemoryReader.setArg(15, sizeInputTileWidthFull);

            //unsigned short sizeInputTileWidthPartial, // (sizeOutputTileWidthPartial - 1)*stride + kernelSize
            kernelMemoryReader.setArg(16 , sizeInputTileWidthPartial);

            //unsigned char sizeInputTileWidthPerColumnFull, // (sizeOutputTileWidthPerColumnFull - 1)*stride + kernelSize
            kernelMemoryReader.setArg(17 , sizeInputTileWidthPerColumnFull);

            //unsigned char sizeInputTileWidthPerColumnPartial, // (sizeOutputTileWidthPerColumnPartial - 1)*stride + kernelSize
            kernelMemoryReader.setArg(18 , sizeInputTileWidthPerColumnPartial);

            //unsigned short strideInputTileWidthFull, //sizeOutputTileWidthFull * stride
            kernelMemoryReader.setArg(19 , strideInputTileWidthFull);

            //unsigned short strideInputTileWidthPartial, //sizeOutputTileWidthPartial * stride
            kernelMemoryReader.setArg(20 , strideInputTileWidthPartial);

            /*
            Output height tiling parameters
            */
            //unsigned short outputHeight, //P
            kernelMemoryReader.setArg(21 , outputHeight);

            //unsigned char sizeOutputHeightTileFull, //TP
            kernelMemoryReader.setArg(22 , sizeOutputHeightTileFull);

            //unsigned char sizeOutputHeightTilePartial, //P mod TP
            kernelMemoryReader.setArg(23 , sizeOutputHeightTilePartial);

            //unsigned char numOutputHightTile, //ceil (P / TP)
            kernelMemoryReader.setArg(24 , numOutputHeightTile);

            //unsigned char numOutputHeightFullTile, // floor (P / TP)
            kernelMemoryReader.setArg(25 , numOutputHeightFullTile);

            //unsigned short sizeInputTileHeightFull, // (sizeOutputTileHeightFull - 1)*stride + kernelSize
            kernelMemoryReader.setArg(26 , sizeInputTileHeightFull);

            //unsigned short sizeInputTileHeightPartial, // (sizeOutputTileHeightPartial - 1)*stride + kernelSize
            kernelMemoryReader.setArg(27 , sizeInputTileHeightPartial);

            //unsigned short strideInputTileHeightFull, //sizeOutputHeightTileFull * stride
            kernelMemoryReader.setArg(28 , strideInputTileHeightFull);

            //unsigned short strideInputTileHeightPartial, //sizeOutputHeightTilePartial * stride
            kernelMemoryReader.setArg(29 , strideInputTileHeightPartial);

            //unsigned short numOutputTiles, //numOutputHeightTile * numOutputWidthTile
            kernelMemoryReader.setArg(30 , numOutputTiles);

            /*
            Input X-Y dimensions
            Without padding around or between input elements
            */
            //unsigned short inputWidth,
            kernelMemoryReader.setArg(31 , (cl_ushort) _inputWidth);

            //unsigned short inputHeight,
            kernelMemoryReader.setArg(32 , (cl_ushort) _inputHeight);

            //Stride between successive strips of IA in terms of dram block
            //unsigned short strideStripIACache, //Stride in terms of dram block
            kernelMemoryReader.setArg(33 , (cl_ushort) strideStripIACache);

            /*
            Paddings.
            Assume border paddings are symmetrical
            Stride padding is for transpose convolution
            index_in_zero_padded_tensor - padding >> stridePaddingShift = actual index
            index_in_zero_padded_tensor - padding & stridePaddingRemainderMask == 0x0 => is actual index
            */
            //unsigned char horizontalBorderPadding,
            kernelMemoryReader.setArg(34 , horizontalBorderPadding );

            //unsigned char verticalBorderPadding,
            kernelMemoryReader.setArg(35 , verticalBorderPadding );

            //unsigned char horizontalStridedPaddingShift,
            kernelMemoryReader.setArg(36 , horizontalStridedPaddingShift);

            //unsigned char horizontalStridedPaddingRemainderMask,
            kernelMemoryReader.setArg(37 , horizontalStridedPaddingRemainderMask);

            //unsigned char verticalStridedPaddingShift,
            kernelMemoryReader.setArg(38 , verticalStridedPaddingShift);

            //unsigned char verticalStridedPaddingRemainderMask,
            kernelMemoryReader.setArg(39 , verticalStridedPaddingRemainderMask);

            /*
            Stride and kernel sizes.
            For transpose convolution, the stride is 1
            */
            //unsigned char kernelSize,
            kernelMemoryReader.setArg(40 , kernelSize);

            //unsigned char stride,
            kernelMemoryReader.setArg(41 , (cl_uchar) 1);

            /*
            Input and output channels
            */
            //unsigned short numFiltersInKernel, //L
            kernelMemoryReader.setArg(42 , numFiltersInKernel);

            //unsigned char numGroups, // L / G
            kernelMemoryReader.setArg(43 , numGroups);

            //unsigned short numFiltersInGroup, // G
            kernelMemoryReader.setArg(44 , numFiltersInGroup);

            //unsigned short numFilterFoldsInGroup, //ceil(numFiltersInGroup / PE_ROWS)
            kernelMemoryReader.setArg(45 , numFilterFoldsInGroup);

            //unsigned short numFullFilterFoldsInGroup,
            kernelMemoryReader.setArg(46 , numFullFilterFoldsInGroup);

            //unsigned char numActiveRowsPartialFold,
            kernelMemoryReader.setArg(47 , numActiveRowsPartialFold);

            //unsigned short numCompressionWindowsInputGroup
            kernelMemoryReader.setArg(48 , numCompressionWindowsInputGroup);

            //unsigned short kernelSizexNumFilterFoldsInGroup
            kernelMemoryReader.setArg(49 , cl_short ((cl_ushort)kernelSize*numFilterFoldsInGroup));
        }

        std::cout <<"5. Setting kernel arguments for the output writer."<<std::endl;
        {
            //Pointer to the output activation
            //volatile __global t_output_dram_block* restrict pOutputActivation,
            kernelOutputWriter.setArg(0 , bufferMemoryWriterWideOutput);

            //Pointer to the output activation transfer block count
            //volatile __global t_streamblock_address* restrict pOAStreamBlockAddress,
#if defined(SPARSE_SYSTEM)
            kernelOutputWriter.setArg(1 , bufferMemoryWriterOutputSBCount);
#else
            kernelOutputWriter.setArg(1 , numWideCountPerOAStrip);
#endif

            //unsigned int strideExternalMemoryOA, //In terms of output dram block
            kernelOutputWriter.setArg(2 , strideExternalMemoryOA);
            /*
            Output width tiling parameters
            */
            //unsigned short outputWidth, //Q
            kernelOutputWriter.setArg(3 , outputWidth);
            //unsigned char sizeOutputTileWidthPerColumnFull, //TQ_A
            kernelOutputWriter.setArg(4 , sizeOutputTileWidthPerColumnFull);
            //unsigned short sizeOutputTileWidthFull, //TQ_A * PE_COLS
            kernelOutputWriter.setArg(5 , sizeOutputTileWidthFull);
            //unsigned char sizeOutputTileWidthPerColumnPartial, //Output tile width per column for the final few columns
            kernelOutputWriter.setArg(6 , sizeOutputTileWidthPerColumnPartial);
            //unsigned short sizeOutputTileWidthPartial, //partialTQ_A * APartial
            kernelOutputWriter.setArg(7 , sizeOutputTileWidthPartial);
            //unsigned char numPartialColumns, //APartial
            kernelOutputWriter.setArg(8 , numPartialColumns);
            //unsigned char numOutputWidthTile, //ceil (Q / (TQ_A * PE_COLS))
            kernelOutputWriter.setArg(9 , numOutputWidthTile);
            //unsigned char numOutputWidthFullTile, // floor (Q / (TQ_A * PE_COLS))
            kernelOutputWriter.setArg(10 , numOutputWidthFullTile);
            /*
            Output height tiling parameters
            */
            //unsigned short outputHeight, //P
            kernelOutputWriter.setArg(11 , outputHeight);
            //unsigned char sizeOutputHeightTileFull, //TP
            kernelOutputWriter.setArg(12 , sizeOutputHeightTileFull);
            //unsigned char sizeOutputHeightTilePartial, //P mod TP
            kernelOutputWriter.setArg(13 , sizeOutputHeightTilePartial);
            //unsigned char numOutputHightTile, //ceil (P / TP)
            kernelOutputWriter.setArg(14 , numOutputHeightTile);
            //unsigned char numOutputHeightFullTile, // floor (P / TP)
            kernelOutputWriter.setArg(15 , numOutputHeightFullTile);
            /*
            Auxillary
            */
            //unsigned short numOutputHxWTiles, //numOutputHeightTile * numOutputWidthTile
            kernelOutputWriter.setArg(16 , numOutputTiles);
            //unsigned short numOutputHxW,
            kernelOutputWriter.setArg(17 , (cl_ushort) (((cl_ushort) (outputWidth)) * ((cl_ushort) (outputHeight))) );
            //Number of groups in the output activations
            //unsigned short numOutputChannels,
            kernelOutputWriter.setArg(18 , numFiltersInKernel);
            //unsigned short numGroupsCurrentLayer,
            kernelOutputWriter.setArg(19 , (cl_ushort) 1);
            //unsigned short numChannelsPerGroupCurrentLayer,
            kernelOutputWriter.setArg(20 , numFiltersInKernel);
            //unsigned short numGroupsNextLayer,
            kernelOutputWriter.setArg(21 , (cl_ushort) 1);
            //unsigned short numChannelsPerGroupNextLayer,
            kernelOutputWriter.setArg(22 , numFiltersInKernel);
            //unsigned char numFoldsInGroupCurrentLayer,
            kernelOutputWriter.setArg(23 , numFilterFoldsInGroup);
            //unsigned char numFullFoldsInGroupCurrentLayer,
            kernelOutputWriter.setArg(24 , numFullFilterFoldsInGroup);
            //unsigned char numActiveRowsInPartialFolds,
            kernelOutputWriter.setArg(25 , numActiveRowsPartialFold);
            /*
            Output modification
            */
            //unsigned char numAccumulatorBitsToRightShift
            kernelOutputWriter.setArg(26, (cl_uchar)( 2*FRAC_WIDTH-FRAC_WIDTH) );
            //unsigned char enableOutputRelu, //argument cannot be bool
            kernelOutputWriter.setArg(27 , enableRelu);
            //unsigned char enableSparsification //argument cannot be bool
            kernelOutputWriter.setArg(28 , enableSparsification);
        }

        std::cout <<"6. Setting kernel arguments for the input and output controllers."<<std::endl;
        {
            //unsigned short numGroupxTiles
            kernelIATileController.setArg(0, numInputGroupxTiles);

            //unsigned short numGroupxTiles
            KernelOATileController.setArg(0, numOutputGroupxTiles);
        }

        /* Transfer buffer content
        */
        cl_int status;

        //Transfer the input
        std::cout <<"7. Transfer the input activations "<<std::endl;
        {
            cl::Event event;
            auto numTransferBlocks = (pInput->getTransferBlockVector()).size();
            auto sizeTransferBlockElement = sizeof(typeof((pInput->getTransferBlockVector()).at(0)));
            auto valueVectorSizeBytes = sizeTransferBlockElement * numTransferBlocks;

            std::cout <<"8. Transfering "<<valueVectorSizeBytes<<" bytes in to bufferMemoryReaderWideInput"<<std::endl;

            status = clCQMemoryReader.enqueueWriteBuffer(bufferMemoryReaderWideInput, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 valueVectorSizeBytes, //size
                                                 (pInput->getTransferBlockVector()).data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the input activation vector");
            clCQMemoryReader.finish();
            cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
            std::cout <<"Transfer the input actvation tensor took "<<elapsedTimeUs<<" us"<<std::endl;
        } // Transfer the input

#if defined(SPARSE_SYSTEM)
        //Transfer the input transfer block count
        std::cout <<"9. Transfer the input transfer block count "<<std::endl;
        {
            cl::Event event;
            auto numBlocks = (pInput->getTransferBlockCountVector()).size();
            auto sizePerElement = sizeof(typeof((pInput->getTransferBlockCountVector()).at(0)));
            auto transferSizeBytes = numBlocks * sizePerElement;

            std::cout <<"Transfering "<<transferSizeBytes<<" bytes in to bufferMemoryReaderInputSBCount"<<std::endl;

            status = clCQMemoryReader.enqueueWriteBuffer(bufferMemoryReaderInputSBCount, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 transferSizeBytes, //size
                                                 (pInput->getTransferBlockCountVector()).data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the input activation transfer block count vector");
            clCQMemoryReader.finish();
            cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
            std::cout <<"Transfer the input actvation count took "<<elapsedTimeUs<<" us"<<std::endl;
        } //Transfer the input block
#endif
        //Transfer the weight
        std::cout <<"10. Transfer the weight "<<std::endl;
        {
            cl::Event event;
            auto numBlocks = (pWeights->getTransferBlockVector()).size();
            auto sizePerElement = sizeof(typeof((pWeights->getTransferBlockVector()).at(0)));
            auto transferSizeBytes = numBlocks * sizePerElement;

            std::cout <<"Transfering "<<transferSizeBytes<<" bytes in to bufferMemoryReaderWideWeights"<<std::endl;

            status = clCQMemoryReader.enqueueWriteBuffer(bufferMemoryReaderWideWeights, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 transferSizeBytes, //size
                                                 (pWeights->getTransferBlockVector()).data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the weight buffer.");
            clCQMemoryReader.finish();
            cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
            std::cout <<"Transfer the weight tensor took "<<elapsedTimeUs<<" us"<<std::endl;
        } // Transfer the weights

#if defined(SPARSE_SYSTEM)
        //Transfer the weight transfer block count
        std::cout <<"11. Transfer the weight block count "<<std::endl;
        {
            cl::Event event;
            auto numBlocks = (pWeights->getTransferBlockCountVector()).size();
            auto sizePerElement = sizeof(typeof((pWeights->getTransferBlockCountVector()).at(0)));
            auto transferSizeBytes = numBlocks * sizePerElement;

            std::cout <<"Transfering "<<transferSizeBytes<<" bytes in to bufferMemoryReaderWeightSBCount"<<std::endl;

            status = clCQMemoryReader.enqueueWriteBuffer(bufferMemoryReaderWeightSBCount, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 transferSizeBytes, //size
                                                 (pWeights->getTransferBlockCountVector()).data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the weight transfer block count vector");
            clCQMemoryReader.finish();
            cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
            std::cout <<"Transfer the weight count took "<<elapsedTimeUs<<" us"<<std::endl;
        }
#endif

        //Transfer the bias vector
        std::cout <<"12. Transfer the bias vector"<<std::endl;
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
            clCQMemoryReader.finish();
            cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
            std::cout <<"Transfer the bias vector took "<<elapsedTimeUs<<" us"<<std::endl;
        } // bias transfer block

#if defined(PROFILE) && defined(C5SOC)
        std::cout <<"Attempting to clear the performance counters."<<std::endl;
        status = clGetProfileDataDeviceIntelFPGA(
                  clDevice(),
                  program(),
                  CL_TRUE,
                  CL_TRUE,
                  CL_FALSE,
                  0,
                  NULL,
                  NULL,
                  NULL
                    );
        aocl_utils_cpp::checkError(status, "Failed to reset the profiling information!");
#endif
        //Launch the kernels
        std::cout<<"13. Launch the kernels and run "<<REPEAT<<" times."<<std::endl;
        std::vector<cl::Event> elist;
       cl_ulong processDurationAgregate = 0;

        //auto processStart = std::chrono::system_clock::now();

        for (int i=0; i<REPEAT; i++)
        {
            cl::Event eventMemoryReader, eventOutputWriter, eventIATileController, eventOATileController;

            status = clCQMemoryReader.enqueueTask(kernelMemoryReader, NULL);
            aocl_utils_cpp::checkError(status, "Failed to launch kernelMemoryReader!");

            status = clCQIATileController.enqueueTask(kernelIATileController, NULL);
            aocl_utils_cpp::checkError(status, "Failed to launch kernelIATileController!");

            status = clCQOATileController.enqueueTask(KernelOATileController, NULL);
            aocl_utils_cpp::checkError(status, "Failed to launch KernelOATileController!");

            status = clCQOutputWriter.enqueueTask(kernelOutputWriter, NULL, &eventOutputWriter);
            aocl_utils_cpp::checkError(status, "Failed to launch kernelOutputWriter!");
            clCQOutputWriter.finish();

            cl_ulong processStart = eventOutputWriter.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong processEnd = eventOutputWriter.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            processDurationAgregate += (processEnd - processStart);
         }

        clCQOutputWriter.finish();

        //auto processEnd = std::chrono::system_clock::now();

        cl_double processAverageDuration = (cl_double)((processDurationAgregate) * (cl_double)(1e-3)) / (cl_double)(REPEAT);
        //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(processEnd-processStart).count();
        //double averageDuration = (double)(duration) / (double)(REPEAT);
#if defined(C5SOC)
        //Hack for ARMv7
        //averageDuration = averageDuration * 1000.0;
#endif


#if defined(PROFILE) && defined(C5SOC)
        std::cout <<"14.b Attempting to retrieve autorun profiling data."<<std::endl;
        status = clGetProfileDataDeviceIntelFPGA(
                  clDevice(),
                  program(),
                  CL_TRUE,
                  CL_TRUE,
                  CL_FALSE,
                  0,
                  NULL,
                  NULL,
                  NULL
                    );
        aocl_utils_cpp::checkError(status, "Failed to retrieve profiling information!");
#endif

        std::cout <<"14. Retrieve the output."<<std::endl;
        cl::Event eventReadOutput, eventReadOutputCount;
        status = clCQOutputWriter.enqueueReadBuffer(
            bufferMemoryWriterWideOutput,
            CL_TRUE,
            0,
            sizeof(typeof((pOutput->getTransferBlockVector()).at(0))) * (pOutput->getTransferBlockVector()).size(),
            (pOutput->getTransferBlockVector()).data(),
            NULL,
            &eventReadOutput
        );
        aocl_utils_cpp::checkError(status, "Failed to read compressed output values!");

#if defined(SPARSE_SYSTEM)
        if (_flagCompressionOutput == true)
        {
            status = clCQOutputWriter.enqueueReadBuffer(
                bufferMemoryWriterOutputSBCount,
                CL_TRUE,
                0,
                sizeof(typeof((pOutput->getTransferBlockCountVector()).at(0))) * (pOutput->getTransferBlockCountVector()).size(),
                (pOutput->getTransferBlockCountVector()).data(),
                NULL,
                &eventReadOutputCount
            );
         }
        aocl_utils_cpp::checkError(status, "Failed to read compressed output counts!");
#endif

        cl_ulong outputValueTransferStart = eventReadOutput.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong outputValueTransferEnd = eventReadOutput.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        cl_double outputValueTransferDuration = (cl_double)((outputValueTransferEnd - outputValueTransferStart) * (cl_double)(1e-3));

        cl_ulong outputCountTransferStart = eventReadOutputCount.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong outputCountTransferEnd = eventReadOutputCount.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        cl_double outputCountTransferDuration = (cl_double)((outputCountTransferEnd - outputCountTransferStart) * (cl_double)(1e-3));

        std::cout <<"Average Convolution time:  (us): "<<processAverageDuration<<std::endl;
        std::cout <<"Output transfer time (us): "<<outputValueTransferDuration<<std::endl;
        std::cout <<"Output count transfer time (us): "<<outputCountTransferDuration<<std::endl;

        //Decompress the output, and check against the input if necessary
        {
            std::cout <<"15. Decode the output"<<std::endl;
            std::vector<fixedPointNumber> outputFPVector;

            pOutput->decodeTensor(outputFPVector, FRAC_WIDTH, INT_WIDTH);

            std::cout <<"16. Check the output"<<std::endl;
            if (eTensorType == TEST)
            {
                for (unsigned char iterHeight=0; iterHeight<outputHeight; iterHeight++)
                {
                    for (unsigned char iterWidth=0; iterWidth<outputWidth; iterWidth++)
                    {
                        for (unsigned char iterInputChannel=0; iterInputChannel<_numInputChannel; iterInputChannel++)
                        {
                            unsigned int outputIndex = (iterHeight*outputWidth + iterWidth)*numFiltersInKernel + iterInputChannel*2;
                            unsigned int inputIndex = (iterHeight*outputWidth + iterWidth)*_numInputChannel + iterInputChannel;

                            fixedPointNumber expectedOutputBiased(inputTensorDense.at(inputIndex).convert2Float() + _bias,
                                                                         FRAC_WIDTH,
                                                                         INT_WIDTH
                                                                         );

                            signed char expectedOutput = (_flagEnableRelu && (expectedOutputBiased.getBits() < ((char) 0x0))) ?
                                        (char) 0x0 : expectedOutputBiased.getBits();

                            signed char actualOutput = outputFPVector.at(outputIndex).getBits();

                            EXPECT_TRUE(expectedOutput == actualOutput)
                            //std::cout<<"Error: iY, iX, iIC, actualOutput, expectedOutput "
                                <<(unsigned int)iterHeight<<" "<<(unsigned int)iterWidth<<" "<<(unsigned int)iterInputChannel<<" 0x"
                                <<std::bitset<8> (actualOutput)<<" 0x"
                                <<std::bitset<8> (expectedOutput)<<std::endl;

                        } // for iterInputChannel
                    } // for iterWidth
                } // for iterHeight
             } // Test condition block
        } // input checking block
    } //launch

};
#ifdef PLAY
TEST_F (testFixture, play) {

    unsigned char inputWidth = 3;
    unsigned char inputHeight = 3;
    unsigned char numInputChannel = 8;
    unsigned char widthBlockSize = 1;
    unsigned char sizeOutputTileWidthPerColFul = 2;
    unsigned char sizeOutputTileHeightFull = 2;
    bool flagEnableRelu = true;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        TEST_TYPE);
}
#else
#if defined(SPARSE_LEVEL_TEST)
TEST_F (testFixture, sparse_16x16x127_0p) {

    unsigned char inputWidth = 16;
    unsigned char inputHeight = 16;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 1;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.0f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_16x16x127_0p1) {

    unsigned char inputWidth = 16;
    unsigned char inputHeight = 16;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 1;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.1f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_16x16x127_0p2) {

    unsigned char inputWidth = 16;
    unsigned char inputHeight = 16;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 1;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.2f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_16x16x127_0p3) {

    unsigned char inputWidth = 16;
    unsigned char inputHeight = 16;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 1;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.3f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_16x16x127_0p4) {

    unsigned char inputWidth = 16;
    unsigned char inputHeight = 16;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 1;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.4f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_16x16x127_0p5) {

    unsigned char inputWidth = 16;
    unsigned char inputHeight = 16;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 1;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.5f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_16x16x127_0p6) {

    unsigned char inputWidth = 16;
    unsigned char inputHeight = 16;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 1;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.6f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_16x16x127_0p7) {

    unsigned char inputWidth = 16;
    unsigned char inputHeight = 16;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 1;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.7f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_16x16x127_0p8) {

    unsigned char inputWidth = 16;
    unsigned char inputHeight = 16;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 1;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.8f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_16x16x127_0p9) {

    unsigned char inputWidth = 16;
    unsigned char inputHeight = 16;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 1;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.9f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_16x16x127_1p) {

    unsigned char inputWidth = 16;
    unsigned char inputHeight = 16;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 1;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 1.0f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}

TEST_F (testFixture, sparse_1x1x127_0p) {

    unsigned char inputWidth = 1;
    unsigned char inputHeight = 1;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.0f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_1x1x127_0p05) {

    unsigned char inputWidth = 1;
    unsigned char inputHeight = 1;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.05f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_1x1x127_0p1) {

    unsigned char inputWidth = 1;
    unsigned char inputHeight = 1;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.1f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_1x1x127_0p2) {

    unsigned char inputWidth = 1;
    unsigned char inputHeight = 1;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.2f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_1x1x127_0p3) {

    unsigned char inputWidth = 1;
    unsigned char inputHeight = 1;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.3f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_1x1x127_0p4) {

    unsigned char inputWidth = 1;
    unsigned char inputHeight = 1;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.4f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_1x1x127_0p5) {

    unsigned char inputWidth = 1;
    unsigned char inputHeight = 1;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.5f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_1x1x127_0p6) {

    unsigned char inputWidth = 1;
    unsigned char inputHeight = 1;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.6f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_1x1x127_0p7) {

    unsigned char inputWidth = 1;
    unsigned char inputHeight = 1;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.7f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_1x1x127_0p8) {

    unsigned char inputWidth = 1;
    unsigned char inputHeight = 1;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.8f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_1x1x127_0p9) {

    unsigned char inputWidth = 1;
    unsigned char inputHeight = 1;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 0.9f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
TEST_F (testFixture, sparse_1x1x127_1p) {

    unsigned char inputWidth = 1;
    unsigned char inputHeight = 1;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;
    float bias = 0.0;
    e_tensor_type tensorType = SPARSE;
    bool flagCompression = false;
    float sparseProb = 1.0f;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        tensorType,
        bias,
        flagCompression,
        1.0f - sparseProb);
}
#else
TEST_F (testFixture, small_5x5) {

    unsigned char inputWidth = 5;
    unsigned char inputHeight = 5;
    unsigned char numInputChannel = 4;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 2;
    unsigned char sizeOutputTileHeightFull = 2;
    bool flagEnableRelu = true;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        TEST_TYPE);
}

TEST_F (testFixture, large_16x16x8_tileSizeCol_8) {

    unsigned char inputWidth = 16;
    unsigned char inputHeight = 16;
    unsigned char numInputChannel = 8;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        TEST_TYPE);
}

TEST_F (testFixture, large_16x16x16_tileSizeCol_8) {

    unsigned char inputWidth = 16;
    unsigned char inputHeight = 16;
    unsigned char numInputChannel = 16;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        TEST_TYPE);
}

TEST_F (testFixture, large_16x16x32_tileSizeCol_8) {

    unsigned char inputWidth = 16;
    unsigned char inputHeight = 16;
    unsigned char numInputChannel = 32;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        TEST_TYPE);
}

TEST_F (testFixture, large_16x16x64_tileSizeCol_8) {

    unsigned char inputWidth = 16;
    unsigned char inputHeight = 16;
    unsigned char numInputChannel = 64;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        TEST_TYPE);
}

TEST_F (testFixture, large_16x16x127_tileSizeCol_2) {

    unsigned char inputWidth = 16;
    unsigned char inputHeight = 16;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 2;
    unsigned char sizeOutputTileHeightFull = 2;
    bool flagEnableRelu = true;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        TEST_TYPE);
}

TEST_F (testFixture, large_16x16x127_tileSizeCol_4) {

    unsigned char inputWidth = 16;
    unsigned char inputHeight = 16;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 4;
    unsigned char sizeOutputTileHeightFull = 4;
    bool flagEnableRelu = true;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        TEST_TYPE);
}

TEST_F (testFixture, large_16x16x127_tileSizeCol_8) {

    unsigned char inputWidth = 16;
    unsigned char inputHeight = 16;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        TEST_TYPE);
}

TEST_F (testFixture, narrow_1x1x8) {

    unsigned char inputWidth = 1;
    unsigned char inputHeight = 1;
    unsigned char numInputChannel = 8;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        TEST_TYPE);
}

TEST_F (testFixture, narrow_1x1x16) {

    unsigned char inputWidth = 1;
    unsigned char inputHeight = 1;
    unsigned char numInputChannel = 16;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        TEST_TYPE);
}

TEST_F (testFixture, narrow_1x1x32) {

    unsigned char inputWidth = 1;
    unsigned char inputHeight = 1;
    unsigned char numInputChannel = 32;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        TEST_TYPE);
}

TEST_F (testFixture, narrow_1x1x64) {

    unsigned char inputWidth = 1;
    unsigned char inputHeight = 1;
    unsigned char numInputChannel = 64;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        TEST_TYPE);
}

TEST_F (testFixture, narrow_1x1x127) {

    unsigned char inputWidth = 1;
    unsigned char inputHeight = 1;
    unsigned char numInputChannel = 127;
    unsigned char widthBlockSize = 3;
    unsigned char sizeOutputTileWidthPerColFul = 8;
    unsigned char sizeOutputTileHeightFull = 8;
    bool flagEnableRelu = true;

    launch(
        inputWidth,
        inputHeight,
        numInputChannel,
        widthBlockSize,
        sizeOutputTileWidthPerColFul,
        sizeOutputTileHeightFull,
        flagEnableRelu,
        TEST_TYPE);
}
#endif //SPARSE_LEVEL_TEST
#endif  //PLAY

int main(int argc, char* argv[]) {

    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}
