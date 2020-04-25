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
#include "layerInstructionGenerator.hpp"

/*Limits on the buffer sizes
 * Assume the biggest test convolves a 32x32x64 tensor with a 128*32*32*64 tensor
 * Add a safety factor of 4
 * */
#define MAX_DRAM_BYTE_INPUT_ACTIVATION 262144
#define MAX_DRAM_BYTE_INPUT_ACTIVATION_SB_COUNT 8192
#define MAX_DRAM_BYTE_INPUT_WEIGHT 16777216
#define MAX_DRAM_BYTE_INPUT_WEIGHT_SB_COUNT 2048
#define MAX_DRAM_BYTE_INPUT_BIAS 2048
#define MAX_DRAM_BYTE_OUTPUT_ACTIVATION 1048576
#define MAX_DRAM_BYTE_OUTPUT_ACTIVATION_SB_COUNT 8192
#define MAX_DRAM_BYTE_INPUT_MOVER_INSTRUCTION (1 << 20)
#define MAX_DRAM_BYTE_INPUT_TILE_CONTROLLER_INSTRUCTION (1 << 20)
#define MAX_DRAM_BYTE_OUTPUT_MOVER_INSTRUCTION (1 << 20)
#define MAX_DRAM_BYTE_WEIGHT_MOVER_INSTRUCTION (1 << 20)
#define MAX_DRAM_BYTE_OUTPUT_TILE_CONTROLLER_INSTRUCTION (1 << 20)
#define MAX_DRAM_BYTE_MISC_CONTROLLER_INSTRUCTION (1 << 20)

#define FRAC_WIDTH 4
#define INT_WIDTH 3
#define OUTPUT_INT_WIDTH 3

#define WEIGHT_SEED 1234
#define INPUT_SEED   7653

//#define PLAY
#define EMULATE

#if defined(C5SOC) //Hack for ARMv7, otherwise chrono won't work
__asm__(".symver _ZNSt6chrono3_V212system_clock3nowEv,_ZNSt6chrono12system_clock3nowEv@GLIBCXX_3.4.11");
#endif

typedef
std::vector<cl_short, boost::alignment::aligned_allocator<cl_short, aocl_utils_cpp::AOCL_ALIGNMENT>>
//std::vector<cl_short>
t_aligned_short_vector;

class testFixture : public ::testing::Test {
protected:
    std::string binaryFile;
    cl::Program program;
    cl::Platform clPlatform;
    cl::Context clContext;
    cl::Device clDevice;

    //Command queues
    cl::CommandQueue clCQIAMover;
    cl::CommandQueue clCQOAMover;
    cl::CommandQueue clCQWMover;
    cl::CommandQueue clCQIATileController;
    cl::CommandQueue clCQOATileController;
    cl::CommandQueue clMKController;

    //The kernels
    cl::Kernel kernelIAMover;
    cl::Kernel kernelOAMover;
    cl::Kernel kernelWMover;
    cl::Kernel kernelMKInstructionMover;
    cl::Kernel kernelIATileController;
    cl::Kernel KernelOATileController;

    //Buffer members associated with the IA Mover kernel
    cl::Buffer bufferIAMoverInstructions;
    cl::Buffer bufferIAMoverIADramBlocks;
#if defined(SPARSE_SYSTEM)
    cl::Buffer bufferIAMoverIATBCounts;
#endif

    //Buffer members associated with the IA tile controller
    cl::Buffer bufferIATileControllerInstructions;

    //Buffer members associated with the OA Mover kernel
    cl::Buffer bufferOAMoverInstructions;
    cl::Buffer bufferOAMoverOADramBlocks;
#if defined(SPARSE_SYSTEM)
    cl::Buffer bufferOAMoverOATBCounts;
#endif

    //Buffer members associated with the OA tile controller
    cl::Buffer bufferOATileControllerInstructions;

    //Buffer members associated with the W Mover kernel
    cl::Buffer bufferWMoverInstructions;
    cl::Buffer bufferWMoverWDramBlocks;
    cl::Buffer bufferWMoverBias;
#if defined(SPARSE_SYSTEM)
    cl::Buffer bufferWMoverWTBCounts;
#endif

    //Buffer members associated with the MK instruction kernel
    cl::Buffer bufferMKInstructions;

    void SetUp() override;

    /*!
     * \brief generateInputTensor
     * \details Generate a tensor of size
     *          _numGroupCurrentLayer*_inputHeight*_inputWidth*(_numInputChannel/_numGroupCurrentLayer).
     *          The width and height must be divisible by 2
     * \param _inputWidth
     * \param _inputHeight
     * \param _numInputChannel
     * \param _numGroupCurrentLayer Number of groups in the current layer.
     *        _numInputChannel must be divisible by _numGroupCurrentLayer
     * \return The tensor generated
     */
    std::vector<fixedPointNumber> generateInputTensor (
                unsigned char _inputWidth,
                unsigned char _inputHeight,
                unsigned char _numInputChannel,
                unsigned char _numGroupCurrentLayer
            );

    /*!
     * \brief generateWeights
     * \details Generate a tensor of weights, following the Filter * OC * W * H * IC layout.
     *          W=H=_kernelSize, IC=_numInputChannel/_numGroups, OC=_numInputChannel
     *          Filters k in [0, IC-1] are the identity filters for input channel k
     * \param _kernelSize Must be odd
     * \param _numInputChannel
     * \param _numGroups
     * \return
     */
    std::vector<fixedPointNumber> generateWeights (
                unsigned char _kernelSize,
                unsigned char _numInputChannel,
                unsigned char _numGroups
            );


    void launch (
            unsigned char _inputWidth,
            unsigned char _inputHeight,
            unsigned char _numInputChannel,
            unsigned char _numInputGroup, //The code will override this to 1 if the operation is not convolution
            unsigned char _inputHeightSPUnitSize, //The code will override this to 1 if the operation is not convolution
            unsigned char _inputWidthSPUnitSize, //The code will overide this to 1 if the operation is not convolution
            unsigned char _sizeOutputTileWidthPerColFull, //The code will override this to 1 if the operation is not convolution
            unsigned char _sizeOutputTileHeight, //The code will overrid this to 1 if the operation is not convolution
            bool _flagEnableRelu,
            bool _flagSparseInput, //The code will override this to FALSE if the operation is not convolution
            bool _flagSparseOutput,
            OPERATION op,
            float _bias = 0.0f //Only matter for convolution
            );
}; //testFixture

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

#endif  //PLAY

int main(int argc, char* argv[]) {

    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}

void testFixture::SetUp()
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
    clContext = cl::Context({clDevice}
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
    kernelIAMover = cl::Kernel(program, "kernelIAMover", &status);
    aocl_utils_cpp::checkError(status, "Failed to create kernelIAMover!");

    kernelWMover = cl::Kernel(program, "kernelWMover", &status);
    aocl_utils_cpp::checkError(status, "Failed to create kernelWMover!");

    kernelOAMover = cl::Kernel(program, "kernelOAMover", &status);
    aocl_utils_cpp::checkError(status, "Failed to create kernelOutputWriter!");

    kernelIATileController = cl::Kernel(program, "kernelIATileController", &status);
    aocl_utils_cpp::checkError(status, "Failed to create kernelIATileController!");

    KernelOATileController = cl::Kernel(program, "kernelOATileController", &status);
    aocl_utils_cpp::checkError(status, "Failed to create kernelOATileController!");

    kernelMKInstructionMover = cl::Kernel(program, "kernelMiscControlMover", &status);
    aocl_utils_cpp::checkError(status, "Failed to create kernelMiscControlMover!");

    //Instantiate the command queues
    clCQIAMover = cl::CommandQueue(
                clContext,
                clDevice,
                CL_QUEUE_PROFILING_ENABLE,
                &status
                );
    aocl_utils_cpp::checkError(status, "Failed to setup the command queue clCQIAMover!");

    clCQOAMover = cl::CommandQueue(
                clContext,
                clDevice,
                CL_QUEUE_PROFILING_ENABLE,
                &status
                );
    aocl_utils_cpp::checkError(status, "Failed to setup the command queue clCQOAMover!");

    clCQWMover = cl::CommandQueue(
                clContext,
                clDevice,
                CL_QUEUE_PROFILING_ENABLE,
                &status
                );
    aocl_utils_cpp::checkError(status, "Failed to setup the command queue clCQWMover!");

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

    clMKController = cl::CommandQueue(
                clContext,
                clDevice,
                CL_QUEUE_PROFILING_ENABLE,
                &status
                );
    aocl_utils_cpp::checkError(status, "Failed to setup the command queue clMKController!");

    //Instantiate the buffers
    cl_ulong maxBufferSizeByte = clDevice.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE> (&status);
    aocl_utils_cpp::checkError(status, "Failed to query the maximum buffer size in bytes!");

    typedef struct {
        cl_ulong bufferSizeByte;
        cl::Buffer& bufferObject;
        cl_mem_flags memFlag;
        std::string bufferName;
    } t_buffer_setup_info;

    std::vector<t_buffer_setup_info> vecBufferInfo;

    cl_ulong weightMoverInstructionBufferSize = maxBufferSizeByte < MAX_DRAM_BYTE_WEIGHT_MOVER_INSTRUCTION ? maxBufferSizeByte : MAX_DRAM_BYTE_WEIGHT_MOVER_INSTRUCTION;
    vecBufferInfo.push_back({weightMoverInstructionBufferSize, bufferWMoverInstructions, CL_MEM_READ_ONLY, "bufferWMoverInstructions"});

    cl_ulong inputWeightBufferSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_WEIGHT ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_WEIGHT;
    vecBufferInfo.push_back(({inputWeightBufferSize, bufferWMoverWDramBlocks, CL_MEM_READ_ONLY, "bufferWMoverWDramBlocks"}));

    cl_ulong inputBiasSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_BIAS ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_BIAS;
    vecBufferInfo.push_back(({inputBiasSize, bufferWMoverBias, CL_MEM_READ_ONLY, "bufferWMoverBias"}));

    cl_ulong inputActivationSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_ACTIVATION ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_ACTIVATION;
    vecBufferInfo.push_back(({inputActivationSize, bufferIAMoverIADramBlocks, CL_MEM_READ_WRITE, "bufferIAMoverIADramBlocks"}));

    cl_ulong inputIAMoverInstructionSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_MOVER_INSTRUCTION ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_MOVER_INSTRUCTION;
    vecBufferInfo.push_back(({inputIAMoverInstructionSize, bufferIAMoverInstructions, CL_MEM_READ_ONLY, "bufferIAMoverInstructions"}));

    cl_ulong inputIATileControllerInstructionSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_TILE_CONTROLLER_INSTRUCTION ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_TILE_CONTROLLER_INSTRUCTION;
    vecBufferInfo.push_back(({inputIATileControllerInstructionSize, bufferIATileControllerInstructions, CL_MEM_READ_ONLY, "bufferIATileControllerInstructions"}));

    cl_ulong outputActivationSize = maxBufferSizeByte < MAX_DRAM_BYTE_OUTPUT_ACTIVATION ? maxBufferSizeByte : MAX_DRAM_BYTE_OUTPUT_ACTIVATION;
    vecBufferInfo.push_back(({outputActivationSize, bufferOAMoverOADramBlocks, CL_MEM_READ_WRITE, "bufferOAMoverOADramBlocks"}));

    cl_ulong outputOAMoverInstructionSize = maxBufferSizeByte < MAX_DRAM_BYTE_OUTPUT_MOVER_INSTRUCTION ? maxBufferSizeByte : MAX_DRAM_BYTE_OUTPUT_MOVER_INSTRUCTION;
    vecBufferInfo.push_back(({outputOAMoverInstructionSize, bufferOAMoverInstructions, CL_MEM_READ_ONLY, "bufferOAMoverInstructions"}));

    cl_ulong outoutOATileControllerInstructionSize = maxBufferSizeByte < MAX_DRAM_BYTE_OUTPUT_TILE_CONTROLLER_INSTRUCTION ? maxBufferSizeByte : MAX_DRAM_BYTE_OUTPUT_TILE_CONTROLLER_INSTRUCTION;
    vecBufferInfo.push_back(({outoutOATileControllerInstructionSize, bufferOATileControllerInstructions, CL_MEM_READ_ONLY, "bufferOATileControllerInstructions"}));

    cl_ulong mkInstructionSize = maxBufferSizeByte < MAX_DRAM_BYTE_MISC_CONTROLLER_INSTRUCTION ? maxBufferSizeByte : MAX_DRAM_BYTE_MISC_CONTROLLER_INSTRUCTION;
    vecBufferInfo.push_back(({mkInstructionSize, bufferMKInstructions, CL_MEM_READ_ONLY, "bufferMKInstructions"}));

#if defined(SPARSE_SYSTEM)
    cl_ulong inputWeightSBSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_WEIGHT_SB_COUNT ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_WEIGHT_SB_COUNT;
    vecBufferInfo.push_back(({inputWeightSBSize, bufferWMoverWTBCounts, CL_MEM_READ_ONLY, "bufferWMoverWTBCounts"}));

    cl_ulong inputIATBCountSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_ACTIVATION_SB_COUNT ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_ACTIVATION_SB_COUNT;
    vecBufferInfo.push_back(({inputIATBCountSize, bufferIAMoverIATBCounts, CL_MEM_READ_ONLY, "bufferIAMoverIATBCounts"}));

    cl_ulong outputOATBCountSize = maxBufferSizeByte < MAX_DRAM_BYTE_OUTPUT_ACTIVATION_SB_COUNT ? maxBufferSizeByte : MAX_DRAM_BYTE_OUTPUT_ACTIVATION_SB_COUNT;
    vecBufferInfo.push_back(({outputOATBCountSize, bufferOAMoverOATBCounts, CL_MEM_READ_ONLY, "bufferOAMoverOATBCounts"}));
#endif

    for (auto& info : vecBufferInfo)
    {
        cl_int localStatus = CL_SUCCESS;
        std::cout <<"Setting the buffer "<<info.bufferName<<". Size (bytes): "<<info.bufferSizeByte<<std::endl;
        info.bufferObject = cl::Buffer (
                    clContext,
                    info.memFlag,
                    info.bufferSizeByte,
                    NULL,
                    &localStatus
                );
        aocl_utils_cpp::checkError(status, "Failed to setup the buffer!");
    }
    std::cout <<"AOCL setup compelete"<<std::endl;
}

std::vector<fixedPointNumber> testFixture::generateInputTensor (
            unsigned char inputWidth,
            unsigned char inputHeight,
            unsigned char numInputChannel,
        )
{

}

std::vector<fixedPointNumber> testFixture::generateWeights (
            unsigned char kernelSize,
            unsigned char numInputChannel
        )
{

}

void testFixture::launch (
        unsigned char _inputWidth,
        unsigned char _inputHeight,
        unsigned char _numInputChannel,
        unsigned char _numInputGroup, //The code will override this to 1 if the operation is not convolution
        unsigned char _inputHeightSPUnitSize, //The code will override this to 1 if the operation is not convolution
        unsigned char _inputWidthSPUnitSize, //The code will overide this to 1 if the operation is not convolution
        unsigned char _sizeOutputTileWidthPerColFull, //The code will override this to 1 if the operation is not convolution
        unsigned char _sizeOutputTileHeight, //The code will overrid this to 1 if the operation is not convolution
        bool _flagEnableRelu,
        bool _flagSparseInput, //The code will override this to FALSE if the operation is not convolution or if the accelerator only supports dense format
        bool _flagSparseOutput, //The code will override this to FALSE if the accelerator only supports dense format.
        OPERATION op,
        float _bias = 0.0f //Only matter for convolution
        )
{
    //Checking the parameters' consistency AND
    //Derive parameters
    assert(_inputHeight % 2 == 0);
    assert(_inputWidth % 2 == 0);
    assert(_numInputChannel <= 127);

    cl_uchar kernelSize;
    cl_uchar stride;
    unsigned char numInputChannel0;
    unsigned char numInputChannel1;
    unsigned char numOutputChannels;
    unsigned char numOutputWidth;
    unsigned char numOutputHeight;
    unsigned char numGroupCurrentLayer;
    unsigned char inputHeightSPUnitSize;
    unsigned char inputWidthSPUnitSize;
    unsigned char inputHeightSPSize;
    unsigned char inputWidthSPSize;
    unsigned char sizeOutputTileWidthPerCol;
    unsigned char sizeOutputTileHeight;
    unsigned char verticalBorderPadding;
    unsigned char horizontalBorderPadding;
    bool flagSparseInput;
    bool flagSparseOutput;

    switch (op) {
        case CONVOLUTION: {
            assert(_numInputChannel % _numInputGroup == 0);
            numInputChannel0 = _numInputChannel;
            numInputChannel1 = 0;
            numOutputChannels = _numInputChannel;
            inputHeightSPUnitSize = _inputHeightSPUnitSize;
            inputWidthSPUnitSize = _inputWidthSPUnitSize;
            numGroupCurrentLayer = _numInputGroup;
            sizeOutputTileWidthPerCol = _sizeOutputTileWidthPerColFull;
            sizeOutputTileHeight = _sizeOutputTileHeight;
            kernelSize = 3;
            stride = 1;

#if defined(SPARSE_SYSTEM)
            flagSparseInput = _flagSparseInput;
            flagSparseOutput = _flagSparseOutput;
#else
            flagSparseInput = false;
            flagSparseOutput = false;
#endif
        }
        break;
        case MAX_POOL: {
            numInputChannel0 = _numInputChannel;
            numInputChannel1 = 0;
            inputHeightSPUnitSize = 1;
            inputWidthSPUnitSize = 1;
            numGroupCurrentLayer = 1;
            sizeOutputTileWidthPerCol = 1;
            sizeOutputTileHeight = 1;
            kernelSize = 2;
            stride = 2;

            flagSparseInput = false;
#if defined(SPARSE_SYSTEM)
            flagSparseOutput = _flagSparseOutput;
#else
            flagSparseOutput = false;
#endif
        }
        break;
        case ELT_ADD: {
            numInputChannel0 = _numInputChannel;
            numInputChannel1 = 0;
            inputHeightSPUnitSize = 1;
            inputWidthSPUnitSize = 1;
            numGroupCurrentLayer = 1;
            sizeOutputTileWidthPerCol = 1;
            sizeOutputTileHeight = 1;
            kernelSize = 1;
            stride = 1;

            flagSparseInput = false;
#if defined(SPARSE_SYSTEM)
            flagSparseOutput = _flagSparseOutput;
#else
            flagSparseOutput = false;
#endif

        }
        break;
        case CONCATENATION: {
            numInputChannel0 = _numInputChannel;
            numInputChannel1 = _numInputChannel;
            inputHeightSPUnitSize = 1;
            inputWidthSPUnitSize = 1;
            numGroupCurrentLayer = 1;
            sizeOutputTileWidthPerCol = 1;
            sizeOutputTileHeight = 1;
            kernelSize = 1;
            stride = 1;

            flagSparseInput = false;
#if defined(SPARSE_SYSTEM)
            flagSparseOutput = _flagSparseOutput;
#else
            flagSparseOutput = false;
#endif
        }
        break;
    } //switch
    numOutputChannels = numInputChannel0 + numInputChannel1;
    inputWidthSPSize = inputWidthSPUnitSize*(_inputWidth-1) + 1;
    inputHeightSPSize = inputHeightSPUnitSize*(_inputHeight-1) + 1;
    numOutputWidth = (op==MAX_POOL) ? inputWidthSPSize / 2 : inputWidthSPSize;
    numOutputHeight = (op==MAX_POOL) ? inputHeightSPSize / 2 : inputHeightSPSize;
    verticalBorderPadding = (op==CONVOLUTION) ? 1 : 0;
    horizontalBorderPadding  = (op==CONVOLUTION) ? 1 : 0;
    unsigned char numActiveColsPartialOutputTile = (op==CONVOLUTION) ?
                1 : (numOutputWidth % PE_COLS);

    /* Generate the dense, fixed point tensors
     * */
    std::cout <<"1. Preparing the test tensors. Test operation type is "<<op<<std::endl;
    std::cout <<"Input SP width:  "<<(unsigned int) inputWidthSPSize<<std::endl
              <<"Input SP height: "<<(unsigned int) inputHeightSPSize<<std::endl
              <<"Input channels 0: "<<(unsigned int) numInputChannel0<<std::endl
              <<"Input channels 1: "<<(unsigned int) numInputChannel1<<std::endl
              <<"Output channels: "<<(unsigned int) numOutputChannels<<std::endl
              <<"Number of groups in current layer: "<<(unsigned int)numGroupCurrentLayer<<std::endl;

    std::vector<fixedPointNumber> inputTensorDense =
            generateInputTensor(_inputWidth, _inputHeight, _numInputChannel, numGroupCurrentLayer);
    std::vector<fixedPointNumber> inputWeightDense;
    if (op == CONVOLUTION)
    {
        inputWeightDense = generateWeights((unsigned char) kernelSize, _numInputChannel, numGroupCurrentLayer);
    }
    t_accumulator fixedBias = (t_accumulator) (round(_bias * (float) (1 << (FRAC_WIDTH + FRAC_WIDTH)) ));
    t_aligned_short_vector biasVector (2*_numInputChannel, fixedBias);

    /* 2. Allocate the aligned tensors and compress them if necessary
     * */
    std::cout <<"2. Allocate, align, and compress the test tensors."<std::endl;
    if (eTensorType == SPARSE)
    {
           std::cout <<"Sparsity level is "<<(1.0f - denseProb)<<std::endl;
    }
    unsigned short maxScalarIndexInChannelGroup = _numInputChannel / numGroupCurrentLayer - 1;
    unsigned short maxClusterIndexInCompressionBlock = COMPRESSION_WINDOW_SIZE-1;
    unsigned short maxClusterIndexInTransferBlock = TRANSFER_SIZE-1;
    unsigned short maxScalarIndexInCluster = CLUSTER_SIZE-1;

    std::unique_ptr<AlignedTensor> pInput, pWeights, pOutput;

    if (flagSparseInput == true)
    {
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
        if (op == CONVOLUTION)
        {
            pWeights.reset(new FlexibleDirectCompressedTensor (
                        inputWeightDense,
                        _numInputChannel, //_num3DTensors
                        _numInputChannel,
                        (unsigned char) kernelSize, //width
                        (unsigned char) kernelSize, //height
                        maxScalarIndexInChannelGroup,
                        maxClusterIndexInCompressionBlock,
                        maxClusterIndexInTransferBlock,
                        maxScalarIndexInCluster,
                        true //isKernel
                    ) );
        }
    }
    else
    {
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
        if (op == CONVOLUTION)
        {
            pWeights.reset( new AlignedTensor (
                        inputWeightDense,
                        _numInputChannel, //_num3DTensors
                        _numInputChannel,
                        (unsigned char) kernelSize, //width
                        (unsigned char) kernelSize, //height
                        maxScalarIndexInChannelGroup,
                        maxClusterIndexInTransferBlock,
                        maxScalarIndexInCluster,
                        true //isKernel
                    ));
        }
    }

    if (flagSparseOutput == true)
    {
        pOutput.reset(new FlexibleDirectCompressedTensor(
                    1, //_num3DTensors
                    numOutputChannels, //_channel
                    numOutputWidth, //_width
                    numOutputHeight, //_height
                    numOutputChannels-1, //_maxScalarIndexInChannelGroup
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
                    numOutputChannels, //_channel
                    numOutputWidth, //_width
                    numOutputHeight, //_height
                    numOutputChannels-1, //_maxScalarIndexInChannelGroup
                    maxClusterIndexInTransferBlock,
                    maxScalarIndexInCluster,
                    false //isKernel
                    ));
    }
    std::cout <<"3. Generate the instructions"<<std::endl;

    /*
     * 3. Generate the instructions
     * */
    t_aligned_ia_mover_instruction_vector vecIAMoverInstruction;
    t_aligned_ia_tile_controller_instruction_vector vecIATileControllerInstruction;
    t_aligned_oa_mover_instruction_vector vecOAMoverInstruction;
    t_aligned_oa_tile_controller_instruction_vector vecOATileControllerInstruction;
    t_aligned_weight_mover_instruction_vector vecWMoverInstruction;
    t_aligned_misc_instruction_vector vecMiscInstruction;

    signed int memDramBlockFilterStride = (pWeights->getExternalMemoryAddressStride()) >> WIDE_SIZE_OFFSET;
    signed int memDramBlockIAColStride = (pInput->getExternalMemoryAddressStride()) >> WIDE_SIZE_OFFSET;
    signed int memDramBlockOAColStride = (pOutput->getExternalMemoryAddressStride()) >> WIDE_SIZE_OFFSET;

    signed int memDramBlockIARowStride = memDramBlockIAColStride*_inputWidth;
    signed int memDramBlockIAGroupStride = memDramBlockIARowStride * _inputHeight;
    unsigned char instFlagSparseOutput = flagSparseOutput ? 0x01:0x00;
    unsigned char instFlagSparseInput = flagSparseInput ? 0x01: 0x00;
    unsigned char instFlagRelu = _flagEnableRelu ? 0x01 : 0x00;
    unsigned char instOutputShiftBits;
    unsigned char instOutputShiftLeft;
    //Figure out the output shift
    if (op == CONVOLUTION)
    {
        if (FRAC_WIDTH+FRAC_WIDTH > (7-OUTPUT_INT_WIDTH))
        {
            instOutputShiftBits = RAC_WIDTH+FRAC_WIDTH - 7 + OUTPUT_INT_WIDTH;
            instOutputShiftLeft = FALSE;
        }
        else
        {
            instOutputShiftBits = 7 - OUTPUT_INT_WIDTH-FRAC_WIDTH-FRAC_WIDTH;
            instOutputShiftLeft = TRUE;
        }
    }
    else if (op == ELT_ADD)
    {
        if (FRAC_WIDTH+1 > (7-OUTPUT_INT_WIDTH))
        {
            instOutputShiftBits = RAC_WIDTH+1 - 7 + OUTPUT_INT_WIDTH;
            instOutputShiftLeft = FALSE;
        }
        else
        {
            instOutputShiftBits = 7 - OUTPUT_INT_WIDTH-FRAC_WIDTH-1;
            instOutputShiftLeft = TRUE;
        }
    }
    else
    {
        instOutputShiftBits = 0x0;
        instOutputShiftLeft = FALSE;
    }

    instruction_generator(
                op,
                vecIAMoverInstruction,
                vecOAMoverInstruction,
                vecIATileControllerInstruction,
                vecOATileControllerInstruction,
                vecWMoverInstruction,
                vecMiscInstruction,

                //signed int memIA0DramBlockStartIndex
                0,
                //signed int memIA1DramBlockStartIndex
                0,
                //signed int memOADramBlockStartIndex
                0,
                //signed int memWeightDramBlockStartIndex
                0,
                //signed int memBiasStartIndex
                0,

                //input 0 strides
                memDramBlockIAColStride,
                memDramBlockIARowStride,
                memDramBlockIAGroupStride,

                //input 1 strides
                memDramBlockIAColStride,
                memDramBlockIARowStride,
                memDramBlockIAGroupStride,

                //output stride
                memDramBlockOAColStride,

                //weight stride
                memDramBlockFilterStride,

            #if defined(SPARSE_SYSTEM)
                //memIATB0CountStart
                0,
                //memIATB0CountColStride,
                1,
                //memOATBCountStart
                0,
                //memOATBCountColStride
                1,
                //memWeightTBCountStart
                0,
            #endif

                instFlagSparseOutput,
                instFlagSparseInput,
                //flagInputSync
                0x0,
                //flagOutputSync
                0x0,
                instFlagRelu,
                instOutputShiftBits,
                instOutputShiftLeft,

                inputWidthSPSize,
                inputHeightSPSize,
                inputWidthSPUnitSize,
                inputHeightSPUnitSize,

                horizontalBorderPadding,
                verticalBorderPadding,

                kernelSize,
                stride,

                sizeOutputTileHeight,
                sizeOutputTileWidthPerCol,
                numActiveColsPartialOutputTile,

                numInputChannel0,
                numInputChannel1,
                numGroupCurrentLayer,
                numOutputChannels,
                //numGroupsNextLayer
                1
                );

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
