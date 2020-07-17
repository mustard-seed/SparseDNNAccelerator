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

//#define PLAY
#define REPEAT 10
//#define EMULATE
#define PERF_TEST
//#NOOP
//#define PROFILE
#if defined(PROFILE)
#include "CL/cl_ext_intelfpga.h"
#endif

#define FRAC_WIDTH 4
#define INT_WIDTH 3
#define OUTPUT_INT_WIDTH 3

#define WEIGHT_SEED 1234
#define INPUT_SEED   7653

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
    cl::CommandQueue clCQMKController;
    #if defined(NOOP)
    cl::CommandQueue clCQNoop;
    #endif

    //The kernels
    cl::Kernel kernelIAMover;
    cl::Kernel kernelOAMover;
    cl::Kernel kernelWMover;
    cl::Kernel kernelMKInstructionMover;
    cl::Kernel kernelIATileController;
    cl::Kernel KernelOATileController;
    #if defined(NOOP)
    cl::Kernel kernelNoop;
    #endif

    //Buffer members associated with the IA Mover kernel
    cl::Buffer bufferIAMoverInstructions;
    cl::Buffer bufferActivationDramBlocks;
#if defined(SPARSE_SYSTEM)
    cl::Buffer bufferActivationTBCounts;
#endif

    //Buffer members associated with the IA tile controller
    cl::Buffer bufferIATileControllerInstructions;

    //Buffer members associated with the OA Mover kernel
    cl::Buffer bufferOAMoverInstructions;
//    cl::Buffer bufferOAMoverOADramBlocks;
//#if defined(SPARSE_SYSTEM)
//    cl::Buffer bufferOAMoverOATBCounts;
//#endif

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

    std::vector<fixedPointNumber> generateSparseInput (unsigned char _inputWidth,
                unsigned char _inputHeight,
                unsigned char _numInputChannel,
                unsigned char _numGroupCurrentLayer,
                float denseProb = 1.0f
                , int channelPruneScale = CLUSTER_SIZE);

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


    std::vector<fixedPointNumber> generateSparseWeights (unsigned char _kernelSize,
                unsigned char _numInputChannel,
                unsigned char _numGroups,
                float denseProb = 1.0f
            , int channelPruneScale = CLUSTER_SIZE);

    void launch (unsigned char _inputWidth,
            unsigned char _inputHeight,
            unsigned char _numInputChannel,
            unsigned char _numInputGroup, //The code will override this to 1 if the operation is not convolution
            unsigned char _numGroupNext, //Number of groups for the next layer
            unsigned char _inputHeightSPUnitSize, //The code will override this to 1 if the operation is not convolution
            unsigned char _inputWidthSPUnitSize, //The code will overide this to 1 if the operation is not convolution
            unsigned char _sizeOutputTileWidthPerColFull, //The code will override this to 1 if the operation is not convolution
            unsigned char _sizeOutputTileHeight, //The code will overrid this to 1 if the operation is not convolution
            bool _flagEnableRelu,
            bool _flagSparseInput, //The code will override this to FALSE if the operation is not convolution
            bool _flagSparseOutput,
            OPERATION op,
            float _bias = 0.0f //Only matter for convolution
            , bool flagMultiLayerConv = false
            , bool _flagPerformanceTest = false
            , float denseProb = 1
            , int pruneScale = CLUSTER_SIZE
            );
}; //testFixture

#ifdef PLAY
TEST_F (testFixture, back_to_back_identity_conv)
{
    unsigned char inputWidth = 4;
    unsigned char inputHeight = 4;
    unsigned char numInputChannel = 8;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 2;
    unsigned char sizeOutputTileHeight = 4;
    bool flagEnableRelu = false;
    bool flagSparseInput = false;
    bool flagSparseOutput = false;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;
    bool flag2Layer = true;

    launch(
                inputWidth,
                inputHeight,
                numInputChannel,
                numInputGroup,
                numOutputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                flagEnableRelu,
                flagSparseInput,
                flagSparseOutput,
                op,
                bias,
                flag2Layer
          );
}
#else
#if defined(PERF_TEST)
TEST_F (testFixture, perf_test_conv_sparse_128x128x3x3x32x16COL)
{
    unsigned char inputWidth = 8*2*PE_COLS;
    unsigned char inputHeight = 32;
    unsigned char numInputChannel = 128;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 8;
    unsigned char sizeOutputTileHeight = 8;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;
    std::vector<float> vecDenseProb = {1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0};
    std::vector<int> vecPruneScale = {1, CLUSTER_SIZE};
    for (auto& pruneScale: vecPruneScale)
    {
        for (auto & prob : vecDenseProb)
        {
            launch(
                        inputWidth,
                        inputHeight,
                        numInputChannel,
                        numInputGroup,
                        numOutputGroup,
                        inputHeightSPUnitSize,
                        inputWidthSPUnitSize,
                        sizeOutputTileWidthPerColFull,
                        sizeOutputTileHeight,
                        flagEnableRelu,
                        flagSparseInput,
                        flagSparseOutput,
                        op,
                        bias,
                        false, //back to back
                        true, //perf test
                        prob, //dense prob
                        pruneScale //channel prune scale
                  );
        }
     }
}

TEST_F (testFixture, depth_sensitivity)
{
    unsigned char inputWidth = 8*2*PE_COLS;
    unsigned char inputHeight = 32;
    std::vector<unsigned char> vecInputChannel = {32, 64, 96, 128};
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 8;
    unsigned char sizeOutputTileHeight = 8;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;
    float prob = 1.0f;
    float pruneScale = CLUSTER_SIZE;

    for (auto& numInputChannel: vecInputChannel) {
        launch(
                    inputWidth,
                    inputHeight,
                    numInputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    flagEnableRelu,
                    flagSparseInput,
                    flagSparseOutput,
                    op,
                    bias,
                    false, //back to back
                    true, //perf test
                    prob,
                    pruneScale

              );
    }
}

TEST_F (testFixture, perf_test_max_pool_sparse_128x32x32)
{
    unsigned char inputWidth = 32;
    unsigned char inputHeight = 32;
    unsigned char numInputChannel = 128;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = ((inputWidth / PE_COLS) > 8) ?
                8 : (inputWidth / PE_COLS);
    unsigned char sizeOutputTileHeight = 8;
    bool flagEnableRelu = false;
    bool flagSparseInput = false;
    bool flagSparseOutput = true;
    OPERATION op = MAX_POOL;
    float bias = 0.0f;
    int channelPruneScale = 1;
    std::vector<float> vecDenseProb = {1.0, 0.0};
    for (auto & prob : vecDenseProb)
    {
        launch(
                    inputWidth,
                    inputHeight,
                    numInputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    flagEnableRelu,
                    flagSparseInput,
                    flagSparseOutput,
                    op,
                    bias,
                    false, //back to back
                    true, //perf test
                    prob, //dense prob
                    channelPruneScale
              );
    }
}

TEST_F (testFixture, perf_test_elt_add_sparse_128x32x32)
{
    unsigned char inputWidth = 32;
    unsigned char inputHeight = 32;
    unsigned char numInputChannel = 128;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = ((inputWidth / PE_COLS) > 8) ?
                8 : (inputWidth / PE_COLS);
    unsigned char sizeOutputTileHeight = 8;
    bool flagEnableRelu = false;
    bool flagSparseInput = false;
    bool flagSparseOutput = true;
    OPERATION op = ELT_ADD;
    float bias = 0.0f;
    int channelPruneScale = 1;
    std::vector<float> vecDenseProb = {1.0, 0.0};
    for (auto & prob : vecDenseProb)
    {
        launch(
                    inputWidth,
                    inputHeight,
                    numInputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    flagEnableRelu,
                    flagSparseInput,
                    flagSparseOutput,
                    op,
                    bias,
                    false, //back to back
                    true, //perf test
                    prob, //dense prob
                    channelPruneScale
              );
    }
}

TEST_F (testFixture, perf_test_concat_sparse_64x32x32)
{
    unsigned char inputWidth = 32;
    unsigned char inputHeight = 32;
    unsigned char numInputChannel = 64;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = ((inputWidth / PE_COLS) > 8) ?
                8 : (inputWidth / PE_COLS);
    unsigned char sizeOutputTileHeight = 8;
    bool flagEnableRelu = false;
    bool flagSparseInput = false;
    bool flagSparseOutput = true;
    OPERATION op = CONCATENATION;
    float bias = 0.0f;
    int channelPruneScale = 1;
    std::vector<float> vecDenseProb = {1.0, 0.0};
    for (auto & prob : vecDenseProb)
    {
        launch(
                    inputWidth,
                    inputHeight,
                    numInputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    flagEnableRelu,
                    flagSparseInput,
                    flagSparseOutput,
                    op,
                    bias,
                    false, //back to back
                    true, //perf test
                    prob, //dense prob
                    channelPruneScale
              );
    }
}
#else //PERF_TEST
TEST_F (testFixture, conv_dense_input_dense_output_plain)
{
    unsigned char inputWidth = 4;
    unsigned char inputHeight = 4;
    unsigned char numInputChannel = 8;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 2;
    unsigned char sizeOutputTileHeight = 4;
    bool flagEnableRelu = false;
    bool flagSparseInput = false;
    bool flagSparseOutput = false;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;

    launch(
                inputWidth,
                inputHeight,
                numInputChannel,
                numInputGroup,
                numOutputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                flagEnableRelu,
                flagSparseInput,
                flagSparseOutput,
                op,
                bias
          );
}

TEST_F (testFixture, conv_dense_input_dense_output_grouped)
{
    unsigned char inputWidth = 4;
    unsigned char inputHeight = 4;
    unsigned char numInputChannel = 8;
    unsigned char numInputGroup = 2;
    unsigned char numOutputGroup = 2;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 2;
    unsigned char sizeOutputTileHeight = 4;
    bool flagEnableRelu = false;
    bool flagSparseInput = false;
    bool flagSparseOutput = false;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;

    launch(
                inputWidth,
                inputHeight,
                numInputChannel,
                numInputGroup,
                numOutputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                flagEnableRelu,
                flagSparseInput,
                flagSparseOutput,
                op,
                bias
          );
}

TEST_F (testFixture, conv_dense_input_dense_output_strided)
{
    unsigned char inputWidth = 4;
    unsigned char inputHeight = 4;
    unsigned char numInputChannel = 8;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 2;
    unsigned char inputWidthSPUnitSize = 2;
    unsigned char sizeOutputTileWidthPerColFull = 2;
    unsigned char sizeOutputTileHeight = 4;
    bool flagEnableRelu = false;
    bool flagSparseInput = false;
    bool flagSparseOutput = false;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;

    launch(
                inputWidth,
                inputHeight,
                numInputChannel,
                numInputGroup,
                numOutputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                flagEnableRelu,
                flagSparseInput,
                flagSparseOutput,
                op,
                bias
          );
}

TEST_F (testFixture, conv_sparse_input_sparse_output)
{
    unsigned char inputWidth = 4;
    unsigned char inputHeight = 4;
    unsigned char numInputChannel = 8;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 2;
    unsigned char sizeOutputTileHeight = 4;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;

    launch(
                inputWidth,
                inputHeight,
                numInputChannel,
                numInputGroup,
                numOutputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                flagEnableRelu,
                flagSparseInput,
                flagSparseOutput,
                op,
                bias
          );
}

TEST_F (testFixture, max_pool_sparse_output_grouped)
{
    unsigned char inputWidth = 4;
    unsigned char inputHeight = 4;
    unsigned char numInputChannel = 8;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 2;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 2;
    unsigned char sizeOutputTileHeight = 4;
    bool flagEnableRelu = false;
    bool flagSparseInput = false;
    bool flagSparseOutput = true;
    OPERATION op = MAX_POOL;
    float bias = 0.0f;

    launch(
                inputWidth,
                inputHeight,
                numInputChannel,
                numInputGroup,
                numOutputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                flagEnableRelu,
                flagSparseInput,
                flagSparseOutput,
                op,
                bias
          );
}

TEST_F (testFixture, elt_add_sparse_output)
{
    unsigned char inputWidth = 4;
    unsigned char inputHeight = 4;
    unsigned char numInputChannel = 8;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 2;
    unsigned char sizeOutputTileHeight = 4;
    bool flagEnableRelu = false;
    bool flagSparseInput = false;
    bool flagSparseOutput = true;
    OPERATION op = ELT_ADD;
    float bias = 0.0f;

    launch(
                inputWidth,
                inputHeight,
                numInputChannel,
                numInputGroup,
                numOutputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                flagEnableRelu,
                flagSparseInput,
                flagSparseOutput,
                op,
                bias
          );
}

TEST_F (testFixture, concat_sparse_output_grouped)
{
    unsigned char inputWidth = 4;
    unsigned char inputHeight = 4;
    unsigned char numInputChannel = 8;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 2;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 2;
    unsigned char sizeOutputTileHeight = 4;
    bool flagEnableRelu = false;
    bool flagSparseInput = false;
    bool flagSparseOutput = true;
    OPERATION op = CONCATENATION;
    float bias = 0.0f;

    launch(
                inputWidth,
                inputHeight,
                numInputChannel,
                numInputGroup,
                numOutputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                flagEnableRelu,
                flagSparseInput,
                flagSparseOutput,
                op,
                bias
          );
}

TEST_F (testFixture, back_to_back_identity_conv)
{
    unsigned char inputWidth = 4;
    unsigned char inputHeight = 4;
    unsigned char numInputChannel = 8;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 2;
    unsigned char sizeOutputTileHeight = 4;
    bool flagEnableRelu = false;
    bool flagSparseInput = false;
    bool flagSparseOutput = false;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;
    bool flag2Layer = true;

    launch(
                inputWidth,
                inputHeight,
                numInputChannel,
                numInputGroup,
                numOutputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                flagEnableRelu,
                flagSparseInput,
                flagSparseOutput,
                op,
                bias,
                flag2Layer
          );
}
#endif //PERF_TEST
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
    binaryFile = "sparse_pe_system.aocx";
#if defined(EMULATE)
    clPlatform = aocl_utils_cpp::findPlatform("Intel(R) FPGA Emulation Platform for OpenCL(TM)");
    //Hacke
    binaryFile = "smallBuffer.aocx";
#else
    clPlatform = aocl_utils_cpp::findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
#endif
#endif

    //Setup and platform and the context
    std::vector<cl::Device> devices;
    status = clPlatform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    aocl_utils_cpp::checkError(status, "Failed to query the devices");
    clDevice = devices[0];
    #if defined(C5SOC)
        clContext = cl::Context({clDevice}
                                ,NULL
                                ,&aocl_utils_cpp::oclContextCallback
                                ,NULL
                                ,&status);
    #else
        clContext = cl::Context(clDevice
                                ,NULL
                                ,&aocl_utils_cpp::oclContextCallback
                                ,NULL
                                ,&status);
        aocl_utils_cpp::checkError(status, "Failed to create context");
    #endif

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

#if defined(NOOP)
    kernelNoop = cl::Kernel(program, "kernelNoop", &status);
    aocl_utils_cpp::checkError(status, "Failed to create kernelNoop!");
#endif
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

    clCQMKController = cl::CommandQueue(
                clContext,
                clDevice,
                CL_QUEUE_PROFILING_ENABLE,
                &status
                );
    aocl_utils_cpp::checkError(status, "Failed to setup the command queue clMKController!");

#if defined(NOOP)
    clCQNoop = cl::CommandQueue(
                clContext,
                clDevice,
                CL_QUEUE_PROFILING_ENABLE,
                &status
                );
    aocl_utils_cpp::checkError(status, "Failed to setup the command queue clCQNoop!");
#endif
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
    vecBufferInfo.push_back({inputWeightBufferSize, bufferWMoverWDramBlocks, CL_MEM_READ_ONLY, "bufferWMoverWDramBlocks"});

    cl_ulong inputBiasSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_BIAS ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_BIAS;
    vecBufferInfo.push_back({inputBiasSize, bufferWMoverBias, CL_MEM_READ_ONLY, "bufferWMoverBias"});

    cl_ulong activationSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_ACTIVATION ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_ACTIVATION;
    vecBufferInfo.push_back({activationSize, bufferActivationDramBlocks, CL_MEM_READ_WRITE, "bufferActivationDramBlocks"});

    cl_ulong inputIAMoverInstructionSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_MOVER_INSTRUCTION ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_MOVER_INSTRUCTION;
    vecBufferInfo.push_back({inputIAMoverInstructionSize, bufferIAMoverInstructions, CL_MEM_READ_ONLY, "bufferIAMoverInstructions"});

    cl_ulong inputIATileControllerInstructionSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_TILE_CONTROLLER_INSTRUCTION ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_TILE_CONTROLLER_INSTRUCTION;
    vecBufferInfo.push_back({inputIATileControllerInstructionSize, bufferIATileControllerInstructions, CL_MEM_READ_ONLY, "bufferIATileControllerInstructions"});

    cl_ulong outputOAMoverInstructionSize = maxBufferSizeByte < MAX_DRAM_BYTE_OUTPUT_MOVER_INSTRUCTION ? maxBufferSizeByte : MAX_DRAM_BYTE_OUTPUT_MOVER_INSTRUCTION;
    vecBufferInfo.push_back({outputOAMoverInstructionSize, bufferOAMoverInstructions, CL_MEM_READ_ONLY, "bufferOAMoverInstructions"});

    cl_ulong outoutOATileControllerInstructionSize = maxBufferSizeByte < MAX_DRAM_BYTE_OUTPUT_TILE_CONTROLLER_INSTRUCTION ? maxBufferSizeByte : MAX_DRAM_BYTE_OUTPUT_TILE_CONTROLLER_INSTRUCTION;
    vecBufferInfo.push_back({outoutOATileControllerInstructionSize, bufferOATileControllerInstructions, CL_MEM_READ_ONLY, "bufferOATileControllerInstructions"});

    cl_ulong mkInstructionSize = maxBufferSizeByte < MAX_DRAM_BYTE_MISC_CONTROLLER_INSTRUCTION ? maxBufferSizeByte : MAX_DRAM_BYTE_MISC_CONTROLLER_INSTRUCTION;
    vecBufferInfo.push_back({mkInstructionSize, bufferMKInstructions, CL_MEM_READ_ONLY, "bufferMKInstructions"});

#if defined(SPARSE_SYSTEM)
    cl_ulong inputWeightSBSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_WEIGHT_SB_COUNT ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_WEIGHT_SB_COUNT;
    vecBufferInfo.push_back({inputWeightSBSize, bufferWMoverWTBCounts, CL_MEM_READ_ONLY, "bufferWMoverWTBCounts"});

    cl_ulong activationTBCountSize = maxBufferSizeByte < MAX_DRAM_BYTE_INPUT_ACTIVATION_SB_COUNT ? maxBufferSizeByte : MAX_DRAM_BYTE_INPUT_ACTIVATION_SB_COUNT;
    vecBufferInfo.push_back({activationTBCountSize, bufferActivationTBCounts, CL_MEM_READ_WRITE, "bufferActivationTBCounts"});
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
            unsigned char _inputWidth,
            unsigned char _inputHeight,
            unsigned char _numInputChannel,
            unsigned char _numGroupCurrentLayer
        )
{
    assert(_numInputChannel % _numGroupCurrentLayer == 0);
    unsigned char numICPerGroup = _numInputChannel / _numGroupCurrentLayer;
    std::vector<fixedPointNumber> inputFPVector;
    for (unsigned char g=0; g<_numGroupCurrentLayer;g++)
    {
        for (unsigned char h=0; h<_inputHeight; h++)
        {
            for (unsigned char w=0; w<_inputWidth; w++)
            {
                for (unsigned char c=0; c<numICPerGroup; c++)
                {
                    unsigned char globalChannel = g*numICPerGroup+c;
                    signed char fpBits = (w % 2 == 0) ? globalChannel : -1*((signed char) globalChannel);
                    fixedPointNumber fpNumber(fpBits, FRAC_WIDTH, INT_WIDTH);
                    inputFPVector.push_back(fpNumber);
                }
            }
        }
    }

    return inputFPVector;
}

std::vector<fixedPointNumber> testFixture::generateSparseInput(
            unsigned char _inputWidth,
            unsigned char _inputHeight,
            unsigned char _numInputChannel,
            unsigned char _numGroupCurrentLayer,
            float denseProb,
            int channelPruneScale
        )
{
     std::mt19937 generator(INPUT_SEED);
     std::bernoulli_distribution bernDistribution(denseProb);
     assert(_numInputChannel % _numGroupCurrentLayer == 0);
     unsigned char numICPerGroup = _numInputChannel / _numGroupCurrentLayer;
     std::vector<fixedPointNumber> inputFPVector;
     bool writePositive = true;
     for (unsigned char g=0; g<_numGroupCurrentLayer;g++)
     {
         for (unsigned char h=0; h<_inputHeight; h++)
         {
             for (unsigned char w=0; w<_inputWidth; w++)
             {
                 bool flagNonZero = false;
                 for (unsigned char c=0; c<numICPerGroup; c++)
                 {
                     if (((int) c) % channelPruneScale == 0)
                     {
                         flagNonZero = bernDistribution(generator);
                     }
                     signed char fpBits = (writePositive) ? 1 : -1;
                     if (flagNonZero == false)
                     {
                         fpBits = 0;
                     }
                     else
                     {
                         writePositive = !writePositive;
                     }
                     fixedPointNumber fpNumber(fpBits, FRAC_WIDTH, INT_WIDTH);
                     inputFPVector.push_back(fpNumber);
                 }
             }
         }
     }

     return inputFPVector;
}

std::vector<fixedPointNumber> testFixture::generateWeights (
        unsigned char _kernelSize,
        unsigned char _numInputChannel,
        unsigned char _numGroups
        )
{
    assert(_kernelSize % 2 == 1);
    assert(_numInputChannel % _numGroups == 0);
    std::vector<fixedPointNumber> fpWeightTensor;
    unsigned char numICPerGroup = _numInputChannel / _numGroups;

    for (unsigned char g=0; g<_numGroups; g++)
    {
        //Number of OC per group equals to the number of IC per group
        for (unsigned char iFilter=0; iFilter<numICPerGroup; iFilter++)
        {
            for (unsigned char iH=0; iH<_kernelSize; iH++)
            {
                bool hCentre = (iH == (_kernelSize / 2));
                for (unsigned char iW=0; iW<_kernelSize; iW++)
                {
                    bool vCentre = (iW == (_kernelSize / 2));
                    for (unsigned char iC=0; iC<numICPerGroup; iC++)
                    {
                        bool isOne = (hCentre == true) && (vCentre == true) && (iFilter == iC);
                        float floatWeight = isOne ? 1.0f : 0.0f;
                        fixedPointNumber fpWeight(floatWeight, FRAC_WIDTH, INT_WIDTH);
                        fpWeightTensor.push_back(fpWeight);
                    }
                }
            }
        }
    }

    return fpWeightTensor;
}

std::vector<fixedPointNumber> testFixture::generateSparseWeights (
        unsigned char _kernelSize,
        unsigned char _numInputChannel,
        unsigned char _numGroups,
        float denseProb,
        int channelPruneScale
        )
{
    assert(_kernelSize % 2 == 1);
    assert(_numInputChannel % _numGroups == 0);
    std::vector<fixedPointNumber> fpWeightTensor;
    unsigned char numICPerGroup = _numInputChannel / _numGroups;

    std::mt19937 generator(WEIGHT_SEED);
    std::bernoulli_distribution bernDistribution(denseProb);
    bool writePositive = true;
    for (unsigned char g=0; g<_numGroups; g++)
    {
        //Number of OC per group equals to the number of IC per group
        for (unsigned char iFilter=0; iFilter<numICPerGroup; iFilter++)
        {
            for (unsigned char iH=0; iH<_kernelSize; iH++)
            {
                for (unsigned char iW=0; iW<_kernelSize; iW++)
                {
                    bool flagNonZero = false;
                    for (unsigned char iC=0; iC<numICPerGroup; iC++)
                    {
                        if (((int) iC) % channelPruneScale == 0)
                        {
                            flagNonZero = bernDistribution(generator);
                        }
                        signed char fpBits = (writePositive) ? 1 : -1;
                        if (flagNonZero == false)
                        {
                            fpBits = 0;
                        }
                        else
                        {
                            writePositive = !writePositive;
                        }
                        fixedPointNumber fpNumber(fpBits, FRAC_WIDTH, INT_WIDTH);
                        fpWeightTensor.push_back(fpNumber);
                    }
                }
            }
        }
    }

    return fpWeightTensor;
}

void testFixture::launch (
        unsigned char _inputWidth,
        unsigned char _inputHeight,
        unsigned char _numInputChannel,
        unsigned char _numInputGroup, //The code will override this to 1 if the operation is not convolution
        unsigned char _numGroupNext, //Number of groups for the next layer
        unsigned char _inputHeightSPUnitSize, //The code will override this to 1 if the operation is not convolution
        unsigned char _inputWidthSPUnitSize, //The code will overide this to 1 if the operation is not convolution
        unsigned char _sizeOutputTileWidthPerColFull, //The code will override this to 1 if the operation is not convolution
        unsigned char _sizeOutputTileHeight, //The code will overrid this to 1 if the operation is not convolution
        bool _flagEnableRelu,
        bool _flagSparseInput, //The code will override this to FALSE if the operation is not convolution or if the accelerator only supports dense format
        bool _flagSparseOutput, //The code will override this to FALSE if the accelerator only supports dense format.
        OPERATION op,
        float _bias, //Only matter for convolution
        bool flagMultiLayerConv,
        bool _flagPerformanceTest,
        float denseProb,
        int pruneScale
        )
{
    //Checking the parameters' consistency AND
    //Derive parameters
    assert(_inputHeight % 2 == 0);
    assert(_inputWidth % 2 == 0);
    //assert(_numInputChannel <= 127);

    cl_uchar kernelSize;
    cl_uchar stride;
    unsigned char numInputChannel0;
    unsigned char numInputChannel1;
    unsigned char numOutputChannels;
    unsigned char numOutputChannelPerGroup;
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
            assert((flagMultiLayerConv==false) || (_numInputGroup == 1));
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

            numOutputChannels = numInputChannel0;

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
            numOutputChannels = numInputChannel0;

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
            numInputChannel1 = _numInputChannel;
            inputHeightSPUnitSize = 1;
            inputWidthSPUnitSize = 1;
            numGroupCurrentLayer = 1;
            sizeOutputTileWidthPerCol = 1;
            sizeOutputTileHeight = 1;
            kernelSize = 1;
            stride = 1;
            numOutputChannels = numInputChannel0;

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
            numOutputChannels = numInputChannel0 + numInputChannel1;

            flagSparseInput = false;
#if defined(SPARSE_SYSTEM)
            flagSparseOutput = _flagSparseOutput;
#else
            flagSparseOutput = false;
#endif
        }
        break;
    } //switch
    inputWidthSPSize = inputWidthSPUnitSize*(_inputWidth-1) + 1;
    inputHeightSPSize = inputHeightSPUnitSize*(_inputHeight-1) + 1;
    numOutputWidth = (op==MAX_POOL) ? inputWidthSPSize / 2 : inputWidthSPSize;
    numOutputHeight = (op==MAX_POOL) ? inputHeightSPSize / 2 : inputHeightSPSize;
    verticalBorderPadding = (op==CONVOLUTION) ? 1 : 0;
    horizontalBorderPadding  = (op==CONVOLUTION) ? 1 : 0;
    unsigned char numActiveColsPartialOutputTile = (op==CONVOLUTION) ?
                1 : (numOutputWidth % PE_COLS);
    assert(numOutputChannels % _numGroupNext == 0);
    numOutputChannelPerGroup = numOutputChannels / _numGroupNext;

    /* Generate the dense, fixed point tensors
     * */
    int stepCount = 1;
    std::cout <<stepCount++<<". Preparing the test tensors. Test operation type is "<<op<<std::endl;
    if (_flagPerformanceTest == true)
    {
        std::cout <<"This is a performance test"<<std::endl
                  <<"Density, prune scale: "<<denseProb<<" "<<pruneScale<<std::endl;
    }
    std::cout <<"Input SP dimensions (H, W):  "<<(unsigned int) inputHeightSPSize<<" "<<(unsigned int) inputWidthSPSize<<std::endl
              <<"PE dimension (H, W): "<<PE_ROWS<<" "<<PE_COLS<<std::endl
              <<"CLUSTER_SIZE: "<<CLUSTER_SIZE<<std::endl
              <<"TRANSFER_SIZE: "<<TRANSFER_SIZE<<std::endl
              <<"WIDE_SIZE "<<WIDE_SIZE<<std::endl
              <<"Output planar dimensions (H, W): "<<(unsigned int)numOutputHeight<<" "<<(unsigned int)numOutputWidth<<std::endl
              <<"Full output tile per col planar sizes (H, W): "<<(unsigned int)sizeOutputTileHeight<<" "<<(unsigned int)sizeOutputTileWidthPerCol<<std::endl
              <<"Input channels 0: "<<(unsigned int) numInputChannel0<<std::endl
              <<"Input channels 1: "<<(unsigned int) numInputChannel1<<std::endl
              <<"Output channels: "<<(unsigned int) numOutputChannels<<std::endl
              <<"Number of output groups: "<<(unsigned int)_numGroupNext<<std::endl
              <<"Number of groups in current layer: "<<(unsigned int)numGroupCurrentLayer<<std::endl;

    std::vector<fixedPointNumber> inputTensorDense;
    if (_flagPerformanceTest == true)
    {
       inputTensorDense = generateSparseInput(_inputWidth, _inputHeight, _numInputChannel, numGroupCurrentLayer, denseProb, pruneScale);
    }
    else
    {
        inputTensorDense = generateInputTensor(_inputWidth, _inputHeight, _numInputChannel, numGroupCurrentLayer);
    }

    std::vector<fixedPointNumber> inputWeightDense;
    if (op == CONVOLUTION)
    {
        if (_flagPerformanceTest == true)
        {
            inputWeightDense = generateSparseWeights((unsigned char) kernelSize, _numInputChannel, numGroupCurrentLayer, denseProb, pruneScale);
        }
        else
        {
            inputWeightDense = generateWeights((unsigned char) kernelSize, _numInputChannel, numGroupCurrentLayer);
        }
    }
    t_accumulator fixedBias = (t_accumulator) (round(_bias * (float) (1 << (FRAC_WIDTH + FRAC_WIDTH)) ));
    t_aligned_short_vector biasVector (_numInputChannel, fixedBias);

    /* 2. Allocate the aligned tensors and compress them if necessary
     * */
    std::cout <<stepCount++<<". Allocate, align, and compress the test tensors."<<std::endl;

    unsigned short maxScalarIndexInChannelGroup = _numInputChannel / numGroupCurrentLayer - 1;
    unsigned short maxClusterIndexInCompressionBlock = COMPRESSION_WINDOW_SIZE-1;
    unsigned short maxClusterIndexInTransferBlock = TRANSFER_SIZE-1;
    unsigned short maxScalarIndexInCluster = CLUSTER_SIZE-1;

    std::unique_ptr<AlignedTensor> pInput, pWeights, pOutput;

    //Prepare the input
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
    }

    //Prepare the output
    if (flagSparseOutput == true)
    {
        pOutput.reset(new FlexibleDirectCompressedTensor(
                    1, //_num3DTensors
                    numOutputChannels, //_channel
                    numOutputWidth, //_width
                    numOutputHeight, //_height
                    numOutputChannelPerGroup-1, //_maxScalarIndexInChannelGroup
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
                    numOutputChannelPerGroup-1, //_maxScalarIndexInChannelGroup
                    maxClusterIndexInTransferBlock,
                    maxScalarIndexInCluster,
                    false //isKernel
                    ));
    }

    //Prepare weights
    if (op==CONVOLUTION)
    {
#if defined(SPARSE_SYSTEM)
        pWeights.reset(new FlexibleDirectCompressedTensor (
                    inputWeightDense,
                    _numInputChannel, //_num3DTensors
                    maxScalarIndexInChannelGroup+1, //channel
                    (unsigned char) kernelSize, //width
                    (unsigned char) kernelSize, //height
                    maxScalarIndexInChannelGroup,
                    maxClusterIndexInCompressionBlock,
                    maxClusterIndexInTransferBlock,
                    maxScalarIndexInCluster,
                    true //isKernel
                ) );
#else
        pWeights.reset( new AlignedTensor (
                    inputWeightDense,
                    _numInputChannel, //_num3DTensors
                    maxScalarIndexInChannelGroup+1, //channel
                    (unsigned char) kernelSize, //width
                    (unsigned char) kernelSize, //height
                    maxScalarIndexInChannelGroup,
                    maxClusterIndexInTransferBlock,
                    maxScalarIndexInCluster,
                    true //isKernel
                ));
#endif
    }
    std::cout <<stepCount++<<". Generate the instructions"<<std::endl;

    /*
     * 3. Generate the instructions
     * */
    t_aligned_ia_mover_instruction_vector vecIAMoverInstruction;
    t_aligned_ia_tile_controller_instruction_vector vecIATileControllerInstruction;
    t_aligned_oa_mover_instruction_vector vecOAMoverInstruction;
    t_aligned_oa_tile_controller_instruction_vector vecOATileControllerInstruction;
    t_aligned_weight_mover_instruction_vector vecWMoverInstruction;
    t_aligned_misc_instruction_vector vecMiscInstruction;

    signed int memDramBlockFilterStride;
    //assign memDramBlockFilterStride conditionally, otherwise there might be seg fault.
    if (op == CONVOLUTION)
    {
        memDramBlockFilterStride = (pWeights->getExternalMemoryAddressStride()) >> WIDE_SIZE_OFFSET;
    }
    else
    {
        memDramBlockFilterStride = 0;
    }

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
            instOutputShiftBits = FRAC_WIDTH+FRAC_WIDTH - 7 + OUTPUT_INT_WIDTH;
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
        if (FRAC_WIDTH > (7-OUTPUT_INT_WIDTH))
        {
            instOutputShiftBits = FRAC_WIDTH - 7 + OUTPUT_INT_WIDTH;
            instOutputShiftLeft = FALSE;
        }
        else
        {
            instOutputShiftBits = 7 - OUTPUT_INT_WIDTH-FRAC_WIDTH;
            instOutputShiftLeft = TRUE;
        }
    }
    else
    {
        instOutputShiftBits = 0x0;
        //Gottcha!!! Need to be shift left!!!
        instOutputShiftLeft = TRUE;
    }

    if (op==CONVOLUTION && flagMultiLayerConv == true)
    {
        //Number of compression block per intermediate channel group
        int numCBPerIMOAChannelGroup =
                1 + (numInputChannel0 / numGroupCurrentLayer - 1) / (CLUSTER_SIZE * COMPRESSION_WINDOW_SIZE);
        int numTBPerCB = COMPRESSION_WINDOW_SIZE / TRANSFER_SIZE + 1;
        int intermediateAColStride = numCBPerIMOAChannelGroup * numTBPerCB;
        int intermediateARowStride = intermediateAColStride * _inputWidth;
        int intermediateAGroupStride = intermediateARowStride * _inputHeight;
        //First set of instructions
        instruction_generator(
                    op,
                    vecIAMoverInstruction,
                    vecOAMoverInstruction,
                    vecIATileControllerInstruction,
                    vecOATileControllerInstruction,
                    vecWMoverInstruction,
                    vecMiscInstruction,

                    //signed int memIA0DramBlockStartIndex
                    MEM_START_ACTIVATION_0,
                    //signed int memIA1DramBlockStartIndex
                    MEM_START_ACTIVATION_0,
                    //signed int memOADramBlockStartIndex
                    MEM_START_ACTIVATION_1,
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
                    intermediateAColStride,

                    //weight stride
                    memDramBlockFilterStride,

                #if defined(SPARSE_SYSTEM)
                    //memIATB0CountStart
                    MEM_START_TB_0,
                    //memIATB0CountColStride,
                    1,
                    //memOATBCountStart
                    MEM_START_TB_1,
                    //memOATBCountColStride
                    1,
                    //memWeightTBCountStart
                    0,
                #endif

                    //flagSparseOutput,
                    TRUE,
                    instFlagSparseInput,
                    //flagInputSync
                    TRUE,
                    //flagOutputSync
                    TRUE,
                    //instFlagRelu,
                    FALSE,
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
                    numGroupCurrentLayer
                    );

        //2 SECOND set of instructions
        instruction_generator(
                    op,
                    vecIAMoverInstruction,
                    vecOAMoverInstruction,
                    vecIATileControllerInstruction,
                    vecOATileControllerInstruction,
                    vecWMoverInstruction,
                    vecMiscInstruction,

                    //signed int memIA0DramBlockStartIndex
                    MEM_START_ACTIVATION_1,
                    //signed int memIA1DramBlockStartIndex
                    MEM_START_ACTIVATION_1,
                    //signed int memOADramBlockStartIndex
                    MEM_START_ACTIVATION_0,
                    //signed int memWeightDramBlockStartIndex
                    0,
                    //signed int memBiasStartIndex
                    0,

                    //input 0 strides
                    intermediateAColStride,
                    intermediateARowStride,
                    intermediateAGroupStride,

                    //input 1 strides
                    intermediateAColStride,
                    intermediateARowStride,
                    intermediateAGroupStride,

                    //output stride
                    memDramBlockOAColStride,

                    //weight stride
                    memDramBlockFilterStride,

                #if defined(SPARSE_SYSTEM)
                    //memIATB0CountStart
                    MEM_START_TB_1,
                    //memIATB0CountColStride,
                    1,
                    //memOATBCountStart
                    MEM_START_TB_0,
                    //memOATBCountColStride
                    1,
                    //memWeightTBCountStart
                    0,
                #endif

                    instFlagSparseOutput,
                    //instFlagSparseInput,
                    TRUE,
                    //flagInputSync
                    FALSE,
                    //flagOutputSync
                    FALSE,
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
                    _numGroupNext
                    );
    } //multilayer-convolution
    else
    {
        instruction_generator(
                    op,
                    vecIAMoverInstruction,
                    vecOAMoverInstruction,
                    vecIATileControllerInstruction,
                    vecOATileControllerInstruction,
                    vecWMoverInstruction,
                    vecMiscInstruction,

                    //signed int memIA0DramBlockStartIndex
                    MEM_START_ACTIVATION_0,
                    //signed int memIA1DramBlockStartIndex
                    MEM_START_ACTIVATION_0,
                    //signed int memOADramBlockStartIndex
                    MEM_START_ACTIVATION_1,
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
                    MEM_START_TB_0,
                    //memIATB0CountColStride,
                    1,
                    //memOATBCountStart
                    MEM_START_TB_1,
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
                    _numGroupNext
                    );
    }



    std::cout <<"Number of IA Mover instructions: "<<vecIAMoverInstruction.size()<<std::endl;
    std::cout <<"Number of IA tile controller instructions: "<<vecIATileControllerInstruction.size()<<std::endl;
    std::cout <<"Number of OA Mover instructions: "<<vecOAMoverInstruction.size()<<std::endl;
    std::cout <<"Number of OA tile contrller instructions: "<<vecOATileControllerInstruction.size()<<std::endl;
    std::cout <<"Number of W Mover instructions: "<<vecWMoverInstruction.size()<<std::endl;
    std::cout <<"Number of MK controller instructions: "<<vecMiscInstruction.size()<<std::endl;

    std::cout <<stepCount++<<". Setting kernel arguments for the IA Mover."<<std::endl;
    {
        cl_uint argIdx = 0;
        //volatile __global t_dram_block* restrict pIA
        kernelIAMover.setArg(argIdx, bufferActivationDramBlocks);
        argIdx++;
        #if defined(SPARSE_SYSTEM)
            //volatile __global t_streamblock_address* restrict pTBCount,
            kernelIAMover.setArg(argIdx, bufferActivationTBCounts);
            argIdx++;
        #endif
        //volatile __global t_ia_mover_instruction* restrict pInstruction,
        kernelIAMover.setArg(argIdx, bufferIAMoverInstructions);
        argIdx++;
        //unsigned int numInstruction
       kernelIAMover.setArg(argIdx, (cl_uint) vecIAMoverInstruction.size());
    }

    std::cout <<stepCount++<<". Setting kernel arguments for the IA Tile controller."<<std::endl;
    {
        cl_uint argIdx = 0;
        //__global volatile t_ia_tile_controller_instruction* restrict pInstruction,
        kernelIATileController.setArg(argIdx, bufferIATileControllerInstructions);
        argIdx++;

        //unsigned int numInstruction
        kernelIATileController.setArg(argIdx, (cl_uint) vecIATileControllerInstruction.size());
    }

    std::cout <<stepCount++<<". Setting kernel arguments for the Filter mover."<<std::endl;
    {
        cl_uint argIdx = 0;
        //volatile __global t_weight_mover_instruction* restrict pInst,
        kernelWMover.setArg(argIdx++, bufferWMoverInstructions);
        //volatile __global t_dram_block* restrict pW,
        kernelWMover.setArg(argIdx++, bufferWMoverWDramBlocks);
        //vola<tile __global t_accumulator* restrict pBias,
        kernelWMover.setArg(argIdx++, bufferWMoverBias);
        #if defined(SPARSE_SYSTEM)
            //volatile __global t_streamblock_address* restrict pFilterTBCount,
            kernelWMover.setArg(argIdx++, bufferWMoverWTBCounts);
        #endif //SPARSE_SYSTEM
        //unsigned int numInstruction
        kernelWMover.setArg(argIdx++, (cl_uint) vecWMoverInstruction.size());
    }

    std::cout <<stepCount++<<". Setting kernel arguments for the Miscellaneous controller."<<std::endl;
    {
        cl_uint argIdx=0;
        //__global t_misc_instruction* restrict pInstruction,
        kernelMKInstructionMover.setArg(argIdx++, bufferMKInstructions);
        //unsigned int numInstruction
        kernelMKInstructionMover.setArg(argIdx++, (cl_uint)vecMiscInstruction.size());
    }

    std::cout <<stepCount++<<". Setting kernel arguments for the OA mover."<<std::endl;
    {
        cl_uint argIdx=0;
        //volatile __global t_output_dram_block* restrict pOA,
        kernelOAMover.setArg(argIdx++, bufferActivationDramBlocks);
        #if defined(SPARSE_SYSTEM)
            //volatile __global t_streamblock_address* restrict pTBCount,
            kernelOAMover.setArg(argIdx++, bufferActivationTBCounts);
        #endif
        //volatile __global t_oa_mover_instruction* restrict pInstruction,
        kernelOAMover.setArg(argIdx++, bufferOAMoverInstructions);
        //unsigned int numInstruction
        kernelOAMover.setArg(argIdx++, (cl_uint) vecOAMoverInstruction.size());
    }

    std::cout <<stepCount++<<". Setting kernel arguments for the OA tile controller."<<std::endl;
    {
        cl_uint argIdx=0;
        //volatile  __global t_oa_tile_controller_instruction* restrict pInst,
        KernelOATileController.setArg(argIdx++, bufferOATileControllerInstructions);
        //unsigned int numInstructions
        KernelOATileController.setArg(argIdx++, (cl_uint) vecOATileControllerInstruction.size());
    }

    cl_int status;

    //Also check the output activation size
    {
        int oaSizeBytes = sizeof(typeof((pOutput->getTransferBlockVector()).at(0))) * (pOutput->getTransferBlockVector()).size();
        int oaOffset = ((op == CONVOLUTION) && (flagMultiLayerConv == true)) ?
                MEM_START_ACTIVATION_0 * BURST_SIZE_BYTE : MEM_START_ACTIVATION_1 * BURST_SIZE_BYTE;
        assert(oaOffset+oaSizeBytes <= MAX_DRAM_BYTE_INPUT_ACTIVATION && "Too many output activation bytes to fit inside the global memory" );
        #if defined(SPARSE_SYSTEM)
            if (flagSparseOutput == true)
            {
                int oaTBSizeBytes = sizeof(typeof((pOutput->getTransferBlockCountVector()).at(0))) * (pOutput->getTransferBlockCountVector()).size();
                int oaTBOffset = ((op == CONVOLUTION) && (flagMultiLayerConv == true)) ?
                        MEM_START_TB_0 * sizeof(t_streamblock_address) : MEM_START_TB_1 * sizeof(t_streamblock_address);
                assert(oaTBOffset+oaTBSizeBytes <= MAX_DRAM_BYTE_INPUT_ACTIVATION_SB_COUNT && "Too many output activation TB counts to fit inside the global memory" );
            }
        #endif
    }

    //Transfer the input
    std::cout <<stepCount++<<". Transfer the input activations "<<std::endl;
    {
        cl::Event event;
        auto numTransferBlocks = (pInput->getTransferBlockVector()).size();
        auto sizeTransferBlockElement = sizeof(typeof((pInput->getTransferBlockVector()).at(0)));
        auto valueVectorSizeBytes = sizeTransferBlockElement * numTransferBlocks;

        int activationOffsetByte = MEM_START_ACTIVATION_0 * BURST_SIZE_BYTE;

        std::cout <<"Transfering "<<valueVectorSizeBytes<<" bytes into bufferActivationDramBlocks"<<std::endl;
        assert(activationOffsetByte+valueVectorSizeBytes <= MAX_DRAM_BYTE_INPUT_ACTIVATION && "Too many input activation bytes to fit inside the global memory" );

        status = clCQIAMover.enqueueWriteBuffer(bufferActivationDramBlocks, //buffer
                                             CL_TRUE, //blocking_write
                                             activationOffsetByte, //offset
                                             valueVectorSizeBytes, //size
                                             (pInput->getTransferBlockVector()).data(), //data pointer
                                             NULL, //dependency list
                                             &event //events generated
                                            );
        aocl_utils_cpp::checkError(status, "Failed to write the input activation vector");
        clCQIAMover.finish();
        cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
        std::cout <<"Transfer the input actvation tensor took "<<elapsedTimeUs<<" us"<<std::endl;
    } // Transfer the input

#if defined(SPARSE_SYSTEM)
    if (flagSparseInput == true)
    {
        std::cout <<stepCount++<<". Transfer the input activation TB count "<<std::endl;

        cl::Event event;
        auto numElements = (pInput->getTransferBlockCountVector()).size();
        auto sizeElement = sizeof(typeof((pInput->getTransferBlockCountVector()).at(0)));
        auto transferBytes = sizeElement * numElements;

        int tbCountOffsetByte = MEM_START_TB_0 * sizeof(t_streamblock_address);

        std::cout <<"Transfering "<<transferBytes<<" bytes into bufferActivationTBCounts"<<std::endl;
        assert(tbCountOffsetByte+transferBytes <= MAX_DRAM_BYTE_INPUT_ACTIVATION_SB_COUNT && "Too many input activation TB count bytes to fit inside the global memory" );

        status = clCQIAMover.enqueueWriteBuffer(bufferActivationTBCounts, //buffer
                                             CL_TRUE, //blocking_write
                                             tbCountOffsetByte, //offset
                                             transferBytes, //size
                                             (pInput->getTransferBlockCountVector()).data(), //data pointer
                                             NULL, //dependency list
                                             &event //events generated
                                            );
        aocl_utils_cpp::checkError(status, "Failed to write the input activation TB count");
        clCQIAMover.finish();
        cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
        std::cout <<"Transfer the input activation TB count took "<<elapsedTimeUs<<" us"<<std::endl;
    }
#endif

    std::cout <<stepCount++<<". Transfer the IA Mover instructions"<<std::endl;
    {
        cl::Event event;
        auto numElements = vecIAMoverInstruction.size();
        auto sizeElement = sizeof(typeof(vecIAMoverInstruction.at(0)));
        auto transferBytes = sizeElement * numElements;

        std::cout <<"Transfering "<<transferBytes<<" bytes into bufferIAMoverInstructions"<<std::endl;
        assert(transferBytes <= MAX_DRAM_BYTE_INPUT_MOVER_INSTRUCTION && "Too many IA mover instructions to fit inside the global memory" );

        status = clCQIAMover.enqueueWriteBuffer(bufferIAMoverInstructions, //buffer
                                             CL_TRUE, //blocking_write
                                             0, //offset
                                             transferBytes, //size
                                             vecIAMoverInstruction.data(), //data pointer
                                             NULL, //dependency list
                                             &event //events generated
                                            );
        aocl_utils_cpp::checkError(status, "Failed to write the IA mover instructions");
        clCQIAMover.finish();
        cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
        std::cout <<"Transfer the IA mover instructions tensor took "<<elapsedTimeUs<<" us"<<std::endl;
    }

    if (op == CONVOLUTION)
    {
        std::cout <<stepCount++<<". Transfer the IA Tile Controller instructions"<<std::endl;
        cl::Event event;
        auto numElements = vecIATileControllerInstruction.size();
        auto sizeElement = sizeof(typeof(vecIATileControllerInstruction.at(0)));
        auto transferBytes = sizeElement * numElements;

        std::cout <<"Transfering "<<transferBytes<<" bytes into bufferIATileControllerInstructions"<<std::endl;
        assert(transferBytes <= MAX_DRAM_BYTE_INPUT_TILE_CONTROLLER_INSTRUCTION && "Too many IA Tile instructions to fit inside the global memory" );

        status = clCQIATileController.enqueueWriteBuffer(bufferIATileControllerInstructions, //buffer
                                             CL_TRUE, //blocking_write
                                             0, //offset
                                             transferBytes, //size
                                             vecIATileControllerInstruction.data(), //data pointer
                                             NULL, //dependency list
                                             &event //events generated
                                            );
        aocl_utils_cpp::checkError(status, "Failed to write the IA tile controller instructions");
        clCQIATileController.finish();
        cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
        std::cout <<"Transfer the IA tile controller instructions tensor took "<<elapsedTimeUs<<" us"<<std::endl;
    }


    std::cout <<stepCount++<<". Transfer the OA Mover instructions"<<std::endl;
    {
        cl::Event event;
        auto numElements = vecOAMoverInstruction.size();
        auto sizeElement = sizeof(typeof(vecOAMoverInstruction.at(0)));
        auto transferBytes = sizeElement * numElements;

        std::cout <<"Transfering "<<transferBytes<<" bytes into bufferOAMoverInstructions"<<std::endl;
        assert(transferBytes <= MAX_DRAM_BYTE_OUTPUT_MOVER_INSTRUCTION && "Too many OA mover instructions to fit inside the global memory" );

        status = clCQOAMover.enqueueWriteBuffer(bufferOAMoverInstructions, //buffer
                                             CL_TRUE, //blocking_write
                                             0, //offset
                                             transferBytes, //size
                                             vecOAMoverInstruction.data(), //data pointer
                                             NULL, //dependency list
                                             &event //events generated
                                            );
        aocl_utils_cpp::checkError(status, "Failed to write the OA mover instructions");
        clCQOAMover.finish();
        cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
        std::cout <<"Transfer the OA mover instructions tensor took "<<elapsedTimeUs<<" us"<<std::endl;
    }

    std::cout <<stepCount++<<". Transfer the OA Tile Controller instructions"<<std::endl;
    {
        cl::Event event;
        auto numElements = vecOATileControllerInstruction.size();
        auto sizeElement = sizeof(typeof(vecOATileControllerInstruction.at(0)));
        auto transferBytes = sizeElement * numElements;

        std::cout <<"Transfering "<<transferBytes<<" bytes into bufferOAMoverInstructions"<<std::endl;
        assert(transferBytes <= MAX_DRAM_BYTE_OUTPUT_TILE_CONTROLLER_INSTRUCTION && "Too many OA TILE instructions to fit inside the global memory" );

        status = clCQOATileController.enqueueWriteBuffer(bufferOATileControllerInstructions, //buffer
                                             CL_TRUE, //blocking_write
                                             0, //offset
                                             transferBytes, //size
                                             vecOATileControllerInstruction.data(), //data pointer
                                             NULL, //dependency list
                                             &event //events generated
                                            );
        aocl_utils_cpp::checkError(status, "Failed to write the IA tile controller instructions");
        clCQOATileController.finish();
        cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
        std::cout <<"Transfer the OA tile controller instructions tensor took "<<elapsedTimeUs<<" us"<<std::endl;
    }

    //If the operation is CONVOLUTION,
    //then transfer the WMover instructions, the weights, the weight TB count, and the biases
    if (op == CONVOLUTION)
    {
        std::cout <<stepCount++<<". Transfer the W Mover instructions"<<std::endl;
        {
            cl::Event event;
            auto numElements = vecWMoverInstruction.size();
            auto sizeElement = sizeof(typeof(vecWMoverInstruction.at(0)));
            auto transferBytes = sizeElement * numElements;

            std::cout <<"Transfering "<<transferBytes<<" bytes into bufferWMoverInstructions"<<std::endl;
            assert(transferBytes <= MAX_DRAM_BYTE_WEIGHT_MOVER_INSTRUCTION && "Too many Weight Mover instructions to fit inside the global memory" );


            status = clCQWMover.enqueueWriteBuffer(bufferWMoverInstructions, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 transferBytes, //size
                                                 vecWMoverInstruction.data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the W Mover instructions");
            clCQWMover.finish();
            cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
            std::cout <<"Transfer the W Mover instructions tensor took "<<elapsedTimeUs<<" us"<<std::endl;
        }

        std::cout <<stepCount++<<". Transfer the filter biases"<<std::endl;
        {
            cl::Event event;
            auto numElements = biasVector.size();
            auto sizeElement = sizeof(typeof(biasVector.at(0)));
            auto transferBytes = sizeElement * numElements;

            std::cout <<"Transfering "<<transferBytes<<" bytes into bufferWMoverBias"<<std::endl;
            assert(transferBytes <= MAX_DRAM_BYTE_INPUT_BIAS && "Too many biases to fit inside the global memory" );

            status = clCQWMover.enqueueWriteBuffer(bufferWMoverBias, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 transferBytes, //size
                                                 biasVector.data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the filter biases");
            clCQWMover.finish();
            cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
            std::cout <<"Transfer the filter biases took "<<elapsedTimeUs<<" us"<<std::endl;
        }

        std::cout <<stepCount++<<". Transfer the filter weights"<<std::endl;
        {
            cl::Event event;
            auto numElements =  (pWeights->getTransferBlockVector()).size();
            auto sizeElement = sizeof(typeof((pWeights->getTransferBlockVector()).at(0)));
            auto transferBytes = sizeElement * numElements;

            std::cout <<"Transfering "<<transferBytes<<" bytes into bufferWMoverWDramBlocks"<<std::endl;
            assert(transferBytes <= MAX_DRAM_BYTE_INPUT_WEIGHT && "Too many weights to fit inside the global memory" );

            status = clCQWMover.enqueueWriteBuffer(bufferWMoverWDramBlocks, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 transferBytes, //size
                                                 (pWeights->getTransferBlockVector()).data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the filter weight tensors");
            clCQWMover.finish();
            cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
            std::cout <<"Transfer the filter weight tensors took "<<elapsedTimeUs<<" us"<<std::endl;
        }

#if defined(SPARSE_SYSTEM)
        {
            std::cout <<stepCount++<<". Transfer the filter weight TB counts"<<std::endl;
            cl::Event event;
            auto numElements =  (pWeights->getTransferBlockCountVector()).size();
            auto sizeElement = sizeof(typeof((pWeights->getTransferBlockCountVector()).at(0)));
            auto transferBytes = sizeElement * numElements;

            std::cout <<"Transfering "<<transferBytes<<" bytes into bufferWMoverWTBCounts"<<std::endl;
            assert(transferBytes <= MAX_DRAM_BYTE_INPUT_WEIGHT_SB_COUNT && "Too many weight TB counts to fit inside the global memory" );

            status = clCQWMover.enqueueWriteBuffer(bufferWMoverWTBCounts, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 transferBytes, //size
                                                 (pWeights->getTransferBlockCountVector()).data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the filter weight TB counts");
            clCQWMover.finish();
            cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
            std::cout <<"Transfer the filter weight TB counts took "<<elapsedTimeUs<<" us"<<std::endl;
        }
#endif
    }
    else
    {
        std::cout <<stepCount++<<". Transfer the MK controller instructions"<<std::endl;
        {
            cl::Event event;
            auto numElements = vecMiscInstruction.size();
            auto sizeElement = sizeof(typeof(vecMiscInstruction.at(0)));
            auto transferBytes = sizeElement * numElements;

            std::cout <<"Transfering "<<transferBytes<<" bytes into bufferMKInstructions"<<std::endl;
            assert(transferBytes <= MAX_DRAM_BYTE_MISC_CONTROLLER_INSTRUCTION && "Too many MK instructions to fit inside the global memory" );

            status = clCQMKController.enqueueWriteBuffer(bufferMKInstructions, //buffer
                                                 CL_TRUE, //blocking_write
                                                 0, //offset
                                                 transferBytes, //size
                                                 vecMiscInstruction.data(), //data pointer
                                                 NULL, //dependency list
                                                 &event //events generated
                                                );
            aocl_utils_cpp::checkError(status, "Failed to write the MK controller instructions");
            clCQMKController.finish();
            cl_ulong startTime = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong endTime = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            cl_double elapsedTimeUs = (cl_double)((endTime - startTime)*(cl_double)(1e-3));
            std::cout <<"Transfer the MK controller instructions tensor took "<<elapsedTimeUs<<" us"<<std::endl;
        }
    }

    //Launch the kernels
    std::cout<<stepCount++<<". Launch the kernels."<<std::endl;
    cl_ulong proccessDuration = 0;
    for (int i=0; i<REPEAT; i++)
    {
        cl::Event eventIAMover, eventWMover, eventOAMover, eventMKController, eventIATileController, eventOATileController;

        status = clCQIAMover.enqueueTask(kernelIAMover, NULL);
        aocl_utils_cpp::checkError(status, "Failed to launch kernelIAMover!");

        status = clCQIATileController.enqueueTask(kernelIATileController, NULL);
        aocl_utils_cpp::checkError(status, "Failed to launch kernelIATileController!");

        status = clCQWMover.enqueueTask(kernelWMover, NULL);
        aocl_utils_cpp::checkError(status, "Failed to launch kernelWMover!");

        status = clCQMKController.enqueueTask(kernelMKInstructionMover, NULL);
        aocl_utils_cpp::checkError(status, "Failed to launch kernelMKInstructionMover!");

        status = clCQOATileController.enqueueTask(KernelOATileController, NULL);
        aocl_utils_cpp::checkError(status, "Failed to launch KernelOATileController!");

        status = clCQOAMover.enqueueTask(kernelOAMover, NULL, &eventOAMover);
        aocl_utils_cpp::checkError(status, "Failed to launch kernelOAMover!");

        clCQOAMover.finish();

        cl_ulong processStart = eventOAMover.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong processEnd = eventOAMover.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        proccessDuration += (processEnd - processStart);
     }
     proccessDuration /= REPEAT;

    //auto processEnd = std::chrono::system_clock::now();

    cl_double proccessDurationUs = ((cl_double) proccessDuration) * (cl_double)(1e-3);
    //auto duration = std::chrono::duration_cast<std::chrono::microseconds>(processEnd-processStart).count();
    //double averageDuration = (double)(duration) / (double)(REPEAT);
#if defined(C5SOC)
    //Hack for ARMv7
    //averageDuration = averageDuration * 1000.0;
#endif


#if defined(PROFILE)
    std::cout <<stepCount++<<". Attempting to retrieve autorun profiling data."<<std::endl;
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

    std::cout <<stepCount++<<". Retrieve the output."<<std::endl;
    int oaOffset = ((op == CONVOLUTION) && (flagMultiLayerConv == true)) ?
                MEM_START_ACTIVATION_0 * BURST_SIZE_BYTE : MEM_START_ACTIVATION_1 * BURST_SIZE_BYTE;
    cl::Event eventReadOutput, eventReadOutputCount;
    status = clCQOAMover.enqueueReadBuffer(
        bufferActivationDramBlocks,
        CL_TRUE,
        oaOffset,
        sizeof(typeof((pOutput->getTransferBlockVector()).at(0))) * (pOutput->getTransferBlockVector()).size(),
        (pOutput->getTransferBlockVector()).data(),
        NULL,
        &eventReadOutput
    );
    aocl_utils_cpp::checkError(status, "Failed to read output values!");

#if defined(SPARSE_SYSTEM)
    if (flagSparseOutput == true)
    {
        int oaTBOffset = ((op == CONVOLUTION) && (flagMultiLayerConv == true)) ?
                    MEM_START_TB_0 * sizeof(t_streamblock_address) : MEM_START_TB_1 * sizeof(t_streamblock_address);
        status = clCQOAMover.enqueueReadBuffer(
            bufferActivationTBCounts,
            CL_TRUE,
            oaTBOffset,
            sizeof(typeof((pOutput->getTransferBlockCountVector()).at(0))) * (pOutput->getTransferBlockCountVector()).size(),
            (pOutput->getTransferBlockCountVector()).data(),
            NULL,
            &eventReadOutputCount
        );
        aocl_utils_cpp::checkError(status, "Failed to read output TB count!");
    }
#endif
    cl_ulong outputValueTransferStart = eventReadOutput.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong outputValueTransferEnd = eventReadOutput.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    cl_double outputValueTransferDuration = (cl_double)((outputValueTransferEnd - outputValueTransferStart) * (cl_double)(1e-3));

    cl_ulong outputCountTransferStart = eventReadOutputCount.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong outputCountTransferEnd = eventReadOutputCount.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    cl_double outputCountTransferDuration = (cl_double)((outputCountTransferEnd - outputCountTransferStart) * (cl_double)(1e-3));

    std::cout <<"Average Convolution time:  (us): "<<proccessDurationUs<<std::endl;
    std::cout <<"Output transfer time (us): "<<outputValueTransferDuration<<std::endl;
#if defined(SPARSE_SYSTEM)
    std::cout <<"Output count transfer time (us): "<<outputCountTransferDuration<<std::endl;
#endif

#if defined(NOOP)
    //Test the noop time
    {
        std::cout <<stepCount++<<". Measure NOOP time"<<std::endl;
        cl_double duration = 0;
        cl::Event eventNoop;

        status = clCQNoop.enqueueTask(kernelNoop, NULL, &eventNoop);
        aocl_utils_cpp::checkError(status, "Failed to launch kernelNoop!");

        clCQNoop.finish();

        cl_ulong processStart = eventNoop.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong processEnd = eventNoop.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        duration = ((cl_double)(processEnd - processStart)) * ((cl_double)(1e-3));
        std::cout <<"Noop time (us): "<<duration<<std::endl;

    }
#endif

    //TODO: Decompress the output, and check against the input if necessary
    {
        std::cout <<stepCount++<<". Decode the output"<<std::endl;
        std::vector<fixedPointNumber> outputFPVector;

        pOutput->decodeTensor(outputFPVector, FRAC_WIDTH, INT_WIDTH);

        if (_flagPerformanceTest == false)
        {
            std::cout <<stepCount++<<". Check the output"<<std::endl;

            for (unsigned int iGroup=0; iGroup<_numGroupNext; iGroup++)
            {
                for (unsigned int iRow=0; iRow<numOutputHeight; iRow++)
                {
                    for (unsigned int iCol=0; iCol<numOutputWidth; iCol++)
                    {
                        for (unsigned int iCh=0; iCh<numOutputChannelPerGroup; iCh++)
                        {
                            //Obtain the actual output
                            unsigned int outputCoord =
                                    iGroup*numOutputHeight*numOutputWidth*numOutputChannelPerGroup
                                    +iRow*numOutputWidth*numOutputChannelPerGroup + iCol*numOutputChannelPerGroup + iCh;
                            fixedPointNumber actualFP = outputFPVector.at(outputCoord);

                            //Compute the expected output
                            float unitFloat = 1.0f / (1 << FRAC_WIDTH);
                            float expectedFloat;
                            switch (op) {
                                case CONVOLUTION: {
                                    bool colIsReal = ((iCol % inputWidthSPUnitSize) == 0);
                                    bool rowIsReal = ((iRow % inputHeightSPUnitSize) == 0);
                                    if ((colIsReal == true) && (rowIsReal == true))
                                    {
                                        unsigned int inputCol = iCol / inputWidthSPUnitSize;
                                        unsigned char iChGlobal = iGroup*numOutputChannelPerGroup + iCh;
                                        expectedFloat = (inputCol % 2 == 0) ? iChGlobal*unitFloat : -1.0f*iChGlobal*unitFloat;
                                        expectedFloat += _bias;
                                    }
                                    else
                                    {
                                        expectedFloat = _bias;
                                    }
                                }
                                break;
                                case MAX_POOL: {
                                    //TODO: Change the ground truth generation if the max pooling kernel size
                                    //stride, or input pattern changes
                                    unsigned char iChGlobal = iGroup*numOutputChannelPerGroup + iCh;
                                    expectedFloat = iChGlobal*unitFloat;
                                }
                                break;
                                case CONCATENATION: {
                                    unsigned char iChGlobal = iGroup*numOutputChannelPerGroup + iCh;
                                    expectedFloat = (iCol % 2 == 0) ?
                                                  (iChGlobal % _numInputChannel)*unitFloat:
                                                  -1.0f*(iChGlobal % _numInputChannel)*unitFloat;

                                }
                                break;
                                case ELT_ADD: {
                                    unsigned char iChGlobal = iGroup*numOutputChannelPerGroup + iCh;
                                    expectedFloat = (iCol % 2 == 0) ? iChGlobal*unitFloat*2.0f : iChGlobal*unitFloat*(-2.0f);
                                }
                                break;
                            }//switch (op)

                            fixedPointNumber expectedFP(expectedFloat, FRAC_WIDTH, INT_WIDTH);

                            signed char actualBits = actualFP.getBits();
                            signed char expectedBits = expectedFP.getBits();

                            //Compare the outputs
                            EXPECT_TRUE(actualBits == expectedBits)
                                    <<"Value disagreement at [outputGroup, outputRow, outputCol, outputCh]: ["
                                    <<iGroup<<" "<<iRow<<" "<<iCol<<" "<<iCh<<"]"<<std::endl
                                    <<"Expected bits: 0x"<<std::bitset<8>(expectedBits)<<std::endl
                                    <<"Actual bits: 0x"<<std::bitset<8>(actualBits)<<std::endl;
                        }
                    }
                }
            }
        } //check the input if necessary
    } // output decode
} //launch
