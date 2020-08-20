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
#include "accelerator_wrapper.hpp"

//#define PLAY
#define REPEAT 1
#define EMULATE
//#define PERF_TEST
//#NOOP
//#define PROFILE
#include "CL/cl_ext_intelfpga.h"

#define FRAC_WIDTH 4
#define INT_WIDTH 3
#define OUTPUT_INT_WIDTH 3

#define WEIGHT_SEED 1234
#define INPUT_SEED   7653

class testFixture : public ::testing::Test {
protected:
    //The accelerator
    GraphRuntime::AcceleratorWrapper accelerator;

    //aocx file
    std::string binaryFile;

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
    std::vector<float> generateInputTensor (
                unsigned char _inputWidth,
                unsigned char _inputHeight,
                unsigned char _numInputChannel,
                unsigned char _numGroupCurrentLayer
            );

    std::vector<float> generateSparseInput (unsigned char _inputWidth,
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
    //Perform a dummy test first to clear the output buffer
    std::cout <<"Performing the dummy test"<<std::endl;
    unsigned char inputWidth = 2*PE_COLS;
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
    float bias = 5.0f;

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

    //Then perform the actual back to back test
    inputWidth = 2*PE_COLS;
    inputHeight = 4;
    numInputChannel = 8;
    numInputGroup = 1;
    numOutputGroup = 1;
    inputHeightSPUnitSize = 1;
    inputWidthSPUnitSize = 1;
    sizeOutputTileWidthPerColFull = 2;
    sizeOutputTileHeight = 4;
    flagEnableRelu = false;
    flagSparseInput = false;
    flagSparseOutput = false;
    op = CONVOLUTION;
    bias = 0.0f;
    bool flag2Layer = true;

    std::cout <<"Actual back-to-back test starts"<<std::endl;

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
#ifdef C5SOC
    binaryFile = "device_utils.aocx";
#else
    binaryFile = "sparse_pe_system.aocx";
#if defined(EMULATE)
    binaryFile = "smallBuffer.aocx";
#endif
#endif
    GraphRuntime::t_accelerator_info acceleratorInfo =
        {   .numPERows=PE_ROWS,
            .numPECols=PE_COLS,
            .numClusterInCompressionBlock=COMPRESSION_WINDOW_SIZE,
            .numClusterInTransferBlock=TRANSFER_SIZE,
            .numScalarInCluster=CLUSTER_SIZE
        };
   accelerator = GraphRuntime::AcceleratorWrapper(binaryFile, acceleratorInfo, 0);
}

std::vector<float> testFixture::generateInputTensor(
            unsigned char _inputWidth,
            unsigned char _inputHeight,
            unsigned char _numInputChannel,
            unsigned char _numGroupCurrentLayer
        )
{
    assert(_numInputChannel % _numGroupCurrentLayer == 0);
    unsigned char numICPerGroup = _numInputChannel / _numGroupCurrentLayer;
    std::vector<float> inputVector;
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
                    inputVector.push_back(fpNumber.convert2Float());
                }
            }
        }
    }

    return inputVector;
}

std::vector<float> testFixture::generateSparseInput(
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
     std::vector<float> inputVector;
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
                     inputVector.push_back(fpNumber.convert2Float());
                 }
             }
         }
     }

     return inputVector;
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

    /*
     * Dervied the parameters required for generating instructions
    */
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
    verticalBorderPadding = (op==CONVOLUTION) ? ((stride-1) * inputWidthSPSize + kernelSize - stride)/2 : 0;
    horizontalBorderPadding  = (op==CONVOLUTION) ? ((stride-1) * inputHeightSPSize + kernelSize - stride)/2 : 0;
    unsigned char numActiveColsPartialOutputTile = (op==CONVOLUTION) ?
                1 : (numOutputWidth % PE_COLS);
    assert(numOutputChannels % _numGroupNext == 0);
    numOutputChannelPerGroup = numOutputChannels / _numGroupNext;

    /* Generate the dense, floating point tensors
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

    std::vector<float> inputTensorDense;
    if (_flagPerformanceTest == true)
    {
       inputTensorDense = generateSparseInput(_inputWidth, _inputHeight, _numInputChannel, numGroupCurrentLayer, denseProb, pruneScale);
    }
    else
    {
        inputTensorDense = generateInputTensor(_inputWidth, _inputHeight, _numInputChannel, numGroupCurrentLayer);
    }

    //Generate qunatized weight tensors
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
    //Generate biases
    t_accumulator fixedBias = (t_accumulator) (round(_bias * (float) (1 << (FRAC_WIDTH + FRAC_WIDTH)) ));
    auto pBiasVector = std::make_shared<t_aligned_short_vector>(_numInputChannel, fixedBias);

    /* 2. Allocate the aligned weight tensors and compress them if necessary
     * */
    std::cout <<stepCount++<<". Allocate, align, and compress the test tensors."<<std::endl;

    unsigned short maxScalarIndexInChannelGroup = _numInputChannel / numGroupCurrentLayer - 1;
    unsigned short maxClusterIndexInCompressionBlock = COMPRESSION_WINDOW_SIZE-1;
    unsigned short maxClusterIndexInTransferBlock = TRANSFER_SIZE-1;
    unsigned short maxScalarIndexInCluster = CLUSTER_SIZE-1;

    std::shared_ptr<AlignedTensor> pWeight;

    //Prepare weights
    if (op==CONVOLUTION)
    {
#if defined(SPARSE_SYSTEM)
        pWeight.reset(new FlexibleDirectCompressedTensor (
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
        pWeight.reset( new AlignedTensor (
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

    /*
     * 3. Generate the executiong graph
    */
    std::cout <<stepCount++<<". Generate the execution graph"<<std::endl;

    GraphRuntime::t_execution_graph graph;

    signed int memDramBlockFilterStride;
    //assign memDramBlockFilterStride conditionally, otherwise there might be seg fault.
    if (op == CONVOLUTION)
    {
        memDramBlockFilterStride = (pWeight->getExternalMemoryAddressStride()) >> WIDE_SIZE_OFFSET;
    }
    else
    {
        memDramBlockFilterStride = 0;
    }

    signed int memDramBlockIAColStride = calculateExternalMemoryAddressStride(
                    maxScalarIndexInChannelGroup+1, //channelsPerGroup
                    _numInputChannel / (maxScalarIndexInChannelGroup+1), //group
                    _inputHeight, //height
                    _inputWidth, //width
                    CLUSTER_SIZE, //clusterSize
                    TRANSFER_SIZE, //transferBlockSize
                    COMPRESSION_WINDOW_SIZE, //compressionWindowSize
                    WIDE_SIZE, //numTransferBlockPerDramBlock
                    false, //isKernel
                    flagSparseInput
                ) >> WIDE_SIZE_OFFSET;
    std::cout <<"memDramBlockIAColStride: "<<memDramBlockIAColStride<<std::endl;

    signed int memDramBlockOAColStride = calculateExternalMemoryAddressStride(
                numOutputChannelPerGroup, //channelsPerGroup
                _numGroupNext, //group
                numOutputHeight, //height
                numOutputWidth, //width
                CLUSTER_SIZE, //clusterSize
                TRANSFER_SIZE, //transferBlockSize
                COMPRESSION_WINDOW_SIZE, //compressionWindowSize
                WIDE_SIZE, //numTransferBlockPerDramBlock
                false, //isKernel
                flagSparseOutput
            ) >> WIDE_SIZE_OFFSET;
    std::cout <<"memDramBlockOAColStride: "<<memDramBlockOAColStride<<std::endl;


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

    //Intermediate buffers for IA mover instructions and OA mover instructions
    t_aligned_ia_mover_instruction_vector vecIAMoverInstruction;
    t_aligned_oa_mover_instruction_vector vecOAMoverInstruction;

    int offsetIAMoverInstruction = 0;
    int offsetOAMoverInstruction = 0;
    int offsetWeightsDramBlock = 0;
    int offsetBiasesDramBlock = 0;
#if defined(SPARSE_SYSTEM)
    int offsetWeightTBCount = 0;
#endif
    if (op==CONVOLUTION && flagMultiLayerConv == true)
    {
        /*
         * If the test is back to back, then
         * the input blob resides on region 0
         * the intermediate blob resides on region 1
         * the final blob resides on region 0
         *
        */
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
                    graph.vecIATileControllerInstruction,
                    graph.vecOATileControllerInstruction,
                    graph.vecWMoverInstruction,
                    graph.vecMiscInstruction,

                    //signed int memIA0DramBlockStartIndex
                    0 * MEM_ACTIVATION_REGION_SIZE_PER_SLICE,
                    //signed int memIA1DramBlockStartIndex
                    0 * MEM_ACTIVATION_REGION_SIZE_PER_SLICE,
                    //signed int memOADramBlockStartIndex
                    1 * MEM_ACTIVATION_REGION_SIZE_PER_SLICE,
                    //signed int memWeightDramBlockStartIndex
                    offsetWeightsDramBlock,
                    //signed int memBiasStartIndex
                    offsetBiasesDramBlock,

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
                    0 * MEM_ACTIVATION_TB_REGION_SIZE_PER_SLICE,
                    //memIATB0CountColStride,
                    1,
                    //memOATBCountStart
                    1 * MEM_ACTIVATION_TB_REGION_SIZE_PER_SLICE,
                    //memOATBCountColStride
                    1,
                    //memWeightTBCountStart
                    offsetWeightTBCount,
                #endif

                    //flagSparseOutput,
                    TRUE,
                    instFlagSparseInput,

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
        //Filling the rest of the information for the first layer
        {
            std::copy(vecIAMoverInstruction.begin(),
                      vecIAMoverInstruction.end(),
                      std::back_inserter(graph.vecIAMoverInstruction)
                      );
            std::copy(vecOAMoverInstruction.begin(),
                      vecOAMoverInstruction.end(),
                      std::back_inserter(graph.vecOAMoverInstruction)
                      );
            int numIAMoverInstructions = vecIAMoverInstruction.size();
            int numOAMoverInstructions = vecOAMoverInstruction.size();
            graph.vecLayerInfo.emplace_back(
                GraphRuntime::t_layer_info {.layerName="layer0",
                 .offsetIAMoverInstruction=offsetIAMoverInstruction,
                 .numIAMoverInstruction=numIAMoverInstructions,
                 .offsetOAMoverInstruction=offsetOAMoverInstruction,
                 .numOAMoverInstructions=numOAMoverInstructions
                 });
            offsetIAMoverInstruction += numIAMoverInstructions;
            offsetOAMoverInstruction += numOAMoverInstructions;

            if (op == CONVOLUTION)
            {
                graph.vecWeightDramBlockStart.push_back(offsetWeightsDramBlock);
                graph.vecBiasStart.push_back(offsetBiasesDramBlock);
                #if defined(SPARSE_SYSTEM)
                    graph.vecWeightTBCountStart.push_back(offsetWeightTBCount);
                #endif
                graph.pWeights.push_back(pWeight);
                graph.pBiasVector.push_back(pBiasVector);
                offsetWeightsDramBlock +=
                        (pWeight->getExternalMemoryAddressStride() >> WIDE_SIZE_OFFSET)
                        * numOutputChannels ;
                offsetBiasesDramBlock += numOutputChannels;
                #if defined(SPARSE_SYSTEM)
                    offsetWeightTBCount += numOutputChannels;
                #endif
             }
        }

        vecIAMoverInstruction.clear();
        vecOAMoverInstruction.clear();


        //2 SECOND set of instructions
        instruction_generator(
                    op,
                    vecIAMoverInstruction,
                    vecOAMoverInstruction,
                    graph.vecIATileControllerInstruction,
                    graph.vecOATileControllerInstruction,
                    graph.vecWMoverInstruction,
                    graph.vecMiscInstruction,

                    //signed int memIA0DramBlockStartIndex
                    1 * MEM_ACTIVATION_REGION_SIZE_PER_SLICE,
                    //signed int memIA1DramBlockStartIndex
                    1 * MEM_ACTIVATION_REGION_SIZE_PER_SLICE,
                    //signed int memOADramBlockStartIndex
                    0 * MEM_ACTIVATION_REGION_SIZE_PER_SLICE,
                    //signed int memWeightDramBlockStartIndex
                    offsetWeightsDramBlock,
                    //signed int memBiasStartIndex
                    offsetBiasesDramBlock,

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
                    1 * MEM_ACTIVATION_TB_REGION_SIZE_PER_SLICE,
                    //memIATB0CountColStride,
                    1,
                    //memOATBCountStart
                    0 * MEM_ACTIVATION_TB_REGION_SIZE_PER_SLICE,
                    //memOATBCountColStride
                    1,
                    //memWeightTBCountStart
                    offsetWeightTBCount,
                #endif

                    instFlagSparseOutput,
                    //instFlagSparseInput,
                    TRUE,

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
        //Filling the rest of the information for the SECOND layer
        {
            std::copy(vecIAMoverInstruction.begin(),
                      vecIAMoverInstruction.end(),
                      std::back_inserter(graph.vecIAMoverInstruction)
                      );
            std::copy(vecOAMoverInstruction.begin(),
                      vecOAMoverInstruction.end(),
                      std::back_inserter(graph.vecOAMoverInstruction)
                      );
            int numIAMoverInstructions = vecIAMoverInstruction.size();
            int numOAMoverInstructions = vecOAMoverInstruction.size();
            graph.vecLayerInfo.emplace_back(
                GraphRuntime::t_layer_info {.layerName="layer1",
                 .offsetIAMoverInstruction=offsetIAMoverInstruction,
                 .numIAMoverInstruction=numIAMoverInstructions,
                 .offsetOAMoverInstruction=offsetOAMoverInstruction,
                 .numOAMoverInstructions=numOAMoverInstructions
                 });
            offsetIAMoverInstruction += numIAMoverInstructions;
            offsetOAMoverInstruction += numOAMoverInstructions;

            if (op == CONVOLUTION)
            {
                graph.vecWeightDramBlockStart.push_back(offsetWeightsDramBlock);
                graph.vecBiasStart.push_back(offsetBiasesDramBlock);
                #if defined(SPARSE_SYSTEM)
                    graph.vecWeightTBCountStart.push_back(offsetWeightTBCount);
                #endif
                graph.pWeights.push_back(pWeight);
                graph.pBiasVector.push_back(pBiasVector);
                offsetWeightsDramBlock +=
                        (pWeight->getExternalMemoryAddressStride() >> WIDE_SIZE_OFFSET)
                        * numOutputChannels ;
                offsetBiasesDramBlock += numOutputChannels;
                #if defined(SPARSE_SYSTEM)
                    offsetWeightTBCount += numOutputChannels;
                #endif
             }
        }
    } //multilayer-convolution
    else
    {
        instruction_generator(
                    op,
                    vecIAMoverInstruction,
                    vecOAMoverInstruction,
                    graph.vecIATileControllerInstruction,
                    graph.vecOATileControllerInstruction,
                    graph.vecWMoverInstruction,
                    graph.vecMiscInstruction,

                    //signed int memIA0DramBlockStartIndex
                    0 * MEM_ACTIVATION_REGION_SIZE_PER_SLICE,
                    //signed int memIA1DramBlockStartIndex
                    0 * MEM_ACTIVATION_REGION_SIZE_PER_SLICE,
                    //signed int memOADramBlockStartIndex
                    1 * MEM_ACTIVATION_REGION_SIZE_PER_SLICE,
                    //signed int memWeightDramBlockStartIndex
                    offsetWeightsDramBlock,
                    //signed int memBiasStartIndex
                    offsetBiasesDramBlock,

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
                    0 * MEM_ACTIVATION_TB_REGION_SIZE_PER_SLICE,
                    //memIATB0CountColStride,
                    1,
                    //memOATBCountStart
                    1 * MEM_ACTIVATION_TB_REGION_SIZE_PER_SLICE,
                    //memOATBCountColStride
                    1,
                    //memWeightTBCountStart
                    offsetWeightTBCount,
                #endif

                    instFlagSparseOutput,
                    instFlagSparseInput,

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
        //Filling the rest of the information for the single layer
        {
            std::copy(vecIAMoverInstruction.begin(),
                      vecIAMoverInstruction.end(),
                      std::back_inserter(graph.vecIAMoverInstruction)
                      );
            std::copy(vecOAMoverInstruction.begin(),
                      vecOAMoverInstruction.end(),
                      std::back_inserter(graph.vecOAMoverInstruction)
                      );
            int numIAMoverInstructions = vecIAMoverInstruction.size();
            int numOAMoverInstructions = vecOAMoverInstruction.size();
            graph.vecLayerInfo.emplace_back(
                GraphRuntime::t_layer_info {.layerName="layer1",
                 .offsetIAMoverInstruction=offsetIAMoverInstruction,
                 .numIAMoverInstruction=numIAMoverInstructions,
                 .offsetOAMoverInstruction=offsetOAMoverInstruction,
                 .numOAMoverInstructions=numOAMoverInstructions
                 });
            offsetIAMoverInstruction += numIAMoverInstructions;
            offsetOAMoverInstruction += numOAMoverInstructions;

            if (op == CONVOLUTION)
            {
                graph.vecWeightDramBlockStart.push_back(offsetWeightsDramBlock);
                graph.vecBiasStart.push_back(offsetBiasesDramBlock);
                #if defined(SPARSE_SYSTEM)
                    graph.vecWeightTBCountStart.push_back(offsetWeightTBCount);
                #endif
                graph.pWeights.push_back(pWeight);
                graph.pBiasVector.push_back(pBiasVector);
                offsetWeightsDramBlock +=
                        (pWeight->getExternalMemoryAddressStride() >> WIDE_SIZE_OFFSET)
                        * numOutputChannels ;
                offsetBiasesDramBlock += numOutputChannels;
                #if defined(SPARSE_SYSTEM)
                    offsetWeightTBCount += numOutputChannels;
                #endif
             }
        }
    } //Instruction generation

    /*Prepare the input/output blobs
     * Note: For THIS demo, if a test case have the multiple input blobs (e.g. concatenation, eltwise-add)
     * then the inputs have identical values.
     * Moreoever, there is only one copy of the input on the FPGA memory.
     * In general, this simplification does not hold.
    */
    graph.vecInputInfo.emplace_back(GraphRuntime::t_blob_info{
                                        .memoryRegionID=0,
                                        .channelPerGroup=numInputChannel0 / numGroupCurrentLayer,
                                        .group=numGroupCurrentLayer,
                                        .height=_inputHeight,
                                        .width=_inputWidth,
                                        .numFracBits=FRAC_WIDTH,
                                        .flagCanBeSparse=flagSparseInput,
                                        .blobName="input"
                                    });
    int oaRegion = ((op == CONVOLUTION) && (flagMultiLayerConv == true)) ? 0 : 1;
    graph.vecOutputInfo.emplace_back(
                GraphRuntime::t_blob_info{
                    .memoryRegionID=oaRegion,
                    .channelPerGroup=numOutputChannelPerGroup,
                    .group= numOutputChannels / numOutputChannelPerGroup,
                    .height= numOutputHeight,
                    .width= numOutputWidth,
                    .numFracBits=FRAC_WIDTH,
                    .flagCanBeSparse=flagSparseOutput,
                    .blobName="output"
                }
                );

    std::cout <<"Number of IA Mover instructions: "<<graph.vecIAMoverInstruction.size()<<std::endl;
    std::cout <<"Number of IA tile controller instructions: "<<graph.vecIATileControllerInstruction.size()<<std::endl;
    std::cout <<"Number of OA Mover instructions: "<<graph.vecOAMoverInstruction.size()<<std::endl;
    std::cout <<"Number of OA tile contrller instructions: "<<graph.vecOATileControllerInstruction.size()<<std::endl;
    std::cout <<"Number of W Mover instructions: "<<graph.vecWMoverInstruction.size()<<std::endl;
    std::cout <<"Number of MK controller instructions: "<<graph.vecMiscInstruction.size()<<std::endl;

    //Load the graph to the FPGA
    std::cout <<stepCount++<<". Load the graph to the FPGA."<<std::endl;
    accelerator.resetGraph();
    accelerator.loadGraph(graph);

    //Setup the input blobs
    std::cout <<stepCount++<<". Setup the input blobs."<<std::endl;
    accelerator.prepareInputBlob(inputTensorDense, 0);

    //Perform inference
    std::cout<<stepCount++<<". Perform inferences."<<std::endl;
    for (int i=0; i<REPEAT; i++)
    {
        accelerator.inference();
    }

    //Retrieve the outputs
    std::cout <<stepCount++<<". Decode and retrieve the output."<<std::endl;
    std::vector<float> outputFloatVector = accelerator.extractOutputBlob(0);


    //Decompress the output, and check against the input if necessary
    {
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
                            float actualFloat = outputFloatVector.at(outputCoord);

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
                            fixedPointNumber actualFP(actualFloat, FRAC_WIDTH, INT_WIDTH);

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
