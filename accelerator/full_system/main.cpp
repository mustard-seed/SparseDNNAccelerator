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
#include "spwTensorCompression.hpp"
#include "vectorType.hpp"
#include "layerInstructionGenerator.hpp"
#include "accelerator_wrapper.hpp"

#define DIVIDE_CEIL(x, y) (1 + (x-1) / (y))
#define SEED 27

#define PLAY
//#define PERF_TEST
//#define THROUGHPUT_DIAGNOSTIC
//#define VALIDATE
//#define TEST1_20201126
//#define TEST2_20201126
//#define ELTADD7_202021129
//Some how if repeat is 100, bad things will happen on concat
#define REPEAT 1
#ifndef C5SOC
#define EMULATE
#endif
//#define PERF_TEST
//#NOOP
//#define PROFILE

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
                unsigned short _inputWidth,
                unsigned short _inputHeight,
                unsigned int _numInputChannel,
                unsigned char _numGroupCurrentLayer,
                bool _alternateSign = true
            );

    //TODO: get rid of _numGroupCurrentLayer
    //TODO: replace dense Prob and channel prune scale with something else
    //TODO: remove this function
//    std::vector<float> generateSparseInput (unsigned short _inputWidth,
//                unsigned short _inputHeight,
//                unsigned int _numInputChannel,
//                unsigned char _numGroupCurrentLayer,
//                float denseProb = 1.0f
//                , int channelPruneScale = CLUSTER_SIZE);

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
                unsigned int _numInputChannel,
                unsigned char _numGroups
            );

    std::vector<fixedPointNumber> generateSparseWeights (
                    unsigned char _kernelSize,
                    unsigned int _numInputChannel,
                    unsigned int _numOutputChannel,
                    unsigned int _numGroups,
                    unsigned int _numNZClustersInPruneRange);

    //TODO:
    //remove _numGroupNext
    //remove _flagSparseInput
    //remove _flagSparseOutput
    //replace pruneScale with balanced block pruning parameters
    void launch (unsigned short _inputWidth,
            unsigned short _inputHeight,
            unsigned int _numInputChannel,
            unsigned int _numOutputChannel,
            unsigned char _numInputGroup, //The code will override this to 1 if the operation is not convolution
            unsigned char _inputHeightSPUnitSize, //The code will override this to 1 if the operation is not convolution
            unsigned char _inputWidthSPUnitSize, //The code will overide this to 1 if the operation is not convolution
            unsigned short _sizeOutputTileWidthPerColFull, //The code will override this to 1 if the operation is not convolution
            unsigned short _sizeOutputTileHeight, //The code will overrid this to 1 if the operation is not convolution
            unsigned char _kernelSize, //convolution kernel size
            bool _flagEnableRelu,
            OPERATION op,
            float _bias = 0.0f //Only matter for convolution
            , float denseProb = 1
            , bool flagMultiLayerConv = false
            , bool _flagPerformanceTest = false
            );
}; //testFixture

#ifdef PLAY
TEST_F (testFixture, conv_dense_input_dense_output_plain)
{
    unsigned char inputWidth = 4;
    unsigned char inputHeight = 4;
    unsigned char numInputChannel = 13;
    unsigned char numOutputChannel = numInputChannel;
    unsigned char numInputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 2;
    unsigned char sizeOutputTileHeight = 4;
    unsigned char kernelSize = 3;
    bool flagEnableRelu = false;
    float denseProb = 1.0 / PRUNE_RANGE_IN_CLUSTER;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;

    launch(
                inputWidth,
                inputHeight,
                numInputChannel,
                numOutputChannel,
                numInputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                kernelSize,
                flagEnableRelu,
                op,
                bias,
                denseProb
          );
}

#endif
#if defined(THROUGHPUT_DIAGNOSTIC)
TEST_F (testFixture, throughput_diagnostic_fully_connected)
{
    unsigned char inputWidth = 1;
    unsigned char inputHeight = 1;
    typedef struct {
          unsigned short inputChannel;
          unsigned short outputChannel;
    } t_fc_pairs;
    std::vector<t_fc_pairs> vecTestsPairs = {
            {.inputChannel=1, .outputChannel=254},
            {.inputChannel=32, .outputChannel=254},
            {.inputChannel=64, .outputChannel=254},
            {.inputChannel=96, .outputChannel=254},
            {.inputChannel=128, .outputChannel=254},
            {.inputChannel=160, .outputChannel=254},
            {.inputChannel=192, .outputChannel=254},
            {.inputChannel=224, .outputChannel=254},
            {.inputChannel=254, .outputChannel=254}
            };
    //std::vector<unsigned char> vecInputChannel = {1};
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned short sizeOutputTileWidthPerColFull = 8;
    unsigned short sizeOutputTileHeight = 8;
    unsigned char kernelSize = 3;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;
    std::vector<float> vecDenseProb = {1.0};
    for (auto& testPairs: vecTestsPairs)
    {
        for (auto & prob : vecDenseProb)
        {
            unsigned short numInputChannel = testPairs.inputChannel;
            unsigned short numOutputChannel = testPairs.outputChannel;
            launch(
                        inputWidth,
                        inputHeight,
                        numInputChannel,
                        numOutputChannel,
                        numInputGroup,
                        numOutputGroup,
                        inputHeightSPUnitSize,
                        inputWidthSPUnitSize,
                        sizeOutputTileWidthPerColFull,
                        sizeOutputTileHeight,
                        kernelSize,
                        flagEnableRelu,
                        flagSparseInput,
                        flagSparseOutput,
                        op,
                        bias,
                        false, //back to back
                        true, //perf test
                        prob, //dense prob
                        1 //channel prune scale
                  );
        }
     }
}

TEST_F (testFixture, throughput_diagnostic_conv1x1_feature_map_size)
{
    //unsigned char inputWidth = 1;
    unsigned char inputHeight = 4;
    unsigned char numInputChannel = 128;
    unsigned char numOutputChannel = 1;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned short sizeOutputTileWidthPerColFull = 8;
    unsigned short sizeOutputTileHeight = inputHeight;
    std::vector<unsigned short> vecInputWidth = {
        PE_COLS*sizeOutputTileWidthPerColFull,
        2*PE_COLS*sizeOutputTileWidthPerColFull,
        4*PE_COLS*sizeOutputTileWidthPerColFull,
        6*PE_COLS*sizeOutputTileWidthPerColFull,
        8*PE_COLS*sizeOutputTileWidthPerColFull,
        10*PE_COLS*sizeOutputTileWidthPerColFull,
        12*PE_COLS*sizeOutputTileWidthPerColFull};

    unsigned char kernelSize = 1;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;
    float prob = 1.0;
    for (auto& inputWidth: vecInputWidth)
    {
        launch(
                    inputWidth,
                    inputHeight,
                    numInputChannel,
                    numOutputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    kernelSize,
                    flagEnableRelu,
                    flagSparseInput,
                    flagSparseOutput,
                    op,
                    bias,
                    false, //back to back
                    true, //perf test
                    prob, //dense prob
                    1 //channel prune scale
              );
     }
}

TEST_F (testFixture, throughput_diagnostic_conv1x1_shallow_output_input_channel_size)
{
    unsigned short inputWidth = 8*PE_COLS;
    unsigned short inputHeight = 8;
    unsigned short numOutputChannel = PE_ROWS;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned short sizeOutputTileWidthPerColFull = 2;
    unsigned short sizeOutputTileHeight = 4;
    unsigned int UNIT_SIZE = 8;
    std::vector<unsigned int> vecInputChannel = {
        UNIT_SIZE,
        8* UNIT_SIZE,
        16* UNIT_SIZE,
        24* UNIT_SIZE,
        32* UNIT_SIZE,
        40* UNIT_SIZE,
        48* UNIT_SIZE,
        56* UNIT_SIZE,
        72* UNIT_SIZE,
        80* UNIT_SIZE,
        88* UNIT_SIZE,
        96* UNIT_SIZE,
        104* UNIT_SIZE,
        112* UNIT_SIZE
        };

    unsigned char kernelSize = 1;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;
    float prob = 1.0;
    for (auto& inputChannel: vecInputChannel)
    {
        launch(
                    inputWidth,
                    inputHeight,
                    inputChannel,
                    numOutputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    kernelSize,
                    flagEnableRelu,
                    flagSparseInput,
                    flagSparseOutput,
                    op,
                    bias,
                    false, //back to back
                    true, //perf test
                    prob, //dense prob
                    1 //channel prune scale
              );
     }
}

TEST_F (testFixture, throughput_diagnostic_conv1x1_deep_output_input_channel_size)
{
    unsigned short inputWidth = 8*PE_COLS;
    unsigned short inputHeight = 8;
    unsigned short numOutputChannel = 8*PE_ROWS;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned short sizeOutputTileWidthPerColFull = 2;
    unsigned short sizeOutputTileHeight = 4;
    unsigned int UNIT_SIZE = 8;
    std::vector<unsigned int> vecInputChannel = {
        UNIT_SIZE,
        8* UNIT_SIZE,
        16* UNIT_SIZE,
        24* UNIT_SIZE,
        32* UNIT_SIZE,
        40* UNIT_SIZE,
        48* UNIT_SIZE,
        56* UNIT_SIZE,
        72* UNIT_SIZE,
        80* UNIT_SIZE,
        88* UNIT_SIZE,
        96* UNIT_SIZE,
        104* UNIT_SIZE,
        112* UNIT_SIZE
        };

    unsigned char kernelSize = 1;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;
    float prob = 1.0;
    for (auto& inputChannel: vecInputChannel)
    {
        launch(
                    inputWidth,
                    inputHeight,
                    inputChannel,
                    numOutputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    kernelSize,
                    flagEnableRelu,
                    flagSparseInput,
                    flagSparseOutput,
                    op,
                    bias,
                    false, //back to back
                    true, //perf test
                    prob, //dense prob
                    1 //channel prune scale
              );
     }
}

TEST_F (testFixture, throughput_diagnostic_conv1x1_output_channel_size)
{
    unsigned short inputWidth = 8*PE_COLS;
    unsigned short inputHeight = 8;
    unsigned char numInputChannel = 8;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned short sizeOutputTileWidthPerColFull = 4;
    unsigned short sizeOutputTileHeight = 4;
    unsigned int UNIT_SIZE = 8;
    std::vector<unsigned int> vecOutputChannel = {
        UNIT_SIZE,
        2* UNIT_SIZE,
        4* UNIT_SIZE,
        6* UNIT_SIZE,
        8* UNIT_SIZE,
        10* UNIT_SIZE,
        12* UNIT_SIZE,
        14* UNIT_SIZE,
        16* UNIT_SIZE,
        18* UNIT_SIZE,
        20* UNIT_SIZE
        };

    unsigned char kernelSize = 1;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;
    float prob = 1.0;
    for (auto& outputChannel: vecOutputChannel)
    {
        launch(
                    inputWidth,
                    inputHeight,
                    numInputChannel,
                    outputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    kernelSize,
                    flagEnableRelu,
                    flagSparseInput,
                    flagSparseOutput,
                    op,
                    bias,
                    false, //back to back
                    true, //perf test
                    prob, //dense prob
                    1 //channel prune scale
              );
     }
}

TEST_F (testFixture, throughput_diagnostic_add)
{
    //unsigned char inputWidth = 1;
    unsigned char inputHeight = 4;
    unsigned char numInputChannel = 128;
    unsigned char numOutputChannel = 1;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 4;
    unsigned char sizeOutputTileHeight = 4;
    std::vector<unsigned short> vecInputWidth = {
        PE_COLS*sizeOutputTileWidthPerColFull,
        2*PE_COLS*sizeOutputTileWidthPerColFull,
        4*PE_COLS*sizeOutputTileWidthPerColFull,
        6*PE_COLS*sizeOutputTileWidthPerColFull,
        8*PE_COLS*sizeOutputTileWidthPerColFull,
        10*PE_COLS*sizeOutputTileWidthPerColFull,
        12*PE_COLS*sizeOutputTileWidthPerColFull};
    unsigned char kernelSize = 1;
    bool flagEnableRelu = false;
    bool flagSparseInput = false;
    bool flagSparseOutput = true;
    OPERATION op = ELT_ADD;
    float bias = 0.0f;
    float prob = 1.0;
    for (auto& inputWidth: vecInputWidth)
    {
        launch(
                    inputWidth,
                    inputHeight,
                    numInputChannel,
                    numOutputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    kernelSize,
                    flagEnableRelu,
                    flagSparseInput,
                    flagSparseOutput,
                    op,
                    bias,
                    false, //back to back
                    true, //perf test
                    prob, //dense prob
                    1 //channel prune scale
              );
     }
}

TEST_F (testFixture, throughput_diagnostic_resnet50_conv6_like)
{
    unsigned short numInputChannel = 64;
    unsigned short numOutputChannel = 256;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned short sizeOutputTileWidthPerColFull = 2;
    unsigned short sizeOutputTileHeight = 8;
    typedef struct {
        unsigned int numTileVertical;
        unsigned int numTileHorizontal;
    } t_num_tile;
    std::vector<t_num_tile> vecTileConfigs = {
      {.numTileVertical=1, .numTileHorizontal=1},
      {.numTileVertical=2, .numTileHorizontal=1},
      {.numTileVertical=3, .numTileHorizontal=1},
      {.numTileVertical=4, .numTileHorizontal=1},
      {.numTileVertical=5, .numTileHorizontal=1},
      {.numTileVertical=6, .numTileHorizontal=1},
      {.numTileVertical=7, .numTileHorizontal=1},

    {.numTileVertical=1, .numTileHorizontal=2},
    {.numTileVertical=2, .numTileHorizontal=2},
    {.numTileVertical=3, .numTileHorizontal=2},
    {.numTileVertical=4, .numTileHorizontal=2},
    {.numTileVertical=5, .numTileHorizontal=2},
    {.numTileVertical=6, .numTileHorizontal=2},
    {.numTileVertical=7, .numTileHorizontal=2}
    };
    unsigned char kernelSize = 1;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;
    float prob = 1.0;
    for (t_num_tile& tileConfig: vecTileConfigs)
    {
        unsigned short inputWidth = tileConfig.numTileHorizontal * PE_COLS * sizeOutputTileWidthPerColFull;
        unsigned short inputHeight = tileConfig.numTileVertical * sizeOutputTileHeight;
        launch(
                    inputWidth,
                    inputHeight,
                    numInputChannel,
                    numOutputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    kernelSize,
                    flagEnableRelu,
                    flagSparseInput,
                    flagSparseOutput,
                    op,
                    bias,
                    false, //back to back
                    true, //perf test
                    prob, //dense prob
                    1 //channel prune scale
              );
     }
}

TEST_F (testFixture, throughput_diagnostic_resnet50_conv6_ic128)
{
    unsigned short numInputChannel = 128;
    unsigned short numOutputChannel = 256;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned short sizeOutputTileWidthPerColFull = 2;
    unsigned short sizeOutputTileHeight = 8;
    typedef struct {
        unsigned int numTileVertical;
        unsigned int numTileHorizontal;
    } t_num_tile;
    std::vector<t_num_tile> vecTileConfigs = {
      {.numTileVertical=1, .numTileHorizontal=1},
      {.numTileVertical=2, .numTileHorizontal=1},
      {.numTileVertical=3, .numTileHorizontal=1},
      {.numTileVertical=4, .numTileHorizontal=1},
      {.numTileVertical=5, .numTileHorizontal=1},
      {.numTileVertical=6, .numTileHorizontal=1},
      {.numTileVertical=7, .numTileHorizontal=1}
    };
    unsigned char kernelSize = 1;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;
    float prob = 1.0;
    for (t_num_tile& tileConfig: vecTileConfigs)
    {
        unsigned short inputWidth = tileConfig.numTileHorizontal * PE_COLS * sizeOutputTileWidthPerColFull;
        unsigned short inputHeight = tileConfig.numTileVertical * sizeOutputTileHeight;
        launch(
                    inputWidth,
                    inputHeight,
                    numInputChannel,
                    numOutputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    kernelSize,
                    flagEnableRelu,
                    flagSparseInput,
                    flagSparseOutput,
                    op,
                    bias,
                    false, //back to back
                    true, //perf test
                    prob, //dense prob
                    1 //channel prune scale
              );
     }
}

TEST_F (testFixture, throughput_diagnostic_resnet50_conv6_ic256)
{
    unsigned short numInputChannel = 256;
    unsigned short numOutputChannel = 256;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned short sizeOutputTileWidthPerColFull = 2;
    unsigned short sizeOutputTileHeight = 8;
    typedef struct {
        unsigned int numTileVertical;
        unsigned int numTileHorizontal;
    } t_num_tile;
    std::vector<t_num_tile> vecTileConfigs = {
      {.numTileVertical=1, .numTileHorizontal=1},
      {.numTileVertical=2, .numTileHorizontal=1},
      {.numTileVertical=3, .numTileHorizontal=1},
      {.numTileVertical=4, .numTileHorizontal=1},
      {.numTileVertical=5, .numTileHorizontal=1},
      {.numTileVertical=6, .numTileHorizontal=1},
      {.numTileVertical=7, .numTileHorizontal=1}
    };
    unsigned char kernelSize = 1;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;
    float prob = 1.0;
    for (t_num_tile& tileConfig: vecTileConfigs)
    {
        unsigned short inputWidth = tileConfig.numTileHorizontal * PE_COLS * sizeOutputTileWidthPerColFull;
        unsigned short inputHeight = tileConfig.numTileVertical * sizeOutputTileHeight;
        launch(
                    inputWidth,
                    inputHeight,
                    numInputChannel,
                    numOutputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    kernelSize,
                    flagEnableRelu,
                    flagSparseInput,
                    flagSparseOutput,
                    op,
                    bias,
                    false, //back to back
                    true, //perf test
                    prob, //dense prob
                    1 //channel prune scale
              );
     }
}

TEST_F (testFixture, throughput_diagnostic_resnet50_conv6_ic320)
{
    unsigned short numInputChannel = 320;
    unsigned short numOutputChannel = 256;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned short sizeOutputTileWidthPerColFull = 2;
    unsigned short sizeOutputTileHeight = 8;
    typedef struct {
        unsigned int numTileVertical;
        unsigned int numTileHorizontal;
    } t_num_tile;
    std::vector<t_num_tile> vecTileConfigs = {
      {.numTileVertical=1, .numTileHorizontal=1},
      {.numTileVertical=2, .numTileHorizontal=1},
      {.numTileVertical=3, .numTileHorizontal=1},
      {.numTileVertical=4, .numTileHorizontal=1},
      {.numTileVertical=5, .numTileHorizontal=1},
      {.numTileVertical=6, .numTileHorizontal=1},
      {.numTileVertical=7, .numTileHorizontal=1}
    };
    unsigned char kernelSize = 1;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;
    float prob = 1.0;
    for (t_num_tile& tileConfig: vecTileConfigs)
    {
        unsigned short inputWidth = tileConfig.numTileHorizontal * PE_COLS * sizeOutputTileWidthPerColFull;
        unsigned short inputHeight = tileConfig.numTileVertical * sizeOutputTileHeight;
        launch(
                    inputWidth,
                    inputHeight,
                    numInputChannel,
                    numOutputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    kernelSize,
                    flagEnableRelu,
                    flagSparseInput,
                    flagSparseOutput,
                    op,
                    bias,
                    false, //back to back
                    true, //perf test
                    prob, //dense prob
                    1 //channel prune scale
              );
     }
}

TEST_F (testFixture, throughput_diagnostic_resnet50_conv6_ic384)
{
    unsigned short numInputChannel = 384;
    unsigned short numOutputChannel = 256;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned short sizeOutputTileWidthPerColFull = 2;
    unsigned short sizeOutputTileHeight = 8;
    typedef struct {
        unsigned int numTileVertical;
        unsigned int numTileHorizontal;
    } t_num_tile;
    std::vector<t_num_tile> vecTileConfigs = {
      {.numTileVertical=1, .numTileHorizontal=1},
      {.numTileVertical=2, .numTileHorizontal=1},
      {.numTileVertical=3, .numTileHorizontal=1},
      {.numTileVertical=4, .numTileHorizontal=1},
      {.numTileVertical=5, .numTileHorizontal=1},
      {.numTileVertical=6, .numTileHorizontal=1},
      {.numTileVertical=7, .numTileHorizontal=1}
    };
    unsigned char kernelSize = 1;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;
    float prob = 1.0;
    for (t_num_tile& tileConfig: vecTileConfigs)
    {
        unsigned short inputWidth = tileConfig.numTileHorizontal * PE_COLS * sizeOutputTileWidthPerColFull;
        unsigned short inputHeight = tileConfig.numTileVertical * sizeOutputTileHeight;
        launch(
                    inputWidth,
                    inputHeight,
                    numInputChannel,
                    numOutputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    kernelSize,
                    flagEnableRelu,
                    flagSparseInput,
                    flagSparseOutput,
                    op,
                    bias,
                    false, //back to back
                    true, //perf test
                    prob, //dense prob
                    1 //channel prune scale
              );
     }
}

TEST_F (testFixture, throughput_diagnostic_resnet50_conv6_oc320)
{
    unsigned short numInputChannel = 64;
    unsigned short numOutputChannel = 320;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned short sizeOutputTileWidthPerColFull = 2;
    unsigned short sizeOutputTileHeight = 8;
    typedef struct {
        unsigned int numTileVertical;
        unsigned int numTileHorizontal;
    } t_num_tile;
    std::vector<t_num_tile> vecTileConfigs = {
      {.numTileVertical=1, .numTileHorizontal=1},
      {.numTileVertical=2, .numTileHorizontal=1},
      {.numTileVertical=3, .numTileHorizontal=1},
      {.numTileVertical=4, .numTileHorizontal=1},
      {.numTileVertical=5, .numTileHorizontal=1},
      {.numTileVertical=6, .numTileHorizontal=1},
      {.numTileVertical=7, .numTileHorizontal=1}
    };
    unsigned char kernelSize = 1;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;
    float prob = 1.0;
    for (t_num_tile& tileConfig: vecTileConfigs)
    {
        unsigned short inputWidth = tileConfig.numTileHorizontal * PE_COLS * sizeOutputTileWidthPerColFull;
        unsigned short inputHeight = tileConfig.numTileVertical * sizeOutputTileHeight;
        launch(
                    inputWidth,
                    inputHeight,
                    numInputChannel,
                    numOutputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    kernelSize,
                    flagEnableRelu,
                    flagSparseInput,
                    flagSparseOutput,
                    op,
                    bias,
                    false, //back to back
                    true, //perf test
                    prob, //dense prob
                    1 //channel prune scale
              );
     }
}

TEST_F (testFixture, throughput_diagnostic_resnet50_conv6_oc384)
{
    unsigned short numInputChannel = 64;
    unsigned short numOutputChannel = 384;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned short sizeOutputTileWidthPerColFull = 2;
    unsigned short sizeOutputTileHeight = 8;
    typedef struct {
        unsigned int numTileVertical;
        unsigned int numTileHorizontal;
    } t_num_tile;
    std::vector<t_num_tile> vecTileConfigs = {
      {.numTileVertical=1, .numTileHorizontal=1},
      {.numTileVertical=2, .numTileHorizontal=1},
      {.numTileVertical=3, .numTileHorizontal=1},
      {.numTileVertical=4, .numTileHorizontal=1},
      {.numTileVertical=5, .numTileHorizontal=1},
      {.numTileVertical=6, .numTileHorizontal=1},
      {.numTileVertical=7, .numTileHorizontal=1}
    };
    unsigned char kernelSize = 1;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;
    float prob = 1.0;
    for (t_num_tile& tileConfig: vecTileConfigs)
    {
        unsigned short inputWidth = tileConfig.numTileHorizontal * PE_COLS * sizeOutputTileWidthPerColFull;
        unsigned short inputHeight = tileConfig.numTileVertical * sizeOutputTileHeight;
        launch(
                    inputWidth,
                    inputHeight,
                    numInputChannel,
                    numOutputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    kernelSize,
                    flagEnableRelu,
                    flagSparseInput,
                    flagSparseOutput,
                    op,
                    bias,
                    false, //back to back
                    true, //perf test
                    prob, //dense prob
                    1 //channel prune scale
              );
     }
}

TEST_F (testFixture, throughput_diagnostic_resnet50_conv6_oc512)
{
    unsigned short numInputChannel = 64;
    unsigned short numOutputChannel = 512;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned short sizeOutputTileWidthPerColFull = 2;
    unsigned short sizeOutputTileHeight = 8;
    typedef struct {
        unsigned int numTileVertical;
        unsigned int numTileHorizontal;
    } t_num_tile;
    std::vector<t_num_tile> vecTileConfigs = {
      {.numTileVertical=1, .numTileHorizontal=1},
      {.numTileVertical=2, .numTileHorizontal=1},
      {.numTileVertical=3, .numTileHorizontal=1},
      {.numTileVertical=4, .numTileHorizontal=1},
      {.numTileVertical=5, .numTileHorizontal=1},
      {.numTileVertical=6, .numTileHorizontal=1},
      {.numTileVertical=7, .numTileHorizontal=1}
    };
    unsigned char kernelSize = 1;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;
    float prob = 1.0;
    for (t_num_tile& tileConfig: vecTileConfigs)
    {
        unsigned short inputWidth = tileConfig.numTileHorizontal * PE_COLS * sizeOutputTileWidthPerColFull;
        unsigned short inputHeight = tileConfig.numTileVertical * sizeOutputTileHeight;
        launch(
                    inputWidth,
                    inputHeight,
                    numInputChannel,
                    numOutputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    kernelSize,
                    flagEnableRelu,
                    flagSparseInput,
                    flagSparseOutput,
                    op,
                    bias,
                    false, //back to back
                    true, //perf test
                    prob, //dense prob
                    1 //channel prune scale
              );
     }
}
#endif //THROUGHPUT_DIAGNOSTIC
#if defined(TEST1_20201126)
TEST_F (testFixture, 20201126_test1)
{
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned short sizeOutputTileWidthPerColFull = 4;
    unsigned short sizeOutputTileHeight = 16;
    unsigned short inputWidth = sizeOutputTileWidthPerColFull * PE_COLS;
    unsigned short inputHeight = 56;
    unsigned char kernelSize = 1;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;
    float prob = 1.0;
    for (unsigned short numInputChannel=64; numInputChannel <= 128; numInputChannel += 16)
    {
        for (unsigned short numOutputChannel= 128; numOutputChannel <= 512; numOutputChannel += 32)
        {
            launch(
                        inputWidth,
                        inputHeight,
                        numInputChannel,
                        numOutputChannel,
                        numInputGroup,
                        numOutputGroup,
                        inputHeightSPUnitSize,
                        inputWidthSPUnitSize,
                        sizeOutputTileWidthPerColFull,
                        sizeOutputTileHeight,
                        kernelSize,
                        flagEnableRelu,
                        flagSparseInput,
                        flagSparseOutput,
                        op,
                        bias,
                        false, //back to back
                        true, //perf test
                        prob, //dense prob
                        1 //channel prune scale
                  );
        }
    }
}
#endif
#if defined(TEST2_20201126)
TEST_F (testFixture, 20201126_test2)
{
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned short inputWidth = 56;
    unsigned short inputHeight = 56;
    unsigned char kernelSize = 1;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;
    float prob = 1.0;
    unsigned short numInputChannel = 64;
    unsigned short numOutputChannel = 256;
    for (unsigned short sizeOutputTileWidthPerColFull=1;
         sizeOutputTileWidthPerColFull <= 4;
         sizeOutputTileWidthPerColFull += 1)
    {
        //Hack, because this test assumes num active cols during partial tile = 1
        if (sizeOutputTileWidthPerColFull != 3)
        {
            for (unsigned short sizeOutputTileHeight= 2;
                 sizeOutputTileHeight <= 32;
                 sizeOutputTileHeight += 2)
            {
                launch(
                            inputWidth,
                            inputHeight,
                            numInputChannel,
                            numOutputChannel,
                            numInputGroup,
                            numOutputGroup,
                            inputHeightSPUnitSize,
                            inputWidthSPUnitSize,
                            sizeOutputTileWidthPerColFull,
                            sizeOutputTileHeight,
                            kernelSize,
                            flagEnableRelu,
                            flagSparseInput,
                            flagSparseOutput,
                            op,
                            bias,
                            false, //back to back
                            true, //perf test
                            prob, //dense prob
                            1 //channel prune scale
                      );
            }
        }
    }
}
#endif
#if defined(ELTADD7_202021129)
TEST_F (testFixture, eltadd7_20201129)
{
    unsigned char inputWidth = 56;
    unsigned char inputHeight = 56;
    unsigned int numInputChannel = 256;
    unsigned int numOutputChannel = numInputChannel;
//    unsigned char inputWidth = 7;
//    unsigned char inputHeight = 2;
//    unsigned int numInputChannel = 64;
//    unsigned int numOutputChannel = numInputChannel;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 1;
    unsigned char sizeOutputTileHeight = 32;
    unsigned char kernelSize = 1;
    bool flagEnableRelu = true;
    bool flagSparseInput = false;
    bool flagSparseOutput = false;
    OPERATION op = ELT_ADD;
    float bias = 0.0f;
    int channelPruneScale = 1;
    launch(
                inputWidth,
                inputHeight,
                numInputChannel,
                numOutputChannel,
                numInputGroup,
                numOutputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                kernelSize,
                flagEnableRelu,
                flagSparseInput,
                flagSparseOutput,
                op,
                bias,
                false, //back to back
                true, //perf test
                1.0, //dense prob
                channelPruneScale
          );
}
#endif
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
    unsigned char kernelSize = 3;
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
            unsigned char numOutputChannel = numInputChannel;
            launch(
                        inputWidth,
                        inputHeight,
                        numInputChannel,
                        numOutputChannel,
                        numInputGroup,
                        numOutputGroup,
                        inputHeightSPUnitSize,
                        inputWidthSPUnitSize,
                        sizeOutputTileWidthPerColFull,
                        sizeOutputTileHeight,
                        kernelSize,
                        flagEnableRelu,
                        flagSparseInput,
                        flagSparseOutput,
                        op,
                        bias,
                        false, //back to back
                        true, //perf test
                        prob, //dense prob
                        pruneScale
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
    unsigned char kernelSize = 3;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;
    float prob = 1.0f;
    float pruneScale = CLUSTER_SIZE;

    for (auto& numInputChannel: vecInputChannel) {
        unsigned char numOutputChannel = numInputChannel;
        launch(
                    inputWidth,
                    inputHeight,
                    numInputChannel,
                    numOutputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    kernelSize,
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
    unsigned char numOutputChannel = numInputChannel;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 1;
    unsigned char sizeOutputTileHeight = 1;
    unsigned char kernelSize = 3;
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
                    numOutputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    kernelSize,
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
    unsigned char numOutputChannel = numInputChannel;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = ((inputWidth / PE_COLS) > 8) ?
                8 : (inputWidth / PE_COLS);
    unsigned char sizeOutputTileHeight = 8;
    unsigned char kernelSize = 3;
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
                    numOutputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    kernelSize,
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
    unsigned char numOutputChannel = numInputChannel + numInputChannel;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 1;
    unsigned char sizeOutputTileHeight = 1;
    unsigned char kernelSize = 3;
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
                    numOutputChannel,
                    numInputGroup,
                    numOutputGroup,
                    inputHeightSPUnitSize,
                    inputWidthSPUnitSize,
                    sizeOutputTileWidthPerColFull,
                    sizeOutputTileHeight,
                    kernelSize,
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
#endif //PERF_TEST
#if defined(VALIDATE)
TEST_F (testFixture, conv_dense_input_dense_output_plain)
{
    unsigned char inputWidth = 4;
    unsigned char inputHeight = 4;
    //DOESN'T WORK!?
    unsigned char numInputChannel = 13;
    //unsigned char numInputChannel = 2;
    unsigned char numOutputChannel = numInputChannel;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 2;
    unsigned char sizeOutputTileHeight = 4;
    unsigned char kernelSize = 3;
    bool flagEnableRelu = false;
    bool flagSparseInput = false;
    bool flagSparseOutput = false;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;

    launch(
                inputWidth,
                inputHeight,
                numInputChannel,
                numOutputChannel,
                numInputGroup,
                numOutputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                kernelSize,
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
    unsigned char numInputChannel = 120;
    unsigned char numOutputChannel = numInputChannel;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 2;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 2;
    unsigned char sizeOutputTileHeight = 4;
    unsigned char kernelSize = 3;
    bool flagEnableRelu = false;
    bool flagSparseInput = false;
    bool flagSparseOutput = true;
    OPERATION op = MAX_POOL;
    float bias = 0.0f;

    launch(
                inputWidth,
                inputHeight,
                numInputChannel,
                numOutputChannel,
                numInputGroup,
                numOutputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                kernelSize,
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
    unsigned char numInputChannel = 127;
    unsigned char numOutputChannel = numInputChannel;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 2;
    unsigned char sizeOutputTileHeight = 4;
    unsigned char kernelSize = 3;
    bool flagEnableRelu = false;
    bool flagSparseInput = false;
    bool flagSparseOutput = true;
    OPERATION op = ELT_ADD;
    float bias = 0.0f;

    launch(
                inputWidth,
                inputHeight,
                numInputChannel,
                numOutputChannel,
                numInputGroup,
                numOutputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                kernelSize,
                flagEnableRelu,
                flagSparseInput,
                flagSparseOutput,
                op,
                bias
          );
}

TEST_F (testFixture, concat_sparse_output_grouped)
{
    unsigned char inputWidth = 2;
    unsigned char inputHeight = 2;
    unsigned char numInputChannel = 120;
    unsigned char numOutputChannel = 2*numInputChannel;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 2;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 2;
    unsigned char sizeOutputTileHeight = 4;
    unsigned char kernelSize = 3;
    bool flagEnableRelu = false;
    bool flagSparseInput = false;
    bool flagSparseOutput = true;
    OPERATION op = CONCATENATION;
    float bias = 0.0f;

    launch(
                inputWidth,
                inputHeight,
                numInputChannel,
                numOutputChannel,
                numInputGroup,
                numOutputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                kernelSize,
                flagEnableRelu,
                flagSparseInput,
                flagSparseOutput,
                op,
                bias
          );
}

TEST_F (testFixture, global_avg_pool_sparse_output_grouped)
{
    unsigned char inputWidth = 4;
    unsigned char inputHeight = 4;
    unsigned char numInputChannel = 120;
    unsigned char numOutputChannel = numInputChannel;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 2;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 2;
    unsigned char sizeOutputTileHeight = 4;
    unsigned char kernelSize = 3;
    bool flagEnableRelu = false;
    bool flagSparseInput = false;
    bool flagSparseOutput = true;
    OPERATION op = AVG_POOL;
    float bias = 0.0f;

    launch(
                inputWidth,
                inputHeight,
                numInputChannel,
                numOutputChannel,
                numInputGroup,
                numOutputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                kernelSize,
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
    unsigned char numInputChannel = 16;
    unsigned char numOutputChannel = numInputChannel;
    unsigned char numInputGroup = 2;
    unsigned char numOutputGroup = 2;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 2;
    unsigned char sizeOutputTileHeight = 4;
    unsigned char kernelSize = 3;
    bool flagEnableRelu = false;
    bool flagSparseInput = false;
    bool flagSparseOutput = false;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;

    launch(
                inputWidth,
                inputHeight,
                numInputChannel,
                numOutputChannel,
                numInputGroup,
                numOutputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                kernelSize,
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
    unsigned char numInputChannel = 16;
    unsigned char numOutputChannel = numInputChannel;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 2;
    unsigned char inputWidthSPUnitSize = 2;
    unsigned char sizeOutputTileWidthPerColFull = 2;
    unsigned char sizeOutputTileHeight = 4;
    unsigned char kernelSize = 3;
    bool flagEnableRelu = false;
    bool flagSparseInput = false;
    bool flagSparseOutput = false;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;

    launch(
                inputWidth,
                inputHeight,
                numInputChannel,
                numOutputChannel,
                numInputGroup,
                numOutputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                kernelSize,
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
    unsigned char numInputChannel = 16;
    unsigned char numOutputChannel = numInputChannel;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 2;
    unsigned char sizeOutputTileHeight = 4;
    unsigned char kernelSize = 3;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;

    launch(
                inputWidth,
                inputHeight,
                numInputChannel,
                numOutputChannel,
                numInputGroup,
                numOutputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                kernelSize,
                flagEnableRelu,
                flagSparseInput,
                flagSparseOutput,
                op,
                bias
          );
}

TEST_F (testFixture, conv_sparse_input_sparse_output_small_tile)
{
    unsigned char inputWidth = 4;
    unsigned char inputHeight = 4;
    unsigned char numInputChannel = 16;
    unsigned char numOutputChannel = numInputChannel;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 1;
    unsigned char sizeOutputTileHeight = 1;
    unsigned char kernelSize = 3;
    bool flagEnableRelu = false;
    bool flagSparseInput = true;
    bool flagSparseOutput = true;
    OPERATION op = CONVOLUTION;
    float bias = 0.0f;

    launch(
                inputWidth,
                inputHeight,
                numInputChannel,
                numOutputChannel,
                numInputGroup,
                numOutputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                kernelSize,
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
    unsigned char numInputChannel =16;
    unsigned char numOutputChannel = numInputChannel;
    unsigned char numInputGroup = 1;
    unsigned char numOutputGroup = 1;
    unsigned char inputHeightSPUnitSize = 1;
    unsigned char inputWidthSPUnitSize = 1;
    unsigned char sizeOutputTileWidthPerColFull = 2;
    unsigned char sizeOutputTileHeight = 4;
    unsigned char kernelSize = 3;
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
                numOutputChannel,
                numInputGroup,
                numOutputGroup,
                inputHeightSPUnitSize,
                inputWidthSPUnitSize,
                sizeOutputTileWidthPerColFull,
                sizeOutputTileHeight,
                kernelSize,
                flagEnableRelu,
                flagSparseInput,
                flagSparseOutput,
                op,
                bias,
                flag2Layer
          );
}
#endif

int main(int argc, char* argv[]) {

    ::testing::InitGoogleTest( &argc, argv);
    return RUN_ALL_TESTS();
}

void testFixture::SetUp()
{
#ifdef C5SOC
    binaryFile = "device_utils.aocx";
#else
    binaryFile = "device_utils.aocx";
#if defined(EMULATE)
    binaryFile = "smallBuffer.aocx";
#endif
#endif

#if defined(EMULATE)
    std::string platformName = "Intel(R) FPGA Emulation Platform for OpenCL(TM)";
#else
    std::string platformName = "Intel(R) FPGA SDK for OpenCL(TM)";
#endif

    GraphRuntime::t_accelerator_info acceleratorInfo =
        {   .numPERows=PE_ROWS,
            .numPECols=PE_COLS,
            .numClusterInCompressionBlock=COMPRESSION_WINDOW_SIZE,
            .numClusterInTransferBlock=TRANSFER_SIZE,
            .numScalarInCluster=CLUSTER_SIZE
        };
   accelerator = GraphRuntime::AcceleratorWrapper(binaryFile, platformName, acceleratorInfo, 0);
}

std::vector<float> testFixture::generateInputTensor(
            unsigned short _inputWidth,
            unsigned short _inputHeight,
            unsigned int _numInputChannel,
            unsigned char _numGroupCurrentLayer,
            bool _alternateSign
        )
{
    assert(_numInputChannel % _numGroupCurrentLayer == 0);
    assert (((signed int) _numInputChannel < 128) && "Too many input channels!");
    unsigned char numICPerGroup = _numInputChannel / _numGroupCurrentLayer;
    std::vector<float> inputVector;
    for (unsigned char g=0; g<_numGroupCurrentLayer;g++)
    {
        for (unsigned short h=0; h<_inputHeight; h++)
        {
            for (unsigned short w=0; w<_inputWidth; w++)
            {
                for (unsigned int c=0; c<numICPerGroup; c++)
                {
                    unsigned int globalChannel = g*numICPerGroup+c;
                    signed char fpBits = ((w % 2 == 1) && _alternateSign)
                            ? -1*((signed char) globalChannel) : globalChannel;
                    fixedPointNumber fpNumber(fpBits, FRAC_WIDTH, INT_WIDTH);
                    inputVector.push_back(fpNumber.convert2Float());
                }
            }
        }
    }

    return inputVector;
}

//std::vector<float> testFixture::generateSparseInput(
//            unsigned short _inputWidth,
//            unsigned short _inputHeight,
//            unsigned int _numInputChannel,
//            unsigned char _numGroupCurrentLayer,
//            float denseProb,
//            int channelPruneScale
//        )
//{
//     std::mt19937 generator(INPUT_SEED);
//     std::bernoulli_distribution bernDistribution(denseProb);
//     assert(_numInputChannel % _numGroupCurrentLayer == 0);
//     unsigned int numICPerGroup = _numInputChannel / _numGroupCurrentLayer;
//     std::vector<float> inputVector;
//     bool writePositive = true;
//     for (unsigned char g=0; g<_numGroupCurrentLayer;g++)
//     {
//         for (unsigned short h=0; h<_inputHeight; h++)
//         {
//             for (unsigned short w=0; w<_inputWidth; w++)
//             {
//                 bool flagNonZero = false;
//                 for (unsigned int c=0; c<numICPerGroup; c++)
//                 {
//                     if (((int) c) % channelPruneScale == 0)
//                     {
//                         flagNonZero = bernDistribution(generator);
//                     }
//                     signed char fpBits = (writePositive) ? 1 : -1;
//                     if (flagNonZero == false)
//                     {
//                         fpBits = 0;
//                     }
//                     else
//                     {
//                         writePositive = !writePositive;
//                     }
//                     fixedPointNumber fpNumber(fpBits, FRAC_WIDTH, INT_WIDTH);
//                     inputVector.push_back(fpNumber.convert2Float());
//                 }
//             }
//         }
//     }

//     return inputVector;
//}

std::vector<fixedPointNumber> testFixture::generateWeights (
        unsigned char _kernelSize,
        unsigned int _numInputChannel,
        unsigned char _numGroups
        )
{
    assert(_kernelSize % 2 == 1);
    assert(_numInputChannel % _numGroups == 0);
    std::vector<fixedPointNumber> fpWeightTensor;
    unsigned int numICPerGroup = _numInputChannel / _numGroups;

    for (unsigned char g=0; g<_numGroups; g++)
    {
        //Number of OC per group equals to the number of IC per group
        for (unsigned int iFilter=0; iFilter<numICPerGroup; iFilter++)
        {
            for (unsigned char iH=0; iH<_kernelSize; iH++)
            {
                bool hCentre = (iH == (_kernelSize / 2));
                for (unsigned char iW=0; iW<_kernelSize; iW++)
                {
                    bool vCentre = (iW == (_kernelSize / 2));
                    for (unsigned int iC=0; iC<numICPerGroup; iC++)
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
        unsigned int _numInputChannel,
        unsigned int _numOutputChannel,
        unsigned int _numGroups,
        unsigned int _numNZClustersInPruneRange
        )
{
    assert(_kernelSize % 2 == 1);
    assert(_numInputChannel % _numGroups == 0);
    fixedPointNumber fpZero(0.0f, FRAC_WIDTH, INT_WIDTH);
    assert ((((unsigned int) _numOutputChannel) % _numGroups == 0) && "Number of output channels is not divisble by the number of groups.");

    unsigned int numOCPerGroup = _numOutputChannel / _numGroups;
    unsigned int numICPerGroup = _numInputChannel / _numGroups;
    unsigned int numPruneRangePerStrip =
            DIVIDE_CEIL(
                numICPerGroup,
                CLUSTER_SIZE * PRUNE_RANGE_IN_CLUSTER);

    std::vector<fixedPointNumber> fpWeightTensor(
                _numOutputChannel * _kernelSize * _kernelSize * numICPerGroup,
                fpZero);

    std::mt19937 generator(WEIGHT_SEED);
    std::uniform_real_distribution<float> distribution(-2.0, 2.0);
    for (unsigned char g=0; g<_numGroups; g++)
    {
        //Number of OC per group equals to the number of IC per group
        for (unsigned int iFilter=0; iFilter<numOCPerGroup; iFilter++)
        {
            for (unsigned char iH=0; iH<_kernelSize; iH++)
            {
                for (unsigned char iW=0; iW<_kernelSize; iW++)
                {
                    for (int iPruneRange=0; iPruneRange<numPruneRangePerStrip; iPruneRange++)
                    {
                        //Randomly select up to _numNZClustersPerPruneRange clusters to set to non-zero
                        std::vector<int> idxClusters (PRUNE_RANGE_IN_CLUSTER, 0x0);
                        std::iota(idxClusters.begin(), idxClusters.end(), 0);
                        std::shuffle(idxClusters.begin(), idxClusters.end(), generator);
                        for (int i=0; i<_numNZClustersInPruneRange; i++)
                        {
                            int idx = idxClusters.at(i);
                            for (int c=0; c<CLUSTER_SIZE; c++)
                            {
                                int channelIdx =
                                        iPruneRange*PRUNE_RANGE_IN_CLUSTER*CLUSTER_SIZE
                                        + idx*CLUSTER_SIZE + c;
                                if (channelIdx < numICPerGroup)
                                {
                                    int weightIdx =
                                            iFilter * _kernelSize * _kernelSize * numICPerGroup
                                            + (iH * _kernelSize + iW) * numICPerGroup
                                            + channelIdx;
                                    fpWeightTensor.at(weightIdx) =
                                            fixedPointNumber(distribution(generator), FRAC_WIDTH, INT_WIDTH);
                                }
                            } //for over cluster size
                        } // for over the nz clusters
                    } //for over the prune ranges
                } //for over kernel width
            } //for over kernel height
        } //for over filters
    } //for over groups

    return fpWeightTensor;
}

void testFixture::launch (unsigned short _inputWidth,
        unsigned short _inputHeight,
        unsigned int _numInputChannel,
        unsigned int  _numOutputChannel,
        unsigned char _numInputGroup, //The code will override this to 1 if the operation is not convolution
        unsigned char _inputHeightSPUnitSize, //The code will override this to 1 if the operation is not convolution
        unsigned char _inputWidthSPUnitSize, //The code will overide this to 1 if the operation is not convolution
        unsigned short _sizeOutputTileWidthPerColFull, //The code will override this to 1 if the operation is not convolution
        unsigned short _sizeOutputTileHeight, //The code will overrid this to 1 if the operation is not convolution
        unsigned char _kernelSize, //Size of the convolution kernel
        bool _flagEnableRelu,
        OPERATION op,
        float _bias,
        float denseProb,
       //Only matter for convolution
        bool flagMultiLayerConv,
        bool _flagPerformanceTest)
{
    //Checking the parameters' consistency AND
    //Derive parameters

    cl_uchar kernelSize;
    cl_uchar stride;
    unsigned int numInputChannel0;
    unsigned int numInputChannel1;
    unsigned int numOutputChannels;
    unsigned int numOutputChannelPerGroup;
    unsigned short numOutputWidth;
    unsigned short numOutputHeight;
    unsigned char numGroupCurrentLayer;
    unsigned char inputHeightSPUnitSize;
    unsigned char inputWidthSPUnitSize;
    unsigned short inputHeightSPSize;
    unsigned short inputWidthSPSize;
    unsigned short sizeOutputTileWidthPerCol;
    unsigned short sizeOutputTileHeight;
    unsigned char verticalBorderPadding;
    unsigned char horizontalBorderPadding;

    /*
     * Dervied the parameters required for generating instructions
    */
    switch (op) {
        case CONVOLUTION: {
            assert(_numInputChannel % _numInputGroup == 0);
            assert((flagMultiLayerConv==false) || (_numInputGroup == 1));
            numInputChannel0 = _numInputChannel;
            numInputChannel1 = 0;
            inputHeightSPUnitSize = _inputHeightSPUnitSize;
            inputWidthSPUnitSize = _inputWidthSPUnitSize;
            numGroupCurrentLayer = _numInputGroup;
            sizeOutputTileWidthPerCol = _sizeOutputTileWidthPerColFull;
            sizeOutputTileHeight = _sizeOutputTileHeight;
            kernelSize = _kernelSize;
            stride = 1;

            numOutputChannels = _flagPerformanceTest ? _numOutputChannel : _numInputChannel;
            assert(numOutputChannels % _numInputGroup == 0);
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
            assert(_inputHeight % 2 == 0);
            assert(_inputWidth % 2 == 0);
        }
        break;
        case ELT_ADD: {
            numInputChannel0 = _numInputChannel;
            numInputChannel1 = _numInputChannel;
            inputHeightSPUnitSize = 1;
            inputWidthSPUnitSize = 1;
            numGroupCurrentLayer = 1;
            sizeOutputTileWidthPerCol = _sizeOutputTileWidthPerColFull;
            sizeOutputTileHeight = _sizeOutputTileHeight;
            kernelSize = 1;
            stride = 1;
            numOutputChannels = numInputChannel0;


        }
        break;
        case CONCATENATION: {
            numInputChannel0 = _numInputChannel;
            numInputChannel1 = _numInputChannel;
            inputHeightSPUnitSize = 1;
            inputWidthSPUnitSize = 1;
            numGroupCurrentLayer = 1;
            sizeOutputTileWidthPerCol = _sizeOutputTileWidthPerColFull;
            sizeOutputTileHeight = _sizeOutputTileHeight;
            kernelSize = 1;
            stride = 1;
            numOutputChannels = (unsigned short) numInputChannel0 + (unsigned short) numInputChannel1;

            assert(_numInputChannel <= 127);
        }
        break;
        case AVG_POOL: {
            assert ((_inputWidth == _inputHeight) && "Input width does not equal input height");
            assert ((_inputHeight % 2 == 0) && "Input height is not divisible by 2");
            numInputChannel0 = _numInputChannel;
            numInputChannel1 = 0;
            inputHeightSPUnitSize = 1;
            inputWidthSPUnitSize = 1;
            numGroupCurrentLayer = 1;
            sizeOutputTileWidthPerCol = 1;
            sizeOutputTileHeight = 1;
            kernelSize = _inputWidth;
            stride = 1;
            numOutputChannels = numInputChannel0;
        }
        break;
        default:
            std::cout<<"Unsupported operation type: "<<op<<std::endl;
            assert(-1);
    } //switch
    inputWidthSPSize = inputWidthSPUnitSize*(_inputWidth-1) + 1;
    inputHeightSPSize = inputHeightSPUnitSize*(_inputHeight-1) + 1;
    numOutputWidth = (op==MAX_POOL || op==AVG_POOL) ? inputWidthSPSize / kernelSize : inputWidthSPSize;
    numOutputHeight = (op==MAX_POOL || op==AVG_POOL) ? inputHeightSPSize / kernelSize : inputHeightSPSize;
    verticalBorderPadding = (op==CONVOLUTION) ? ((stride-1) * inputWidthSPSize + kernelSize - stride)/2 : 0;
    horizontalBorderPadding  = (op==CONVOLUTION) ? ((stride-1) * inputHeightSPSize + kernelSize - stride)/2 : 0;
    unsigned char numActiveColsPartialOutputTile = 1;
    numOutputChannelPerGroup = numOutputChannels / numGroupCurrentLayer;

    /* Generate the dense, floating point tensors
     * */
    int stepCount = 1;
    std::cout <<stepCount++<<". Preparing the test tensors. Test operation type is "<<op<<std::endl;
    if (_flagPerformanceTest == true)
    {
        std::cout <<"This is a performance test"<<std::endl
                  <<"dense prob: "<<denseProb<<std::endl;
    }
    std::cout <<"Input SP dimensions (H, W):  "<<(unsigned int) inputHeightSPSize<<" "<<(unsigned int) inputWidthSPSize<<std::endl
              <<"PE dimension (H, W): "<<PE_ROWS<<" "<<PE_COLS<<std::endl
              <<"CLUSTER_SIZE: "<<CLUSTER_SIZE<<std::endl
              <<"PE_SIMD_SIZE: "<<TRANSFER_SIZE<<std::endl
              <<"PRUNE_RANGE_IN_CLUSTER "<<PRUNE_RANGE_IN_CLUSTER<<std::endl
              <<"ACTIVATION_WIDE_SIZE "<<WIDE_SIZE<<std::endl
              <<"WEIGHT_WIDE_SIZE "<<WEIGHT_WIDE_SIZE<<std::endl
              <<"Output planar dimensions (H, W): "<<(unsigned int)numOutputHeight<<" "<<(unsigned int)numOutputWidth<<std::endl
              <<"Full output tile per col planar sizes (H, W): "<<(unsigned int)sizeOutputTileHeight<<" "<<(unsigned int)sizeOutputTileWidthPerCol<<std::endl
              <<"Input channels 0: "<<(unsigned int) numInputChannel0<<std::endl
              <<"Input channels 1: "<<(unsigned int) numInputChannel1<<std::endl
              <<"Output channels: "<<(unsigned int) numOutputChannels<<std::endl
              <<"Number of groups in current layer: "<<(unsigned int)numGroupCurrentLayer<<std::endl;

    std::vector<float> inputTensorDense;
    bool alternateSign = (op == AVG_POOL) ? false : true;
    inputTensorDense = generateInputTensor(_inputWidth, _inputHeight, _numInputChannel, numGroupCurrentLayer, alternateSign);

    //Generate qunatized weight tensors
    std::vector<fixedPointNumber> inputWeightDense;
    int numNZClustersPerPrungingRange = std::ceil((float) PRUNE_RANGE_IN_CLUSTER * denseProb);
    if (op == CONVOLUTION)
    {
        if (_flagPerformanceTest == true)
        {
            inputWeightDense = generateSparseWeights(
                            _kernelSize,
                            _numInputChannel,
                            numOutputChannels,
                            numGroupCurrentLayer,
                            numNZClustersPerPrungingRange
                        );
        }
        else
        {
            inputWeightDense = generateWeights((unsigned char) kernelSize, _numInputChannel, numGroupCurrentLayer);
        }
    }
    //Generate biases
    t_bias fixedBias = (t_bias) (std::nearbyint(_bias * (float) (1 << (FRAC_WIDTH + FRAC_WIDTH)) ));
    auto pBiasVector = std::make_shared<t_aligned_short_vector>(numOutputChannels, fixedBias);

    /* 2. Allocate the aligned weight tensors and compress them if necessary
     * */
    std::cout <<stepCount++<<". Allocate, align, and compress the test tensors."<<std::endl;

    unsigned short numICPerGroup = _numInputChannel / numGroupCurrentLayer;
    unsigned short maxClusterIndexInCompressionBlock = COMPRESSION_WINDOW_SIZE-1;
    unsigned short maxClusterIndexInTransferBlock = TRANSFER_SIZE-1;
    unsigned short maxScalarIndexInCluster = CLUSTER_SIZE-1;

    std::shared_ptr<DeviceWeightTensor> pWeight;

    //Prepare weights
    if (op==CONVOLUTION)
    {
#if defined(SPW_SYSTEM)
        pWeight.reset(new DeviceSpWTensor (
                    inputWeightDense,
                    //_outputChannel
                    numOutputChannels,
                    //_inputChannel
                    numICPerGroup,
                    //_width
                    _kernelSize,
                    //_height,
                    _kernelSize,
                    //_peSimdSize
                    PE_SIMD_SIZE,
                    //_clusterSize
                    CLUSTER_SIZE,
                    //_numClustersInPruningRange
                    PRUNE_RANGE_IN_CLUSTER,
                    //_numNZClustersPerPruningRange
                    numNZClustersPerPrungingRange
                ) );
#else
        pWeight.reset( new DeviceWeightTensor (
                   numOutputChannels,
                   //_inputChannel
                   numICPerGroup,
                   //_width
                   _kernelSize,
                   //_height,
                   _kernelSize,
                   //_peSimdSize
                   PE_SIMD_SIZE,
                   //_clusterSize
                   CLUSTER_SIZE
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
        memDramBlockFilterStride = pWeight->getDramBlocksInFilter();
    }
    else
    {
        memDramBlockFilterStride = 0;
    }

    //Stride across activation tensor strip in terms of activation words
    signed int memIAColStride =
            DIVIDE_CEIL(numInputChannel0, ACTIVATION_BURST_SIZE_BYTE) * ACTIVATION_BURST_SIZE_BYTE;
    std::cout <<"memIA0ColStride: "<<memIAColStride<<std::endl;

    signed int memOAColStride =
            (numGroupCurrentLayer - 1) * numOutputChannelPerGroup
            + DIVIDE_CEIL(numOutputChannelPerGroup, ACTIVATION_BURST_SIZE_BYTE) * ACTIVATION_BURST_SIZE_BYTE;
    std::cout <<"memOAColStride: "<<memOAColStride<<std::endl;


    signed int memIARowStride = memIAColStride*_inputWidth;
    signed int memIAGroupStride = memIARowStride * _inputHeight;
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
    else if (op == AVG_POOL)
    {
        int divisorShift = (int) std::ceil(log2(kernelSize*kernelSize));
        assert ( (divisorShift >=0) && "Average pool divisor is less than 1");
        int tentativeLeftShift = 7 - OUTPUT_INT_WIDTH - FRAC_WIDTH - divisorShift;
        if (tentativeLeftShift >= 0)
        {
            instOutputShiftLeft = TRUE;
            instOutputShiftBits = tentativeLeftShift;
        }
        else
        {
            instOutputShiftLeft = FALSE;
            instOutputShiftBits = ((-1) * tentativeLeftShift);
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
//        int numCBPerIMOAChannelGroup =
//                1 + (numInputChannel0 / numGroupCurrentLayer - 1) / (CLUSTER_SIZE * COMPRESSION_WINDOW_SIZE);
//        int numTBPerCB = COMPRESSION_WINDOW_SIZE / TRANSFER_SIZE + 1;
        int intermediateAColStride =
                (numGroupCurrentLayer-1) * numOutputChannelPerGroup
                + DIVIDE_CEIL(numOutputChannelPerGroup, ACTIVATION_BURST_SIZE_BYTE) * ACTIVATION_BURST_SIZE_BYTE;
        int intermediateARowStride = intermediateAColStride * _inputWidth;
        //First set of instructions
        instruction_generator(
                    op,
                    vecIAMoverInstruction,
                    vecOAMoverInstruction,
                    graph.vecIATileControllerInstruction,
                    graph.vecOATileControllerInstruction,
                    graph.vecWMoverInstruction,
                    graph.vecMiscInstruction,

                    // bool flagIA0ShiftLeft,
                    true,
                    // unsigned int numIA0ShiftAmount,
                    0,
                    // bool flagIA1ShiftLeft,
                    true,
                    // unsigned int numIA1ShiftAmount,
                    0,

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
                    memIAColStride,
                    memIARowStride,

                    //input 1 strides
                    memIAColStride,
                    memIARowStride,

                    //output stride
                    intermediateAColStride,

                    //weight stride
                    memDramBlockFilterStride,

                    //unsigned char flagTensorSync,
                    FALSE,

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

            #if defined(SPW_SYSTEM)
                    numNZClustersPerPrungingRange,
            #endif

                    sizeOutputTileHeight,
                    sizeOutputTileWidthPerCol,
                    numActiveColsPartialOutputTile,

                    numInputChannel0,
                    numInputChannel1,
                    numGroupCurrentLayer,
                    numOutputChannels
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
                 .numOAMoverInstructions=numOAMoverInstructions,
                .outputTileHeight = sizeOutputTileHeight,
                .outputTileWidthPerCol = sizeOutputTileWidthPerCol,
                .numActiveColsPartialOutputTile = numActiveColsPartialOutputTile
                 });
            offsetIAMoverInstruction += numIAMoverInstructions;
            offsetOAMoverInstruction += numOAMoverInstructions;

            graph.vecWeightDramBlockStart.push_back(offsetWeightsDramBlock);
            graph.vecBiasStart.push_back(offsetBiasesDramBlock);
            graph.pWeights.push_back(pWeight);
            graph.pBiasVector.push_back(pBiasVector);
            offsetWeightsDramBlock +=
                    pWeight->getDramBlocksInFilter() * numOutputChannels ;
            offsetBiasesDramBlock += numOutputChannels;
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

                    // bool flagIA0ShiftLeft,
                    true,
                    // unsigned int numIA0ShiftAmount,
                    0,
                    // bool flagIA1ShiftLeft,
                    true,
                    // unsigned int numIA1ShiftAmount,
                    0,

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

                    //input 1 strides
                    intermediateAColStride,
                    intermediateARowStride,

                    //output stride
                    memOAColStride,

                    //weight stride
                    memDramBlockFilterStride,

                    //unsigned char flagTensorSync,
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

            #if defined(SPW_SYSTEM)
                    numNZClustersPerPrungingRange,
            #endif
                    sizeOutputTileHeight,
                    sizeOutputTileWidthPerCol,
                    numActiveColsPartialOutputTile,

                    numInputChannel0,
                    numInputChannel1,
                    numGroupCurrentLayer,
                    numOutputChannels
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
                 .numOAMoverInstructions=numOAMoverInstructions,
                .outputTileHeight = sizeOutputTileHeight,
                .outputTileWidthPerCol = sizeOutputTileWidthPerCol,
                .numActiveColsPartialOutputTile = numActiveColsPartialOutputTile
                 });
            offsetIAMoverInstruction += numIAMoverInstructions;
            offsetOAMoverInstruction += numOAMoverInstructions;

            graph.vecWeightDramBlockStart.push_back(offsetWeightsDramBlock);
            graph.vecBiasStart.push_back(offsetBiasesDramBlock);
            graph.pWeights.push_back(pWeight);
            graph.pBiasVector.push_back(pBiasVector);
            offsetWeightsDramBlock +=
                    pWeight->getDramBlocksInFilter() * numOutputChannels;
            offsetBiasesDramBlock += numOutputChannels;
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

                    // bool flagIA0ShiftLeft,
                    true,
                    // unsigned int numIA0ShiftAmount,
                    0,
                    // bool flagIA1ShiftLeft,
                    true,
                    // unsigned int numIA1ShiftAmount,
                    0,

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
                    memIAColStride,
                    memIARowStride,

                    //input 1 strides
                    memIAColStride,
                    memIARowStride,

                    //output stride
                    memOAColStride,

                    //weight stride
                    memDramBlockFilterStride,

                    //unsigned char flagTensorSync,
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

            #if defined(SPW_SYSTEM)
                    numNZClustersPerPrungingRange,
            #endif

                    sizeOutputTileHeight,
                    sizeOutputTileWidthPerCol,
                    numActiveColsPartialOutputTile,

                    numInputChannel0,
                    numInputChannel1,
                    numGroupCurrentLayer,
                    numOutputChannels
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
                GraphRuntime::t_layer_info {.layerName="layer0",
                 .offsetIAMoverInstruction=offsetIAMoverInstruction,
                 .numIAMoverInstruction=numIAMoverInstructions,
                 .offsetOAMoverInstruction=offsetOAMoverInstruction,
                 .numOAMoverInstructions=numOAMoverInstructions,
                .outputTileHeight = sizeOutputTileHeight,
                .outputTileWidthPerCol = sizeOutputTileWidthPerCol,
                .numActiveColsPartialOutputTile = numActiveColsPartialOutputTile
                 });
            offsetIAMoverInstruction += numIAMoverInstructions;
            offsetOAMoverInstruction += numOAMoverInstructions;

            if (op == CONVOLUTION)
            {
                graph.vecWeightDramBlockStart.push_back(offsetWeightsDramBlock);
                graph.vecBiasStart.push_back(offsetBiasesDramBlock);
                graph.pWeights.push_back(pWeight);
                graph.pBiasVector.push_back(pBiasVector);
                offsetWeightsDramBlock +=
                        pWeight->getDramBlocksInFilter() * numOutputChannels ;
                offsetBiasesDramBlock += numOutputChannels;
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
                                        .channel=numInputChannel0,
                                        .height=_inputHeight,
                                        .width=_inputWidth,
                                        .stripStrideSeenBySource=numInputChannel0,
                                        .numFracBits=FRAC_WIDTH,
                                        .blobName="input"
                                    });
    int oaRegion = ((op == CONVOLUTION) && (flagMultiLayerConv == true)) ? 0 : 1;
    int oaStripStrideSeenBySource =
            numOutputChannelPerGroup * (numGroupCurrentLayer - 1)
            + DIVIDE_CEIL(numOutputChannelPerGroup, ACTIVATION_BURST_SIZE_BYTE) * ACTIVATION_BURST_SIZE_BYTE;
    graph.vecOutputInfo.emplace_back(
                GraphRuntime::t_blob_info{
                    .memoryRegionID=oaRegion,
                    .channel=numOutputChannels,
                    .height= numOutputHeight,
                    .width= numOutputWidth,
                    .stripStrideSeenBySource = oaStripStrideSeenBySource,
                    .numFracBits=FRAC_WIDTH,
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
        #if defined(PROFILE)
            accelerator.inference(true);
        #else
             accelerator.inference(false);
        #endif
    }


    //Retrieve the outputs
    std::cout <<stepCount++<<". Decode and retrieve the output."<<std::endl;
    std::vector<float> outputFloatVector = accelerator.extractOutputBlob(0);

    std::cout <<stepCount++<<". Runtime stats."<<std::endl;
    std::cout <<accelerator.reportRuntime();


    //Decompress the output, and check against the input if necessary
    {
        if (_flagPerformanceTest == false)
        {
            std::cout <<stepCount++<<". Check the output"<<std::endl;

            for (unsigned int iGroup=0; iGroup<numGroupCurrentLayer; iGroup++)
            {
                for (unsigned int iRow=0; iRow<numOutputHeight; iRow++)
                {
                    for (unsigned int iCol=0; iCol<numOutputWidth; iCol++)
                    {
                        for (unsigned int iCh=0; iCh<numOutputChannelPerGroup; iCh++)
                        {
                            //Obtain the actual output
                            unsigned int outputCoord =
                                    (iRow*numOutputWidth + iCol) * numOutputChannels
                                    + iGroup * numOutputChannelPerGroup
                                    + iCh;
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
                                case AVG_POOL: {
                                    unsigned char iChGlobal = iGroup*numOutputChannelPerGroup + iCh;
                                    expectedFloat = iChGlobal*unitFloat;
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
    float invocationOverhead = accelerator.getInvocationOverhead();
    std::cout <<"Invocation overhead is "<<invocationOverhead<<" us"<<std::endl;
} //launch
