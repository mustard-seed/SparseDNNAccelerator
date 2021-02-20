#include "graph_factory.hpp"
#include "gtest/gtest.h"

#include "tile.hpp"
#include "model_container.hpp"

#include <iostream>
#include <vector>
#include <map>
#include <memory>

#ifndef C5SOC
//#define EMULATE
#endif
#define INFERENCE_REPEAT 100
#define WARMUP 100
#define CHECKOUTPUT
//#define PROFILE

using namespace GraphRuntime;
using namespace  std;
class testFixture : public ::testing::Test {
protected:
    std::string aocxBinaryFile;
    GraphRuntime::AcceleratorWrapper accelerator;

    void SetUp() override;

    typedef struct {
        int inputChannel;
        int outputChannel;
        int inputWidth;
        int inputHeight;
        int borderPadding;
        int kernel;
        int stride;
        int transConvPadding;
        int numGroups;
        int numSourceGroups;
        bool relu;
        int inputFracBits;
        int weightFracBits;
        int outputFracBits;
        float sparsity;
    } t_conv_spec;

    int launch(
                t_conv_spec _convInfo,
                int sizeOutputFullTileHeight,
                int sizeOutputFullTileWidthPerCol
            );
};

TEST_F(testFixture, 128x128x3x3x56x56)
{
    testFixture::t_conv_spec spec {
        .inputChannel=128,
        .outputChannel=128,
        .inputWidth=56,
        .inputHeight=56,
        .borderPadding=1,
        .kernel=3,
        .stride=1,
        .transConvPadding=0,
        .numGroups=1,
        .numSourceGroups=1,
        .relu=true,
        .inputFracBits=2,
        .weightFracBits=2,
        .outputFracBits=2,
        .sparsity=0.0
    };

    typedef struct {
        int sizeOutputFullTileHeight;
        int sizeOutputFullTileWidthPerCol;
    } t_tile;

    std::vector<t_tile> vecTileConfigs {
        {.sizeOutputFullTileHeight = 28, .sizeOutputFullTileWidthPerCol = 4},
        {.sizeOutputFullTileHeight = 14, .sizeOutputFullTileWidthPerCol = 4},
        {.sizeOutputFullTileHeight = 7, .sizeOutputFullTileWidthPerCol = 4},
        {.sizeOutputFullTileHeight = 1, .sizeOutputFullTileWidthPerCol = 4},
        {.sizeOutputFullTileHeight = 28, .sizeOutputFullTileWidthPerCol = 2},
        {.sizeOutputFullTileHeight = 14, .sizeOutputFullTileWidthPerCol = 2},
        {.sizeOutputFullTileHeight = 7, .sizeOutputFullTileWidthPerCol = 2},
        {.sizeOutputFullTileHeight = 1, .sizeOutputFullTileWidthPerCol = 2},
        {.sizeOutputFullTileHeight = 28, .sizeOutputFullTileWidthPerCol = 1},
        {.sizeOutputFullTileHeight = 14, .sizeOutputFullTileWidthPerCol = 1},
        {.sizeOutputFullTileHeight = 7, .sizeOutputFullTileWidthPerCol = 1},
        {.sizeOutputFullTileHeight = 1, .sizeOutputFullTileWidthPerCol = 1},
    };

    for (auto &config : vecTileConfigs) {
        launch(spec, config.sizeOutputFullTileHeight, config.sizeOutputFullTileWidthPerCol);
    }
}

void testFixture::SetUp()
{
#ifdef C5SOC
    aocxBinaryFile = "device_utils.aocx";
#else
#if defined(EMULATE)
    aocxBinaryFile = "c5_mac8bitx4_c_model.aocx";
#else
    aocxBinaryFile = "sparse_pe_system.aocx";
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
            .numScalarInCluster=CLUSTER_SIZE
        };
   accelerator = GraphRuntime::AcceleratorWrapper(aocxBinaryFile, platformName, acceleratorInfo, 0);
}

int testFixture::launch(t_conv_spec _convInfo,
                         int sizeOutputFullTileHeight,
                         int sizeOutputFullTileWidthPerCol)
{
    int stepCount = 1;
    const int convInputLocation = 0, convOutputLocation = 1;

    std::cout <<"============================================"<<std::endl;
    std::cout<<"Conv IC, OC, IW, IH, K, Stride, TransPading, Border Pading, sparsity = "<<std::endl;
    std::cout<<_convInfo.inputChannel<<" "<<_convInfo.outputChannel<<" "<<_convInfo.inputWidth<<" "<<_convInfo.inputHeight<<" ";
    std::cout<<_convInfo.kernel<<" "<<_convInfo.stride<<" "<<_convInfo.transConvPadding<<" "<<_convInfo.borderPadding<<" "<<_convInfo.sparsity<<std::endl;
    std::cout<<"sizeOutputFullTileHeight, sizeOutputFullTileWidthPerCol = "<<std::endl;
    std::cout <<sizeOutputFullTileHeight<<" "<<sizeOutputFullTileWidthPerCol<<std::endl;
    //Load the trace file and the parameter file
    std::cout <<"Step "<<stepCount++<<": Construct the graph."<<std::endl;
    GraphRuntime::GraphFactory graph;
    auto pInputBlob = make_shared<QuantLayer>(QuantLayer());
    auto pConv = make_shared<ConvLayer>(ConvLayer());
    auto pOutputBlob = make_shared<DeQuantLayer>(DeQuantLayer());

    int convEffectiveInputHeight = _convInfo.inputHeight*(1+_convInfo.transConvPadding) + 2*_convInfo.borderPadding;
    int convOutputHeight = (convEffectiveInputHeight - _convInfo.kernel) / _convInfo.stride + 1;
    int convEffectiveInputWidth = _convInfo.inputWidth*(1+_convInfo.transConvPadding) + 2*_convInfo.borderPadding;
    int convOutputWidth = (convEffectiveInputWidth - _convInfo.kernel) / _convInfo.stride + 1;

    std:vector<float> vecInput(_convInfo.inputHeight * _convInfo.inputWidth * _convInfo.inputChannel, 1.3f);
    std::vector<float> vecWeights;
    vecWeights.resize(_convInfo.inputChannel * _convInfo.outputChannel * _convInfo.kernel * _convInfo.kernel);
    std::vector<float> vecBiases(_convInfo.outputChannel, 1.2f);

    //Set the QuantLayer
    pInputBlob->setLayerID(0);
    pInputBlob->setInputHeights({_convInfo.inputHeight});
    pInputBlob->setInputWidths({_convInfo.inputWidth});
    pInputBlob->setInputChannels({_convInfo.inputChannel});
    pInputBlob->setCurrentNumberGroups(1);
    pInputBlob->setInputFracBits({_convInfo.inputFracBits});
    pInputBlob->setOutputMemoryLocation(convInputLocation);
    pInputBlob->setInputMemoryLocations({convInputLocation});
    pInputBlob->setInputGroupsSeenBySource({1});
    pInputBlob->setOutputFracBits(_convInfo.inputFracBits);
    pInputBlob->setOutputHeight(_convInfo.inputHeight);
    pInputBlob->setOutputWidth(_convInfo.inputWidth);
    pInputBlob->setOutputChannel(_convInfo.inputChannel);
    graph.addLayer(pInputBlob);

    //Set the ConvLayer
    pConv->setLayerID(1);
    pConv->setInputHeights({_convInfo.inputHeight});
    pConv->setInputWidths({_convInfo.inputWidth});
    pConv->setInputChannels({_convInfo.inputChannel});
    pConv->setInputFracBits({_convInfo.inputFracBits});
    pConv->setInputMemoryLocations({convInputLocation});
    pConv->setOutputMemoryLocation(convOutputLocation);
    pConv->setInputGroupsSeenBySource({1});
    pConv->setOutputFracBits(_convInfo.outputFracBits);
    pConv->setOutputHeight(convOutputHeight);
    pConv->setOutputWidth(convOutputWidth);
    pConv->setOutputChannel(_convInfo.outputChannel);
    pConv->setBiasFlag(true);
    pConv->setOutputReluFlag(true);
    pConv->setCurrentNumberGroups(_convInfo.numGroups);
    pConv->setKernelSize(_convInfo.kernel);
    pConv->setKernelStride(_convInfo.stride);
    pConv->setInputBorderPadding(_convInfo.borderPadding);
    pConv->setTransConvPadding(_convInfo.transConvPadding);
    pConv->setWeightFracBits(_convInfo.weightFracBits);
    pConv->setWeightPruneClusterSize(CLUSTER_SIZE);
    pConv->setWeightPruneRangeSizeInCluster(PRUNE_RANGE_IN_CLUSTER);
    pConv->setWeightSparsity(_convInfo.sparsity);
    pConv->loadWeights(vecWeights.data());
    pConv->loadBiases(vecBiases.data());
    t_graph_output_tile_info tileConfig = deriveConvOutputTileShape(
                    convOutputHeight,
                    convOutputWidth,
                    sizeOutputFullTileHeight,
                    sizeOutputFullTileWidthPerCol,
                    true
                );
    if (graph.addLayer(pConv, &tileConfig)) {
        return -1;
    }

    //Set the output blob
    pOutputBlob->setLayerID(2);
    pOutputBlob->setInputHeights({convOutputHeight});
    pOutputBlob->setInputWidths({convOutputWidth});
    pOutputBlob->setInputChannels({_convInfo.outputChannel});
    pOutputBlob->setInputFracBits({_convInfo.outputFracBits});
    pOutputBlob->setOutputMemoryLocation(convOutputLocation);
    pOutputBlob->setInputMemoryLocations({convOutputLocation});
    pOutputBlob->setInputGroupsSeenBySource({_convInfo.numGroups});
    pOutputBlob->setCurrentNumberGroups(_convInfo.numGroups);
    pOutputBlob->setOutputFracBits(_convInfo.outputFracBits);
    pOutputBlob->setOutputHeight(convOutputHeight);
    pOutputBlob->setOutputWidth(convOutputWidth);
    pOutputBlob->setOutputChannel(_convInfo.outputChannel);
    graph.addLayer(pOutputBlob);

    std::cout <<"Step "<<stepCount++<<": Generate the execution graph."<<std::endl;
    auto pGraph = std::move(graph.generateGraph());

    std::cout <<"Step "<<stepCount++<<": Load the graph to the accelerator"<<std::endl;
    accelerator.resetGraph();
    accelerator.loadGraph(*(pGraph.get()));

    std::cout <<"Step "<<stepCount++<<": Load the inputs and send them to the accelerator."<<std::endl;
    {
       accelerator.prepareInputBlob(vecInput, 0);
    }

    std::cout <<"Step "<<stepCount++<<": Perform inference."<<std::endl;
#if defined(PROFILE)
    bool profile = true;
#else
    bool profile = false;
#endif
    for (int i=0; i<(INFERENCE_REPEAT+WARMUP); i++)
    {
        bool includeInCount = (i<WARMUP) ? false : true;
        accelerator.inference(includeInCount, profile);
    }

    std::cout <<"Step "<<stepCount++<<": Performance counts"<<std::endl;
    std::cout <<accelerator.reportRuntime();
}
