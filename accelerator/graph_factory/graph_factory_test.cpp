#include "graph_factory.hpp"
#include "gtest/gtest.h"

#include <iostream>
#include <vector>
#include <map>

//#define PLAY
//#define EMULATE
#define INFERENCE_REPEAT 10

class testFixture : public ::testing::Test {
protected:
    std::string aocxBinaryFile;
    GraphRuntime::AcceleratorWrapper accelerator;

    void SetUp() override;

    void launch(std::string _traceFileName,
                std::string _parameterFileName,
                std::string _inoutFileName,
                std::map<std::string, std::string> _traceName2BlobName);
};

TEST_F(testFixture, miniResNet)
{
    /*
     *Test trace: https://drive.google.com/drive/folders/1k9m5-DMOAJaM3-psX6jmItSoer11TBqf?usp=sharing
    */
    std::string traceFileName = "testTrace_trace.yaml";
    std::string traceParameterFile = "testTrace_parameters.yaml";
    std::string inoutFile = "testTrace_inout.yaml";
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quantstub_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequantstub_0", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName);
}

void testFixture::SetUp()
{
#ifdef C5SOC
    aocxBinaryFile = "device_utils.aocx";
#else
    aocxBinaryFile = "sparse_pe_system.aocx";
#if defined(EMULATE)
    aocxBinaryFile = "smallBuffer.aocx";
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
   accelerator = GraphRuntime::AcceleratorWrapper(aocxBinaryFile, platformName, acceleratorInfo, 0);
}

void testFixture::launch(std::string _traceFileName,
                         std::string _parameterFileName,
                         std::string _inoutFileName,
                         std::map<std::string, std::string> _traceName2BlobName)
{
    int stepCount = 1;

    //Load the trace file and the parameter file
    std::cout <<"Step "<<stepCount++<<": Loading trace file and parameter file."<<std::endl;
    std::cout <<_traceFileName<<" "<<_parameterFileName<<std::endl;
    GraphRuntime::GraphFactory graphFactory(_traceFileName, _parameterFileName);

    std::cout <<"Step "<<stepCount++<<": Generate the execution graph."<<std::endl;
    auto pGraph = std::move(graphFactory.generateGraph());

    std::cout <<"Step "<<stepCount++<<": Load the graph to the accelerator"<<std::endl;
    accelerator.resetGraph();
    accelerator.loadGraph(*(pGraph.get()));

    std::cout <<"Step "<<stepCount++<<": Load the inputs and send them to the accelerator."<<std::endl;
    {
       YAML::Node rawBlobs = YAML::LoadFile(_inoutFileName);
       auto vecInputInfo = accelerator.getInputBlobsInfo();
       int blobID = 0;
       for (const auto& inputInfo: vecInputInfo)
       {
           std::string layerName = inputInfo.blobName;
           YAML::Node blob = rawBlobs[layerName];
           int size = inputInfo.group
                   * inputInfo.channelPerGroup
                   * inputInfo.height
                   * inputInfo.width;
           std::vector<float> inputReordered(size, 0.0f);
           int iter=0;
           for (int g=0; g<inputInfo.group; g++)
           {
               for (int h=0; h<inputInfo.height; h++)
               {
                   for (int w=0; h<inputInfo.width; w++)
                   {
                       for (int c=0; c<inputInfo.channelPerGroup; c++)
                       {
                           int rawIter = (c + g*inputInfo.channelPerGroup) * inputInfo.height * inputInfo.width
                                   + h * inputInfo.width + w;
                           inputReordered.at(iter++) = blob[rawIter].as<float>();
                       }
                   }
               }
           }
           accelerator.prepareInputBlob(inputReordered, blobID);
           blobID++;
       }
    }

    std::cout <<"Step "<<stepCount++<<": Perform inference."<<std::endl;
    for (int i=0; i<INFERENCE_REPEAT; i++)
    {
        accelerator.inference();
    }

    std::cout <<"Step "<<stepCount++<<": Extract output and perform checks"<<std::endl;
    {
       YAML::Node rawBlobs = YAML::LoadFile(_inoutFileName);
       auto vecBlobInfo = accelerator.getOutputBlobsInfo();
       int blobID = 0;
       for (const auto& blobInfo: vecBlobInfo)
       {
           std::string layerName = _traceName2BlobName[blobInfo.blobName];
           YAML::Node blob = rawBlobs[layerName];
           std::vector<float> actualResult = accelerator.extractOutputBlob(blobID);
           int iter=0;
           for (int g=0; g<blobInfo.group; g++)
           {
               for (int h=0; h<blobInfo.height; h++)
               {
                   for (int w=0; h<blobInfo.width; w++)
                   {
                       for (int c=0; c<blobInfo.channelPerGroup; c++)
                       {
                           int rawIter = (c + g*blobInfo.channelPerGroup) * blobInfo.height * blobInfo.width
                                   + h * blobInfo.width + w;
                           float expected = blob[rawIter].as<float>();
                           float actual = actualResult.at(iter++);
                           EXPECT_FLOAT_EQ(actual, expected)
                                   <<"Inference output disagreement at [tensor, group, channel, height, col]: ["
                                   <<blobID<<" "<<g<<" "<<c<<" "<<h<<" "<<w<<std::endl
                                   <<"Expected: "<<expected<<std::endl
                                   <<"Actual: "<<actual<<std::endl;
                       }
                   }
               }
           }
           blobID++;
       }
    }
}
