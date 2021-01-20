#include "graph_factory.hpp"
#include "gtest/gtest.h"

#include <iostream>
#include <vector>
#include <map>

//Prirority of the MACRO flags:
//PLAY > VALIDATE > RESNET56
//#define PLAY
//#define VALIDATE
//#define RESNET50_CONV12
//#define RESNET56
#define RESNET50
#ifndef C5SOC
 //#define EMULATE
#endif
#define INFERENCE_REPEAT 40
#define CHECKOUTPUT
//#define PROFILE

class testFixture : public ::testing::Test {
protected:
    std::string aocxBinaryFile;
    GraphRuntime::AcceleratorWrapper accelerator;
    std::string testPrefix = "/home/jamesliu/thesis/SparseDNNAccelerator/accelerator/test0/";

    void SetUp() override;

    void launch(std::string _traceFileName,
                std::string _parameterFileName,
                std::string _inoutFileName,
                std::map<std::string, std::string> _traceName2BlobName,
                bool _scatterInput=false);
};
#if defined(PLAY) //focus on one test
TEST_F(testFixture, testTrace)
{
    /*
     *Test trace: https://drive.google.com/drive/folders/1HZ5jjIw-71bSwvaNlaOGVJSW_FvTdY5D?usp=sharing
    */
    std::string traceFileName = "testTrace_trace.yaml";
    std::string traceParameterFile = "testTrace_parameters.npz";
    std::string inoutFile = "testTrace_inout.yaml";
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_15", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName);
}
//TEST_F(testFixture, tinyNet)
//{
//    /*
//     *Test trace: https://drive.google.com/drive/folders/1HZ5jjIw-71bSwvaNlaOGVJSW_FvTdY5D?usp=sharing
//    */
//    std::string traceFileName = "tinyTrace_trace.yaml";
//    std::string traceParameterFile = "tinyTrace_parameters.npz";
//    std::string inoutFile = "tinyTrace_inout.yaml";
//    std::map<std::string, std::string> traceName2BlobName;
//    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
//    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_6", "output"));
//    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName);
//}
//TEST_F(testFixture, pointConv)
//{
//    /*
//     *Test trace: https://drive.google.com/drive/folders/1HZ5jjIw-71bSwvaNlaOGVJSW_FvTdY5D?usp=sharing
//    */
//    std::string traceFileName = "pointconvTrace_trace.yaml";
//    std::string traceParameterFile = "pointconvTrace_parameters.npz";
//    std::string inoutFile = "pointconvTrace_inout.yaml";
//    std::map<std::string, std::string> traceName2BlobName;
//    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
//    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_2", "output"));
//    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName);
//}
//TEST_F(testFixture, conv)
//{
//    /*
//     *Test trace: https://drive.google.com/drive/folders/1HZ5jjIw-71bSwvaNlaOGVJSW_FvTdY5D?usp=sharing
//    */
//    std::string traceFileName = "convTrace_trace.yaml";
//    std::string traceParameterFile = "convTrace_parameters.npz";
//    std::string inoutFile = "convTrace_inout.yaml";
//    std::map<std::string, std::string> traceName2BlobName;
//    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
//    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_2", "output"));
//    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName);
//}
#endif
#if defined(VALIDATE)
TEST_F(testFixture, conv)
{
    /*
     *Test trace: https://drive.google.com/drive/folders/1HZ5jjIw-71bSwvaNlaOGVJSW_FvTdY5D?usp=sharing
    */
    std::string traceFileName = "convTrace_trace.yaml";
    std::string traceParameterFile = "convTrace_parameters.npz";
    std::string inoutFile = "convTrace_inout.yaml";
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_2", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName);
}

TEST_F(testFixture, pointConv)
{
    /*
     *Test trace: https://drive.google.com/drive/folders/1HZ5jjIw-71bSwvaNlaOGVJSW_FvTdY5D?usp=sharing
    */
    std::string traceFileName = "pointconvTrace_trace.yaml";
    std::string traceParameterFile = "pointconvTrace_parameters.npz";
    std::string inoutFile = "pointconvTrace_inout.yaml";
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_2", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName);
}

TEST_F(testFixture, tinyNet)
{
    /*
     *Test trace: https://drive.google.com/drive/folders/1HZ5jjIw-71bSwvaNlaOGVJSW_FvTdY5D?usp=sharing
    */
    std::string traceFileName = "tinyTrace_trace.yaml";
    std::string traceParameterFile = "tinyTrace_parameters.npz";
    std::string inoutFile = "tinyTrace_inout.yaml";
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_6", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName);
}

TEST_F(testFixture, testTrace)
{
    /*
     *Test trace: https://drive.google.com/drive/folders/1HZ5jjIw-71bSwvaNlaOGVJSW_FvTdY5D?usp=sharing
    */
    std::string traceFileName = "testTrace_trace.yaml";
    std::string traceParameterFile = "testTrace_parameters.npz";
    std::string inoutFile = "testTrace_inout.yaml";
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_15", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName);
}

#endif
#if defined(RESNET56)
TEST_F(testFixture, resnet56_cifar10)
{
    /*
     *Test trace: https://drive.google.com/drive/folders/1rp9Mnggpa9UpGUBAwuhS-BlgAVVlwqgj?usp=sharing
    */
    std::string traceFileName = "resnet56_cifar10_trace.yaml";
    std::string traceParameterFile = "resnet56_cifar10_parameters.npz";
    std::string inoutFile = "resnet56_cifar10_inout.yaml";
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_87", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName);
}
#endif
#if defined(RESNET50)
TEST_F(testFixture, resnet50_imagenet1k)
{

    std::string traceFileName = "resnet50_imagenet_trace.yaml";
    std::string traceParameterFile = "resnet50_imagenet_parameters.npz";
    std::string inoutFile = "resnet50_imagenet_inout.yaml";
    bool scatterInput = true;
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_73", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput);
}
#endif
#if defined(RESNET50_CONV12)
TEST_F(testFixture, resnet50_conv12)
{

    std::string traceFileName = "resnet50_conv12_trace.yaml";
    std::string traceParameterFile = "resnet50_conv12_parameters.npz";
    std::string inoutFile = "resnet50_conv12_inout.yaml";
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_2", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName);
}
#endif

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

void testFixture::launch(std::string _traceFileName,
                         std::string _parameterFileName,
                         std::string _inoutFileName,
                         std::map<std::string, std::string> _traceName2BlobName,
                         bool _scatterInput)
{
    int stepCount = 1;

    //Load the trace file and the parameter file
    std::cout <<"Step "<<stepCount++<<": Loading trace file and parameter file."<<std::endl;
    std::cout <<testPrefix+_traceFileName<<" "<<testPrefix+_parameterFileName<<std::endl;
    GraphRuntime::GraphFactory graphFactory(testPrefix+_traceFileName, testPrefix+_parameterFileName, _scatterInput);

    std::cout <<"Step "<<stepCount++<<": Generate the execution graph."<<std::endl;
    auto pGraph = std::move(graphFactory.generateGraph());

    std::cout <<"Step "<<stepCount++<<": Load the graph to the accelerator"<<std::endl;
    accelerator.resetGraph();
    accelerator.loadGraph(*(pGraph.get()));

    std::cout <<"Step "<<stepCount++<<": Load the inputs and send them to the accelerator."<<std::endl;
    {
       YAML::Node rawBlobs = YAML::LoadFile(testPrefix+_inoutFileName);
       auto vecInputInfo = accelerator.getInputBlobsInfo();
       int blobID = 0;
       for (const auto& inputInfo: vecInputInfo)
       {
           std::string layerName = _traceName2BlobName[inputInfo.blobName];
           YAML::Node blob = rawBlobs[layerName];
           int size = inputInfo.channel
                   * inputInfo.height
                   * inputInfo.width;
           std::vector<float> inputReordered(size, 0.0f);
           int iter=0;
           for (int h=0; h<inputInfo.height; h++)
           {
               for (int w=0; w<inputInfo.width; w++)
               {
                   for (int c=0; c<inputInfo.channel; c++)
                   {
                       int rawIter = c * inputInfo.height * inputInfo.width
                               + h * inputInfo.width + w;
                       inputReordered.at(iter++) = blob[rawIter].as<float>();
                   }
               }
           }
           accelerator.prepareInputBlob(inputReordered, blobID);
           blobID++;
       }
    }

    std::cout <<"Step "<<stepCount++<<": Perform inference."<<std::endl;
#if defined(PROFILE)
    bool profile = true;
#else
    bool profile = false;
#endif
    for (int i=0; i<INFERENCE_REPEAT; i++)
    {
        accelerator.inference(profile);
    }

    std::cout <<"Step "<<stepCount++<<": Performance counts"<<std::endl;
    std::cout <<accelerator.reportRuntime();

    std::size_t dotPos = _traceFileName.find(".");
    std::string csvFileName = _traceFileName.substr(0, dotPos) + ".csv";
    accelerator.dumpRuntimeToCSV(csvFileName);

#if defined(CHECKOUTPUT)
    std::cout <<"Step "<<stepCount++<<": Extract output and perform checks"<<std::endl;
    {
       YAML::Node rawBlobs = YAML::LoadFile(testPrefix+_inoutFileName);
       auto vecBlobInfo = accelerator.getOutputBlobsInfo();
       int blobID = 0;
       for (const auto& blobInfo: vecBlobInfo)
       {
           std::string layerName = _traceName2BlobName[blobInfo.blobName];
           YAML::Node blob = rawBlobs[layerName];
           std::vector<float> actualResult = accelerator.extractOutputBlob(blobID);
           int iter=0;
           signed char numFracBitsDifference = blobInfo.numFracBits - 1;
//           float tolerance = (numFracBitsDifference >= 0) ?
//                      1.0 / (1 << numFracBitsDifference) : 1 << (-1 * numFracBitsDifference);
           float tolerance = 1e-3;
           for (int h=0; h<blobInfo.height; h++)
           {
               for (int w=0; w<blobInfo.width; w++)
               {
                   for (int c=0; c<blobInfo.channel; c++)
                   {
                       int rawIter = c * blobInfo.height * blobInfo.width
                               + h * blobInfo.width + w;
                       float expected = blob[rawIter].as<float>();
                       float actual = actualResult.at(iter++);
                       //The computation is like adding two signed numbers with numFracBits
                       //hence the difference's number of frac bits is numFracBits - 1
                       EXPECT_TRUE(std::abs(actual -expected) <= tolerance)
                               <<"Inference output disagreement at [tensor, channel, height, col]: ["
                               <<blobID<<" "<<c<<" "<<h<<" "<<w<<"]"<<std::endl
                               <<"Expected: "<<expected<<std::endl
                               <<"Actual: "<<actual<<std::endl;
                   }
               }
           }
           blobID++;
       }
    }
#endif //CHECKOUTPUT
}
