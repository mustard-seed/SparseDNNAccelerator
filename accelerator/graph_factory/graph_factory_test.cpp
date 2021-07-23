#include "graph_factory.hpp"
#include "gtest/gtest.h"

#include <iostream>
#include <vector>
#include <map>
#include <numeric> //iota
#include <algorithm> //stable_sort

//Prirority of the MACRO flags:
//PLAY > VALIDATE > RESNET56
//#define PLAY
#define RESNET50_IMAGENET_C2R4
//#define RESNET50_IMAGENET_C1R4
//#define RESNET50_IMAGENET_C4R4
//#define RESNET50_DUMMY_C2R4
//#define RESNET50_DUMMY_C1R4
//#define RESNET50_DUMMY_C4R4
//#define VGG16_IMAGENET_C2R4
//#define VGG16_IMAGENET_C1R4
//#define VGG16_IMAGENET_C4R4
//#define VGG16_DUMMY_C2R4
//#define VGG16_DUMMY_C1R4
//#define VGG16_DUMMY_C4R4
#ifndef C5SOC
//#define EMULATE
#endif
//#define INFERENCE_REPEAT 50
//#define WARMUP 50
#define INFERENCE_REPEAT 500
#define WARMUP 50
//Define checkoutput 0 means compare the output blob with the reference
//1 means simply printing the output
//#define CHECKOUTPUT 0
//#define PROFILE

class testFixture : public ::testing::Test {
protected:
    std::string aocxBinaryFile;
    GraphRuntime::AcceleratorWrapper accelerator;
    //TODO: change the path in anounymous submission
    //std::string testPrefix = "/home/jamesliu/thesis/SparseDNNAccelerator/accelerator/test0/FPL_traces/";
    std::string testPrefix = "/home/jamesliu/thesis/SparseDNNAccelerator/accelerator/test0/gcp_traces/traces/";

    void SetUp() override;

    void launch(std::string _traceFileName,
                std::string _parameterFileName,
                std::string _inoutFileName,
                std::map<std::string, std::string> _traceName2BlobName,
                bool _scatterInput=false,
                int _targetLayerID=-1);
};

typedef struct {
    int idx;
    float val;
} t_topK_elem;

std::vector<t_topK_elem> getTopK(std::vector<float> _vec, int k=10);
#if defined(PLAY) //focus on one test
//TEST_F(testFixture, resnet50_imagenet_c2r4p75)
//{

//    std::string traceFileName = "resnet56_cifar_trace.yaml";
//    std::string traceParameterFile = "resnet56_cifar_parameters.npz";
////    std::string inoutFile = "resnet50_imagenet_pretrained_quantize_bias_inout_img00000008_1.yaml";
//     std::string inoutFile = "resnet56_cifar_inout.yaml";
//    bool scatterInput = true;
//    std::map<std::string, std::string> traceName2BlobName;
//    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
//    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_87", "output"));
//    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput);
//}
TEST_F(testFixture, vgg16_imagenet_c2r4p75)
{
    std::string traceFileName = "vgg16_imagenet_c2r4p75_trace.yaml";
    std::string traceParameterFile = "vgg16_imagenet_c2r4p75_parameters.npz";
    std::string inoutFile = "vgg16_imagenet_c2r4p75_inout_img00000008.yaml";
    bool scatterInput = true;
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_20", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput, 19);
}
#endif

#if defined(RESNET50_IMAGENET_C2R4)
// TEST_F(testFixture, resnet50_imagenet_dense_c2r4)
// {

//     std::string traceFileName = "resnet50_imagenet_dense_c2r4_trace.yaml";
//     std::string traceParameterFile = "resnet50_imagenet_dense_c2r4_parameters.npz";
//     std::string inoutFile = "resnet50_imagenet_dense_c2r4_inout_img00000008.yaml";
//     bool scatterInput = true;
//     std::map<std::string, std::string> traceName2BlobName;
//     traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
//     traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_73", "output"));
//     launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput);
// }

// TEST_F(testFixture, resnet50_imagenet_c2r4p25)
// {

//     std::string traceFileName = "resnet50_imagenet_c2r4p25_trace.yaml";
//     std::string traceParameterFile = "resnet50_imagenet_c2r4p25_parameters.npz";
//     std::string inoutFile = "resnet50_imagenet_c2r4p25_inout_img00000008.yaml";
//     bool scatterInput = true;
//     std::map<std::string, std::string> traceName2BlobName;
//     traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
//     traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_73", "output"));
//     launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput);
// }

// TEST_F(testFixture, resnet50_imagenet_c2r4p50)
// {

//     std::string traceFileName = "resnet50_imagenet_c2r4p50_trace.yaml";
//     std::string traceParameterFile = "resnet50_imagenet_c2r4p50_parameters.npz";
//     std::string inoutFile = "resnet50_imagenet_c2r4p50_inout_img00000008.yaml";
//     bool scatterInput = true;
//     std::map<std::string, std::string> traceName2BlobName;
//     traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
//     traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_73", "output"));
//     launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput);
// }

TEST_F(testFixture, resnet50_imagenet_c2r4p75)
{

    std::string traceFileName = "resnet50_imagenet_c2r4p75_trace.yaml";
    std::string traceParameterFile = "resnet50_imagenet_c2r4p75_parameters.npz";
    std::string inoutFile = "resnet50_imagenet_c2r4p75_inout_img00000008.yaml";
    bool scatterInput = true;
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_73", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput);
}
#endif
#if defined(RESNET50_IMAGENET_C1R4)
// TEST_F(testFixture, resnet50_imagenet_dense_c1r4)
// {

//     std::string traceFileName = "resnet50_imagenet_dense_c1r4_trace.yaml";
//     std::string traceParameterFile = "resnet50_imagenet_dense_c1r4_parameters.npz";
//     std::string inoutFile = "resnet50_imagenet_dense_c1r4_inout_img00000008.yaml";
//     bool scatterInput = true;
//     std::map<std::string, std::string> traceName2BlobName;
//     traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
//     traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_73", "output"));
//     launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput);
// }
// TEST_F(testFixture, resnet50_imagenet_c1r4p25)
// {

//     std::string traceFileName = "resnet50_imagenet_c1r4p25_trace.yaml";
//     std::string traceParameterFile = "resnet50_imagenet_c1r4p25_parameters.npz";
//     std::string inoutFile = "resnet50_imagenet_c1r4p25_inout_img00000008.yaml";
//     bool scatterInput = true;
//     std::map<std::string, std::string> traceName2BlobName;
//     traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
//     traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_73", "output"));
//     launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput);
// }
// TEST_F(testFixture, resnet50_imagenet_c1r4p50)
// {

//     std::string traceFileName = "resnet50_imagenet_c1r4p50_trace.yaml";
//     std::string traceParameterFile = "resnet50_imagenet_c1r4p50_parameters.npz";
//     std::string inoutFile = "resnet50_imagenet_c1r4p50_inout_img00000008.yaml";
//     bool scatterInput = true;
//     std::map<std::string, std::string> traceName2BlobName;
//     traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
//     traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_73", "output"));
//     launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput);
// }
TEST_F(testFixture, resnet50_imagenet_c1r4p75)
{

    std::string traceFileName = "resnet50_imagenet_c1r4p75_trace.yaml";
    std::string traceParameterFile = "resnet50_imagenet_c1r4p75_parameters.npz";
    std::string inoutFile = "resnet50_imagenet_c1r4p75_inout_img00000008.yaml";
    bool scatterInput = true;
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_73", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput);
}
#endif
#if defined(RESNET50_IMAGENET_C4R4)
TEST_F(testFixture, resnet50_imagenet_c4r4p75)
{

    std::string traceFileName = "resnet50_imagenet_c4r4p75_trace.yaml";
    std::string traceParameterFile = "resnet50_imagenet_c4r4p75_parameters.npz";
    std::string inoutFile = "resnet50_imagenet_c4r4p75_inout_img00000008.yaml";
    bool scatterInput = true;
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_73", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput);
}
#endif
#if defined(RESNET50_DUMMY_C2R4)
TEST_F(testFixture, resnet50_dummy_c2r4p75)
{

    std::string traceFileName = "resnet50_dummy_c2r4p75_trace.yaml";
    std::string traceParameterFile = "resnet50_dummy_c2r4p75_parameters.npz";
    std::string inoutFile = "resnet50_dummy_c2r4p75_inout.yaml";
    bool scatterInput = true;
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_73", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput);
}
#endif
#if defined(RESNET50_DUMMY_C1R4)
TEST_F(testFixture, resnet50_dummy_c2r4p75)
{

    std::string traceFileName = "resnet50_dummy_c1r4p75_trace.yaml";
    std::string traceParameterFile = "resnet50_dummy_c1r4p75_parameters.npz";
    std::string inoutFile = "resnet50_dummy_c1r4p75_inout.yaml";
    bool scatterInput = true;
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_73", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput);
}
#endif
#if defined(RESNET50_DUMMY_C4R4)
TEST_F(testFixture, resnet50_dummy_c4r4p75)
{

    std::string traceFileName = "resnet50_dummy_c4r4p75_trace.yaml";
    std::string traceParameterFile = "resnet50_dummy_c4r4p75_parameters.npz";
    std::string inoutFile = "resnet50_dummy_c4r4p75_inout.yaml";
    bool scatterInput = true;
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_73", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput);
}
#endif
#if defined(VGG16_DUMMY_C2R4)
TEST_F(testFixture, vgg16_dummy_c2r4p75)
{
    std::string traceFileName = "vgg16_dummy_c2r4p75_trace.yaml";
    std::string traceParameterFile = "vgg16_dummy_c2r4p75_parameters.npz";
    std::string inoutFile = "vgg16_dummy_c2r4p75_inout.yaml";
    bool scatterInput = true;
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_22", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput, -1);
}
#endif
#if defined(VGG16_DUMMY_C1R4)
TEST_F(testFixture, vgg16_dummy_c1r4p75)
{
    std::string traceFileName = "vgg16_dummy_c1r4p75_trace.yaml";
    std::string traceParameterFile = "vgg16_dummy_c1r4p75_parameters.npz";
    std::string inoutFile = "vgg16_dummy_c1r4p75_inout.yaml";
    bool scatterInput = true;
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_22", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput, -1);
}
#endif
#if defined(VGG16_DUMMY_C4R4)
TEST_F(testFixture, vgg16_dummy_c4r4p75)
{
    std::string traceFileName = "vgg16_dummy_c4r4p75_trace.yaml";
    std::string traceParameterFile = "vgg16_dummy_c4r4p75_parameters.npz";
    std::string inoutFile = "vgg16_dummy_c4r4p75_inout.yaml";
    bool scatterInput = true;
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_22", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput, -1);
}
#endif

#if defined(VGG16_IMAGENET_C2R4)
// TEST_F(testFixture, vgg16_imagenet_dense_c2r4)
// {
//     std::string traceFileName = "vgg16_imagenet_dense_c2r4_trace.yaml";
//     std::string traceParameterFile = "vgg16_imagenet_dense_c2r4_parameters.npz";
//     std::string inoutFile = "vgg16_imagenet_dense_c2r4_inout_img00000008.yaml";
//     bool scatterInput = true;
//     std::map<std::string, std::string> traceName2BlobName;
//     traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
//     traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_22", "output"));
//     launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput, -1);
// }

// TEST_F(testFixture, vgg16_imagenet_c2r4p25)
// {
//     std::string traceFileName = "vgg16_imagenet_c2r4p25_trace.yaml";
//     std::string traceParameterFile = "vgg16_imagenet_c2r4p25_parameters.npz";
//     std::string inoutFile = "vgg16_imagenet_c2r4p25_inout_img00000008.yaml";
//     bool scatterInput = true;
//     std::map<std::string, std::string> traceName2BlobName;
//     traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
//     traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_22", "output"));
//     launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput, -1);
// }

// TEST_F(testFixture, vgg16_imagenet_c2r4p50)
// {
//     std::string traceFileName = "vgg16_imagenet_c2r4p50_trace.yaml";
//     std::string traceParameterFile = "vgg16_imagenet_c2r4p50_parameters.npz";
//     std::string inoutFile = "vgg16_imagenet_c2r4p50_inout_img00000008.yaml";
//     bool scatterInput = true;
//     std::map<std::string, std::string> traceName2BlobName;
//     traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
//     traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_22", "output"));
//     launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput, -1);
// }

TEST_F(testFixture, vgg16_imagenet_c2r4p75)
{
    std::string traceFileName = "vgg16_imagenet_c2r4p75_trace.yaml";
    std::string traceParameterFile = "vgg16_imagenet_c2r4p75_parameters.npz";
    std::string inoutFile = "vgg16_imagenet_c2r4p75_inout_img00000008.yaml";
    bool scatterInput = true;
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_22", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput, -1);
}
#endif
#if defined(VGG16_IMAGENET_C1R4)
// TEST_F(testFixture, vgg16_imagenet_dense_c1r4)
// {
//     std::string traceFileName = "vgg16_imagenet_dense_c1r4_trace.yaml";
//     std::string traceParameterFile = "vgg16_imagenet_dense_c1r4_parameters.npz";
//     std::string inoutFile = "vgg16_imagenet_dense_c1r4_inout_img00000008.yaml";
//     bool scatterInput = true;
//     std::map<std::string, std::string> traceName2BlobName;
//     traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
//     traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_22", "output"));
//     launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput, -1);
// }
// TEST_F(testFixture, vgg16_imagenet_c1r4p25)
// {
//     std::string traceFileName = "vgg16_imagenet_c1r4p25_trace.yaml";
//     std::string traceParameterFile = "vgg16_imagenet_c1r4p25_parameters.npz";
//     std::string inoutFile = "vgg16_imagenet_c1r4p25_inout_img00000008.yaml";
//     bool scatterInput = true;
//     std::map<std::string, std::string> traceName2BlobName;
//     traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
//     traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_22", "output"));
//     launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput, -1);
// }
// TEST_F(testFixture, vgg16_imagenet_c1r4p50)
// {
//     std::string traceFileName = "vgg16_imagenet_c1r4p50_trace.yaml";
//     std::string traceParameterFile = "vgg16_imagenet_c1r4p50_parameters.npz";
//     std::string inoutFile = "vgg16_imagenet_c1r4p50_inout_img00000008.yaml";
//     bool scatterInput = true;
//     std::map<std::string, std::string> traceName2BlobName;
//     traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
//     traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_22", "output"));
//     launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput, -1);
// }
TEST_F(testFixture, vgg16_imagenet_c1r4p75)
{
    std::string traceFileName = "vgg16_imagenet_c1r4p75_trace.yaml";
    std::string traceParameterFile = "vgg16_imagenet_c1r4p75_parameters.npz";
    std::string inoutFile = "vgg16_imagenet_c1r4p75_inout_img00000008.yaml";
    bool scatterInput = true;
    std::map<std::string, std::string> traceName2BlobName;
    traceName2BlobName.insert(std::pair<std::string, std::string>("quant_0", "input"));
    traceName2BlobName.insert(std::pair<std::string, std::string>("dequant_22", "output"));
    launch(traceFileName, traceParameterFile, inoutFile, traceName2BlobName, scatterInput, -1);
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
                         bool _scatterInput,
                         int _targetLayerID)
{
    int stepCount = 1;

    //Load the trace file and the parameter file
    std::cout <<"Step "<<stepCount++<<": Loading trace file and parameter file."<<std::endl;
    std::cout <<testPrefix+_traceFileName<<" "<<testPrefix+_parameterFileName<<std::endl;
    GraphRuntime::GraphFactory graphFactory(testPrefix+_traceFileName, testPrefix+_parameterFileName, _scatterInput, _targetLayerID);

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
    for (int i=0; i<(INFERENCE_REPEAT+WARMUP); i++)
    {
        bool includeInCount = (i<WARMUP) ? false : true;
        accelerator.inference(includeInCount, profile);
    }

    std::cout <<"Step "<<stepCount++<<": Performance counts"<<std::endl;
    std::cout <<accelerator.reportRuntime();

    std::size_t dotPos = _traceFileName.find(".");
    std::string csvFileName = _traceFileName.substr(0, dotPos) + ".csv";
    accelerator.dumpRuntimeToCSV(csvFileName);

    std::cout <<"Step "<<stepCount++<<": Extract output and perform checks"<<std::endl;
    {
       YAML::Node rawBlobs = YAML::LoadFile(testPrefix+_inoutFileName);
       auto vecBlobInfo = accelerator.getOutputBlobsInfo();
       int blobID = 0;
       for (const auto& blobInfo: vecBlobInfo)
       {
           std::string layerName = _traceName2BlobName[blobInfo.blobName];
           std::vector<float> blob = rawBlobs[layerName].as<std::vector<float>>();
           std::vector<float> actualResult = accelerator.extractOutputBlob(blobID);
           std::cout<<"Actual blob size: "<<actualResult.size()<<std::endl;
           std::cout<<"Reference blob size: "<<blob.size()<<std::endl;
           int iter=0;
           //float tolerance = std::pow(2.0f, -1.0 * blobInfo.numFracBits);
           float tolerance = 1e-6;
           #if defined(CHECKOUTPUT) && (CHECKOUTPUT == 0)
           for (int h=0; h<blobInfo.height; h++)
           {
               for (int w=0; w<blobInfo.width; w++)
               {
                   for (int c=0; c<blobInfo.channel; c++)
                   {
                       int rawIter = c * blobInfo.height * blobInfo.width
                               + h * blobInfo.width + w;
                       float expected = blob.at(rawIter);
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
           std::cout <<"Tolerance is "<<tolerance<<std::endl;
           //Compare the top-10 from each output
           int K = 5;
           std::vector<t_topK_elem> referenceTopK = getTopK(blob, K);
           std::cout <<"Reference Top "<<K<<": "<<std::endl;
           iter=0;
           for (const auto& elem: referenceTopK) {
               int idx = elem.idx;
               int ch = idx / (blobInfo.height * blobInfo.width);
               int row = idx % (blobInfo.height * blobInfo.width) / blobInfo.width;
               int col = idx  % blobInfo.width;
               std::cout<<iter<<". [ch="<<ch<<", row="<<row<<", col="<<col<<"]: "<<elem.val<<std::endl;
               iter++;
           }
           iter = 0;
           std::cout <<"Actual Top "<<K<<": "<<std::endl;
           std::vector<t_topK_elem> actualTopK = getTopK(actualResult, K);
           for (const auto& elem: actualTopK) {
               int idx = elem.idx;
               int ch = idx % blobInfo.channel;
               int row = idx / (blobInfo.channel * blobInfo.width);
               int col = idx % (blobInfo.channel * blobInfo.width) / blobInfo.channel;
               std::cout<<iter<<". [ch="<<ch<<", row="<<row<<", col="<<col<<"]: "<<elem.val<<std::endl;
               iter++;
           }
           #elif defined(CHECKOUTPUT) && (CHECKOUTPUT == 1)
           for (int h=0; h<blobInfo.height; h++)
           {
               for (int w=0; w<blobInfo.width; w++)
               {
                   for (int c=0; c<blobInfo.channel; c++)
                   {
                       float actual = actualResult.at(iter++);
                       //The computation is like adding two signed numbers with numFracBits
                       //hence the difference's number of frac bits is numFracBits - 1
                       std::cout <<"Inference output at [tensor, channel, height, col]: ["
                                <<blobID<<" "<<c<<" "<<h<<" "<<w<<"]"<<std::endl
                                <<"Actual: "<<actual<<std::endl;
                   }
               }
           }
           #endif //CHECKOUTPUT


           blobID++;
       }
    }
}

std::vector<t_topK_elem> getTopK(std::vector<float> _vec, int k)
{
    int size = _vec.size();
    k = std::min(k, size);
    std::vector<int> indices(size, 0);
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(
                indices.begin(),
                indices.end(),
                [&_vec](size_t i1, size_t i2) {return _vec[i1] > _vec[i2];}
                     );
    std::vector<t_topK_elem> result;
    for (int i=0; i<k; i++) {
        t_topK_elem elem;
        elem.idx = indices.at(i);
        elem.val = _vec.at(elem.idx);
        result.push_back(elem);
    }
    return result;
}
