#include "model_container.hpp"
#include "gtest/gtest.h"

#include <vector>
#include <memory>
#include <iostream>

using namespace GraphRuntime;
using namespace std;
TEST(MODEL_CONTAINER_TEST, LoadTrace)
{
    //TODO: Change the file names
    std::string traceFileName = "testTrace_trace.yaml";
    std::string parameterFileName = "testTrace_parameters.yaml";

    YAML::Node traceNodes = YAML::LoadFile(traceFileName);
    YAML::Node parameterNodes = YAML::LoadFile(parameterFileName);
    vector<shared_ptr<Layer>> vecLayers;

    std::cout <<"Loading the trace file and instantiating the layers."<<endl;
    for (int i=0; i<traceNodes.size(); i++) {
        YAML::Node traceLayer = traceNodes[i];
        //Hash the string to a enum type, so we can use switch-case statement
        //See https://stackoverflow.com/a/650307
        LayerType opType = hashLayerTypeString(
                    traceLayer["operationType"].as<string>());

        //cout <<"Detected layer type: "<<traceLayer["operationType"].as<string>()<<endl;

        std::string layerID = traceLayer["layerID"].as<string>();
        switch (opType) {
            case CONVOLUTION:
                vecLayers.emplace_back(make_shared<ConvLayer>(
                            ConvLayer(traceLayer
                                ,parameterNodes[layerID+string("_weight")]
                                ,parameterNodes[layerID+string("_bias")])
                        ));
            break;
            case ELTADD:
                vecLayers.emplace_back(make_shared<EltAddLayer>(
                            EltAddLayer(traceLayer)
                         ));
            break;
            case MAXPOOL:
                vecLayers.emplace_back(make_shared<MaxPoolLayer>(
                            MaxPoolLayer(traceLayer)
                         ));
            break;
            case AVGPOOL:
                vecLayers.emplace_back(make_shared<AveragePoolLayer>(
                            AveragePoolLayer(traceLayer)
                         ));
            break;
            case QUANT:
                vecLayers.emplace_back(make_shared<QuantLayer>(
                            QuantLayer(traceLayer)
                         ));
            break;
            case DEQUANT:
                vecLayers.emplace_back(make_shared<DeQuantLayer>(
                            DeQuantLayer(traceLayer)
                         ));
            break;
        } // end of case
    } //end of for loop iterating through all nodes

    cout <<"Perform access evaluations."<<std::endl;
    //Check the correct number of layers have been loaded
    EXPECT_EQ(vecLayers.size(), 16);

    //Spot check convoution loading stats
    EXPECT_EQ(vecLayers.at(3)->getLayerType(), CONVOLUTION);
    EXPECT_EQ((dynamic_pointer_cast<ConvLayer>(vecLayers.at(3)))->getKernelSize(), 3);
    EXPECT_FLOAT_EQ((dynamic_pointer_cast<ConvLayer>(vecLayers.at(3)))->getBiases().at(0), 0.0f);
    EXPECT_EQ((dynamic_pointer_cast<ConvLayer>(vecLayers.at(3)))->getInputMemoryLocations().at(0), 2);


    //Spot check eltadd loading stats
    EXPECT_EQ(vecLayers.at(8)->getLayerType(), ELTADD);
    EXPECT_EQ((dynamic_pointer_cast<EltAddLayer>(vecLayers.at(8)))->getInputMemoryLocations().at(0), 0);
    EXPECT_EQ((dynamic_pointer_cast<EltAddLayer>(vecLayers.at(8)))->getInputMemoryLocations().at(1), 1);
    EXPECT_EQ((dynamic_pointer_cast<EltAddLayer>(vecLayers.at(8)))->getInputFracBits().at(0), 0);
    EXPECT_EQ((dynamic_pointer_cast<EltAddLayer>(vecLayers.at(8)))->getInputFracBits().at(1), 5);
    EXPECT_EQ((dynamic_pointer_cast<EltAddLayer>(vecLayers.at(8)))->getOutputFracBits(), 0);

    //Spot check quant loading stats
    EXPECT_EQ(vecLayers.at(0)->getLayerType(), QUANT);
    EXPECT_EQ((dynamic_pointer_cast<Layer>(vecLayers.at(0)))->getOutputHeight(), 32);
    EXPECT_EQ((dynamic_pointer_cast<Layer>(vecLayers.at(0)))->getOutputFracBits(), 4);

    //Spot check dequant loading stats
    EXPECT_EQ(vecLayers.at(15)->getLayerType(), DEQUANT);
    EXPECT_EQ((dynamic_pointer_cast<Layer>(vecLayers.at(15)))->getOutputChannel(), 10);
    EXPECT_EQ((dynamic_pointer_cast<Layer>(vecLayers.at(15)))->getInputFracBits().at(0), 9);
}
