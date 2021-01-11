#include "model_container.hpp"

namespace GraphRuntime {
    Layer::Layer(const YAML::Node &_node) : node(_node)
    { }

    int Layer::getLayerID()
    {
        return node["layerID"].as<int>();
    }

    IntVec Layer::getInputHeights()
    {
        IntVec result;
        YAML::Node item = node["inputHeights"];
        for (unsigned int i=0; i<item.size(); i++)
        {
            result.push_back(item[i].as<int>());
        }
        return result;
    }

    IntVec Layer::getInputWidths()
    {
        IntVec result;
        YAML::Node item = node["inputWidths"];
        for (unsigned int i=0; i<item.size(); i++)
        {
            result.push_back(item[i].as<int>());
        }
        return result;
    }

    IntVec Layer::getInputChannels()
    {
        IntVec result;
        YAML::Node item = node["inputChannels"];
        for (unsigned int i=0; i<item.size(); i++)
        {
            result.push_back(item[i].as<int>());
        }
        return result;
    }

    IntVec Layer::getInputFracBits()
    {
        IntVec result;
        YAML::Node item = node["inputFracBits"];
        for (size_t i=0; i<item.size(); i++)
        {
            result.push_back(item[i].as<int>());
        }
        return result;
    }

    IntVec Layer::getInputMemoryLocations()
    {
        IntVec result;
        YAML::Node item = node["inputMemoryLocations"];
        for (size_t i=0; i<item.size(); i++)
        {
            result.push_back(item[i].as<int>());
        }
        return result;
    }

    IntVec Layer::getInputGroupsSeenBySource()
    {
        IntVec result;
        YAML::Node item = node["inputGroupsSeenBySource"];
        for (size_t i=0; i<item.size(); i++)
        {
            result.push_back(item[i].as<int>());
        }
        return result;
    }

    bool Layer::getInputSparseFlag()
    {
        return node["sparseInput"].as<bool>();
    }

    int Layer::getOutputFracBits()
    {
        return node["outputFracBits"].as<int>();
    }

    int Layer::getOutputHeight()
    {
        return node["outputHeight"].as<int>();
    }
    int Layer::getOutputWidth()
    {
        return node["outputWidth"].as<int>();
    }
    int Layer::getOutputChannel()
    {
        return node["outputChannels"].as<int>();
    }
    int Layer::getOutputMemoryLocation()
    {
        return node["outputMemoryLocation"].as<int>();
    }
    bool Layer::getOutputReluFlag()
    {
        return node["outputRelu"].as<bool>();
    }
    bool Layer::getOutputSparseFlag()
    {
        return node["outputCanBeSparse"].as<bool>();
    }
    int Layer::getCurrentNumberGroups()
    {
        return node["outputCurrentNumGroups"].as<int>();
    }
    int Layer::getNextNumberGroups()
    {
        return node["outputNextNumGroups"].as<int>();
    }

    ConvLayer::ConvLayer(const YAML::Node& _node, const cnpy::NpyArray& _weightNode, const cnpy::NpyArray& _biasNode)
        : Layer(_node)
    {
        //Reorder the weight from NCHW to NHWC
        this->loadWeights(_weightNode);

        //Store biases
        this->loadBiases(_biasNode);
    }

    void ConvLayer::loadWeights(const cnpy::NpyArray& _weightNode)
    {
        int outputChannel = Layer::getOutputChannel();
        int inputChannel = Layer::getInputChannels().at(0);
        int kernelSize = getKernelSize();

        vecWeights.resize(outputChannel*inputChannel*kernelSize*kernelSize);
        const float* pWeights = _weightNode.data<float>();
        //std::cout <<"Weight word size is "<<_weightNode.word_size<<std::endl;

        //Store weights
        int weightTraceIndex = 0;
        int weightLocalIndexOCContrib = 0;
        for (int oc=0; oc<outputChannel; oc++)
        {
            int weightLocalIndexICContrib = 0;
            for (int ic=0; ic<inputChannel; ic++)
            {
                int weightLocalIndexPlanarContrib = 0;
                for (int k=0; k<kernelSize*kernelSize; k++)
                {
                    int weightLocalIndex = weightLocalIndexOCContrib+weightLocalIndexICContrib+weightLocalIndexPlanarContrib;
                    vecWeights.at(weightLocalIndex) = pWeights[weightTraceIndex];
                    //std::cout <<"[oc, ic, k, weight]"<<oc<<" "<<ic<<" "<<k<<" "<<pWeights[weightTraceIndex]<<std::endl;

                    weightLocalIndexPlanarContrib += inputChannel;
                    weightTraceIndex++;
                }
                weightLocalIndexICContrib++;
            }
            weightLocalIndexOCContrib += kernelSize*kernelSize*inputChannel;
        }
    }

    void ConvLayer::loadBiases(const cnpy::NpyArray &_biasNode)
    {
        int outputChannel = Layer::getOutputChannel();
        vecBiases.resize(outputChannel);

        const float* pBiases = _biasNode.data<float>();
        //Store biases
        for (int iB=0; iB<outputChannel; iB++)
        {
            vecBiases.at(iB) = pBiases[iB];
        }
    }

    LayerType ConvLayer::getLayerType()
    {
        return CONVOLUTION;
    }

    int ConvLayer::getKernelStride()
    {
        return node["kernelStride"].as<int>();
    }
    int ConvLayer::getKernelSize()
    {
        return node["kernelSize"].as<int>();
    }
    int ConvLayer::getInputBorderPadding()
    {
        return node["inputBorderPadding"].as<int>();
    }
    int ConvLayer::getTransConvPadding()
    {
        return node["inputTransConvPadding"].as<int>();
    }

    int ConvLayer::getWeightFracBits()
    {
        return node["weightFracBits"].as<int>();
    }

    bool ConvLayer::getBiasFlag()
    {
        return node["hasBias"].as<bool>();
    }

    FloatVec ConvLayer::getWeights()
    {
        return vecWeights;
    }

    FloatVec ConvLayer::getBiases()
    {
        return vecBiases;
    }

    float ConvLayer::getWeightSparsity()
    {
        return node["sparsity"].as<float>();
    }

    int ConvLayer::getWeightPruneClusterSize()
    {
        return node["pruneClusterSize"].as<int>();
    }

    int ConvLayer::getWeightPruneRangeSizeInCluster()
    {
        return node["pruneRangeInCluster"].as<int>();
    }

    MaxPoolLayer::MaxPoolLayer(const YAML::Node& _node)
        :Layer(_node)
    {

    }

    LayerType MaxPoolLayer::getLayerType()
    {
        return MAXPOOL;
    }

    int MaxPoolLayer::getKernelStride()
    {
        return node["kernelStride"].as<int>();
    }
    int MaxPoolLayer::getKernelSize()
    {
        return node["kernelSize"].as<int>();
    }
    int MaxPoolLayer::getInputBorderPadding()
    {
        return node["inputBorderPadding"].as<int>();
    }

    AveragePoolLayer::AveragePoolLayer(const YAML::Node &_node)
        :Layer(_node)
    {

    }

    LayerType AveragePoolLayer::getLayerType()
    {
        return AVGPOOL;
    }

    int AveragePoolLayer::getKernelStride()
    {
        return node["kernelStride"].as<int>();
    }
    int AveragePoolLayer::getKernelSize()
    {
        return node["kernelSize"].as<int>();
    }
    int AveragePoolLayer::getInputBorderPadding()
    {
        return node["inputBorderPadding"].as<int>();
    }

    float AveragePoolLayer::getDivisor()
    {
        return node["divisor"].as<float>();
    }

    EltAddLayer::EltAddLayer(const YAML::Node &_node)
        :Layer(_node)
    {

    }

    LayerType EltAddLayer::getLayerType()
    {
        return ELTADD;
    }

    QuantLayer::QuantLayer(const YAML::Node &_node)
        :Layer(_node)
    {

    }

    LayerType QuantLayer::getLayerType()
    {
        return QUANT;
    }

    DeQuantLayer::DeQuantLayer(const YAML::Node &_node)
        :Layer(_node)
    {

    }

    LayerType DeQuantLayer::getLayerType()
    {
        return DEQUANT;
    }

    LayerType hashLayerTypeString(std::string stringValue)
    {
        if (stringValue == "quantstub") {return QUANT;}
        if (stringValue == "dequantstub") {return DEQUANT;}
        if (stringValue == "conv") {return CONVOLUTION;}
        if (stringValue == "maxpool") {return MAXPOOL;}
        if (stringValue == "avgpool") {return AVGPOOL;}
        if (stringValue == "eltadd") {return ELTADD;}
    }
}
