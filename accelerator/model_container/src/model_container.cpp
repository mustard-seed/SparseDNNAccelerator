#include "model_container.hpp"
#include "params.hpp"

#define DIVIDE_CEIL(x, y) (1 + (x-1) / (y))

namespace GraphRuntime {
    Layer::Layer(const YAML::Node &_node) : node(_node)
    { }

    int Layer::getLayerID()
    {
        return node["layerID"].as<int>();
    }

    void Layer::setLayerID(int _id)
    {
        node["layerID"] = _id;
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

    void Layer::setInputHeights(IntVec _inputHeights)
    {
        node["inputHeights"] = _inputHeights;
    }

    void Layer::setInputWidths(IntVec _inputWidths)
    {
        node["inputWidths"] = _inputWidths;
    }

    void Layer::setInputChannels(IntVec _inputChannels)
    {
        node["inputChannels"] = _inputChannels;
    }

    void Layer::setInputFracBits(IntVec _inputFracBits)
    {
        node["inputFracBits"] = _inputFracBits;
    }

    void Layer::setInputMemoryLocations(IntVec _inputMemLocations)
    {
        node["inputMemoryLocations"] = _inputMemLocations;
    }

    void Layer::setInputGroupsSeenBySource(IntVec _inputGroupsSeenBySource)
    {
        node["inputGroupsSeenBySource"] = _inputGroupsSeenBySource;
    }

    void Layer::setInputSparseFlag(bool _inputSparseFlag)
    {
        node["sparseInput"] = _inputSparseFlag;
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

    void Layer::setOutputFracBits(int _outputFracBits)
    {
        node["outputFracBits"] = _outputFracBits;
    }

    void Layer::setOutputHeight(int _outputHeight)
    {
        node["outputHeight"] = _outputHeight;
    }

    void Layer::setOutputWidth(int _outputWidth)
    {
        node["outputWidth"] = _outputWidth;
    }

    void Layer::setOutputChannel(int _oc)
    {
        node["outputChannels"] = _oc;
    }

    void Layer::setOutputMemoryLocation(int _omem)
    {
        node["outputMemoryLocation"] = _omem;
    }

    void Layer::setOutputReluFlag(bool _flag)
    {
        node["outputRelu"] = _flag;
    }

    void Layer::setOutputSparseFlag(bool _flag)
    {
        node["outputCanBeSparse"] = _flag;
    }

    int Layer::getCurrentNumberGroups()
    {
        return node["outputCurrentNumGroups"].as<int>();
    }
    int Layer::getNextNumberGroups()
    {
        return node["outputNextNumGroups"].as<int>();
    }

    void Layer::setCurrentNumberGroups(int _groups)
    {
        node["outputCurrentNumGroups"] = _groups;
    }

    void Layer::setNextNumberGroups(int _groups)
    {
        node["outputNextNumGroups"] = _groups;
    }

    bool Layer::cacheBoundaryCheck(t_graph_output_tile_info _tileCandidate)
    {
        return (_tileCandidate.sizeOutputTileFullHeight <= MAX_OUTPUT_TILE_HEIGHT)
                && (_tileCandidate.sizeOutputTileFullWidthPerCol <= MAX_OUTPUT_TILE_WIDTH_PER_COL);
    }

    t_latency_info Layer::deriveLatency(t_graph_output_tile_info _tileCandidate)
    {
        t_latency_info info {.inputTransferLatency=0,
                    .weightTransferLatency=0,
                     .outputTransferLatency=0,
                     .computeLatency=0,
                     .computeLatencyWithOverhead=0,
                      .ddrLatency=0,
                      .totalLatency=0,
                      .isComputeBound=false};

        return info;
    }

    int Layer::deriveOps()
    {
        return 0;
    }

    ConvLayer::ConvLayer(const YAML::Node& _node, const cnpy::NpyArray& _weightNode, const cnpy::NpyArray& _biasNode)
        : Layer(_node)
    {
        //Reorder the weight from NCHW to NHWC
        this->loadWeights(_weightNode);

        //Store biases
        this->loadBiases(_biasNode);
    }

    void ConvLayer::setKernelStride(int _stride)
    {
        node["kernelStride"] = _stride;
    }

    void ConvLayer::setKernelSize(int _size)
    {
        node["kernelSize"] = _size;
    }

    void ConvLayer::setInputBorderPadding(int _padding)
    {
        node["inputBorderPadding"] = _padding;
    }

    void ConvLayer::setTransConvPadding(int _padding)
    {
        node["inputTransConvPadding"] = _padding;
    }

    void ConvLayer::setBiasFlag(bool _flag)
    {
        node["hasBias"] = _flag;
    }

    void ConvLayer::setWeightFracBits(int _bits)
    {
        node["weightFracBits"] = _bits;
    }

    void ConvLayer::setWeightSparsity(float _sparsity)
    {
        node["sparsity"] = _sparsity;
    }

    void ConvLayer::setWeightPruneClusterSize(int _size)
    {
        node["pruneClusterSize"] = _size;
    }

    void ConvLayer::setWeightPruneRangeSizeInCluster(int _size)
    {
        node["pruneRangeInCluster"] = _size;
    }

    void ConvLayer::setIsAfterInput (bool _flag)
    {
        node["isAfterInput"] = _flag;
    }

    void ConvLayer::loadWeights(const float* _pWeights)
    {
        int outputChannel = Layer::getOutputChannel();
        int inputChannel = Layer::getInputChannels().at(0);
        int kernelSize = getKernelSize();
        bool needToPermuteWeight = true;
        if (node["needToPermuteWeight"]) {
            if (node["needToPermuteWeight"].as<bool>() == false) {
                needToPermuteWeight = false;
            }
        }

        vecWeights.resize(outputChannel*inputChannel*kernelSize*kernelSize);

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
                    vecWeights.at(weightLocalIndex) = _pWeights[weightTraceIndex];
                    //std::cout <<"[oc, ic, k, weight]"<<oc<<" "<<ic<<" "<<k<<" "<<pWeights[weightTraceIndex]<<std::endl;

                    if (needToPermuteWeight) {
                        weightLocalIndexPlanarContrib += inputChannel;
                    }
                    else {
                        weightLocalIndexPlanarContrib++;
                    }
                    weightTraceIndex++;
                }
                if (needToPermuteWeight) {
                    weightLocalIndexICContrib++;
                }
                else
                {
                    weightLocalIndexICContrib += kernelSize*kernelSize;
                }
            }
            weightLocalIndexOCContrib += kernelSize*kernelSize*inputChannel;
        }
    }

    void ConvLayer::loadWeights(const cnpy::NpyArray& _weightNode)
    {
        const float* pWeights = _weightNode.data<float>();
        this->loadWeights(pWeights);
    }

    void ConvLayer::loadBiases (const float* _pBiases)
    {
        int outputChannel = Layer::getOutputChannel();
        vecBiases.resize(outputChannel);
        for (int iB=0; iB<outputChannel; iB++)
        {
            vecBiases.at(iB) = _pBiases[iB];
        }
    }

    void ConvLayer::loadBiases(const cnpy::NpyArray &_biasNode)
    {
        const float* pBiases = _biasNode.data<float>();

        this->loadBiases(pBiases);
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

    bool ConvLayer::getIsAfterInput()
    {
        return node["isAfterInput"].as<bool>();
    }

    bool ConvLayer::cacheBoundaryCheck(t_graph_output_tile_info _tileCandidate)
    {
        //Check max tiling dimensions
        if (!Layer::cacheBoundaryCheck(_tileCandidate)) {
            return false;
        }

        //Check that the tile configuration can work wit the IA/OA cache limit
        unsigned int sizeInputTileFullHeight = deriveConvInputDimension1D(
                       _tileCandidate.sizeOutputTileFullHeight,
                       getKernelSize(),
                       getKernelStride()
                    );
        unsigned int sizeInputTileFullWidthPerCol = deriveConvInputDimension1D(
                       _tileCandidate.sizeOutputTileFullWidthPerCol,
                       getKernelSize(),
                       getKernelStride()
                    );
        unsigned int sizeInputTilePartialWidthPerCol = deriveConvInputDimension1D(
                       _tileCandidate.sizeOutputTilePartialWidthPerCol,
                        getKernelSize(),
                        getKernelStride()
                    );

        int iaCachePerColRequirement =
                ia_cache_boundary_check(
                    sizeInputTileFullHeight,
                    sizeInputTileFullWidthPerCol,
                    DIVIDE_CEIL(getInputChannels().at(0), ACTIVATION_DRAM_SIZE_BYTE)
                    );
        int iaCachePerPartialColRequirement =
                ia_cache_boundary_check(
                    sizeInputTileFullHeight,
                    sizeInputTilePartialWidthPerCol,
                    DIVIDE_CEIL(getInputChannels().at(0), ACTIVATION_DRAM_SIZE_BYTE)
                    );
        //TODO: Change the arguments to the OA cache requirement checker
        int oaCachePerColRequirement =
                oa_cache_boundary_check(
                    _tileCandidate.sizeOutputTileFullHeight,
                    _tileCandidate.sizeOutputTileFullWidthPerCol,
                    getOutputChannel() / getCurrentNumberGroups()
                    );
        int oaCachePerPartialColRequirement =
                oa_cache_boundary_check(
                    _tileCandidate.sizeOutputTileFullHeight,
                    _tileCandidate.sizeOutputTilePartialWidthPerCol,
                    getOutputChannel() / getCurrentNumberGroups()
                    );
        bool passCacheRequirement =
                (iaCachePerColRequirement <= IA_CACHE_DEPTH)
                && (iaCachePerPartialColRequirement <= IA_CACHE_DEPTH)
                && (oaCachePerColRequirement <= OA_CACHE_DEPTH)
                && (oaCachePerPartialColRequirement <= OA_CACHE_DEPTH)
                && (sizeInputTileFullHeight <= MAX_INPUT_TILE_HEIGHT)
                && (sizeInputTileFullWidthPerCol <= MAX_INPUT_TILE_WIDTH_PER_COL)
                && (sizeInputTilePartialWidthPerCol <= MAX_INPUT_TILE_WIDTH_PER_COL);

        return passCacheRequirement;
    }

    t_latency_info ConvLayer::deriveLatency(t_graph_output_tile_info _tileCandidate)
    {
        unsigned int outputHeight = getOutputHeight();
        unsigned int outputChannelsPerCurrentGroup = getOutputChannel() / getCurrentNumberGroups();
        unsigned int inputChannels = getInputChannels().at(0);
        unsigned int inputChannelsPerGroup = inputChannels / getCurrentNumberGroups();

        unsigned int computeLatency;
        unsigned int weightOnChipLatency, weightDDRLatency;
#if defined(SPW_SYSTEM)
       computeLatency = deriveSparseConvComputationLatency(
                   _tileCandidate,
                   outputChannelsPerCurrentGroup,
                   inputChannelsPerGroup,
                   getCurrentNumberGroups(),
                   getKernelSize(),
                   (int) std::ceil( (1.0f - getWeightSparsity()) * PRUNE_RANGE_IN_CLUSTER)
                   );
        weightOnChipLatency = deriveSparseConvWeightTransferLatency(
                       _tileCandidate,
                       inputChannelsPerGroup,
                       outputChannelsPerCurrentGroup,
                       getCurrentNumberGroups(),
                       getKernelSize(),
                       PE_SIMD_SIZE,
                       CLUSTER_SIZE,
                       PRUNE_RANGE_IN_CLUSTER,
                       (int) std::ceil( (1.0f - getWeightSparsity()) * PRUNE_RANGE_IN_CLUSTER),
                       WEIGHT_DRAM_SIZE_VALUE_BYTE
                   );
        //Need to adjust the weight latency by taking the index into account
        weightDDRLatency = deriveSparseConvWeightTransferLatency(
                   _tileCandidate,
                   inputChannelsPerGroup,
                   outputChannelsPerCurrentGroup,
                   getCurrentNumberGroups(),
                   getKernelSize(),
                   PE_SIMD_SIZE,
                   CLUSTER_SIZE,
                   PRUNE_RANGE_IN_CLUSTER,
                   (int) std::ceil( (1.0f - getWeightSparsity()) * PRUNE_RANGE_IN_CLUSTER),
                   DDR_BYTES_PER_CYCLE
               );
        weightDDRLatency = (unsigned int) ((float) weightDDRLatency * (1.0f + (float) WEIGHT_DRAM_SIZE_INDEX_BYTE / (float) WEIGHT_DRAM_SIZE_VALUE_BYTE));
#else
        computeLatency = deriveDenseConvComputationLatency(
                    _tileCandidate,
                    outputChannelsPerCurrentGroup,
                    inputChannelsPerGroup,
                    getCurrentNumberGroups(),
                    getKernelSize()
                    );
        weightOnChipLatency = deriveDenseConvWeightTransferLatency(
                    _tileCandidate,
                    inputChannelsPerGroup,
                    outputChannelsPerCurrentGroup,
                    getCurrentNumberGroups(),
                    getKernelSize(),
                     WEIGHT_DRAM_SIZE_VALUE_BYTE
                    );
        weightDDRLatency = deriveDenseConvWeightTransferLatency(
                    _tileCandidate,
                    inputChannelsPerGroup,
                    outputChannelsPerCurrentGroup,
                    getCurrentNumberGroups(),
                    getKernelSize(),
                     DDR_BYTES_PER_CYCLE
                    );
#endif
       unsigned int inputOnChipLatency = deriveInputTransferLatency(
                   _tileCandidate,
                   inputChannelsPerGroup,
                   getCurrentNumberGroups(),
                   getKernelSize(),
                   getKernelStride(),
                   true,
                   ACTIVATION_DRAM_SIZE_BYTE
                   );
       unsigned int inputDDRLatency = deriveInputTransferLatency(
                   _tileCandidate,
                   inputChannelsPerGroup,
                   getCurrentNumberGroups(),
                   getKernelSize(),
                   getKernelStride(),
                   true,
                   DDR_BYTES_PER_CYCLE
                   );

       unsigned int outputOnChipLatency = deriveOutputTransferLatency(
                   _tileCandidate,
                   outputHeight,
                   outputChannelsPerCurrentGroup,
                   getCurrentNumberGroups(),
                   true,
                   ACTIVATION_DRAM_SIZE_BYTE
                   );
       unsigned int outputDDRLatency = deriveOutputTransferLatency(
                   _tileCandidate,
                   outputHeight,
                   outputChannelsPerCurrentGroup,
                   getCurrentNumberGroups(),
                   true,
                   DDR_BYTES_PER_CYCLE
                   );
       //maxLatency = outputLatency > maxLatency ? outputLatency : maxLatency;


       unsigned int firstTileInputLatency = deriveFirstTileConvInputTransferLatency
               (
                   _tileCandidate,
                   inputChannelsPerGroup,
                   getKernelSize(),
                   getKernelStride()
               );

       unsigned int lastTileOutputLatency = deriveLastTileOutputTransferLatency
               (
                   _tileCandidate,
                   outputChannelsPerCurrentGroup
               );

       unsigned int computeLatencyWithOverhead =
               computeLatency + firstTileInputLatency + lastTileOutputLatency;

       unsigned int totalDDRLatency = inputDDRLatency + outputDDRLatency + weightDDRLatency;

       unsigned int totalLatency = computeLatencyWithOverhead;
       if (totalLatency < totalDDRLatency)
       {
           totalLatency = totalDDRLatency;
       }
       if (totalLatency < inputOnChipLatency)
       {
           totalLatency = inputOnChipLatency;
       }
       if (totalLatency < outputOnChipLatency)
       {
           totalLatency = outputOnChipLatency;
       }
       if (totalLatency < weightOnChipLatency)
       {
           totalLatency = weightOnChipLatency;
       }

       t_latency_info latInfo {
                   .inputTransferLatency = inputOnChipLatency,
                   .weightTransferLatency = weightOnChipLatency,
                   .outputTransferLatency = outputOnChipLatency,
                   .computeLatency = computeLatency,
                   .computeLatencyWithOverhead = computeLatencyWithOverhead,
                   .ddrLatency = totalDDRLatency,
                   .totalLatency = totalLatency,
                   .isComputeBound = computeLatencyWithOverhead >= totalLatency
       };

       return latInfo;
    }

    int ConvLayer::deriveOps()
    {
        unsigned int inputChannelsPerGroup = getInputChannels()[0] / getCurrentNumberGroups();
        unsigned int outputChannelsPerCurrentGroup = getOutputChannel() / getCurrentNumberGroups();
        return getCurrentNumberGroups() * (
                    outputChannelsPerCurrentGroup
                        * getOutputHeight() * getOutputWidth()
                        * getKernelSize() * getKernelSize() * inputChannelsPerGroup
                        *2);
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

    void MaxPoolLayer::setKernelStride(int _stride)
    {
        node["kernelStride"] = _stride;
    }
    void MaxPoolLayer::setKernelSize(int _size)
    {
        node["kernelSize"] = _size;
    }
    void MaxPoolLayer::setInputBorderPadding(int _padding)
    {
        node["inputBorderPadding"] = _padding;
    }

    t_latency_info MaxPoolLayer::deriveLatency(t_graph_output_tile_info _tileCandidate)
    {
        _tileCandidate = deriveConvOutputTileShape(
                        getOutputHeight(),
                        getOutputWidth(),
                        1, //output tile height, full
                        1, //output tile width per col
                        false //not conv
                    );
        int numEffectiveGroups = DIVIDE_CEIL(getOutputChannel(), ACTIVATION_DRAM_SIZE_BYTE);

        int inputOnChipLatency = deriveInputTransferLatency(
                        _tileCandidate,
                        ACTIVATION_DRAM_SIZE_BYTE,
                        numEffectiveGroups,
                        getKernelSize(),
                        getKernelStride(),
                        false,
                        ACTIVATION_DRAM_SIZE_BYTE
                    );
        int inputDDRLatency = deriveInputTransferLatency(
                        _tileCandidate,
                        ACTIVATION_DRAM_SIZE_BYTE,
                        numEffectiveGroups,
                        getKernelSize(),
                        getKernelStride(),
                        false,
                        DDR_BYTES_PER_CYCLE
                    );
        int outputOnChipLatency = deriveOutputTransferLatency(
                        _tileCandidate,
                        getOutputHeight(),
                        ACTIVATION_DRAM_SIZE_BYTE,
                        numEffectiveGroups,
                        false,
                        ACTIVATION_DRAM_SIZE_BYTE
                    );
        int outputDDRLatency = deriveOutputTransferLatency(
                    _tileCandidate,
                    getOutputHeight(),
                    ACTIVATION_DRAM_SIZE_BYTE,
                    numEffectiveGroups,
                    false,
                    DDR_BYTES_PER_CYCLE
                );

        int totalDDRLatency = outputDDRLatency + inputDDRLatency;
        int actualLatency = inputOnChipLatency > outputOnChipLatency ? inputOnChipLatency : outputOnChipLatency;
        actualLatency = actualLatency > totalDDRLatency ? actualLatency : totalDDRLatency;
        t_latency_info latInfo {
            .inputTransferLatency = inputOnChipLatency,
            .weightTransferLatency = 0,
            .outputTransferLatency = outputOnChipLatency,
            .computeLatency = 0,
            .computeLatencyWithOverhead = 0,
            .ddrLatency = totalDDRLatency,
            .totalLatency = actualLatency,
            .isComputeBound = false
        };
        return latInfo;
    }

    int MaxPoolLayer::deriveOps()
    {
        return getOutputHeight()*getOutputWidth()*getOutputChannel()*getKernelSize()*getKernelSize();
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

    void AveragePoolLayer::setKernelStride(int _stride)
    {
        node["kernelStride"] = _stride;
    }
    void AveragePoolLayer::setKernelSize(int _size)
    {
        node["kernelSize"] = _size;
    }
    void AveragePoolLayer::setInputBorderPadding(int _padding)
    {
        node["inputBorderPadding"] = _padding;
    }

    void AveragePoolLayer::setDivisor(float _divisor)
    {
        node["divisor"] = _divisor;
    }

    t_latency_info AveragePoolLayer::deriveLatency(t_graph_output_tile_info _tileCandidate)
    {
        _tileCandidate = deriveConvOutputTileShape(
                        getOutputHeight(),
                        getOutputWidth(),
                        1, //output tile height, full
                        1, //output tile width per col
                        false //not conv
                    );
        int numEffectiveGroups = DIVIDE_CEIL(getOutputChannel(), ACTIVATION_DRAM_SIZE_BYTE);

        int inputOnChipLatency = deriveInputTransferLatency(
                        _tileCandidate,
                        ACTIVATION_DRAM_SIZE_BYTE,
                        numEffectiveGroups,
                        getKernelSize(),
                        getKernelStride(),
                        false,
                        ACTIVATION_DRAM_SIZE_BYTE
                    );
        int inputDDRLatency = deriveInputTransferLatency(
                        _tileCandidate,
                        ACTIVATION_DRAM_SIZE_BYTE,
                        numEffectiveGroups,
                        getKernelSize(),
                        getKernelStride(),
                        false,
                        DDR_BYTES_PER_CYCLE
                    );
        int outputOnChipLatency = deriveOutputTransferLatency(
                        _tileCandidate,
                        getOutputHeight(),
                        ACTIVATION_DRAM_SIZE_BYTE,
                        numEffectiveGroups,
                        false,
                        ACTIVATION_DRAM_SIZE_BYTE
                    );
        int outputDDRLatency = deriveOutputTransferLatency(
                    _tileCandidate,
                    getOutputHeight(),
                    ACTIVATION_DRAM_SIZE_BYTE,
                    numEffectiveGroups,
                    false,
                    DDR_BYTES_PER_CYCLE
                );

        int totalDDRLatency = outputDDRLatency + inputDDRLatency;
        int actualLatency = inputOnChipLatency > outputOnChipLatency ? inputOnChipLatency : outputOnChipLatency;
        actualLatency = actualLatency > totalDDRLatency ? actualLatency : totalDDRLatency;
        t_latency_info latInfo {
            .inputTransferLatency = inputOnChipLatency,
            .weightTransferLatency = 0,
            .outputTransferLatency = outputOnChipLatency,
            .computeLatency = 0,
            .computeLatencyWithOverhead = 0,
            .ddrLatency = totalDDRLatency,
            .totalLatency = actualLatency,
            .isComputeBound = false
        };
        return latInfo;
    }

    int AveragePoolLayer::deriveOps()
    {
        return getOutputHeight()*getOutputWidth()*getOutputChannel()*getKernelSize()*getKernelSize();
    }

    EltAddLayer::EltAddLayer(const YAML::Node &_node)
        :Layer(_node)
    {

    }

    LayerType EltAddLayer::getLayerType()
    {
        return ELTADD;
    }

    t_latency_info EltAddLayer::deriveLatency(t_graph_output_tile_info _tileCandidate)
    {
        _tileCandidate = deriveConvOutputTileShape(
                        getOutputHeight(),
                        getOutputWidth(),
                        1, //output tile height, full
                        1, //output tile width per col
                        false //not conv
                    );
        int numEffectiveGroups = DIVIDE_CEIL(getOutputChannel(), ACTIVATION_DRAM_SIZE_BYTE);

        int inputOnChipLatency = deriveInputTransferLatency(
                        _tileCandidate,
                        ACTIVATION_DRAM_SIZE_BYTE,
                        numEffectiveGroups,
                        1,
                        1,
                        false,
                        ACTIVATION_DRAM_SIZE_BYTE
                    ) * 2;
        int inputDDRLatency = deriveInputTransferLatency(
                        _tileCandidate,
                        ACTIVATION_DRAM_SIZE_BYTE,
                        numEffectiveGroups,
                        1,
                        1,
                        false,
                        DDR_BYTES_PER_CYCLE
                    ) * 2;
        int outputOnChipLatency = deriveOutputTransferLatency(
                        _tileCandidate,
                        getOutputHeight(),
                        ACTIVATION_DRAM_SIZE_BYTE,
                        numEffectiveGroups,
                        false,
                        ACTIVATION_DRAM_SIZE_BYTE
                    );
        int outputDDRLatency = deriveOutputTransferLatency(
                    _tileCandidate,
                    getOutputHeight(),
                    ACTIVATION_DRAM_SIZE_BYTE,
                    numEffectiveGroups,
                    false,
                    DDR_BYTES_PER_CYCLE
                );

        int totalDDRLatency = outputDDRLatency + inputDDRLatency;
        int actualLatency = inputOnChipLatency > outputOnChipLatency ? inputOnChipLatency : outputOnChipLatency;
        actualLatency = actualLatency > totalDDRLatency ? actualLatency : totalDDRLatency;
        t_latency_info latInfo {
            .inputTransferLatency = inputOnChipLatency,
            .weightTransferLatency = 0,
            .outputTransferLatency = outputOnChipLatency,
            .computeLatency = 0,
            .computeLatencyWithOverhead = 0,
            .ddrLatency = totalDDRLatency,
            .totalLatency = actualLatency,
            .isComputeBound = false
        };
        return latInfo;
    }

    int EltAddLayer::deriveOps()
    {
        return getOutputHeight()*getOutputWidth()*getOutputChannel();
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

        return QUANT;
    }
}
