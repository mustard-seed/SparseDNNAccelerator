#include "graph_factory.hpp"
#include "layerInstructionGenerator.hpp"
#include "params.hpp"
#include "floatFixedPointConversion.hpp"
//#include "tensorCompression.hpp" //calculateExternalMemoryAddressStride
#include "spwTensorCompression.hpp"
#include <cfenv> //For rounding modes

#include <memory>
#include <cmath>

#include "cnpy.hpp" //Third party library used to load numpy array weights

#define DIVIDE_CEIL(x, y) (1 + (x-1) / (y))

using namespace std;
using namespace GraphRuntime;

/*!
 * Some helper functions
 */

/*!
 * \brief calculateTileWidthPerUnit
 * \details calculate the best tile configuration
 * all there is to know about the layer.
 * \param _convLayer[const GraphRuntime::ConvLayer &]
 * \return
 */
t_tile_pair calculateTileSizePerUnit(ConvLayer &_convLayer);

t_tile_pair calculateTileSizePerUnit(EltAddLayer &_eltAddLayer);

namespace GraphRuntime {
    GraphFactory::GraphFactory(std::string _traceFileName, std::string _parameterFileName, bool _inputScatter, int _lastLayerID)
    {
       typedef YAML::Node YN;
       flagInputScatter = _inputScatter;
       YN traceNodes = YAML::LoadFile(_traceFileName);
       std::cout <<"Loaded YAML trace file "<<_traceFileName<<"."<<std::endl;
       //YN parameterNodes = YAML::LoadFile(_parameterFileName);
       cnpy::npz_t parameterNodes;
       try {
        parameterNodes = cnpy::npz_load(_parameterFileName);
       }
       catch (std::runtime_error)
       {
           std::cout <<"Failed to load npz_load, but the test might be ok."<<std::endl;
       }
       bool appendDequant = _lastLayerID >= 0;
       int size = (_lastLayerID < 0) ? traceNodes.size() : std::min((int)_lastLayerID+1, (int)traceNodes.size());
       for (int i=0; i<size; i++) {
           YAML::Node traceLayer = traceNodes[i];
           //Hash the string to a enum type, so we can use switch-case statement
           //See https://stackoverflow.com/a/650307
           LayerType opType = hashLayerTypeString(
                       traceLayer["operationType"].as<string>());
           vecFlagManualTile.push_back(false);
           t_graph_output_tile_info tempTile;
           vecTileInfo.push_back(tempTile);

           //cout <<"[Graph factory] Detected layer type: "<<traceLayer["operationType"].as<string>()<<endl;
#if defined(HOST_DEBUG)
           cout <<"LayerType-ID: "<<traceLayer["operationType"].as<string>()<<"-"<<i<<endl;
#endif
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

       if (appendDequant) {
           int layerID = vecLayers.back()->getLayerID() + 1;
           int outputMemoryLocation = vecLayers.back()->getOutputMemoryLocation();
           int height = vecLayers.back()->getOutputHeight();
           int width = vecLayers.back()->getOutputWidth();
           int channel = vecLayers.back()->getOutputChannel();
           int fracBits = vecLayers.back()->getOutputFracBits();
           int groups = vecLayers.back()->getCurrentNumberGroups();

           auto pOutputBlob = std::make_shared<DeQuantLayer>(DeQuantLayer());

           pOutputBlob->setLayerID(layerID);
           pOutputBlob->setInputHeights({height});
           pOutputBlob->setInputWidths({width});
           pOutputBlob->setInputChannels({channel});
           pOutputBlob->setInputFracBits({fracBits});
           pOutputBlob->setOutputMemoryLocation(outputMemoryLocation);
           pOutputBlob->setInputMemoryLocations({outputMemoryLocation});
           pOutputBlob->setInputGroupsSeenBySource({groups});
           pOutputBlob->setCurrentNumberGroups(groups);
           pOutputBlob->setOutputFracBits(fracBits);
           pOutputBlob->setOutputHeight(height);
           pOutputBlob->setOutputWidth(width);
           pOutputBlob->setOutputChannel(channel);
           this->addLayer(pOutputBlob);
       }
    }

    void GraphFactory::setInputScatter(bool _flag)
    {
       flagInputScatter = _flag;
    }

    int GraphFactory::addLayer(std::shared_ptr<Layer> _pLayer, t_graph_output_tile_info* _pTileInfo)
    {
        t_graph_output_tile_info tempTileInfo;
        bool manualFlag = false;
        if (_pTileInfo != NULL) {
            //Tile configuration is applied.
            //Try to see if it is valid
            bool pass = _pLayer->cacheBoundaryCheck(*_pTileInfo);
            if (pass) {
                tempTileInfo = *_pTileInfo;
                manualFlag = true;
            }
            else {
                return -1;
            }
        }
        //return branch
        vecLayers.push_back(_pLayer);
        vecFlagManualTile.push_back(manualFlag);
        vecTileInfo.push_back(tempTileInfo);
        return 0;
    }

    std::unique_ptr<t_execution_graph> GraphFactory::generateGraph()
    {
        //std::cout <<"Hi."<<std::endl;
        std::unique_ptr<t_execution_graph> pGraph = std::unique_ptr<t_execution_graph>(new GraphRuntime::t_execution_graph);

        /*
         * Helper counters
        */
        int offsetIAMoverInstruction = 0;
        int offsetOAMoverInstruction = 0;
        int offsetWeightsDramBlock = 0;
        int offsetBiasesDramBlock = 0;

        //Counter of the number of compute layers;
        unsigned int countComputeLayer = 0;
        unsigned int idxLayer = 0;
                //Iterate through the layers
        for (const auto& pLayer: vecLayers)
        {
            LayerType layerType = pLayer->getLayerType();

            /*
             * Arguments for the instruction generator. May or may not be required.
             * Assign default values to everything, and fix-up later.
            */
            unsigned int numInputChannel0 = pLayer->getInputChannels().at(0);
            unsigned int numOutputChannels = pLayer->getOutputChannel();
            unsigned int numOutputChannelPerGroup = numOutputChannels / (pLayer->getCurrentNumberGroups());
            unsigned int numInputHeight0 = pLayer->getInputHeights().at(0);
            unsigned int numInputHeight1 = 0; //override
            unsigned int numInputWidth0 = pLayer->getInputWidths().at(0);
            unsigned int numInputWidth1 = 0; //override
            unsigned char numGroupCurrentLayer = pLayer->getCurrentNumberGroups();
            unsigned int numInputChannelPerGroup0 = numInputChannel0 / numGroupCurrentLayer;
            unsigned int numInputChannelPerGroup1 = 0; //override
            #if defined(SPW_SYSTEM)
            int numNZClustersPerPruningRange = PRUNE_RANGE_IN_CLUSTER; //override
            #endif


            unsigned char verticalBorderPadding = 0; //override this later
            unsigned char horizontalBorderPadding = 0; //override this later

            unsigned char kernelSize = 1; //override
            unsigned char stride = 1; //override
            unsigned int numInputChannel1 = 0; //override

            unsigned char inputHeightSPUnitSize = 1; //override if transpose conv
            unsigned char inputWidthSPUnitSize = 1; //override if transpose conv
            unsigned char sizeOutputTileFullWidthPerCol = 1; //override
            unsigned char numActiveColsPartialOutputTile = 1; //override
            unsigned char sizeOutputTileFullHeight = 1; //override

            //Arguments related to pSum binary-point shifting
            unsigned char outputShiftBits = 0; //override
            unsigned char outputShiftLeft = TRUE; //overrided

            //Arguments realted to input binary-point shifting
            unsigned int input0ShiftBits = 0; //override
            bool input0ShiftLeft = true;
            unsigned int input1ShiftBits = 0; //override
            bool input1ShiftLeft = true;

            //Input memory regions
            unsigned int input0MemoryRegion = 0; //override
            unsigned int input1MemoryRegion = 0;
            //Output memory regions
            unsigned int outputMemoryRegion = pLayer->getOutputMemoryLocation();

            //Memory strides
            signed int memIA0ColStride = 0;
            signed int memIA1ColStride = 0;
            signed int memOAColStride = 0;
            signed int memDramBlockFilterStride = 0; //override

            int offsetWeightsDramBlockIncrement = 0; //override
            int offsetBiasesDramBlockIncrement = 0; //override

            bool isComputeLayer = true; //override
            OPERATION op = ::CONVOLUTION; //override
            std::string layerName;

            //Latency estimation
            t_latency_info latInfo {
               .inputTransferLatency = 0,
               .weightTransferLatency = 0,
               .outputTransferLatency = 0,
               .computeLatency = 0,
               .computeLatencyWithOverhead = 0,
               .ddrLatency = 0,
               .totalLatency = 0
            };
            float weightSparsity = 0.0f;
            unsigned int ops = 0; //override
#if defined(HOST_DEBUG)
           cout <<"Generating for Layer(ID): "<<pLayer->getLayerID()<<endl;
#endif
            switch (layerType) {
                case CONVOLUTION: {
                    auto pLayerLocal = dynamic_pointer_cast<ConvLayer>(pLayer);
                    layerName = "conv_"+to_string(pLayerLocal->getLayerID());
                    numInputChannel1 = 0;
                    t_graph_output_tile_info tileInfo;
                    if (vecFlagManualTile.at(idxLayer) == true) {
                        tileInfo = vecTileInfo.at(idxLayer);
                        latInfo = pLayerLocal->deriveLatency(tileInfo);
                    }
                    else {
                        t_tile_pair tileConfig = calculateTileSizePerUnit(*pLayerLocal.get());
                        tileInfo = tileConfig.tileInfo;
                        latInfo = tileConfig.latencyInfo;
                    }

                    sizeOutputTileFullWidthPerCol = tileInfo.sizeOutputTileFullWidthPerCol;
                    numActiveColsPartialOutputTile = tileInfo.numActiveColsForPartialWidthTile;
                    sizeOutputTileFullHeight = tileInfo.sizeOutputTileFullHeight;
                    weightSparsity = pLayerLocal->getWeightSparsity();
                    ops = pLayerLocal->deriveOps();

                    kernelSize = pLayerLocal->getKernelSize();
                    stride = pLayerLocal->getKernelStride();
                    verticalBorderPadding = pLayerLocal->getInputBorderPadding();
                    horizontalBorderPadding = pLayerLocal->getInputBorderPadding();

                    //output precision control
                    int weightFracBits = pLayerLocal->getWeightFracBits();
                    int inputFracBits = pLayerLocal->getInputFracBits().at(0);
                    int outputFracBits = pLayerLocal->getOutputFracBits();
                    int pSumFracBits = weightFracBits + inputFracBits;
                    if (pSumFracBits > outputFracBits)
                    {
                        outputShiftBits = pSumFracBits - outputFracBits;
                        outputShiftLeft = FALSE;
                    }
                    else
                    {
                        outputShiftBits = outputFracBits - pSumFracBits;
                        outputShiftLeft = TRUE;
                    }

                    //Prepare the weights
                    //Convert the weights from float to fixed point
                    std::vector<float> floatWeights = pLayerLocal->getWeights();
                    std::vector<fixedPointNumber> fixedPointWeight;
#if defined(SPW_SYSTEM)
                    bool flagAdjustInputChannelSize =  false;
                    if ((pLayerLocal->getIsAfterInput() == true) && flagInputScatter) {
                        if (numInputChannel0 <= (CLUSTER_SIZE * PE_SIMD_SIZE)) {
                            numInputChannel0 = CLUSTER_SIZE * PE_SIMD_SIZE * PRUNE_RANGE_IN_CLUSTER;
                            flagAdjustInputChannelSize = true;
                            numInputChannelPerGroup0 = numInputChannel0;
                            if (numGroupCurrentLayer > 1) {
                                std::cout <<"Modifying the number of input channels for a convolution layer after input, but its input has more than one channel groups."
                                         <<std::endl;
                                throw;
                            }
                        }
                    }

                    fixedPointWeight.resize(
                                pLayerLocal->getOutputChannel()
                                * pLayerLocal->getKernelSize()
                                * pLayerLocal->getKernelSize()
                                * numInputChannel0);

                    if (flagAdjustInputChannelSize) {
                        unsigned int oCh=0, iKxK=0, iInCluster=0, iClusterInPR=0, iPR=0;
                        unsigned int KxK = pLayerLocal->getKernelSize() * pLayerLocal->getKernelSize();
                        for (unsigned int i=0; i<fixedPointWeight.size(); i++){
                            float floatVal;
                            int idxICInOriginal = iPR * CLUSTER_SIZE + iInCluster;
                            int idxInOriginal =
                                    oCh * KxK * pLayerLocal->getInputChannels().at(0)
                                    + iKxK * pLayerLocal->getInputChannels().at(0)
                                    + idxICInOriginal;
                            if ((iClusterInPR == 0) && (idxICInOriginal < pLayerLocal->getInputChannels().at(0))) {
                                floatVal = floatWeights.at(idxInOriginal);
                            }
                            else {
                                floatVal = 0.0f;
                            }

                            fixedPointWeight.at(i) = fixedPointNumber(floatVal, weightFracBits, 7-weightFracBits);

                            //Update the counters
                            iInCluster++;
                            if (iInCluster == CLUSTER_SIZE) {
                                iInCluster = 0;
                                iClusterInPR++;
                                if (iClusterInPR == PRUNE_RANGE_IN_CLUSTER) {
                                    iClusterInPR = 0;
                                    iPR++;
                                    if (iPR == PE_SIMD_SIZE) {
                                        iPR = 0;
                                        iKxK++;
                                        if (iKxK == KxK) {
                                            iKxK=0;
                                            oCh++;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else {
                        for (unsigned int i=0; i<fixedPointWeight.size(); i++) {
                             fixedPointWeight.at(i) = fixedPointNumber(floatWeights.at(i), weightFracBits, 7-weightFracBits);
                        }
                    }
#else
                    fixedPointWeight.resize(floatWeights.size());
                    {
                        for (unsigned int i=0; i<floatWeights.size(); i++)
                        {
                             fixedPointWeight.at(i) = fixedPointNumber(floatWeights.at(i), weightFracBits, 7-weightFracBits);
                        }
                    }
#endif

                    //Align and compress the weight tensor
                    std::shared_ptr<DeviceWeightTensor> pWeight;
                    #if defined(SPW_SYSTEM)
                        if (pLayerLocal->getWeightPruneClusterSize() != CLUSTER_SIZE)
                        {
                            std::cout <<"The accelerator's prune cluster size does not agree with the graph's prune cluster size."<<std::endl;
                            throw;
                        }

                        if (pLayerLocal->getWeightPruneRangeSizeInCluster() != PRUNE_RANGE_IN_CLUSTER)
                        {
                            std::cout <<"The accelerator's prune range size does not agree with the graph's prune range size."<<std::endl;
                            throw;
                        }
                        numNZClustersPerPruningRange = flagAdjustInputChannelSize ?
                            1 : std::ceil((1.0f - pLayerLocal->getWeightSparsity()) * PRUNE_RANGE_IN_CLUSTER);
                        pWeight.reset(new DeviceSpWTensor (
                                    fixedPointWeight,
                                    numOutputChannels, //_num3DTensors
                                    numInputChannelPerGroup0, //_inputChannel
                                    (unsigned char) kernelSize, //width
                                    (unsigned char) kernelSize, //height
                                    PE_SIMD_SIZE, //_peSimdSize
                                    CLUSTER_SIZE, //_clusterSize
                                    PRUNE_RANGE_IN_CLUSTER, //_numClustersInPruningRange
                                    numNZClustersPerPruningRange
                                ) );
                    #else
                        pWeight.reset( new DeviceWeightTensor (
                                            fixedPointWeight,
                                            numOutputChannels, //_num3DTensors
                                            numInputChannelPerGroup0, //_inputChannel
                                            (unsigned char) kernelSize, //width
                                            (unsigned char) kernelSize, //height
                                           PE_SIMD_SIZE, //_peSimdSize
                                           CLUSTER_SIZE //_clusterSize
                                        ));
                    #endif
                    //Filter stride
                    memDramBlockFilterStride =  pWeight->getDramBlocksInFilter();

                    //Prepare the fixed-point bias vector
                    std::shared_ptr<t_aligned_bias_vector> pBiasVector = std::make_shared<t_aligned_bias_vector>(numOutputChannels, 0x0);
                    bool hasBias = pLayerLocal->getBiasFlag();
                    if (hasBias)
                    {
                        if (pSumFracBits < 0) {
                            std::cout <<"[graph factory] pSumFracBits cannot be less than 0 when preparing bias!"<<std::endl;
                            throw;
                        }
                        float tolerance = 1.0f / (float)(1 << pSumFracBits);
                        std::fesetround(FE_TONEAREST); //round to even
                        std::vector<float> biasVector = pLayerLocal->getBiases();
                        for (int i=0; i<biasVector.size(); i++)
                        {
                            float bias = biasVector.at(i);
                            //pBiasVector->at(i) = (t_bias) (std::round(bias * (float) (1 << pSumFracBits )) );
                            t_bias quantBias = (t_bias) (std::nearbyint(bias * (float) (1 << pSumFracBits )) );
                            pBiasVector->at(i) = quantBias;
                            float diff = std::fabs(bias - (float) quantBias / (float) (1 << pSumFracBits));
                            //std::cout <<"Bias quantization error: "<<diff<<std::endl;
                            if (diff > tolerance) {
                                std::cout <<"Warning: quantization error of bias is larger than tolerance. "
                                         <<"Difference, tolerance: "<<diff<<" "<<tolerance<<std::endl;
                            }
                        }
                    }

                    //update weight count vecotr, bias count vector and offsets
                    pGraph->vecWeightDramBlockStart.push_back(offsetWeightsDramBlock);
                    pGraph->vecBiasStart.push_back(offsetBiasesDramBlock);
                    pGraph->pWeights.push_back(pWeight);
                    pGraph->pBiasVector.push_back(pBiasVector);
                    offsetWeightsDramBlockIncrement = memDramBlockFilterStride * numOutputChannels;
                    offsetBiasesDramBlockIncrement = numOutputChannels;

                    //Memory region
                    input0MemoryRegion = pLayer->getInputMemoryLocations().at(0);


                } //CONVOLUTION
                break;
                case ELTADD: {
                    op = ELT_ADD;
                    auto pLayerLocal = dynamic_pointer_cast<EltAddLayer>(pLayer);
                    numInputHeight1 = pLayerLocal->getInputHeights().at(1);
                    numInputWidth1 = pLayerLocal->getInputWidths().at(1);
                    numInputChannel1 = pLayerLocal->getInputChannels().at(1);
                    numInputChannelPerGroup1 = numInputChannel1 / numGroupCurrentLayer;

                    t_graph_output_tile_info tileInfo;
                    if (vecFlagManualTile.at(idxLayer) == true) {
                        tileInfo = vecTileInfo.at(idxLayer);
                        latInfo = pLayerLocal->deriveLatency(tileInfo);
                    }
                    else {
                        t_tile_pair tileConfig = calculateTileSizePerUnit(*pLayerLocal.get());
                        tileInfo = tileConfig.tileInfo;
                        latInfo = tileConfig.latencyInfo;
                    }

                    t_tile_pair tileConfig = calculateTileSizePerUnit(*pLayerLocal.get());
                    sizeOutputTileFullWidthPerCol = tileConfig.tileInfo.sizeOutputTileFullWidthPerCol;
                    numActiveColsPartialOutputTile = 1;
                    sizeOutputTileFullHeight = tileConfig.tileInfo.sizeOutputTileFullHeight;
                    ops = tileConfig.ops;

                    //Align input bits
                    int inputFracBits0 = pLayerLocal->getInputFracBits().at(0);
                    int inputFracBits1 = pLayerLocal->getInputFracBits().at(1);
                    int outputFracBits = pLayerLocal->getOutputFracBits();
                    int pSumFracBits = 0;
                    if (inputFracBits1 > inputFracBits0)
                    {
                        input0ShiftBits = inputFracBits1 - inputFracBits0;
                        input0ShiftLeft = TRUE;
                        input1ShiftBits = 0;
                        input1ShiftLeft = TRUE;
                        pSumFracBits = inputFracBits1;
                    }
                    else if (inputFracBits1 == inputFracBits0)
                    {
                        input0ShiftBits = 0;
                        input0ShiftLeft = TRUE;
                        input1ShiftBits = 0;
                        input1ShiftLeft = TRUE;
                        pSumFracBits = inputFracBits0;
                    }
                    else // inputFracBits1 < inputFracBits0
                    {
                        input0ShiftBits = 0;
                        input0ShiftLeft = TRUE;
                        input1ShiftBits = inputFracBits0 - inputFracBits1;
                        input1ShiftLeft = TRUE;
                        pSumFracBits = inputFracBits0;
                    }

                    //Figure out output bits
                    if (pSumFracBits > outputFracBits)
                    {
                        outputShiftBits = pSumFracBits - outputFracBits;
                        outputShiftLeft = FALSE;
                    }
                    else
                    {
                        outputShiftBits = outputFracBits - pSumFracBits;
                        outputShiftLeft = TRUE;
                    }

                    //Input memory regions
                    input0MemoryRegion = pLayer->getInputMemoryLocations().at(0);
                    input1MemoryRegion = pLayerLocal->getInputMemoryLocations().at(1);

                    isComputeLayer = true;
                    layerName = "eltadd_"+to_string(pLayerLocal->getLayerID());
                } //ELTADD
                break;
                case MAXPOOL: {
                    op = MAX_POOL;
                    auto pLayerLocal = dynamic_pointer_cast<MaxPoolLayer>(pLayer);
                    kernelSize = pLayerLocal->getKernelSize();
                    stride = pLayerLocal->getKernelStride();
                    verticalBorderPadding = pLayerLocal->getInputBorderPadding();
                    horizontalBorderPadding = pLayerLocal->getInputBorderPadding();
                    numActiveColsPartialOutputTile = 1;

                    //TODO: add precision stuff
                    int inputFracBits = pLayerLocal->getInputFracBits().at(0);
                    int outputFacBits = pLayerLocal->getOutputFracBits();
                    if (inputFracBits > outputFacBits)
                    {
                        outputFacBits = inputFracBits - outputFacBits;
                        outputShiftLeft = FALSE;
                    }
                    else
                    {
                        outputFacBits = outputFacBits - inputFracBits;
                        outputShiftLeft = TRUE;
                    }

                    //Input memory regions
                    input0MemoryRegion = pLayer->getInputMemoryLocations().at(0);

                    isComputeLayer = true;
                    layerName = "maxpool_"+to_string(pLayerLocal->getLayerID());
                    ops = pLayerLocal->deriveOps();
                    t_graph_output_tile_info tileInfoLocal;
                    latInfo = pLayerLocal->deriveLatency(tileInfoLocal);
                } //MAXPOOL
                break;
                case AVGPOOL:{
                    op = AVG_POOL;
                    auto pLayerLocal = dynamic_pointer_cast<AveragePoolLayer>(pLayer);
                    kernelSize = pLayerLocal->getKernelSize();
                    stride = pLayerLocal->getKernelStride();
                    verticalBorderPadding = pLayerLocal->getInputBorderPadding();
                    horizontalBorderPadding = pLayerLocal->getInputBorderPadding();
                    numActiveColsPartialOutputTile = 1;
                    ops = pLayerLocal->getOutputHeight() * pLayerLocal->getOutputWidth() * pLayerLocal->getOutputChannel();
                    //TODO: add precision stuff
                    //TODO: Modify the shift direction and amounts, to simulate the effect of the integer divisor
                    int divisorShift = (int) std::ceil(log2(pLayerLocal->getDivisor()));
                    assert ( (divisorShift >=0) && "Average pool divisor is less than 1");

                    int inputFracBits = pLayerLocal->getInputFracBits().at(0);
                    int outputFacBits = pLayerLocal->getOutputFracBits();

                    int tentativeLeftShift = outputFacBits - inputFracBits - divisorShift;
                    if (tentativeLeftShift >= 0)
                    {
                        outputShiftLeft = TRUE;
                        outputShiftBits = tentativeLeftShift;
                    }
                    else
                    {
                        outputShiftLeft = FALSE;
                        outputShiftBits = ((-1) * tentativeLeftShift);
                    }

                    //Input memory regions
                    input0MemoryRegion = pLayer->getInputMemoryLocations().at(0);

                    isComputeLayer = true;
                    layerName = "avgpool_"+to_string(pLayerLocal->getLayerID());
                    ops = pLayerLocal->deriveOps();
                    t_graph_output_tile_info tileInfoLocal;
                    latInfo = pLayerLocal->deriveLatency(tileInfoLocal);

                } //AVGPOOL
                break;
                case QUANT: {
                    isComputeLayer = false;
                    auto pLayerLocal = dynamic_pointer_cast<QuantLayer>(pLayer);
                    int numChannels = numOutputChannels;
//#if defined(SPW_SYSTEM)
//                    if (flagInputScatter)
//                    {
//                        if (numOutputChannels < (PE_SIMD_SIZE * CLUSTER_SIZE))
//                        {
//                            numChannels = PE_SIMD_SIZE * CLUSTER_SIZE * PRUNE_RANGE_IN_CLUSTER;
//                        }
//                    }
//#endif
                    //Add an input
                    pGraph->vecInputInfo.emplace_back(
                                t_blob_info{
                                    .memoryRegionID=pLayerLocal->getOutputMemoryLocation(),
                                    .channel = numOutputChannels,
                                    .height=pLayerLocal->getOutputHeight(),
                                    .width=pLayerLocal->getOutputWidth(),
                                    .stripStrideSeenBySource= DIVIDE_CEIL(numChannels, ACTIVATION_BURST_SIZE_BYTE) * ACTIVATION_BURST_SIZE_BYTE,
                                    .numFracBits=pLayerLocal->getOutputFracBits(),
                                    .blobName="quant_"+to_string(pLayerLocal->getLayerID()),
                                    .flagInputScatter = flagInputScatter
                                    }
                                );
                } //QUANT
                break;
                case DEQUANT: {
                    isComputeLayer = false;
                    auto pLayerLocal = dynamic_pointer_cast<DeQuantLayer>(pLayer);
                    //Add an output
                    pGraph->vecOutputInfo.emplace_back(
                                t_blob_info{
                                    .memoryRegionID=pLayerLocal->getInputMemoryLocations().at(0),
                                    .channel = numOutputChannels,
                                    .height=pLayerLocal->getInputHeights().at(0),
                                    .width=pLayerLocal->getInputWidths().at(0),
                                    .stripStrideSeenBySource= DIVIDE_CEIL(numOutputChannels, ACTIVATION_BURST_SIZE_BYTE) * ACTIVATION_BURST_SIZE_BYTE,
                                    .numFracBits=pLayerLocal->getInputFracBits().at(0),
                                    .blobName="dequant_"+to_string(pLayerLocal->getLayerID())
                                    }
                                );
                } //DEQUANT
                break;
            } //switch

#if defined(HOST_DEBUG)
            std::cout <<"Processing layer: "<<layerName<<std::endl;
#endif
            // Generate instruction if this is a computation layer
            if (isComputeLayer == true)
            {
                unsigned int inputHeightSPSize = inputHeightSPUnitSize*(numInputHeight0-1) + 1;
                unsigned int inputWidthSPSize = inputWidthSPUnitSize*(numInputWidth0-1) + 1;
                char instEnableRelu = pLayer->getOutputReluFlag() ? TRUE : FALSE;

                //strides
                //TODO: Need to add support for concatentation if we expand the work to YOLO!
                int numChannelsPerGroupSeenBySource0 =
                        numInputChannel0 / pLayer->getInputGroupsSeenBySource().at(0);
                memIA0ColStride =
                        ((pLayer->getInputGroupsSeenBySource().at(0)) - 1) * numChannelsPerGroupSeenBySource0
                        + DIVIDE_CEIL(numChannelsPerGroupSeenBySource0, ACTIVATION_BURST_SIZE_BYTE) * ACTIVATION_BURST_SIZE_BYTE;

                memOAColStride = (numGroupCurrentLayer - 1) * numOutputChannelPerGroup
                        + DIVIDE_CEIL(numOutputChannelPerGroup, ACTIVATION_BURST_SIZE_BYTE) * ACTIVATION_BURST_SIZE_BYTE;

                if (pLayer->getInputMemoryLocations().size() > 1)
                {
                    int numChannelsPerGroupSeenBySource1 =
                            numInputChannel1 / pLayer->getInputGroupsSeenBySource().at(1);
                    memIA1ColStride =
                            ((pLayer->getInputGroupsSeenBySource().at(0)) - 1) * numChannelsPerGroupSeenBySource1
                            + DIVIDE_CEIL(numChannelsPerGroupSeenBySource1, ACTIVATION_BURST_SIZE_BYTE) * ACTIVATION_BURST_SIZE_BYTE;
                }
                else
                {
                    memIA1ColStride = 0;
                }

                //Synchornization flags
                //The first layer: the IAMover doesn't need to wait
                //The first layer: the OAMover doesn't need to wait
                unsigned char flagTensorSync = (countComputeLayer==0) ? 0x00 : 0x01;

                t_aligned_ia_mover_instruction_vector vecIAMoverInstruction;
                t_aligned_oa_mover_instruction_vector vecOAMoverInstruction;
                instruction_generator (//Type of the operation
                        //OPERATION op,
                        op,

                        //t_aligned_ia_mover_instruction_vector & vecIAMoverInstruction,
                        vecIAMoverInstruction,
                        //t_aligned_oa_mover_instruction_vector & vecOAMoverInstruction,
                        vecOAMoverInstruction,
                        //t_aligned_ia_tile_controller_instruction_vector & vecIATileControlInstruction,
                        pGraph->vecIATileControllerInstruction,
                        //t_aligned_oa_tile_controller_instruction_vector & vecOATileControlInstruction,
                        pGraph->vecOATileControllerInstruction,
                        //t_aligned_weight_mover_instruction_vector & vecWeightMoverInstruction,
                        pGraph->vecWMoverInstruction,
                        //t_aligned_misc_instruction_vector & vecMiscInstruction,
                        pGraph->vecMiscInstruction,

                        //bool flagIA0ShiftLeft,
                        input0ShiftLeft,
                        //unsigned int numIA0ShiftAmount,
                        input0ShiftBits,
                        //bool flagIA1ShiftLeft,
                        input1ShiftLeft,
                        //unsigned int numIA1ShiftAmount,
                        input1ShiftBits,

                        //signed int memIA0DramBlockStartIndex,
                        input0MemoryRegion * MEM_ACTIVATION_REGION_SIZE_PER_SLICE,
                        //signed int memIA1DramBlockStartIndex,
                        input1MemoryRegion * MEM_ACTIVATION_REGION_SIZE_PER_SLICE,

                        //signed int memOADramBlockStartIndex,
                        outputMemoryRegion * MEM_ACTIVATION_REGION_SIZE_PER_SLICE,

                        //memWeightDramBlockStartIndex,
                        offsetWeightsDramBlock,
                        //signed int memBiasStartIndex,
                        offsetBiasesDramBlock,

                        //signed int memIA0ColStride,
                        memIA0ColStride,
                        //signed int memIA0RowStride,
                        memIA0ColStride * numInputWidth0,

                        //signed int memIA1ColStride,
                        memIA1ColStride,
                        //signed int memIA1RowStride,
                        memIA1ColStride * numInputWidth1,

                        //signed int memOADColStride,
                        memOAColStride,

                        //signed int memWeightDramBlockFilterStride,
                        memDramBlockFilterStride,

                        //unsigned char flagTensorSync,
                        flagTensorSync,

                        //unsigned char flagRelu,
                        instEnableRelu,
                        //unsigned char outputShiftBits,
                        outputShiftBits,
                        //unsigned char flagOutputShiftLeft,
                        outputShiftLeft,

                        //unsigned short inputSPWidth,
                        inputWidthSPSize,
                        //unsigned short inputSPHeight,
                        inputHeightSPSize,

                        //unsigned char inputSPWidthUnit,
                        inputWidthSPUnitSize,
                        //unsigned char inputSPHeightUnit,
                        inputHeightSPUnitSize,

                        //unsigned char inputWidthPadding,
                        horizontalBorderPadding,
                        //unsigned char inputHeightPadding,
                        verticalBorderPadding,

                        //unsigned char kernelSize,
                        kernelSize,
                        //unsigned char kernelStride,
                        stride,

                        #if defined(SPW_SYSTEM)
                            //unsigned int numNZClustersInPruningRange,
                            numNZClustersPerPruningRange,
                       #endif

                        //unsigned char _sizeOutputTileFullHeight,
                        sizeOutputTileFullHeight,
                        //unsigned char _sizeOutputTileFullWidthPerCol,
                        sizeOutputTileFullWidthPerCol,
                        //unsigned char _numActiveColsPartialOutputTile,
                        numActiveColsPartialOutputTile,
                        //unsigned short numInputChannels0,
                        numInputChannel0,
                        //unsigned short numInputChannels1,
                        numInputChannel1,
                        //unsigned short numGroupsCurrentLayer,
                        numGroupCurrentLayer,
                        //unsigned short numOutputChannels,
                        numOutputChannels
                        );                   {
                       std::copy(vecIAMoverInstruction.begin(),
                                 vecIAMoverInstruction.end(),
                                 std::back_inserter(pGraph->vecIAMoverInstruction)
                                 );
                       std::copy(vecOAMoverInstruction.begin(),
                                 vecOAMoverInstruction.end(),
                                 std::back_inserter(pGraph->vecOAMoverInstruction)
                                 );
                       int numIAMoverInstructions = vecIAMoverInstruction.size();
                       int numOAMoverInstructions = vecOAMoverInstruction.size();
                       pGraph->vecLayerInfo.emplace_back(
                           GraphRuntime::t_layer_info {.layerName=layerName,
                            .offsetIAMoverInstruction=offsetIAMoverInstruction,
                            .numIAMoverInstruction=numIAMoverInstructions,
                            .offsetOAMoverInstruction=offsetOAMoverInstruction,
                            .numOAMoverInstructions=numOAMoverInstructions,
                           .outputTileHeight = sizeOutputTileFullHeight,
                           .outputTileWidthPerCol = sizeOutputTileFullWidthPerCol,
                           .numActiveColsPartialOutputTile = numActiveColsPartialOutputTile,
                           .inputTransferLatency = latInfo.inputTransferLatency,
                           .weightTransferLatency = latInfo.weightTransferLatency,
                           .outputTransferLatency = latInfo.outputTransferLatency,
                           .rawComputeLatency = latInfo.computeLatency,
                           .computeLatencyWithOverhead = latInfo.computeLatencyWithOverhead,
                           .expectedLatency = latInfo.totalLatency,
                           .isComputeBound = latInfo.isComputeBound ? 1 : 0,
                           .ddrLatency = latInfo.ddrLatency,
                           .weightSparsity = weightSparsity,
                           .ops = ops
                            });
                       offsetIAMoverInstruction += numIAMoverInstructions;
                       offsetOAMoverInstruction += numOAMoverInstructions;
                    }

                    offsetWeightsDramBlock += offsetWeightsDramBlockIncrement;
                    offsetBiasesDramBlock += offsetBiasesDramBlockIncrement;
                    #if defined(SPARSE_SYSTEM)
                        offsetWeightTBCount += offsetWeightTBCountIncrement;
                    #endif

                    assert(offsetBiasesDramBlock <= MAX_DRAM_BYTE_INPUT_WEIGHT);
                    #if defined(SPARSE_SYSTEM)
                        assert(offsetWeightTBCount <= MAX_DRAM_BYTE_INPUT_WEIGHT_SB_COUNT);
                    #endif

                    countComputeLayer++;
            } // if compute layer
#if defined(HOST_DEBUG)
            std::cout <<"Finished layer: "<<layerName<<std::endl;
#endif
            idxLayer++;
        } // for layer

        return pGraph;
    }
}

t_tile_pair calculateTileSizePerUnit(ConvLayer& _convLayer)
{
    unsigned int maxOutputTileWidthPerCol =
            (MAX_INPUT_TILE_WIDTH_PER_COL - _convLayer.getKernelSize()) / _convLayer.getKernelStride() + 1;
    unsigned int maxOutputTileHeight =
            (MAX_INPUT_TILE_HEIGHT - _convLayer.getKernelSize()) / _convLayer.getKernelStride() + 1;

    //Generate the starting point of for the width of the tile per col
    unsigned int tempOutputTileWidthPerCol = 1 + (_convLayer.getOutputWidth()-1)/ PE_COLS;
    unsigned int outputTileWidthPerCol = tempOutputTileWidthPerCol < maxOutputTileWidthPerCol ?
        tempOutputTileWidthPerCol : maxOutputTileWidthPerCol;

    unsigned int outputHeight = _convLayer.getOutputHeight();
    unsigned int outputWidth = _convLayer.getOutputWidth();
    unsigned int outputChannels = _convLayer.getOutputChannel();

    t_graph_output_tile_info bestTileInfo;
    unsigned int minLatency = 0xFFFFFFFF;
    t_latency_info bestLatInfo;
    bool isComputeBound = true;

    for (;outputTileWidthPerCol > 0; outputTileWidthPerCol--)
    {
        //Generate a candidate tile configuration
        unsigned int sizeOutputFullTileHeightTemp =
                OA_CACHE_DEPTH * ACTIVATION_BURST_SIZE_BYTE /
                (outputTileWidthPerCol * (DIVIDE_CEIL(outputChannels, PE_ROWS_PER_GROUP) * PE_ROWS_PER_GROUP));
        sizeOutputFullTileHeightTemp = sizeOutputFullTileHeightTemp < outputHeight ?
                    sizeOutputFullTileHeightTemp : outputHeight;
        unsigned int sizeOutputFullTileHeight = sizeOutputFullTileHeightTemp < maxOutputTileHeight?
                sizeOutputFullTileHeightTemp : maxOutputTileHeight;

         for (; sizeOutputFullTileHeight > 0; sizeOutputFullTileHeight--)
         {
             t_graph_output_tile_info candidateTileInfo =
                     deriveConvOutputTileShape(
                            outputHeight,
                            outputWidth,
                            sizeOutputFullTileHeight,
                            outputTileWidthPerCol,
                            true //isConv
                         );

             bool passCacheRequirement = _convLayer.cacheBoundaryCheck(candidateTileInfo);

             if (passCacheRequirement == true)
             {
                t_latency_info tileLat = _convLayer.deriveLatency(candidateTileInfo);

                if (tileLat.totalLatency < (minLatency * 0.9f))
                {
                    minLatency = tileLat.totalLatency;
                    bestTileInfo = candidateTileInfo;
                    isComputeBound = (tileLat.totalLatency <= tileLat.computeLatencyWithOverhead);
                    bestLatInfo = tileLat;
                }

             } // if passCacheRequirement == true
         } //for over possible tile height

    } //for over possible tile width

    if (minLatency == 0xFFFFFFFF)
    {
        std::cout <<"Warning: Cannot find a suitable tile configuration for Conv Layer "<<_convLayer.getLayerID()<<std::endl;
        throw;
    }
    unsigned int ops = _convLayer.deriveOps();

    t_tile_pair result = {.tileInfo = bestTileInfo,
                          .latencyInfo = bestLatInfo,
                          .ops = ops
                         };
    return result;
}

t_tile_pair calculateTileSizePerUnit(EltAddLayer &_eltAddLayer)
{
    unsigned int maxOutputTileHeight = MAX_OUTPUT_TILE_HEIGHT;

    //Search all possible solutions tile solution exhautively.
    unsigned int outputTileWidthPerCol = MAX_OUTPUT_TILE_WIDTH_PER_COL;

    unsigned int outputHeight = _eltAddLayer.getOutputHeight();
    unsigned int outputWidth = _eltAddLayer.getOutputWidth();

    //Generate a candidate tile configuration
    unsigned int sizeOutputFullTileHeight = outputHeight < maxOutputTileHeight?
            outputHeight : maxOutputTileHeight;
     t_graph_output_tile_info candidateTileInfo =
             deriveConvOutputTileShape(
                    outputHeight,
                    outputWidth,
                    sizeOutputFullTileHeight,
                    outputTileWidthPerCol,
                    false //isConv
                 );
    t_latency_info latInfo = _eltAddLayer.deriveLatency(candidateTileInfo);
    int ops = _eltAddLayer.deriveOps();

    t_tile_pair result = {.tileInfo = candidateTileInfo,
                          .latencyInfo = latInfo,
                          .ops = ops};
    return result;
}
