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
typedef struct {
    t_graph_output_tile_info tileInfo;
    int     inputTransferLatency;
    int     weightTransferLatency;
    int     outputTransferLatency;
    int     computeLatency;
    int     computeLatencyWithOverhead;
    unsigned int latency;
    bool flagComputeBound;
    unsigned int ops;
} t_tile_pair;
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
    GraphFactory::GraphFactory(std::string _traceFileName, std::string _parameterFileName)
    {
       typedef YAML::Node YN;
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
       for (int i=0; i<traceNodes.size(); i++) {
           YAML::Node traceLayer = traceNodes[i];
           //Hash the string to a enum type, so we can use switch-case statement
           //See https://stackoverflow.com/a/650307
           LayerType opType = hashLayerTypeString(
                       traceLayer["operationType"].as<string>());

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
    }

    std::unique_ptr<t_execution_graph> GraphFactory::generateGraph()
    {
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
            unsigned int numOutputChannelPerGroup = numOutputChannels / (pLayer->getNextNumberGroups());
            unsigned int numOutputWidth = pLayer->getOutputWidth();
            unsigned int numOutputHeight = pLayer->getOutputHeight();
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
            unsigned int estimatedLatency = 0; //override
            unsigned int inputTransferLatency = 0;
            unsigned int weightTransferLatency = 0;
            unsigned int outputTransferLatency = 0;
            unsigned int computeLatency = 0;
            unsigned int computeLatencyWithOverhead = 0;
            bool flagComputeBound = false; //override
            unsigned int ops = 0; //override
#if defined(HOST_DEBUG)
           cout <<"Generating for Layer(ID): "<<pLayer->getLayerID()<<endl;
#endif
            switch (layerType) {
                case CONVOLUTION: {
                    auto pLayerLocal = dynamic_pointer_cast<ConvLayer>(pLayer);
                    layerName = "conv_"+to_string(pLayerLocal->getLayerID());
                    numInputChannel1 = 0;
                    t_tile_pair tileConfig = calculateTileSizePerUnit(*pLayerLocal.get());
                    sizeOutputTileFullWidthPerCol = tileConfig.tileInfo.sizeOutputTileFullWidthPerCol;
                    numActiveColsPartialOutputTile = tileConfig.tileInfo.numActiveColsForPartialWidthTile;
                    sizeOutputTileFullHeight = tileConfig.tileInfo.sizeOutputTileFullHeight;
                    estimatedLatency = tileConfig.latency;
                    inputTransferLatency = tileConfig.inputTransferLatency;
                    weightTransferLatency = tileConfig.weightTransferLatency;
                    outputTransferLatency = tileConfig.outputTransferLatency;
                    computeLatency = tileConfig.computeLatency;
                    computeLatencyWithOverhead = tileConfig.computeLatencyWithOverhead;
                    flagComputeBound = tileConfig.flagComputeBound;
                    ops = tileConfig.ops;

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
                    fixedPointWeight.resize(floatWeights.size());
                    {
                        for (unsigned int i=0; i<floatWeights.size(); i++)
                        {
                             fixedPointWeight.at(i) = fixedPointNumber(floatWeights.at(i), weightFracBits, 7-weightFracBits);
                        }
                    }

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
                        numNZClustersPerPruningRange =
                            std::ceil((1.0f - pLayerLocal->getWeightSparsity()) * PRUNE_RANGE_IN_CLUSTER);
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
                    std::shared_ptr<t_aligned_short_vector> pBiasVector = std::make_shared<t_aligned_short_vector>(numOutputChannels, 0x0);
                    bool hasBias = pLayerLocal->getBiasFlag();
                    if (hasBias)
                    {
                        std::fesetround(FE_TONEAREST); //round to even
                        std::vector<float> biasVector = pLayerLocal->getBiases();
                        for (int i=0; i<biasVector.size(); i++)
                        {
                            float bias = biasVector.at(i);
                            pBiasVector->at(i) = (t_bias) (std::nearbyint(bias * (float) (1 << pSumFracBits )) );
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

                    t_tile_pair tileConfig = calculateTileSizePerUnit(*pLayerLocal.get());
                    sizeOutputTileFullWidthPerCol = tileConfig.tileInfo.sizeOutputTileFullWidthPerCol;
                    numActiveColsPartialOutputTile = 1;
                    sizeOutputTileFullHeight = tileConfig.tileInfo.sizeOutputTileFullHeight;
                    inputTransferLatency = tileConfig.inputTransferLatency;
                    weightTransferLatency = 0;
                    outputTransferLatency = tileConfig.outputTransferLatency;
                    computeLatency = tileConfig.computeLatency;
                    estimatedLatency = tileConfig.latency;
                    flagComputeBound = tileConfig.flagComputeBound;
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
                    numActiveColsPartialOutputTile = numOutputWidth % MISC_COLS;

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
                } //MAXPOOL
                break;
                case AVGPOOL:{
                    op = AVG_POOL;
                    auto pLayerLocal = dynamic_pointer_cast<AveragePoolLayer>(pLayer);
                    kernelSize = pLayerLocal->getKernelSize();
                    stride = pLayerLocal->getKernelStride();
                    verticalBorderPadding = pLayerLocal->getInputBorderPadding();
                    horizontalBorderPadding = pLayerLocal->getInputBorderPadding();
                    numActiveColsPartialOutputTile = numOutputWidth % MISC_COLS;
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

                } //AVGPOOL
                break;
                case QUANT: {
                    isComputeLayer = false;
                    auto pLayerLocal = dynamic_pointer_cast<QuantLayer>(pLayer);
                    //Add an input
                    pGraph->vecInputInfo.emplace_back(
                                t_blob_info{
                                    .memoryRegionID=pLayerLocal->getOutputMemoryLocation(),
                                    .channel = numOutputChannels,
                                    .height=pLayerLocal->getOutputHeight(),
                                    .width=pLayerLocal->getOutputWidth(),
                                    .stripStrideSeenBySource= DIVIDE_CEIL(numOutputChannels, ACTIVATION_BURST_SIZE_BYTE) * ACTIVATION_BURST_SIZE_BYTE,
                                    .numFracBits=pLayerLocal->getOutputFracBits(),
                                    .blobName="quant_"+to_string(pLayerLocal->getLayerID())
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

            std::cout <<"Processing layer: "<<layerName<<std::endl;

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
                           .inputTransferLatency = inputTransferLatency,
                           .weightTransferLatency = weightTransferLatency,
                           .outputTransferLatency = outputTransferLatency,
                           .rawComputeLatency = computeLatency,
                           .computeLatencyWithOverhead = computeLatencyWithOverhead,
                           .expectedLatency = estimatedLatency,
                           .isComputeBound = flagComputeBound ? 1 : 0,
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
            std::cout <<"Finished layer: "<<layerName<<std::endl;
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
    unsigned int outputChannelsPerNextGroup = _convLayer.getOutputChannel() / _convLayer.getNextNumberGroups();
    unsigned int outputChannelsPerCurrentGroup = _convLayer.getOutputChannel() / _convLayer.getCurrentNumberGroups();
    unsigned int inputChannels = _convLayer.getInputChannels().at(0);
    unsigned int inputChannelsPerGroup = inputChannels / _convLayer.getCurrentNumberGroups();

    t_graph_output_tile_info bestTileInfo;
    unsigned int minLatency = 0xFFFFFFFF;
    bool solutionExists = false;
    int     bestInputTransferLatency = 0x0;
    int     bestWeightTransferLatency = 0x0;
    int     bestOutputTransferLatency = 0x0;
    int     bestRawComputeLatency = 0x0;
    int     bestCompuateLatencyWithOverhead = 0x0;
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

             //Check that the tile configuration can work wit the IA/OA cache limit
             unsigned int sizeInputTileFullHeight = deriveConvInputDimension1D(
                            sizeOutputFullTileHeight,
                            _convLayer.getKernelSize(),
                            _convLayer.getKernelStride()
                         );
             unsigned int sizeInputTileFullWidthPerCol = deriveConvInputDimension1D(
                            candidateTileInfo.sizeOutputTileFullWidthPerCol,
                            _convLayer.getKernelSize(),
                            _convLayer.getKernelStride()
                         );
             unsigned int sizeInputTilePartialWidthPerCol = deriveConvInputDimension1D(
                            candidateTileInfo.sizeOutputTilePartialWidthPerCol,
                            _convLayer.getKernelSize(),
                            _convLayer.getKernelStride()
                         );
             //TODO: change the arguments passed to ia_cache_boundary_check
             //In terms of ACTIVATION_DRAM_BLOCKS
             int iaCachePerColRequirement =
                     ia_cache_boundary_check(
                         sizeInputTileFullHeight,
                         sizeInputTileFullWidthPerCol,
                         DIVIDE_CEIL(inputChannels, ACTIVATION_BURST_SIZE_BYTE)
                         );
             int iaCachePerPartialColRequirement =
                     ia_cache_boundary_check(
                         sizeInputTileFullHeight,
                         sizeInputTilePartialWidthPerCol,
                         DIVIDE_CEIL(inputChannels, ACTIVATION_BURST_SIZE_BYTE)
                         );
             //TODO: Change the arguments to the OA cache requirement checker
             int oaCachePerColRequirement =
                     oa_cache_boundary_check(
                         candidateTileInfo.sizeOutputTileFullHeight,
                         candidateTileInfo.sizeOutputTileFullWidthPerCol,
                         outputChannelsPerCurrentGroup
                         );
             int oaCachePerPartialColRequirement =
                     oa_cache_boundary_check(
                         candidateTileInfo.sizeOutputTileFullHeight,
                         candidateTileInfo.sizeOutputTilePartialWidthPerCol,
                         outputChannelsPerCurrentGroup
                         );
             bool passCacheRequirement =
                     (iaCachePerColRequirement <= IA_CACHE_DEPTH)
                     && (iaCachePerPartialColRequirement <= IA_CACHE_DEPTH)
                     && (oaCachePerColRequirement <= OA_CACHE_DEPTH)
                     && (oaCachePerPartialColRequirement <= OA_CACHE_DEPTH)
                     && (sizeInputTileFullHeight <= MAX_INPUT_TILE_HEIGHT)
                     && (sizeInputTileFullWidthPerCol <= MAX_INPUT_TILE_WIDTH_PER_COL)
                     && (sizeInputTilePartialWidthPerCol <= MAX_INPUT_TILE_WIDTH_PER_COL);

             if (passCacheRequirement == true)
             {
                unsigned int computeLatency = deriveConvComputationLatency(
                            candidateTileInfo,
                            outputChannelsPerCurrentGroup,
                            inputChannelsPerGroup,
                            _convLayer.getCurrentNumberGroups(),
                            _convLayer.getKernelSize()
                            );

                unsigned int inputLatency = deriveConvInputTransferLatency(
                            candidateTileInfo,
                            inputChannelsPerGroup,
                            _convLayer.getCurrentNumberGroups(),
                            _convLayer.getKernelSize(),
                            _convLayer.getKernelStride()
                            );
                //maxLatency = inputLatency > maxLatency ? inputLatency : maxLatency;

                unsigned int outputLatency = deriveOutputTransferLatency(
                            candidateTileInfo,
                            outputHeight,
                            outputChannelsPerCurrentGroup,
                            _convLayer.getCurrentNumberGroups()
                            );
                //maxLatency = outputLatency > maxLatency ? outputLatency : maxLatency;

                unsigned int weightLatency = deriveConvWeightTransferLatency(
                            candidateTileInfo,
                            inputChannelsPerGroup,
                            outputChannelsPerCurrentGroup,
                            _convLayer.getCurrentNumberGroups(),
                            _convLayer.getKernelSize()
                            );

                unsigned int firstTileInputLatency = deriveFirstTileConvInputTransferLatency
                        (
                            candidateTileInfo,
                            inputChannelsPerGroup,
                            _convLayer.getKernelSize(),
                            _convLayer.getKernelStride()
                        );

                unsigned int firstTileComputeLatency = deriveFirstTileConvComputationLatency
                        (
                            candidateTileInfo,
                            outputChannelsPerCurrentGroup,
                            inputChannelsPerGroup,
                            _convLayer.getKernelSize()
                         );

                unsigned int lastTileOutputLatency = deriveLastTileOutputTransferLatency
                        (
                            candidateTileInfo,
                            outputChannelsPerCurrentGroup
                        );

                unsigned int computeLatencyWithOverhead =
                        computeLatency + firstTileInputLatency + lastTileOutputLatency;

                unsigned int weightBoundedLatency =
                        weightLatency + firstTileInputLatency + lastTileOutputLatency;
                //maxLatency = weightLatency > maxLatency ? weightLatency : maxLatency;

               unsigned int adjustedOutputLatency =
                        firstTileInputLatency + firstTileComputeLatency + outputLatency;

               //Get the input/output activation bound
                unsigned int actualTransferLatency = inputLatency;
                if (actualTransferLatency < weightLatency)
                {
                    actualTransferLatency = weightLatency;
                }
                if (actualTransferLatency <adjustedOutputLatency)
                {
                    actualTransferLatency = adjustedOutputLatency;
                }

                //Pick from the compute bound, weight bound, and input/output activation transfer bound
                unsigned int maxLatency = computeLatencyWithOverhead;
                if (maxLatency < weightBoundedLatency)
                {
                    maxLatency = weightBoundedLatency;
                }
                if (maxLatency < actualTransferLatency)
                {
                    maxLatency = actualTransferLatency;
                }

                if (maxLatency < (minLatency * 0.9f))
                {
                    minLatency = maxLatency;
                    bestTileInfo = candidateTileInfo;
                    isComputeBound = (maxLatency <= computeLatencyWithOverhead);
                    bestRawComputeLatency = computeLatency;
                    bestCompuateLatencyWithOverhead = computeLatencyWithOverhead;
                    bestInputTransferLatency = inputLatency;
                    bestWeightTransferLatency = weightLatency;
                    bestOutputTransferLatency = outputLatency;
                }

             } // if passCacheRequirement == true
         } //for over possible tile height

    } //for over possible tile width

    if (minLatency == 0xFFFFFFFF)
    {
        std::cout <<"Warning: Cannot find a suitable tile configuration for Conv Layer "<<_convLayer.getLayerID()<<std::endl;
        throw;
    }
    unsigned int ops = _convLayer.getCurrentNumberGroups() * (
                    outputChannelsPerCurrentGroup
                        * outputHeight * outputWidth
                        * _convLayer.getKernelSize() * _convLayer.getKernelSize() * inputChannelsPerGroup
                        *2
                );
    t_tile_pair result = {.tileInfo = bestTileInfo,
                          .inputTransferLatency = bestInputTransferLatency,
                          .weightTransferLatency = bestWeightTransferLatency,
                          .outputTransferLatency = bestOutputTransferLatency,
                          .computeLatency = bestRawComputeLatency,
                          .computeLatencyWithOverhead = bestCompuateLatencyWithOverhead,
                          .latency = minLatency,
                          .flagComputeBound=isComputeBound,
                          .ops = ops
                         };
    return result;
}

//t_tile_pair calculateTileSizePerUnit(EltAddLayer &_eltAddLayer)
//{
//    unsigned int maxOutputTileHeight = MAX_OUTPUT_TILE_HEIGHT;

//    //Search all possible solutions tile solution exhautively.
//    unsigned int outputTileWidthPerCol = MAX_OUTPUT_TILE_WIDTH_PER_COL;

//    unsigned int outputHeight = _eltAddLayer.getOutputHeight();
//    unsigned int outputWidth = _eltAddLayer.getOutputWidth();
//    unsigned int outputChannels = _eltAddLayer.getOutputChannel();
//    unsigned int outputChannelsPerNextGroup = _eltAddLayer.getOutputChannel() / _eltAddLayer.getNextNumberGroups();

//    unsigned int numClustersPerOutputStrip = 1 + (outputChannels-1) / CLUSTER_SIZE;

//    t_graph_output_tile_info bestTileInfo;
//    unsigned int minLatency = 0xFFFFFFFF;
//    int     bestInputTransferLatency = 0x0;
//    int     bestOutputTransferLatency = 0x0;
//    while (outputTileWidthPerCol > 0)
//    {
//        //Generate a candidate tile configuration
//        unsigned int sizeOutputFullTileHeightTemp =
//                OA_CACHE_DEPTH / (outputTileWidthPerCol*numClustersPerOutputStrip);
//        sizeOutputFullTileHeightTemp = sizeOutputFullTileHeightTemp < outputHeight ?
//                    sizeOutputFullTileHeightTemp : outputHeight;
//        unsigned int sizeOutputFullTileHeight = sizeOutputFullTileHeightTemp < maxOutputTileHeight?
//                sizeOutputFullTileHeightTemp : maxOutputTileHeight;
//         t_graph_output_tile_info candidateTileInfo =
//                 deriveConvOutputTileShape(
//                        outputHeight,
//                        outputWidth,
//                        sizeOutputFullTileHeight,
//                        outputTileWidthPerCol,
//                        false //isConv
//                     );

//         //TODO: Remove the call to the OA cache requirement check
//         int oaCachePerColRequirementInWords =
//                 oa_cache_boundary_check(
//                     candidateTileInfo.sizeOutputTileFullHeight,
//                     candidateTileInfo.sizeOutputTileFullWidthPerCol,
//                     outputChannels
//                     );
//         int oaCachePerPartialColRequirementInWords =
//                 oa_cache_boundary_check(
//                     candidateTileInfo.sizeOutputTileFullHeight,
//                     candidateTileInfo.sizeOutputTilePartialWidthPerCol,
//                     outputChannels
//                     );
//         bool passCacheRequirement =
//                (oaCachePerColRequirementInWords <= (OA_CACHE_DEPTH*CLUSTER_SIZE))
//                 && (oaCachePerPartialColRequirementInWords <= (OA_CACHE_DEPTH*CLUSTER_SIZE));

//         if (passCacheRequirement == true)
//         {

//            //Times 3 to account for two input transfers, and one output transfer for each
//            //block of addition
//            unsigned int inputLatency = deriveConvInputTransferLatency(
//                        candidateTileInfo,
//                        outputChannels,
//                        1,
//                        1,
//                        1
//                        );
//            unsigned int outputLatency = deriveOutputTransferLatency(
//                        candidateTileInfo,
//                        outputHeight,
//                        outputChannelsPerNextGroup,
//                        _eltAddLayer.getNextNumberGroups()
//                        );
//            if (outputLatency < minLatency)
//            {
//                minLatency = outputLatency;
//                bestTileInfo = candidateTileInfo;
//                bestInputTransferLatency = inputLatency;
//                bestOutputTransferLatency = outputLatency;
//                break;
//            }
//            outputTileWidthPerCol--;
//         } // if passCacheRequirement == true
//         else
//         {
//             if (maxOutputTileHeight < outputTileWidthPerCol)
//             {
//                 outputTileWidthPerCol--;
//             }
//             else
//             {
//                 maxOutputTileHeight--;
//             }
//         }
//    }

//    if (minLatency == 0xFFFFFFFF)
//    {
//        std::cout <<"Warning: Cannot find a suitable tile configuration for EltAdd Layer "<<_eltAddLayer.getLayerID()<<std::endl;
//        throw;
//    }
//    unsigned int ops = outputChannels * outputHeight * outputWidth;
//    t_tile_pair result = {.tileInfo = bestTileInfo,
//                          .inputTransferLatency = bestInputTransferLatency,
//                          .weightTransferLatency = 0x0,
//                          .outputTransferLatency = bestOutputTransferLatency,
//                          .computeLatency = 0x0,
//                          .computeLatencyWithOverhead = 0x0,
//                          .latency = minLatency,
//                          .flagComputeBound=false,
//                          .ops = ops};
//    return result;
//}

t_tile_pair calculateTileSizePerUnit(EltAddLayer &_eltAddLayer)
{
    unsigned int maxOutputTileHeight = MAX_OUTPUT_TILE_HEIGHT;

    //Search all possible solutions tile solution exhautively.
    unsigned int outputTileWidthPerCol = MAX_OUTPUT_TILE_WIDTH_PER_COL;

    unsigned int outputHeight = _eltAddLayer.getOutputHeight();
    unsigned int outputWidth = _eltAddLayer.getOutputWidth();
    unsigned int outputChannels = _eltAddLayer.getOutputChannel();

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



    //Times 3 to account for two input transfers, and one output transfer for each
    //block of addition
    unsigned int inputLatency = deriveConvInputTransferLatency(
                candidateTileInfo,
                outputChannels,
                1,
                1,
                1
                );
    unsigned int outputLatency = deriveOutputTransferLatency(
                candidateTileInfo,
                outputHeight,
                outputChannels,
                1
                );

    unsigned int ops = outputChannels * outputHeight * outputWidth;
    t_tile_pair result = {.tileInfo = candidateTileInfo,
                          .inputTransferLatency = inputLatency,
                          .weightTransferLatency = 0x0,
                          .outputTransferLatency = outputLatency,
                          .computeLatency = 0x0,
                          .computeLatencyWithOverhead = 0x0,
                          .latency = inputLatency + outputLatency,
                          .flagComputeBound=false,
                          .ops = ops};
    return result;
}

