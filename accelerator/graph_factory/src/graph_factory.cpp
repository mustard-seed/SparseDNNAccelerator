#include "graph_factory.hpp"
#include "layerInstructionGenerator.hpp"
#include "params.hpp"
#include "floatFixedPointConversion.hpp"

#include <memory>
#include <cmath>

using namespace std;
using namespace GraphRuntime;

/*!
 * Some helper functions
 */
typedef struct {
    int tileSizePerUnitFull;
    int numUnitsWhenPartial;
} t_tile_pair;
/*!
 * \brief calculateTileWidthPerUnit
 * \details calculate the best tile size per unit (e.g. PE column/row)
 *  given the number of units and all there is to know about the layer.
 * \param _convLayer[const GraphRuntime::ConvLayer &]
 * \param _numUnits[int]
 * \param _isWidth[bool]
 * \return
 */
t_tile_pair calculateTileSizePerUnit(ConvLayer& _convLayer, int _numUnits, bool _isWidth);

namespace GraphRuntime {
    GraphFactory::GraphFactory(std::string _traceFileName, std::string _parameterFileName)
    {
       typedef YAML::Node YN;
       YN traceNodes = YAML::LoadFile(_traceFileName);
       YN parameterNodes = YAML::LoadFile(_parameterFileName);
       for (int i=0; i<traceNodes.size(); i++) {
           YAML::Node traceLayer = traceNodes[i];
           //Hash the string to a enum type, so we can use switch-case statement
           //See https://stackoverflow.com/a/650307
           LayerType opType = hashLayerTypeString(
                       traceLayer["operationType"].as<string>());

           //cout <<"Detected layer type: "<<traceLayer["operationType"].as<string>()<<endl;
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
    #if defined(SPARSE_SYSTEM)
        int offsetWeightTBCount = 0;
    #endif

        unsigned short maxClusterIndexInCompressionBlock = COMPRESSION_WINDOW_SIZE-1;
        unsigned short maxClusterIndexInTransferBlock = TRANSFER_SIZE-1;
        unsigned short maxScalarIndexInCluster = CLUSTER_SIZE-1;

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
            signed int memDramBlockIA0ColStride = 0;
            signed int memDramBlockIA1ColStride = 0;
            signed int memDramBlockOAColStride = 0;
            signed int memDramBlockFilterStride = 0; //override

            int offsetWeightsDramBlockIncrement = 0; //override
            int offsetBiasesDramBlockIncrement = 0; //override
        #if defined(SPARSE_SYSTEM)
            int offsetWeightTBCountIncrement = 0; //override
        #endif

#if defined(SPARSE_SYSTEM)
            bool flagSparseInput = pLayer->getInputSparseFlag();
            bool flagSparseOutput = pLayer->getOutputSparseFlag();
#else
            bool flagSparseInput = false;
            bool flagSparseOutput = false;
#endif
            bool isComputeLayer = true; //override
            OPERATION op = ::CONVOLUTION; //override
            std::string layerName;
#if defined(HOST_DEBUG)
           cout <<"Generating for Layer(ID): "<<pLayer->getLayerID()<<endl;
#endif
            switch (layerType) {
                case CONVOLUTION: {
                    auto pLayerLocal = dynamic_pointer_cast<ConvLayer>(pLayer);
                    layerName = "conv_"+to_string(pLayerLocal->getLayerID());


                    numInputChannel1 = 0;
                    t_tile_pair widthTileInfo = calculateTileSizePerUnit(*pLayerLocal.get(), PE_COLS, true);
                    sizeOutputTileFullWidthPerCol = widthTileInfo.tileSizePerUnitFull;
                    numActiveColsPartialOutputTile = widthTileInfo.numUnitsWhenPartial;
                    t_tile_pair heightTileInfo = calculateTileSizePerUnit(*pLayerLocal.get(), 1, false);
                    sizeOutputTileFullHeight = heightTileInfo.tileSizePerUnitFull;
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
//                        unsigned int i=0;
//                        for (unsigned int h=0; h<kernelSize; h++)
//                        {
//                            for (unsigned int w=0; w<kernelSize; w++)
//                            {
//                                for (unsigned int c=0; c<numInputChannelPerGroup0; c++)
//                                {
//                                    unsigned int traceIdx = c*kernelSize*kernelSize + h*kernelSize + w;
//                                    fixedPointWeight.at(i++) = fixedPointNumber(floatWeights.at(traceIdx), weightFracBits, 7-weightFracBits);
//                                    //fixedPointWeight.at(i++) = fixedPointNumber(0.0f, weightFracBits, 7-weightFracBits);
//                                }
//                            }
//                        }
                        for (unsigned int i=0; i<floatWeights.size(); i++)
                        {
                             fixedPointWeight.at(i) = fixedPointNumber(floatWeights.at(i), weightFracBits, 7-weightFracBits);
                        }
                    }

                    //Align and compress the weight tensor
                    std::shared_ptr<AlignedTensor> pWeight;
                    #if defined(SPARSE_SYSTEM)
                        pWeight.reset(new FlexibleDirectCompressedTensor (
                                    fixedPointWeight,
                                    numOutputChannels, //_num3DTensors
                                    numInputChannelPerGroup0, //channel
                                    (unsigned char) kernelSize, //width
                                    (unsigned char) kernelSize, //height
                                    numInputChannelPerGroup0-1, //_maxScalarIndexInChannelGroup
                                    maxClusterIndexInCompressionBlock, //_maxClusterIndexInCompressionBlock
                                    maxClusterIndexInTransferBlock, //_maxClusterIndexInTransferBlock
                                    maxScalarIndexInCluster, //_maxScalarIndexInCluster
                                    true //isKernel
                                ) );
                    #else
                        pWeight.reset( new AlignedTensor (
                                            fixedPointWeight,
                                            numOutputChannels, //_num3DTensors
                                            numInputChannelPerGroup0, //channel
                                            (unsigned char) kernelSize, //width
                                            (unsigned char) kernelSize, //height
                                            numInputChannelPerGroup0-1,
                                            maxClusterIndexInTransferBlock,
                                            maxScalarIndexInCluster,
                                            true //isKernel
                                        ));
                    #endif
                    //Filter stride
                    memDramBlockFilterStride = (pWeight->getExternalMemoryAddressStride()) >> WIDE_SIZE_OFFSET;

                    //Prepare the fixed-point bias vector
                    std::shared_ptr<t_aligned_short_vector> pBiasVector = std::make_shared<t_aligned_short_vector>(numOutputChannels, 0x0);
                    bool hasBias = pLayerLocal->getBiasFlag();
                    if (hasBias)
                    {
                        std::vector<float> biasVector = pLayerLocal->getBiases();
                        for (int i=0; i<biasVector.size(); i++)
                        {
                            float bias = biasVector.at(i);
                            pBiasVector->at(i) = (t_accumulator) (round(bias * (float) (1 << pSumFracBits )) );
                        }
                    }

                    //update weight count vecotr, bias count vector and offsets
                    pGraph->vecWeightDramBlockStart.push_back(offsetWeightsDramBlock);
                    pGraph->vecBiasStart.push_back(offsetBiasesDramBlock);
                    #if defined(SPARSE_SYSTEM)
                        pGraph->vecWeightTBCountStart.push_back(offsetWeightTBCount);
                    #endif
                    pGraph->pWeights.push_back(pWeight);
                    pGraph->pBiasVector.push_back(pBiasVector);
                    offsetWeightsDramBlockIncrement =
                            (pWeight->getExternalMemoryAddressStride() >> WIDE_SIZE_OFFSET) * numOutputChannels;
                    offsetBiasesDramBlockIncrement = numOutputChannels;
                    #if defined(SPARSE_SYSTEM)
                        offsetWeightTBCountIncrement = numOutputChannels;
                    #endif

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

                    sizeOutputTileFullWidthPerCol = 1;
                    numActiveColsPartialOutputTile = numOutputWidth % PE_COLS;
                    sizeOutputTileFullHeight = 1;

                    //Align input bits
                    int inputFracBits0 = pLayerLocal->getInputFracBits().at(0);
                    int inputFracBits1 = pLayerLocal->getInputFracBits().at(1);
                    int outputFracBits = pLayerLocal->getOutputFracBits();
                    int pSumFracBits = 0;
                    if (inputFracBits1 > inputFracBits0)
                    {
                        input0ShiftBits = 0;
                        input0ShiftLeft = TRUE;
                        input1ShiftBits = inputFracBits1 - inputFracBits0;
                        input1ShiftLeft = FALSE;
                        pSumFracBits = inputFracBits0;
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
                        input0ShiftBits = inputFracBits0 - inputFracBits1;
                        input0ShiftLeft = FALSE;
                        input1ShiftBits = 0;
                        input1ShiftLeft = TRUE;
                        pSumFracBits = inputFracBits1;
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
                    numActiveColsPartialOutputTile = numOutputWidth % PE_COLS;

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
                    numActiveColsPartialOutputTile = numOutputWidth % PE_COLS;
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
                                    .channelPerGroup=numOutputChannels / (pLayerLocal->getNextNumberGroups()),
                                    .group=pLayerLocal->getNextNumberGroups(),
                                    .height=pLayerLocal->getOutputHeight(),
                                    .width=pLayerLocal->getOutputWidth(),
                                    .numFracBits=pLayerLocal->getOutputFracBits(),
                                    .flagCanBeSparse=flagSparseOutput,
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
                                    .channelPerGroup=numOutputChannelPerGroup,
                                    .group=pLayerLocal->getCurrentNumberGroups(),
                                    .height=pLayerLocal->getInputHeights().at(0),
                                    .width=pLayerLocal->getInputWidths().at(0),
                                    .numFracBits=pLayerLocal->getInputFracBits().at(0),
                                    .flagCanBeSparse=flagSparseInput,
                                    .blobName="dequant_"+to_string(pLayerLocal->getLayerID())
                                    }
                                );
                } //DEQUANT
                break;
            } //switch

            // Generate instruction if this is a computation layer
            if (isComputeLayer == true)
            {
                unsigned int inputHeightSPSize = inputHeightSPUnitSize*(numInputHeight0-1) + 1;
                unsigned int inputWidthSPSize = inputWidthSPUnitSize*(numInputWidth0-1) + 1;
                char instFlagSparseInput = flagSparseInput ? TRUE : FALSE;
                char instFlagSparseOutput = flagSparseOutput ? TRUE : FALSE;
                char instEnableRelu = pLayer->getOutputReluFlag() ? TRUE : FALSE;

                //strides
                memDramBlockIA0ColStride = calculateExternalMemoryAddressStride(
                            numInputChannelPerGroup0, //channelsPerGroup
                            numGroupCurrentLayer, //group
                            numInputHeight0, //height
                            numInputWidth0, //width
                            CLUSTER_SIZE, //clusterSize
                            TRANSFER_SIZE, //transferBlockSize
                            COMPRESSION_WINDOW_SIZE, //compressionWindowSize
                            WIDE_SIZE, //numTransferBlockPerDramBlock
                            false, //isKernel
                            !flagSparseInput //isDense
                      ) >> WIDE_SIZE_OFFSET;

                memDramBlockOAColStride = calculateExternalMemoryAddressStride(
                                numOutputChannelPerGroup, //channelsPerGroup
                                pLayer->getNextNumberGroups(), //group
                                numOutputHeight, //height
                                numOutputWidth, //width
                                CLUSTER_SIZE, //clusterSize
                                TRANSFER_SIZE, //transferBlockSize
                                COMPRESSION_WINDOW_SIZE, //compressionWindowSize
                                WIDE_SIZE, //numTransferBlockPerDramBlock
                                false, //isKernel
                                !flagSparseOutput //isDense
                            ) >> WIDE_SIZE_OFFSET;

                if (pLayer->getInputMemoryLocations().size() > 1)
                {
                    memDramBlockIA1ColStride = calculateExternalMemoryAddressStride(
                                numInputChannelPerGroup1, //channelsPerGroup
                                numGroupCurrentLayer, //group
                                numInputHeight1, //height
                                numInputWidth1, //width
                                CLUSTER_SIZE, //clusterSize
                                TRANSFER_SIZE, //transferBlockSize
                                COMPRESSION_WINDOW_SIZE, //compressionWindowSize
                                WIDE_SIZE, //numTransferBlockPerDramBlock
                                false, //isKernel
                                !flagSparseInput //isDense
                          ) >> WIDE_SIZE_OFFSET;
                }
                else
                {
                    memDramBlockIA1ColStride = 0;
                }

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

                        //signed int memIA0DramBlockColStride,
                        memDramBlockIA0ColStride,
                        //signed int memIA0DramBlockRowStride,
                        memDramBlockIA0ColStride * numInputWidth0,
                        //signed int _memIA0DramBlockGroupStride,
                        memDramBlockIA1ColStride * numInputWidth0 * numInputHeight0,


                        //signed int memIA1DramBlockColStride,
                        memDramBlockIA1ColStride,
                        //signed int memIA1DramBlockRowStride,
                        memDramBlockIA1ColStride * numInputWidth1,
                        //signed int _memIA1DramBlockGroupStride,
                        memDramBlockIA1ColStride * numInputWidth1 * numInputHeight1,

                        //signed int memOADramBlockColStride,
                        memDramBlockOAColStride,

                        //signed int memWeightDramBlockFilterStride,
                        memDramBlockFilterStride,

                        //TB count memory information. Only one input blob is supported for sparse operation
                        #if defined(SPARSE_SYSTEM)
                            //signed int memIATB0CountStart,
                            input0MemoryRegion * MEM_ACTIVATION_TB_REGION_SIZE_PER_SLICE,
                            //unsigned int memIATB0CountColStride,
                            1,
                            //signed int memOATBCountStart,
                            outputMemoryRegion * MEM_ACTIVATION_TB_REGION_SIZE_PER_SLICE,
                            //unsigned int memOATBCountColStride,
                            1,
                            //signed int memWeightTBCountStart,
                            offsetWeightTBCount,
                        #endif

                        //unsigned char flagSparseOutput,
                        instFlagSparseOutput,
                        //unsigned char flagSparseInput,
                        instFlagSparseInput,

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
                        numOutputChannels,
                        //unsigned short numGroupsNextLayer
                        pLayer->getNextNumberGroups()
                        );

                    //Filling the rest of the information for the first layer
                   {
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
                            .numOAMoverInstructions=numOAMoverInstructions
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
            } // if compute layer

        } // for layer

        return pGraph;
    }
}

t_tile_pair calculateTileSizePerUnit(ConvLayer& _convLayer, int _numUnits, bool _isWidth)
{
    //TODO: To compute the tile size and the number of active roles properly, we need to develop an analytical model.
    int tileSizePerUnitFull;
    int numUnitsWhenPartial;
    if (_isWidth)
    {
        int width = _convLayer.getOutputWidth();
        tileSizePerUnitFull = ((width / PE_COLS) > 4) ? 4 : (width / PE_COLS);
        tileSizePerUnitFull = (tileSizePerUnitFull == 0) ? 1 : tileSizePerUnitFull;
        numUnitsWhenPartial = 1;
    }
    else
    {
        tileSizePerUnitFull = 4;
        numUnitsWhenPartial = 1;
    }

    t_tile_pair result{.tileSizePerUnitFull=tileSizePerUnitFull, .numUnitsWhenPartial=numUnitsWhenPartial};
    return result;
}

