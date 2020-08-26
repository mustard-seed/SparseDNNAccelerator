#include "graph_factory.hpp"
#include "layerInstructionGenerator.hpp"
#include "params.hpp"
#include "floatFixedPointConversion.hpp"

#include <memory>

using namespace std;
using namespace GraphRuntime;

/*!
 * Some helper functions
 */
/*!
 * \brief calculateTileWidthPerUnit
 * \details calculate the best tile size per unit (e.g. PE column/row)
 *  given the number of units and all there is to know about the layer.
 * \param _convLayer[const GraphRuntime::ConvLayer &]
 * \param _numUnits[int]
 * \param _isWidth[bool]
 * \return
 */
int calculateTileSizePerUnit(const ConvLayer& _convLayer, int _numUnits, bool _isWidth);

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
            unsigned char numGroupCurrentLayer = pLayer->getCurrentNumberGroups();
            unsigned int numInputChannelPerGroup0 = numInputChannel0 / numGroupCurrentLayer;

            unsigned char verticalBorderPadding = 0; //override this later
            unsigned char horizontalBorderPadding = 0; //override this later

            unsigned char kernelSize = 1; //override
            unsigned char stride = 1; //override
            unsigned int numInputChannel1 = 0; //override

            unsigned char inputHeightSPUnitSize = 1; //override if transpose conv
            unsigned char inputWidthSPUnitSize = 1; //override if transpose conv
            unsigned char sizeOutputTileWidthPerCol = 1; //override
            unsigned char sizeOutputTileHeight = 1; //override

            //Arguments related to pSum binary-point shifting
            unsigned char outputShiftBits = 0; //override
            unsigned char outputShiftLeft = TRUE; //overrided

            //Arguments realted to input binary-point shifting
            unsigned int input0ShiftBits = 0; //override
            bool input0ShiftLeft = true;
            unsigned int input1ShiftBits = 0; //override
            bool input1ShiftLeft = true;

            //Input memory regions
            unsigned int input0MemoryRegion = 0;
            unsigned int input1MemoryRegion = 1;
            //Output memory regions
            unsigned int outputMemoryRegion = 2;

#if defined(SPARSE_SYSTEM)
            bool flagSparseInput = pLayer->getInputSparseFlag();
            bool flagSparseOutput = pLayer->getOutputSparseFlag();
#else
            bool flagSparseInput = false;
            bool flagSparseOutput = false;
#endif
            bool isComputeLayer = true;
            switch (layerType) {
                case CONVOLUTION: {
                    auto pLayerLocal = dynamic_pointer_cast<ConvLayer>(pLayer);
                    numInputChannel1 = 0;
                    sizeOutputTileWidthPerCol = calculateTileSizePerUnit(*pLayerLocal.get(), PE_COLS, true);
                    sizeOutputTileHeight = calculateTileSizePerUnit(*pLayerLocal.get(), 1, false);
                    kernelSize = pLayerLocal->getKernelSize();
                    stride = pLayerLocal->getKernelStride();
                    verticalBorderPadding = pLayerLocal->getInputBorderPadding();
                    horizontalBorderPadding = pLayerLocal->getInputBorderPadding();

                    //output precision control
                    int weightFracBits = pLayerLocal->getWeightFracBits();
                    int inputFracBits = pLayerLocal->getInputFracBits().at(0);
                    int outputFracBits = pLayerLocal->getOutputFracBits().at(0);
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
                    for (unsigned int i=0; i<floatWeights.size(); i++)
                    {
                        fixedPointWeight.at(i) = fixedPointNumber(floatWeights.at(i), weightFracBits, 7-weightFracBits);
                    }
                    //Align and compress the weight tensor
                    std::shared_ptr<AlignedTensor> pWeight;
                    #if defined(SPARSE_SYSTEM)
                        pWeight.reset(new FlexibleDirectCompressedTensor (
                                    pLayerLocal->getWeights(),
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
                                            pLayerLocal->getWeights(),
                                            numOutputChannels, //_num3DTensors
                                            numInputChannelPerGroup0, //channel
                                            (unsigned char) kernelSize, //width
                                            (unsigned char) kernelSize, //height
                                            maxScalarIndexInChannelGroup,
                                            maxClusterIndexInTransferBlock,
                                            maxScalarIndexInCluster,
                                            true //isKernel
                                        ));
                    #endif

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

                } //CONVOLUTION
                break;
                case ELTADD: {
                    auto pLayerLocal = dynamic_pointer_cast<EltAddLayer>(pLayer);
                    numInputChannel1 = pLayerLocal->getInputChannels().at(1);
                } //ELTADD
                break;
                case MAXPOOL: {
                    auto pLayerLocal = dynamic_pointer_cast<MaxPoolLayer>(pLayer);
                    kernelSize = pLayerLocal->getKernelSize();
                    stride = pLayerLocal->getKernelStride();

                     //TODO: add precision stuff
                } //MAXPOOL
                break;
                case AVGPOOL:{
                    auto pLayerLocal = dynamic_pointer_cast<AveragePoolLayer>(pLayer);
                    kernelSize = pLayerLocal->getKernelSize();
                    stride = pLayerLocal->getKernelStride();
                    //TODO: add precision stuff
                    //TODO: Modify the shift direction and amounts, to simulate the effect of the integer divisor

                } //AVGPOOL
                break;
                case QUANT: {
                    isComputeLayer = false;
                    auto pLayerLocal = dynamic_pointer_cast<QuantLayer>(pLayer);
                    //Add an input
                    pGraph->vecInputInfo.emplace_back(
                                t_blob_info(
                                    .memoryRegionID=pLayerLocal->getOutputMemoryLocation() ,
                                    .channelPerGroup=numOutputChannels / (pLayerLocal->getNextNumberGroups()),
                                    .group=pLayerLocal->getNextNumberGroups(),
                                    .height=pLayerLocal->getOutputHeight(),
                                    .width=pLayerLocal->getOutputWidth(),
                                    .numFracBits=pLayerLocal->getInputFracBits(),
                                    .flagCanBeSparse=flagSparseOutput,
                                    .blobName="quant_"+to_string(pLayerLocal->getLayerID())
                                    )
                                );
                } //QUANT
                break;
                case DEQUANT: {
                    isComputeLayer = false;
                    auto pLayerLocal = dynamic_pointer_cast<DeQuantLayer>(pLayer);
                    //Add an output
                    pGraph->vecOutputInfo.emplace_back(
                                t_blob_info(
                                    .memoryRegionID=pLayerLocal->getInputMemoryLocations().at(0),
                                    .channelPerGroup=numOutputChannelPerGroup,
                                    .group=pLayerLocal->getCurrentNumberGroups(),
                                    .height=pLayerLocal->getInputHeights().at(0),
                                    .width=pLayerLocal->getInputWidths().at(0),
                                    .numFracBits=pLayerLocal->getInputFracBits().at(0),
                                    .flagCanBeSparse=flagSparseInput,
                                    .blobName="dequant_"+to_string(pLayerLocal->getLayerID())
                                    )
                                );
                } //DEQUANT
                break;
            }

            unsigned int inputHeightSPSize;
            unsigned int inputWidthSPSize;
        } // for layer



        return pGraph;
    }
}

int calculateTileSizePerUnit(const ConvLayer& _convLayer, int _numUnits, bool _isWidth)
{
    return 4;
}

