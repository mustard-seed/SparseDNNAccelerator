#include "layerInstructionGenerator.hpp"

#include <cassert>
#include <iostream>

void instruction_generator(
        //Type of the operation
        OPERATION op,

        //Instruction buffers
        t_aligned_ia_mover_instruction_vector & vecIAMoverInstruction,
        t_aligned_oa_mover_instruction_vector & vecOAMoverInstruction,
        t_aligned_ia_tile_controller_instruction_vector & vecIATileControlInstruction,
        t_aligned_oa_tile_controller_instruction_vector & vecOATileControlInstruction,
        t_aligned_weight_mover_instruction_vector & vecWeightMoverInstruction,
        t_aligned_misc_instruction_vector & vecMiscInstruction,

        bool flagIA0ShiftLeft,
        unsigned int numIA0ShiftAmount,
        bool flagIA1ShiftLeft,
        unsigned int numIA1ShiftAmount,

        //Starting location of activation tensors in the OpenCL Buffer
        //IA and OA occupies the same OpenCL buffer

        //Starting location of the input tensor dram blocks.
        //Support up to 2 input activation tensors
        //Assuming GHWC layout. Strips are at aligned memory location
        signed int memIA0DramBlockStartIndex,
        signed int memIA1DramBlockStartIndex,

        //Starting location of the output tensor dram blocks
        //Supports only one output tensor
        //Assuming GHWC layout. Strips are at aligned memory location
        signed int memOADramBlockStartIndex,

        //Starting location of the weight tensor dram blocks
        signed int memWeightDramBlockStartIndex,
        //Starting location of bias
        signed int memBiasStartIndex,

        //Input activation blob 0 strides in terms of DRAM block
        //Assuming GHWC layout
        signed int memIA0DramBlockColStride,
        signed int memIA0DramBlockRowStride,
        //If the operation is not CONVOLUTION or CONCATENATION, then _memIA0DramBlockGroupStride can be overwritten
        signed int _memIA0DramBlockGroupStride,

        //Input activation blob 1 strides in terms of DRAM block
        //Assuming GHWC layout
        signed int memIA1DramBlockColStride,
        signed int memIA1DramBlockRowStride,
        //If the operation is not CONVOLUTION or CONCATENATION, then _memIA0DramBlockGroupStride can be overwritten
        signed int _memIA1DramBlockGroupStride,

        //Output activation blob stride in terms of DRAM block
        //Assuming GHWC layout
        signed int memOADramBlockColStride,
        signed int memWeightDramBlockFilterStride,

        //TB count memory information. Only one input blob is supported for sparse operation
        #if defined(SPARSE_SYSTEM)
            signed int memIATB0CountStart,
            unsigned int memIATB0CountColStride,

            signed int memOATBCountStart,
            unsigned int memOATBCountColStride,

            signed int memWeightTBCountStart,
        #endif

        unsigned char flagSparseOutput,
        unsigned char flagSparseInput,
        //Whether the IA mover kernel should wait for the output from the previous tensor to commit before moving
        //on to the current tensor
        unsigned char flagTensorSync,
        //unsigned char flagOutputSync,
        unsigned char flagRelu,
        unsigned char outputShiftBits,
        unsigned char flagOutputShiftLeft,

        //Input stride-padded width and height, not including border padding
        unsigned short inputSPWidth,
        unsigned short inputSPHeight,
        //Input stride-padded unit sizes
        unsigned char inputSPWidthUnit,
        unsigned char inputSPHeightUnit,
        //Input border paddings
        unsigned char inputWidthPadding,
        unsigned char inputHeightPadding,

        unsigned char kernelSize,
        unsigned char kernelStride,

        unsigned char _sizeOutputTileFullHeight,
        unsigned char _sizeOutputTileFullWidthPerCol,
        unsigned char _numActiveColsPartialOutputTile,

        //Number of channels in input blobs 0 and 1
        //Only element-wise addition and pooling should use the second  blob
        unsigned short numInputChannels0,
        unsigned short numInputChannels1,

        //Number of groups in the current layer's output.
        //Only relevant for convolution
        unsigned short _numGroupsCurrentLayer,

        //Number of output channels
        //Only relevant for convolution
        unsigned short numOutputChannels,

        //Number of groups in the next layer
        unsigned short numGroupsNextLayer

        )
{
    unsigned int sizeOutputTileFullHeight = ((op == CONVOLUTION) || (op == ELT_ADD) || (op == CONCATENATION)) ?
                _sizeOutputTileFullHeight : 1;
    unsigned int sizeOutputTileFullWidthPerCol = ((op == CONVOLUTION) || (op == ELT_ADD) || (op == CONCATENATION)) ?
                _sizeOutputTileFullWidthPerCol : 1;

//    unsigned int sizeOutputTileFullHeight = _sizeOutputTileFullHeight;
//    unsigned int sizeOutputTileFullWidthPerCol = _sizeOutputTileFullWidthPerCol;

    unsigned int outputHeight = ( ((unsigned int) inputSPHeight)
            + 2*((unsigned int) inputHeightPadding) - ((unsigned int) kernelSize))
            / ((unsigned int) kernelStride) + 1;

    unsigned int outputWidth = ( ((unsigned int) inputSPWidth)
            + 2* ((unsigned int) inputWidthPadding) - ((unsigned int) kernelSize))
            / ((unsigned int) kernelStride) + 1;

//    unsigned char numActiveColsPartialOutputTile = ((op == CONVOLUTION) || (op == ELT_ADD) || (op == CONCATENATION)) ?
//                _numActiveColsPartialOutputTile : (outputWidth % PE_COLS);
    unsigned char numActiveColsPartialOutputTile = (op == CONCATENATION) ?
                1 : (((op == MAX_POOL) || (op == AVG_POOL)) ? (outputWidth % PE_COLS) : _numActiveColsPartialOutputTile
                    );

    unsigned char numActiveColsFullOutputTile = (op != CONCATENATION) ?
                PE_COLS : 1;

    //Input height and width before stretch and padding
    if ((inputSPHeightUnit != 1) && ((inputSPHeight-1) % inputSPHeightUnit != 0))
    {
        std::cout <<"[layer instruction generation] The stretch and padded height is not compatiable with the height strech-padding unit"<<std::endl;
        throw;
    }
    if ((inputSPWidthUnit != 1) && ((inputSPWidth-1) % inputSPWidthUnit != 0))
    {
        std::cout <<"[layer instruction generation] The stretch and padded width is not compatiable with the width strech-padding unit"<<std::endl;
        throw;
    }
    unsigned int inputDenseHeight = 1 + (inputSPHeight-1) / inputSPHeightUnit;
    unsigned int inputDenseWidth = 1 + (inputSPWidth-1) / inputSPWidthUnit;

    unsigned int numFullOutputTileY =
            ((unsigned int) outputHeight) / ((unsigned int) sizeOutputTileFullHeight);

    unsigned int numOutputTileY =
            1 + ((unsigned int) (outputHeight - 1)) / ((unsigned int) sizeOutputTileFullHeight);

    unsigned int sizeOutputTilePartialHeight =
            ((unsigned int) outputHeight) % ((unsigned int) sizeOutputTileFullHeight);

    unsigned int numFullOutputTileX =
            ((unsigned int) outputWidth) /
            ( ((unsigned int) numActiveColsFullOutputTile) * ((unsigned int) sizeOutputTileFullWidthPerCol) );

    unsigned int numOutputTileX =
            1 + ((unsigned int) (outputWidth-1) ) /
                ( ((unsigned int) numActiveColsFullOutputTile) * ((unsigned int) sizeOutputTileFullWidthPerCol) );

    unsigned int sizeOutputTilePartialWidthPerCol = (numActiveColsPartialOutputTile != 0) ?
                ( ((unsigned int) outputWidth) %
                    ( ((unsigned int) numActiveColsFullOutputTile) * ((unsigned int) sizeOutputTileFullWidthPerCol) )
                ) / ((unsigned int) numActiveColsPartialOutputTile)
              : sizeOutputTileFullWidthPerCol;

    unsigned int sizeInputTileFullHeight =
            ((unsigned int) (sizeOutputTileFullHeight-1))* ((unsigned int) kernelStride)
            + ((unsigned int) kernelSize);

    unsigned int sizeInputTilePartialHeight =
            ((unsigned int) (sizeOutputTilePartialHeight-1))* ((unsigned int) kernelStride)
            + ((unsigned int) kernelSize);

    unsigned int sizeInputTileFullWidthPerCol =
            ((unsigned int) (sizeOutputTileFullWidthPerCol-1))* ((unsigned int) kernelStride)
            + ((unsigned int) kernelSize);

    unsigned int sizeInputTilePartialWidthPerCol =
            ((unsigned int) (sizeOutputTilePartialWidthPerCol-1))* ((unsigned int) kernelStride)
            + ((unsigned int) kernelSize);

    //Number of input channels per group as seen by the IA mover
    unsigned int numIAMoverInputChannelsPerGroup0;
    unsigned int numIAMoverInputChannelsPerGroup1;
    //Number of input channel groups as seen by the IA mover
    unsigned int numIAMoverGroup0;
    unsigned int numIAMoverGroup1;
    //Group stride in terms of DRAM BLOCK as seen by the IA Mover
    unsigned int memIA0DramBlockGroupStride;
    unsigned int memIA1DramBlockGroupStride;


    //Number of groups current layer
    unsigned int numOAGroupsCurrentLayer;

    //Number of active elements in a full compute fold;
    unsigned int numActiveElementsInFullComputeFold;

    //Number of output channels caused by each blob that the MISC kernels have to process
    unsigned int numOutputChannelsBlob0MK;
    unsigned int numOutputChannelsBlob1MK;
    //Number of misc engine output blocks per *current layer* output tile
//    unsigned int numOutputBlocksBlob0MK;
//    unsigned int numOutputBlocksBlob1MK;
    //Number of misc engine output blocks per "current layer" strip
    unsigned int numOutputBlocksBlob0PerStripMK;
    unsigned int numOutputBlocksBlob1PerStripMK;
    //Number of dram block each MK should reduce in order to produce a block of output;
    unsigned int numDramBlocksToReduceMK;
    switch(op) {
        case CONVOLUTION : {
            numOAGroupsCurrentLayer = _numGroupsCurrentLayer;
            if ((numOutputChannels % _numGroupsCurrentLayer != 0)
                    || (numInputChannels0 % _numGroupsCurrentLayer != 0))
            {
                std::cout <<"[layer instruction generator] Number of input or output channels is not divisble by the number of groups in the current layer."<<std::endl;
                throw;
            }
            numIAMoverInputChannelsPerGroup0 = numInputChannels0 / _numGroupsCurrentLayer;
            numIAMoverInputChannelsPerGroup1  = 0;
            numIAMoverGroup0 = _numGroupsCurrentLayer;
            numIAMoverGroup1 = 0;
            numActiveElementsInFullComputeFold = PE_ROWS;
            memIA0DramBlockGroupStride = _memIA0DramBlockGroupStride;
            numOutputChannelsBlob0MK = 0;
            numOutputChannelsBlob1MK = 0;
//            numOutputBlocksBlob0MK = 0;
//            numOutputBlocksBlob1MK = 0;
            numOutputBlocksBlob0PerStripMK = 0;
            numOutputBlocksBlob1PerStripMK = 0;
            numDramBlocksToReduceMK = 0;

            int filterCacheRequirement = filter_cache_boundary_check(
                            kernelSize,
                            numInputChannels0
                        );
            if (filterCacheRequirement > KERNEL_CACHE_DEPTH)
            {
                std::cout <<"Individual fitler size is too big to fit inside the filter cache."<<std::endl;
                std::cout <<"[kernelSize, number of input channels]: "<<(int) kernelSize<<" "<<(int) numInputChannels0<<std::endl;
                throw;
            }
        }
        break;
        case CONCATENATION : {
            if (numOutputChannels != (numInputChannels0 + numInputChannels1))
            {
                std::cout <<"[layer instruction generator] Operation is concatenation, "
                            "but the two input tensor's input channels do not add up to the number of output channels."<<std::endl;
                throw;
            }
            if (flagSparseInput != FALSE)
            {
                std::cout <<"[layer instruction generator] Operation is concatenation, "
                            "but the input sparse flag is TRUE"<<std::endl;
                throw;
            }
            if (kernelSize != 1)
            {
                std::cout <<"[layer instruction generator] Operation is concatenation, "
                            "but the kernel size is not 1"<<std::endl;
                throw;
            }
            numIAMoverInputChannelsPerGroup0 = numInputChannels0;
            numIAMoverInputChannelsPerGroup1 = numInputChannels1;
            numIAMoverGroup0 = 1;
            numIAMoverGroup1 = 1;
            numActiveElementsInFullComputeFold = numOutputChannels;
            numOAGroupsCurrentLayer = 1;
            memIA0DramBlockGroupStride = _memIA0DramBlockGroupStride;
            memIA1DramBlockGroupStride = _memIA1DramBlockGroupStride;
            numOutputChannelsBlob0MK = numInputChannels0;
            numOutputChannelsBlob1MK = numInputChannels1;
//            numOutputBlocksBlob0MK = 1+ (numOutputChannelsBlob0MK-1) / BURST_SIZE_BYTE;
//            numOutputBlocksBlob1MK = 1+ (numOutputChannelsBlob1MK-1) / BURST_SIZE_BYTE;
            numOutputBlocksBlob0PerStripMK = 1+ (numOutputChannelsBlob0MK-1) / BURST_SIZE_BYTE;;
            numOutputBlocksBlob1PerStripMK = 1+ (numOutputChannelsBlob1MK-1) / BURST_SIZE_BYTE;
            numDramBlocksToReduceMK = 1;
        }
        break;
        case MAX_POOL : {
            if (numOutputChannels != numInputChannels0)
            {
                std::cout <<"[layer instruction generator] Operation is max pool, "
                            "but the number of input channels does not match the number of output channels."<<std::endl;
                throw;
            }
            if (flagSparseInput != FALSE)
            {
                std::cout <<"[layer instruction generator] Operation is max pool, "
                            "but the input sparse flag is TRUE"<<std::endl;
                throw;
            }
            numIAMoverInputChannelsPerGroup0 = BURST_SIZE_BYTE;
            numIAMoverInputChannelsPerGroup1 = 0;
            numIAMoverGroup0 = (1 + (numInputChannels0-1) / BURST_SIZE_BYTE);
            numIAMoverGroup1 = 0;
            numActiveElementsInFullComputeFold = BURST_SIZE_BYTE;
            numOAGroupsCurrentLayer = 1;
            memIA0DramBlockGroupStride = 1;
            numOutputChannelsBlob0MK = numInputChannels0;
            numOutputChannelsBlob1MK = 0;
            numOutputBlocksBlob0PerStripMK = 1+ (numOutputChannelsBlob0MK-1) / BURST_SIZE_BYTE;
            numOutputBlocksBlob1PerStripMK = 0;
//            numOutputBlocksBlob0PerStripMK = numOutputBlocksBlob0MK;
//            numOutputBlocksBlob1PerStripMK = 0;
            numDramBlocksToReduceMK = kernelSize * kernelSize;
        }
        break;
        case ELT_ADD : {
            if ( (numOutputChannels != numInputChannels0) || (numInputChannels0!=numInputChannels1))
            {
                std::cout <<"[layer instruction generator] Operation is elt-add, "
                            "but the number of input channels and the number of output channels do not match."<<std::endl;
                throw;
            }
            if (flagSparseInput != FALSE)
            {
                std::cout <<"[layer instruction generator] Operation is elt-add, "
                            "but the input sparse flag is TRUE."<<std::endl;
                throw;
            }
            if (kernelSize != 1)
            {
                std::cout <<"[layer instruction generator] Operation is elt-add, "
                            "but the kernel size is not 1."<<std::endl;
                throw;
            }
            numIAMoverInputChannelsPerGroup0 = numInputChannels0;
            numIAMoverInputChannelsPerGroup1 = numInputChannels1;
            numIAMoverGroup0 = 1;
            numIAMoverGroup1 = 1;
            numActiveElementsInFullComputeFold = numInputChannels0;
            numOAGroupsCurrentLayer = 1;
            memIA0DramBlockGroupStride = _memIA0DramBlockGroupStride;
            memIA1DramBlockGroupStride = _memIA1DramBlockGroupStride;
            numOutputChannelsBlob0MK = numInputChannels0;
            numOutputChannelsBlob1MK = 0;
            numOutputBlocksBlob0PerStripMK = 1+ (numOutputChannelsBlob0MK-1) / BURST_SIZE_BYTE;
            numOutputBlocksBlob1PerStripMK = 0;
            numDramBlocksToReduceMK = 2;
        }
        break;
        case AVG_POOL : {
            if ( (numOutputChannels != numInputChannels0))
            {
                std::cout <<"[layer instruction generator] Operation is avg-pool, "
                            "but the number of output channels does not match the number of input channels."<<std::endl;
                throw;
            }
            if (flagSparseInput != FALSE)
            {
                std::cout <<"[layer instruction generator] Operation is avg-pool, "
                            "but the input sparse flag is TRUE."<<std::endl;
                throw;
            }
            numIAMoverInputChannelsPerGroup0 = BURST_SIZE_BYTE;
            numIAMoverInputChannelsPerGroup1 = 0;
            numIAMoverGroup0 = (1 + (numInputChannels0-1) / BURST_SIZE_BYTE);
            numIAMoverGroup1 = 0;
            numActiveElementsInFullComputeFold = BURST_SIZE_BYTE;
            numOAGroupsCurrentLayer = 1;
            memIA0DramBlockGroupStride = 1;
            numOutputChannelsBlob0MK = numInputChannels0;
            numOutputChannelsBlob1MK = 0;
            numOutputBlocksBlob0PerStripMK = 1+ (numOutputChannelsBlob0MK-1) / BURST_SIZE_BYTE;
            numOutputBlocksBlob1PerStripMK = 0;
            numDramBlocksToReduceMK = (unsigned int) kernelSize * (unsigned int) kernelSize;
        }
    break;
        default: {
            std::cout <<"Instruction generator: unsupported operation type: "<<op<<std::endl;
            throw;
        }
        break;
    }
    unsigned int numOutputChannelsPerGroupCurrentLayer =
            numOutputChannels / numOAGroupsCurrentLayer;

    if (numOutputChannels % numGroupsNextLayer != 0)
    {
        std::cout <<"[layer instruction generator] The number of output channels is not divisble by the number of groups "
                    "in the next layer."<<std::endl;
        throw;
    }
    unsigned int numOutputChannelsPerGroupNextLayer = numOutputChannels / numGroupsNextLayer;

    unsigned int numComputeFoldPerGroup = (numOutputChannelsPerGroupCurrentLayer-1) / numActiveElementsInFullComputeFold + 1;
    unsigned int numFullComputeFoldPerGroup = numOutputChannelsPerGroupCurrentLayer / numActiveElementsInFullComputeFold;
    unsigned int numActiveElementsInPartialComputeFold = numOutputChannelsPerGroupCurrentLayer % numActiveElementsInFullComputeFold;

    int maxOATileHeight = sizeOutputTileFullHeight > sizeOutputTilePartialHeight ?
                    sizeOutputTileFullHeight : sizeOutputTilePartialHeight;
    int maxOATileWidth = sizeOutputTileFullWidthPerCol > sizeOutputTilePartialWidthPerCol ?
                sizeOutputTileFullWidthPerCol : sizeOutputTilePartialWidthPerCol;



    //x & y planar coordinates in the input planar dimension
    unsigned int iterPGlobal = 0;
    unsigned int iterMGlobal = 0;
    for (unsigned int iterPTile=0; iterPTile < numOutputTileY; iterPTile++)
    {
        unsigned int maxTP = (iterPTile >= numFullOutputTileY) ?
                    sizeOutputTilePartialHeight : sizeOutputTileFullHeight;
        unsigned int maxTM = (iterPTile >= numFullOutputTileY) ?
                    sizeInputTilePartialHeight : sizeInputTileFullHeight;

        unsigned int iterQGlobal = 0;
        unsigned int iterNGlobal = 0;
        for (unsigned int iterQTile=0; iterQTile < numOutputTileX; iterQTile++)
        {
            unsigned char numActiveCols = (iterQTile >= numFullOutputTileX) ?
                        numActiveColsPartialOutputTile : numActiveColsFullOutputTile;

            unsigned int maxTQPerCol = (iterQTile >= numFullOutputTileX) ?
                        sizeOutputTilePartialWidthPerCol : sizeOutputTileFullWidthPerCol;

            unsigned int maxTQ = numActiveCols * maxTQPerCol;

            unsigned int maxTNPerCol = (iterQTile >= numFullOutputTileX) ?
                        sizeInputTilePartialWidthPerCol : sizeInputTileFullWidthPerCol;

            unsigned int maxTN = maxTNPerCol * numActiveCols
                    - (((signed int) kernelSize) - ((signed int) kernelStride) ) * (numActiveCols-1);

            /*
             * Cache limit check when performing convolution
            */
            if (op == CONVOLUTION)
            {
                int oaCacheRequirement = oa_cache_boundary_check(
                            //heightTile
                            maxTP,
                            //widthTile
                            maxTQPerCol,
                            //numChannels,
                            numOutputChannels
                            );
                if (oaCacheRequirement > (OA_CACHE_DEPTH * CLUSTER_SIZE))
                {
                    std::cout << "OA tile size is too big to fit inside the oa cache. "
                              << "output height per tile, output width per col: "
                              << maxTP <<" "<<maxTQPerCol<<std::endl;
                    throw;
                }

                int iaCacheRequirementInDramBlock = ia_cache_boundary_check(
                            //heightTile
                            maxTM,
                            //widthTile
                            maxTNPerCol,
                            //numDramBlockPerDenseStrip,
                            memIA0DramBlockColStride
                            );
                if (iaCacheRequirementInDramBlock > IA_CACHE_DEPTH)
                {
                    std::cout <<"IA tile size is too big to fit inside the IA buffer cache"<<std::endl;
                    std::cout <<"input tile height, input tile width per col, numDramBlockPerStrip: "
                             <<" "<<maxTM<<" "<<maxTNPerCol<<" "<<memIA0DramBlockColStride<<std::endl;
                    throw;
                }

                #if defined(SPARSE_SYSTEM)
                    int iaTBCountRequirement = ia_tbcount_cache_boundary_check(
                                maxTM,
                                maxTNPerCol
                                );
                    if (iaTBCountRequirement > IA_BUFFER_TBCOUNT_CACHE_SIZE)
                    {
                        std::cout <<"IA TB count size is too big to fit inside the IA buffer cache"<<std::endl;
                        std::cout <<"input tile height, input tile width per col: "
                                 <<" "<<maxTM<<" "<<maxTNPerCol<<std::endl;
                        throw;
                    }
                #endif
            }

            /*! Generate the output tile controller instruction */
            {
                if (((numGroupsNextLayer != 1) && ((numOutputChannelsPerGroupNextLayer % CLUSTER_SIZE) != 0)))
                {
                    std::cout <<"Either the number of groups in the next layer should be 1, "
                                "or the number of channel per next group should be divisible by CLUSTER_SIZE"
                               <<std::endl;
                    throw;
                }
                t_oa_tile_controller_instruction instructionOAControl;

                instructionOAControl.numLocalTilePerColHxW = (t_uchar)(maxTQPerCol*maxTP);
                instructionOAControl.numRoundedLocalChannels = (t_ushort)((1 + (numOutputChannels-1) / CLUSTER_SIZE) * CLUSTER_SIZE);
                instructionOAControl.numDrainInstructions = (t_ushort) ( numComputeFoldPerGroup * numOAGroupsCurrentLayer);
                instructionOAControl.numMemInstructions = (t_uchar) numGroupsNextLayer;
                instructionOAControl.numFoldsInGroupCurrentLayer = (t_ushort) numComputeFoldPerGroup;
                instructionOAControl.numFullFoldsInCurrentLayer = (t_ushort) numFullComputeFoldPerGroup;
                instructionOAControl.numActiveElementsInFullFold = (t_ushort) numActiveElementsInFullComputeFold;
                instructionOAControl.numActiveElementsInPartialFold = (t_ushort) numActiveElementsInPartialComputeFold;
                instructionOAControl.numLocalChannelsPerCurrentGroup = (t_ushort) numOutputChannelsPerGroupCurrentLayer;
                instructionOAControl.numLocalChannelsPerNextGroup = (t_ushort) numOutputChannelsPerGroupNextLayer;
                instructionOAControl.numActiveCols = (t_uchar) numActiveCols;
                unsigned char leftShift = flagOutputShiftLeft;
                unsigned char scaleShift = outputShiftBits;

                if (!(((flagOutputShiftLeft == 0x00) && (outputShiftBits > 0))
                     || ((flagOutputShiftLeft == 0x01) && (outputShiftBits >= 0))))
                {
                    std::cout << "If output shift direction is RIGHT, then the number of shift must be greater than 0."
                              << std::endl
                              << "shift left flag, shift amount: "
                              <<(unsigned int) flagOutputShiftLeft<<" "
                              <<(unsigned int) outputShiftBits<<std::endl;
                    throw;
                }
                unsigned char sourceIsMisc = (op != CONVOLUTION) ? 0x01 : 0x00;
                instructionOAControl.flagSparseCatFlagReluCatFlagSourceCatShift = (t_uchar)
                        (   ((t_uchar) (scaleShift & 0x0F))
                            | ((t_uchar) ((leftShift & 0x01) << 0x4))
                            | (t_uchar)((flagRelu & 0x01) << 0x7)
                            | (t_uchar)((flagSparseOutput & 0x01) << 0x05)
                            | (t_uchar)((sourceIsMisc & 0x01) << 0x6) //drain source is convolution
                        );
                vecOATileControlInstruction.push_back(instructionOAControl);
            }

            /*! Generate the output OA instruction */
            for (int iOutputGroup=0; iOutputGroup < numGroupsNextLayer; iOutputGroup++)
            {
                t_oa_mover_instruction instructionOA;
                bool isFirstTile =
                        ( iOutputGroup == 0)
                        && (iterPTile == 0)
                        && (iterQTile == 0);
//                unsigned char actualFlagOutputSync = (flagOutputSync == 0x01) ?
//                            ((isLastOutputTile == true) ? 0x1 : 0x0)
//                            : 0x0;
                unsigned char actualFlagOutputSync = isFirstTile ? flagTensorSync : 0x0;
                instructionOA.memSelectCatSparseFlagCatSyncFlagCatNumActiveCols =
                        ((t_uchar) numActiveCols & 0x0F)
                        | ((((t_uchar) actualFlagOutputSync) & 0x01) << 0x04)
                        | ((((t_uchar) flagSparseOutput) & 0x01) << 0x06);
                //TODO: reinstate the memory region is necessary

                instructionOA.memOAStart = (t_int)
                        (memOADramBlockStartIndex
                          + (   (unsigned int) iOutputGroup*outputHeight*outputWidth +
                                (unsigned int) iterPGlobal*outputWidth +
                                (unsigned int) iterQGlobal
                            )
                            * memOADramBlockColStride);

                instructionOA.memOAPEColStride = (t_uint)((unsigned int) maxTQPerCol * memOADramBlockColStride);
                instructionOA.memOARowStride = (t_ushort)((unsigned int) outputWidth * memOADramBlockColStride);
                instructionOA.memOAColStride = (t_ushort) memOADramBlockColStride;

    #if defined(SPARSE_SYSTEM)
                instructionOA.memTBStart = (t_int)
                        (memOATBCountStart
                            + (   (unsigned int) iOutputGroup*outputHeight*outputWidth +
                                  (unsigned int) iterPGlobal*outputWidth +
                                  (unsigned int) iterQGlobal
                              )
                              * memOATBCountColStride);

                instructionOA.memTBPEColStride = (t_uint)((unsigned int) maxTQPerCol * memOATBCountColStride);
                instructionOA.memTBRowStride = (t_ushort)((unsigned int) outputWidth * memOATBCountColStride);
                instructionOA.memTBColStride = (t_ushort) memOATBCountColStride;
    #else
                unsigned int numTBPerGroupNextLayer =
                        1 + (numOutputChannelsPerGroupNextLayer-1) / CLUSTER_SIZE / TRANSFER_SIZE;
                instructionOA.numDramBlockPerStrip =
                        (t_uint) (1 + (numTBPerGroupNextLayer-1)/WIDE_SIZE);
    #endif
                instructionOA.tileHeight = (t_uchar) maxTP;
                instructionOA.columnTileWidth = (t_uchar) maxTQPerCol;
                instructionOA.numColumnTileWidthxTileHeightxNumActiveCols
                        = (t_ushort) (maxTQPerCol * maxTP * numActiveCols);

                vecOAMoverInstruction.push_back(instructionOA);
            } //OA mover instructions for the current planar tile

            unsigned int iterMClipped = (iterMGlobal < inputHeightPadding ) ?
                        0 : ( (iterMGlobal >= (inputHeightPadding + inputSPHeight)) ?
                                  (inputSPHeight - 1) : (iterMGlobal - inputHeightPadding));
            unsigned int iterMDense = iterMClipped / inputSPHeightUnit;
            unsigned int iterSPMIndex = iterMClipped % inputSPHeightUnit;

            unsigned int iterNClipped = (iterNGlobal < inputWidthPadding ) ?
                        0 : ( (iterNGlobal >= (inputWidthPadding + inputSPWidth)) ?
                                  (inputSPWidth - 1) : (iterNGlobal - inputWidthPadding));
            unsigned int iterNDense = iterNClipped / inputSPWidthUnit;
            unsigned int iterSPNIndex = iterNClipped % inputSPWidthUnit;

            /*!
              Transfer the inputs for non-concatenation ops
              For the operations that have two input tensors, the number of IA groups are 1
            */
            if (op != CONCATENATION)
            {
                for (unsigned int iterInputGroup0=0; iterInputGroup0<numIAMoverGroup0; iterInputGroup0++)
                {
                    unsigned int inputStripStartIndex =
                                            iterInputGroup0 * inputDenseHeight * inputDenseWidth
                                            + iterMDense * inputDenseWidth + iterNDense;

                    //Transfer the first input blob, and convolution related items (if necessary)
                    //IA mover instruciton
                    {
                        bool isFirstTile =
                                ( iterInputGroup0 == 0)
                                && (iterPTile == 0)
                                && (iterQTile == 0);

                        t_ia_mover_instruction instructionIA;
                        //Set the transport target. 0x0 means convolution, 0x1 means MISC
                        unsigned char flagTarget= (op == CONVOLUTION) ?  0x00 : 0x01;
                        unsigned char inputArrangement = 0x0;
                        if (op == ELT_ADD)
                        {
                            inputArrangement = 0x01;
                        }

                        unsigned char actualSyncFlag = isFirstTile ? flagTensorSync : 0x00;

                        instructionIA.flagSyncCatInputArrangementCatSparseFlagCatDestinationCatNumActiveCols = (t_uchar)
                                ( ( ((t_uchar) numActiveCols)& 0x0F)
                                 | ((((t_uchar) flagTarget) & 0x01) << 0x04)
                                 | ((((t_uchar) flagSparseInput) & 0x01) << 0x05) //Sparse flag for the input tensor
                                 | ((((t_uchar) inputArrangement) & 0x01) << 0x06)
                                 | ((((t_uchar) actualSyncFlag) & 0x01) << 0x07)
                                );
                        if (!(((flagIA0ShiftLeft == true) && (numIA0ShiftAmount >= 0))
                              && ((flagIA1ShiftLeft == true) && (numIA1ShiftAmount >= 0))))
                        {
                            std::cout <<"Input left shift amount must be greater or equal to 0."<<std::endl;
                            throw;
                        }
                        t_uchar inputShiftAmounts = ((numIA1ShiftAmount & 0x0F) << 0x04) | (numIA0ShiftAmount & 0x0F);
                        instructionIA.inputShiftAmounts = inputShiftAmounts;
                        instructionIA.memBlockStart0 = (t_int) (
                                    memIA0DramBlockStartIndex
                                    + memIA0DramBlockGroupStride * iterInputGroup0
                                    + memIA0DramBlockRowStride * iterMDense
                                    + memIA0DramBlockColStride * iterNDense);

                        instructionIA.memBlockStart1 = (t_int) (
                                    memIA1DramBlockStartIndex
                                    + memIA1DramBlockGroupStride * iterInputGroup0
                                    + memIA1DramBlockRowStride * iterMDense
                                    + memIA1DramBlockColStride * iterNDense);

                        instructionIA.memBlockColStripStride = (t_ushort)memIA0DramBlockColStride;
                        instructionIA.memBlockRowStripStride = (t_ushort)memIA0DramBlockRowStride;

                        #if defined(SPARSE_SYSTEM)
                            instructionIA.memTBCountStart = (t_int)
                                    (memIATB0CountStart
                                     + inputStripStartIndex * memIATB0CountColStride);
                            instructionIA.memTBCountColStride = (t_ushort) memIATB0CountColStride;
                            instructionIA.memTBCountRowStride = (t_ushort) (memIATB0CountColStride * inputDenseWidth);
                            instructionIA.numCWOrTBInGroup = (flagSparseInput == 0x1) ?
                                                (t_ushort) (
                                                    1 + (numIAMoverInputChannelsPerGroup0-1) / COMPRESSION_WINDOW_SIZE / CLUSTER_SIZE
                                                    )
                                                : (t_ushort) (1 + (numIAMoverInputChannelsPerGroup0-1) / TRANSFER_SIZE / CLUSTER_SIZE);
                        #else
                            instructionIA.numTBPerStrip = (t_ushort) (1 + (numIAMoverInputChannelsPerGroup0-1) / TRANSFER_SIZE / CLUSTER_SIZE);
                        #endif
                            instructionIA.tileSPHeight = (t_uchar) maxTM;
                            instructionIA.tileSPWidth = (t_uchar) maxTN;
                            unsigned char inputTileLeftPadding = (iterNGlobal < inputWidthPadding) ?
                                        inputWidthPadding : 0;
                            unsigned char inputTileRightPadding = ((iterNGlobal + maxTN) > (inputWidthPadding + inputDenseWidth)) ?
                                        inputWidthPadding : 0;
                            unsigned char inputTileTopPadding = (iterMGlobal < inputHeightPadding) ?
                                        inputHeightPadding : 0;
                            unsigned char inputTileBottomPadding = ((iterMGlobal + maxTM) > (inputHeightPadding + inputDenseHeight)) ?
                                        inputHeightPadding : 0;
                            instructionIA.concatPadding = (t_uchar) (
                                              (inputTileLeftPadding & 0x03)
                                            | ((inputTileRightPadding & 0x03) << 2)
                                            | ((inputTileTopPadding & 0x03) << 4)
                                            | ((inputTileBottomPadding & 0x03) << 6)
                                        );
                            instructionIA.concatInitSPIndices = (t_uchar) (
                                             ( ((t_uchar) iterSPNIndex) & 0x0F)
                                            | (( ((t_uchar) iterSPMIndex) & 0x0F) << 0x04)
                                        );
                            instructionIA.concatSPSize = (t_uchar) (
                                            (((t_uchar) inputSPWidthUnit) & 0x0F)
                                            | ((((t_uchar) inputSPHeightUnit & 0x0F)) << 0x04)
                                        );
                            instructionIA.columnWidthStride = (t_uchar) (kernelStride * maxTQPerCol);
                            instructionIA.columnSPWidth = (t_uchar) maxTNPerCol;

                            instructionIA.tileSPWidthxTileSPHeight = ((t_ushort) maxTN) * ((t_ushort) maxTM);

                            vecIAMoverInstruction.push_back(instructionIA);
                    } // generate t_ia_mover_instruction 1

                    //Instruction that only matters for convolution
                    if (op == CONVOLUTION)
                    {
                         /*! Send the input IA controller instruction */
                        {
                            t_ia_tile_controller_instruction instructionIAControl;
                            instructionIAControl.localTileWidth = (t_uchar) maxTNPerCol;
                            instructionIAControl.localTileHeight = (t_uchar) maxTM;
                            instructionIAControl.kernelStride = (t_uchar) kernelStride;
                            instructionIAControl.kernelSize = (t_uchar) kernelSize;
                            instructionIAControl.numOutputInstructions = (t_ushort)
                                    ((t_ushort) maxTP * (t_ushort) maxTQPerCol * numComputeFoldPerGroup);
                            //Number of TBs in an uncompressed strip
                            unsigned short cacheIAStripColStrideTBCount = 1 + (numIAMoverInputChannelsPerGroup0-1) / TRANSFER_SIZE / CLUSTER_SIZE;
                            //Number of TBs needed for encoding the bitmask
                            unsigned short cacheIAStripColStrideBitmaskCount = 1 + (numIAMoverInputChannelsPerGroup0-1)/COMPRESSION_WINDOW_SIZE / CLUSTER_SIZE;
                            instructionIAControl.cacheIAStripColStride = (flagSparseInput == 0x00) ?
                                        (t_ushort) (1 + (cacheIAStripColStrideTBCount-1) / WIDE_SIZE)
                                      : (t_ushort) (1 + (cacheIAStripColStrideBitmaskCount+cacheIAStripColStrideTBCount-1) / WIDE_SIZE);
//                            instructionIAControl.numOutputChannelsInGroup = (t_ushort) numOutputChannelsPerGroupCurrentLayer;
                            instructionIAControl.numFullFoldsPerStripInGroup = (t_ushort) numFullComputeFoldPerGroup;
                            instructionIAControl.numActiveRowsInPartialFold = (t_uchar) numActiveElementsInPartialComputeFold;
                            unsigned char inputNeedsBitmaskPadding = (flagSparseInput == 0x00) ? 0x80 : 0x00;
                            instructionIAControl.flagPadBitmaskCatNumActiveCols = (t_uchar)
                                    (inputNeedsBitmaskPadding | (0x7F & numActiveCols));
    #if defined(SPARSE_SYSTEM)
                            if (COMPRESSION_WINDOW_SIZE % 8 != 0)
                            {
                                std::cout <<"The compression window size is not divisible by 8."<<std::endl;
                                throw;
                            }
                            unsigned int numClusterinChannelGroup = 1 + (numIAMoverInputChannelsPerGroup0-1) / CLUSTER_SIZE;
                            unsigned int partialBitmask = (numClusterinChannelGroup % COMPRESSION_WINDOW_SIZE == 0)
                                    ? 0 : ((1 << (numClusterinChannelGroup % COMPRESSION_WINDOW_SIZE)) - 1);
                            for (int i=0; i < (COMPRESSION_WINDOW_SIZE / 8); i++)
                            {
                                instructionIAControl.partialBitmask[i] = (partialBitmask >> (i*8)) & 0x0FF;
                            }
    #endif

                            vecIATileControlInstruction.push_back(instructionIAControl);
                         } // generate the ia controller instruction

                        /*! Generate the weight mover instruction*/
                        {
                            unsigned int filterIndex = iterInputGroup0 * numOutputChannelsPerGroupCurrentLayer;
                            t_weight_mover_instruction instructionWMover;
                            instructionWMover.numFiltersInGroup = (t_ushort) numOutputChannelsPerGroupCurrentLayer;
                            instructionWMover.numFullFilterFold = (t_ushort) numFullComputeFoldPerGroup;
                            instructionWMover.numFiltersInPartialFold = (t_uchar) numActiveElementsInPartialComputeFold;
                            instructionWMover.filterReuse = (t_ushort) maxTQPerCol * (t_ushort) maxTP;
                            instructionWMover.numActivePeCols = (t_uchar) numActiveCols;
                            instructionWMover.memBiasStart = (t_int)(memBiasStartIndex + filterIndex);
                            instructionWMover.memWeightStart = (t_int) memWeightDramBlockStartIndex +(t_int) filterIndex * memWeightDramBlockFilterStride;
                            instructionWMover.memWeightFilterStride = (t_int) memWeightDramBlockFilterStride;
                            #if defined(SPARSE_SYSTEM)
                                instructionWMover.memTBCountStart = (t_int) memWeightTBCountStart + (t_int) filterIndex;
                            #else
                                unsigned int numTBPerInputStrip =
                                        1+(numIAMoverInputChannelsPerGroup0-1)/CLUSTER_SIZE/TRANSFER_SIZE;
                                instructionWMover.numTBPerFilter = (t_uint) numTBPerInputStrip*kernelSize*kernelSize;
                            #endif
                            vecWeightMoverInstruction.push_back(instructionWMover);
                        } //Generate the weight mover instruction
                    }   //Instruction that only matters for convolution
                } //For. Transfer the first input blob, and convolution related items (if necessary)
            }

            //Generate the MISC instructions for misc operations
            //EXCEPT CONCATENATION
            if ((op != CONVOLUTION) && (op != CONCATENATION))
            {
                unsigned char opCodeField = 0X0;
                if ((op == ELT_ADD) || (op == AVG_POOL))
                {
                    //Like elmentwise addition, average pooling first requires adding the elements together
                    opCodeField = 0x0;
                }
                else if (op == MAX_POOL)
                {
                    opCodeField = 0x10;
                }

                t_misc_instruction instructionMisc;
                instructionMisc.controlBits = (t_uchar) (opCodeField |( numActiveCols & 0x0F));
                instructionMisc.numDramBlocksToReduce = (cl_ushort) numDramBlocksToReduceMK;
                instructionMisc.numOutputBlocksPerStrip = (cl_ushort) numOutputBlocksBlob0PerStripMK;
                instructionMisc.numOutputBlocks =
                        (cl_ushort) (maxTP * maxTQPerCol * numOutputBlocksBlob0PerStripMK) ;
                instructionMisc.numEffectiveValuesInLastStrip = (t_uchar) (
                            numOutputChannelsBlob0MK - (numOutputBlocksBlob0PerStripMK-1)*BURST_SIZE_BYTE);

                vecMiscInstruction.push_back(instructionMisc);
            } //Non-convolution stuff

            //Concatenation's IA instruction and misc instructions requires special attention
            if (op == CONCATENATION)
            {
                for (unsigned int iterMLocal=iterMGlobal;
                     iterMLocal < iterMGlobal+maxTP;
                     iterMLocal++)
                {
                    for (unsigned int iterNLocal=iterNGlobal;
                         iterNLocal < iterNGlobal+maxTQ;
                         iterNLocal++)
                    {
                        //Generage the IA instructions
                        {
                            bool isFirstTile = (iterMLocal == 0) && (iterNLocal == 0);
                            t_uchar actualSyncFlag = isFirstTile ? flagTensorSync : 0x0;

                            unsigned int iterMClipped = (iterMLocal < inputHeightPadding ) ?
                                        0 : ( (iterMLocal >= (inputHeightPadding + inputSPHeight)) ?
                                                  (inputSPHeight - 1) : (iterMLocal - inputHeightPadding));
                            unsigned int iterMDense = iterMClipped / inputSPHeightUnit;
                            unsigned int iterSPMIndex = iterMClipped % inputSPHeightUnit;

                            unsigned int iterNClipped = (iterNLocal < inputWidthPadding ) ?
                                        0 : ( (iterNLocal >= (inputWidthPadding + inputSPWidth)) ?
                                                  (inputSPWidth - 1) : (iterNLocal - inputWidthPadding));
                            unsigned int iterNDense = iterNClipped / inputSPWidthUnit;
                            unsigned int iterSPNIndex = iterNClipped % inputSPWidthUnit;

                            t_ia_mover_instruction instructionIA;
                            //Set the transport target. 0x1 means MISC
                            unsigned char flagTarget=  0x01;
                            unsigned char inputArrangement = 0x0;

                            t_uchar commonFlag = ( ( ((t_uchar) numActiveCols)& 0x0F)
                                                   | ((((t_uchar) flagTarget) & 0x01) << 0x04)
                                                   | ((((t_uchar) flagSparseInput) & 0x01) << 0x05) //Sparse flag for the input tensor
                                                   | ((((t_uchar) inputArrangement) & 0x01) << 0x06)
                                                  );

                            instructionIA.flagSyncCatInputArrangementCatSparseFlagCatDestinationCatNumActiveCols = (t_uchar)
                                    commonFlag | (actualSyncFlag << 0x07);

                            if (numActiveCols != 1)
                            {
                                std::cout <<"The operation is conatenation, but the number of active column is not 1."
                                          <<std::endl;
                                throw;
                            }

                            if (!(((flagIA0ShiftLeft == true) && (numIA0ShiftAmount >= 0))
                                 && ((flagIA1ShiftLeft == true) && (numIA1ShiftAmount >= 0))))
                            {
                                std::cout <<"Input left shift amount must be greater or equal to 0"
                                          <<std::endl;
                                throw;
                            }
                            instructionIA.inputShiftAmounts = (cl_uchar) (numIA0ShiftAmount & 0x0F);
                            instructionIA.memBlockStart0 = (t_int) (
                                        memIA0DramBlockStartIndex
                                        + memIA0DramBlockRowStride * iterMDense
                                        + memIA0DramBlockColStride * iterNDense);

                            //Concatenation can only work with dense input
                            #if defined(SPARSE_SYSTEM)
                                instructionIA.numCWOrTBInGroup =
                                        (t_ushort) (1 + (numIAMoverInputChannelsPerGroup0-1) / TRANSFER_SIZE / CLUSTER_SIZE);
                            #else
                                instructionIA.numTBPerStrip = (t_ushort) (1 + (numIAMoverInputChannelsPerGroup0-1) / TRANSFER_SIZE / CLUSTER_SIZE);
                            #endif
                                instructionIA.tileSPHeight = (t_uchar) 1;
                                instructionIA.tileSPWidth = (t_uchar) 1;
                                unsigned char inputTileLeftPadding = (iterNLocal < inputWidthPadding) ?
                                            inputWidthPadding : 0;
                                unsigned char inputTileRightPadding = ((iterNLocal + 1) > (inputWidthPadding + inputDenseWidth)) ?
                                            inputWidthPadding : 0;
                                unsigned char inputTileTopPadding = (iterMLocal < inputHeightPadding) ?
                                            inputHeightPadding : 0;
                                unsigned char inputTileBottomPadding = ((iterMLocal + 1) > (inputHeightPadding + inputDenseHeight)) ?
                                            inputHeightPadding : 0;
                                instructionIA.concatPadding = (t_uchar) (
                                                  (inputTileLeftPadding & 0x03)
                                                | ((inputTileRightPadding & 0x03) << 2)
                                                | ((inputTileTopPadding & 0x03) << 4)
                                                | ((inputTileBottomPadding & 0x03) << 6)
                                            );
                                instructionIA.concatInitSPIndices = (t_uchar) (
                                                 ( ((t_uchar) iterSPNIndex) & 0x0F)
                                                | (( ((t_uchar) iterSPMIndex) & 0x0F) << 0x04)
                                            );
                                instructionIA.concatSPSize = (t_uchar) (
                                                (((t_uchar) inputSPWidthUnit) & 0x0F)
                                                | ((((t_uchar) inputSPHeightUnit & 0x0F)) << 0x04)
                                            );
                                instructionIA.columnWidthStride = (t_uchar) 1;
                                instructionIA.columnSPWidth = (t_uchar) 1;

                                instructionIA.tileSPWidthxTileSPHeight = (t_ushort) 1;

                                vecIAMoverInstruction.push_back(instructionIA);

                                //Generate the IA instruction for the second tensor in the strip
                                instructionIA.flagSyncCatInputArrangementCatSparseFlagCatDestinationCatNumActiveCols = (t_uchar)
                                        commonFlag;
                                instructionIA.inputShiftAmounts = (cl_uchar) (numIA1ShiftAmount & 0x0F);
                                //Still use memBlockStart0
                                instructionIA.memBlockStart0 = (t_int) (
                                            memIA1DramBlockStartIndex
                                            + memIA1DramBlockRowStride * iterMDense
                                            + memIA1DramBlockColStride * iterNDense);

                                //Concatenation can only work with dense input
                                #if defined(SPARSE_SYSTEM)
                                    instructionIA.numCWOrTBInGroup =
                                            (t_ushort) (1 + (numIAMoverInputChannelsPerGroup1-1) / TRANSFER_SIZE / CLUSTER_SIZE);
                                #else
                                    instructionIA.numTBPerStrip = (t_ushort) (1 + (numIAMoverInputChannelsPerGroup1-1) / TRANSFER_SIZE / CLUSTER_SIZE);
                                #endif

                                vecIAMoverInstruction.push_back(instructionIA);
                            }

                        //Generate the Misc instructions
                        unsigned char opCodeField = 0X20;
                        //Instruciton 0
                        {
                            t_misc_instruction instructionMisc;
                            instructionMisc.controlBits = (t_uchar) (opCodeField |( numActiveCols & 0x0F));
                            instructionMisc.numDramBlocksToReduce = (cl_ushort) numDramBlocksToReduceMK;
                            instructionMisc.numOutputBlocksPerStrip = (cl_ushort) numOutputBlocksBlob0PerStripMK;
                            instructionMisc.numOutputBlocks = (cl_ushort) numOutputBlocksBlob0PerStripMK;
                            instructionMisc.numEffectiveValuesInLastStrip = (t_uchar) (
                                        numOutputChannelsBlob0MK - (numOutputBlocksBlob0PerStripMK-1)*BURST_SIZE_BYTE);

                            vecMiscInstruction.push_back(instructionMisc);
                         } //Misc instruction 0

                        //Instruction 1
                        {
                            t_misc_instruction instructionMisc;
                            instructionMisc.controlBits = (t_uchar) (opCodeField |( numActiveCols & 0x0F));
                            instructionMisc.numDramBlocksToReduce = (cl_ushort) numDramBlocksToReduceMK;
                            instructionMisc.numOutputBlocksPerStrip = (cl_ushort) numOutputBlocksBlob1PerStripMK;
                            instructionMisc.numOutputBlocks = (cl_ushort) numOutputBlocksBlob1PerStripMK;
                            instructionMisc.numEffectiveValuesInLastStrip = (t_uchar) (
                                        numOutputChannelsBlob1MK - (numOutputBlocksBlob1PerStripMK-1)*BURST_SIZE_BYTE);

                            vecMiscInstruction.push_back(instructionMisc);
                        }   //Misc instruction 1
                    } //iterNLocal for concatenation
                } //iterMLocal for concatenation
            } //If concatenation

            iterQGlobal += maxTQ;
            iterNGlobal += ((unsigned int)kernelStride)*maxTQ;
        } //for iterQTile
        iterPGlobal += maxTP;
        iterMGlobal += ((unsigned int)kernelStride)*maxTP;
    } //for iterPTile
}

int ia_cache_boundary_check(
      int heightTile,
      int widthTile,
      int numDramBlockPerDenseStrip
        )
{
    int requirement = numDramBlockPerDenseStrip*heightTile*widthTile;
    return requirement;
}

int ia_tbcount_cache_boundary_check(
      int heightTile,
      int widthTile
        )
{
    return (heightTile * widthTile);
}


int oa_cache_boundary_check(
      int heightTile,
      int widthTile,
      int numChannels
        )
{
    int roundedNumChannels = ((numChannels - 1) / CLUSTER_SIZE + 1) * CLUSTER_SIZE;
    return (heightTile*widthTile* roundedNumChannels);
}

int filter_cache_boundary_check(
      int kernelSize,
      int inputChannelSize
        )
{
    //Nubmer of dram blocks required to store one filter
#if defined(SPARSE_SYSTEM)
    unsigned int numTransferBlocksPerCompressionBlock
            =  (COMPRESSION_WINDOW_SIZE + TRANSFER_SIZE - 1) / TRANSFER_SIZE + 1;
    unsigned int numCompressionBlocksInChannel =
            1 + (inputChannelSize - 1) / (COMPRESSION_WINDOW_SIZE * CLUSTER_SIZE);
    unsigned int tempStride = kernelSize*kernelSize*((unsigned int) numTransferBlocksPerCompressionBlock* (unsigned int) numCompressionBlocksInChannel);
    //externalMemoryAddressStride = lcm(tempStride, (unsigned int) WIDE_SIZE); //DRAM stride needs to be a multiple of DRAM width and the storage requirement per filter
    int requirement = (tempStride-1) / WIDE_SIZE + 1;
#else
    int requirement = (inputChannelSize * kernelSize * kernelSize - 1) / BURST_SIZE_BYTE + 1;
#endif

    return requirement;
}

t_graph_output_tile_info deriveConvOutputTileShape(
        unsigned int outputHeight,
        unsigned int outputWidth,
        unsigned int sizeOutputFullTileHeight,
        unsigned int sizeOutputFullTileWidthPerCol
        )
{
    t_graph_output_tile_info tileInfo;
    tileInfo.sizeOutputTileFullHeight = sizeOutputFullTileHeight;
    tileInfo.sizeOutputTileFullWidthPerCol = sizeOutputFullTileWidthPerCol;
    tileInfo.numOutputTileAlongHeight =
            1 + (outputHeight-1) / sizeOutputFullTileHeight;
    tileInfo.numFullOutputTileAlongHeight =
            outputHeight / sizeOutputFullTileHeight;

    unsigned int sizeOutputTileFullWidth =
            sizeOutputFullTileWidthPerCol * PE_COLS;
    tileInfo.numOutputTileAlongWidth =
            1 + (outputWidth-1) / sizeOutputTileFullWidth;
    tileInfo.numFullOutputTileAlongWidth =
            outputWidth / sizeOutputTileFullWidth;

    tileInfo.sizeOutputTilePartialHeight =
            outputHeight % sizeOutputFullTileHeight;

    unsigned int sizeOutputTilePartialWidth =
            outputWidth % sizeOutputTileFullWidth;

    if (sizeOutputTilePartialWidth == 0)
    {
        tileInfo.numActiveColsForPartialWidthTile = 0;
        tileInfo.sizeOutputTilePartialWidthPerCol =
                sizeOutputFullTileWidthPerCol;
    }
    else
    {
        unsigned int numColsForPartialWidthTile = PE_COLS;
        while (sizeOutputTilePartialWidth % numColsForPartialWidthTile != 0)
        {
            numColsForPartialWidthTile--;
        }
        tileInfo.sizeOutputTilePartialWidthPerCol =
                sizeOutputTilePartialWidth / numColsForPartialWidthTile;
        tileInfo.numActiveColsForPartialWidthTile =
                numColsForPartialWidthTile;
    }

    return tileInfo;
}


unsigned int deriveConvInputDimension1D(
        unsigned int outputDimension1D,
        unsigned int kernelSize,
        unsigned int kernelStride
        )
{
    unsigned int inputSPWithBorderPaddingDimension1D =
            (outputDimension1D - 1) * kernelStride + kernelSize;
    return inputSPWithBorderPaddingDimension1D;
}

unsigned int deriveConvComputationLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numOutputChannelsPerGroup,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _numGroups,
        unsigned int _sizeKernel
        )
{
    unsigned int numPERowFoldPerGroup =
            1 + (_numOutputChannelsPerGroup-1) / PE_ROWS;
    unsigned int numTranferBlockPerInputGroup =
            1 + (_numInputChannelsPerGroup-1) / (TRANSFER_SIZE * CLUSTER_SIZE);
    unsigned int latency =
            _numGroups * numPERowFoldPerGroup * numTranferBlockPerInputGroup * _sizeKernel * _sizeKernel
            * (
                _outputTileInfo.numFullOutputTileAlongHeight
                    * ( _outputTileInfo.numFullOutputTileAlongWidth
                        * _outputTileInfo.sizeOutputTileFullWidthPerCol
                        * _outputTileInfo.sizeOutputTileFullHeight
                        + (_outputTileInfo.numOutputTileAlongWidth - _outputTileInfo.numFullOutputTileAlongWidth)
                          * _outputTileInfo.sizeOutputTileFullHeight * _outputTileInfo.sizeOutputTilePartialWidthPerCol
                      )
                +
                  (_outputTileInfo.numOutputTileAlongHeight -  _outputTileInfo.numFullOutputTileAlongHeight)
                    * ( _outputTileInfo.numFullOutputTileAlongWidth
                        * _outputTileInfo.sizeOutputTileFullWidthPerCol
                        * _outputTileInfo.sizeOutputTilePartialHeight
                        + (_outputTileInfo.numOutputTileAlongWidth - _outputTileInfo.numFullOutputTileAlongWidth)
                          * _outputTileInfo.sizeOutputTilePartialHeight * _outputTileInfo.sizeOutputTilePartialWidthPerCol
                      )
              );
    return latency;
}

unsigned int deriveConvInputTransferLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _numGroups,
        unsigned int _sizeKernel,
        unsigned _sizeStride
        )
{
    unsigned int numFullOutputTileAlongHeight =
            _outputTileInfo.numFullOutputTileAlongHeight;
    unsigned int numPartialTileAlongHeight =
            _outputTileInfo.numOutputTileAlongHeight - numFullOutputTileAlongHeight;
    unsigned int numFullOutputTileAlongWidth =
            _outputTileInfo.numFullOutputTileAlongWidth;
    unsigned int numPartialTileAlongWidth =
            _outputTileInfo.numOutputTileAlongWidth - numFullOutputTileAlongWidth;
    unsigned int sizeFullTileInputHeight =
            deriveConvInputDimension1D(
                    //unsigned int outputDimension1D,
                    _outputTileInfo.sizeOutputTileFullHeight,
                    //unsigned int kernelSize,
                    _sizeKernel,
                    //unsigned int kernelStride
                    _sizeStride
                );
    unsigned int sizePartialTileInputHeight =
            deriveConvInputDimension1D(
                    //unsigned int outputDimension1D,
                    _outputTileInfo.sizeOutputTilePartialHeight,
                    //unsigned int kernelSize,
                    _sizeKernel,
                    //unsigned int kernelStride
                    _sizeStride
                );
    unsigned int sizeFullTileInputWidth =
            deriveConvInputDimension1D(
                    //unsigned int outputDimension1D,
                    _outputTileInfo.sizeOutputTileFullWidthPerCol * PE_COLS,
                    //unsigned int kernelSize,
                    _sizeKernel,
                    //unsigned int kernelStride
                    _sizeStride
                );
    unsigned int sizePartialTileInputWidth =
            deriveConvInputDimension1D(
                    //unsigned int outputDimension1D,
                    _outputTileInfo.sizeOutputTilePartialWidthPerCol * _outputTileInfo.numActiveColsForPartialWidthTile,
                    //unsigned int kernelSize,
                    _sizeKernel,
                    //unsigned int kernelStride
                    _sizeStride
                );

    unsigned int numTransferBlockPerChannelGroup = 1 + ( _numInputChannelsPerGroup -1) / (CLUSTER_SIZE*TRANSFER_SIZE);
    unsigned int numDramBlockPerStrip = 1 + (numTransferBlockPerChannelGroup-1) / WIDE_SIZE;

    unsigned int latency =
            _numGroups * numDramBlockPerStrip
            * (
                numFullOutputTileAlongHeight*numFullOutputTileAlongWidth*sizeFullTileInputHeight*sizeFullTileInputWidth
                + numFullOutputTileAlongHeight*numPartialTileAlongWidth*sizeFullTileInputHeight*sizePartialTileInputWidth
                + numPartialTileAlongHeight*numFullOutputTileAlongWidth*sizePartialTileInputHeight*sizeFullTileInputWidth
                + numPartialTileAlongHeight*numPartialTileAlongWidth*sizePartialTileInputHeight*sizePartialTileInputWidth
              );
    return latency;
 }

unsigned int deriveConvOutputTransferLatency(
        unsigned int _outputHeight,
        unsigned int _outputWidth,
        unsigned int _numOutputChannelsPerNextGroup,
        unsigned int _numNextGroups
        )
{
    unsigned int numTransferBlockPerChannelGroup = 1 + ( _numOutputChannelsPerNextGroup -1) / (CLUSTER_SIZE*TRANSFER_SIZE);
    unsigned int numDramBlockPerStrip = 1 + (numTransferBlockPerChannelGroup-1) / WIDE_SIZE;

    unsigned int latency = _outputHeight * _outputWidth * numDramBlockPerStrip * _numNextGroups;
    return latency;
}

unsigned int deriveConvWeightTransferLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _numOutputChannelsPerGroup,
        unsigned int _numGroups,
        unsigned int _sizeKernel
        )
{
    unsigned int numTileAlongHeight = _outputTileInfo.numOutputTileAlongHeight;
    unsigned int numTileAlongWidth = _outputTileInfo.numOutputTileAlongWidth;
    unsigned int numTBPerStrip = 1 + (_numInputChannelsPerGroup - 1) / (CLUSTER_SIZE * TRANSFER_SIZE);
    unsigned int numDramBlocksInFilter =
            1 + (_sizeKernel*_sizeKernel*numTBPerStrip - 1) / WIDE_SIZE;
    unsigned int latency =
            _numGroups * _numOutputChannelsPerGroup * numTileAlongHeight * numTileAlongWidth * numDramBlocksInFilter;

    return latency;
}

