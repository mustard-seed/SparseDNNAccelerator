#include "layerInstructionGenerator.hpp"

#include <cassert>
#include <iostream>

#define DIVIDE_CEIL(x, y) (1 + (x-1) / (y))

void instruction_generator(//Type of the operation
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

        //Input activation blob 0 strides in terms of activation words
        //Assuming GHWC layout
        signed int memIA0ColStride,
        signed int memIA0RowStride,

        //Input activation blob 1 strides in terms of activation words
        //Assuming GHWC layout
        signed int memIA1ColStride,
        signed int memIA1RowStride,

        //Output activation blob stride in terms of DRAM block
        //Assuming GHWC layout
        signed int memOAColStride,
        signed int memWeightDramBlockFilterStride,
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

        #if defined(SPW_SYSTEM)
          //Number of NZ clusters in a pruning range. Only useful for convolution
          unsigned int numNZClustersInPruningRange,
        #endif

        unsigned char _sizeOutputTileFullHeight,
        unsigned char _sizeOutputTileFullWidthPerCol,
        unsigned char _numActiveColsPartialOutputTile,

        //Number of channels in input blobs 0 and 1
        //Only element-wise addition and pooling should use the second  blob
        unsigned short numInputChannels0,
        unsigned short numInputChannels1,

        //Number of groups in the current layer's output.
        //Only relevant for convolution
        unsigned short numGroupsCurrentLayer,

        //Number of output channels
        //Only relevant for convolution
        unsigned short numOutputChannels)
{
    /*!
     * Important (20201224): Only allow the output tile height size to exceed 1 if the operation is
     * convolution, elt_add, or concatenation
     * Pooling operation cannot handle verticle size greater than 1 because MISC unit cannot handle input overlap
     */
    unsigned int sizeOutputTileFullHeight = ((op == CONVOLUTION) || (op == ELT_ADD) || (op == CONCATENATION) ) ?
                _sizeOutputTileFullHeight : 1;

    unsigned int sizeOutputTileFullWidthPerCol = ( (op == CONVOLUTION)  || (op == ELT_ADD) || (op == CONCATENATION) ) ?
                _sizeOutputTileFullWidthPerCol : 1;


    unsigned int outputHeight = ( ((unsigned int) inputSPHeight)
            + 2*((unsigned int) inputHeightPadding) - ((unsigned int) kernelSize))
            / ((unsigned int) kernelStride) + 1;

    unsigned int outputWidth = ( ((unsigned int) inputSPWidth)
            + 2* ((unsigned int) inputWidthPadding) - ((unsigned int) kernelSize))
            / ((unsigned int) kernelStride) + 1;

    unsigned char numActiveColsFullOutputTile =
            (op == CONVOLUTION) ?  PE_COLS : 1;

    unsigned char numActiveColsPartialOutputTile =
            ((op == MAX_POOL) || (op == AVG_POOL) || (op == ELT_ADD) || (op == CONCATENATION)) ?
                (outputWidth % numActiveColsFullOutputTile)
                : _numActiveColsPartialOutputTile;

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

    //NEW: added for the SpW system
    //Number of groups seen by the system.
    //Override
    unsigned int numEffectiveGroups = numGroupsCurrentLayer;


    //Number of groups current layer
    unsigned int numOAGroupsCurrentLayer;
    unsigned int numOAChannelsPerGroup = 0;
    //Number of nominal dram blocks in an output strip,
    //override if necessary
    unsigned int numNominalDramBlocksPerOutputStrip0 = 0;
    unsigned int numNominalDramBlocksPerOutputStrip1 = 0;


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
            if ((numOutputChannels % numGroupsCurrentLayer != 0)
                    || (numInputChannels0 % numGroupsCurrentLayer != 0))
            {
                std::cout <<"[layer instruction generator] Number of input or output channels is not divisble by the number of groups in the current layer."<<std::endl;
                throw;
            }
            numEffectiveGroups = numGroupsCurrentLayer;
            numOAGroupsCurrentLayer = numEffectiveGroups;
            numOAChannelsPerGroup = numOutputChannels / numEffectiveGroups;
            numNominalDramBlocksPerOutputStrip0 =
                   (t_ushort) DIVIDE_CEIL(numOAChannelsPerGroup, ACTIVATION_BURST_SIZE_BYTE);
            numIAMoverInputChannelsPerGroup0 = numInputChannels0 / numGroupsCurrentLayer;
            numIAMoverInputChannelsPerGroup1  = 0;
            numIAMoverGroup0 = numEffectiveGroups;
            numIAMoverGroup1 = 0;
            numActiveElementsInFullComputeFold = PE_ROWS;
//            memIA0DramBlockGroupStride = memIA0GroupStride;
            numOutputChannelsBlob0MK = 0;
            numOutputChannelsBlob1MK = 0;
//            numOutputBlocksBlob0MK = 0;
//            numOutputBlocksBlob1MK = 0;
            numOutputBlocksBlob0PerStripMK = 0;
            numOutputBlocksBlob1PerStripMK = 0;
            numDramBlocksToReduceMK = 0;

            int filterCacheRequirement = filter_cache_boundary_check(
                            kernelSize,
                            numInputChannels0,
                            PE_SIMD_SIZE * CLUSTER_SIZE,
                            #if defined(SPW_SYSTEM)
                                PRUNE_RANGE_IN_CLUSTER,
                                numNZClustersInPruningRange
                            #else
                                1,
                                1
                            #endif
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
            if (kernelSize != 1)
            {
                std::cout <<"[layer instruction generator] Operation is concatenation, "
                            "but the kernel size is not 1"<<std::endl;
                throw;
            }
            if (numGroupsCurrentLayer != 1)
            {
                std::cout <<"[layer instruciton generator] Operation is CONCATENATION, "
                             "but the number of groups of current layer is not 1."<<std::endl;
                throw;
            }
            numEffectiveGroups = 1;
            numOAGroupsCurrentLayer = numEffectiveGroups;
            numOAChannelsPerGroup = numOutputChannels;
            numIAMoverInputChannelsPerGroup0 = numInputChannels0;
            numIAMoverInputChannelsPerGroup1 = numInputChannels1;
            numIAMoverGroup0 = 1;
            numIAMoverGroup1 = 1;
            numActiveElementsInFullComputeFold = ACTIVATION_BURST_SIZE_BYTE;
            numOutputChannelsBlob0MK = numInputChannels0;
            numOutputChannelsBlob1MK = numInputChannels1;
//            numOutputBlocksBlob0MK = 1+ (numOutputChannelsBlob0MK-1) / BURST_SIZE_BYTE;
//            numOutputBlocksBlob1MK = 1+ (numOutputChannelsBlob1MK-1) / BURST_SIZE_BYTE;
            numOutputBlocksBlob0PerStripMK = DIVIDE_CEIL(numOutputChannelsBlob0MK, ACTIVATION_BURST_SIZE_BYTE);
            numOutputBlocksBlob1PerStripMK = DIVIDE_CEIL(numOutputChannelsBlob1MK, ACTIVATION_BURST_SIZE_BYTE);
            numNominalDramBlocksPerOutputStrip0 = DIVIDE_CEIL(numInputChannels0, ACTIVATION_BURST_SIZE_BYTE);
            numNominalDramBlocksPerOutputStrip1 = DIVIDE_CEIL(numInputChannels1, ACTIVATION_BURST_SIZE_BYTE);
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
            if (numGroupsCurrentLayer != 1)
            {
                std::cout <<"[layer instruciton generator] Operation is MAX_POOL, "
                             "but the number of groups of current layer is not 1."<<std::endl;
                throw;
            }
            numOAChannelsPerGroup = ACTIVATION_BURST_SIZE_BYTE;
            numNominalDramBlocksPerOutputStrip0 = 1;
            numEffectiveGroups = DIVIDE_CEIL(numOutputChannels, numOAChannelsPerGroup);
            numIAMoverInputChannelsPerGroup0 = ACTIVATION_BURST_SIZE_BYTE;
            numIAMoverInputChannelsPerGroup1 = 0;
            numIAMoverGroup0 = numEffectiveGroups;
            numIAMoverGroup1 = 0;
            numActiveElementsInFullComputeFold = ACTIVATION_BURST_SIZE_BYTE;
            numOAGroupsCurrentLayer = numEffectiveGroups;
            numOutputChannelsBlob0MK = numInputChannels0;
            numOutputChannelsBlob1MK = 0;
            numOutputBlocksBlob0PerStripMK = DIVIDE_CEIL(numOutputChannelsBlob0MK, ACTIVATION_BURST_SIZE_BYTE);
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
            if (kernelSize != 1)
            {
                std::cout <<"[layer instruction generator] Operation is elt-add, "
                            "but the kernel size is not 1."<<std::endl;
                throw;
            }
            if (numGroupsCurrentLayer != 1)
            {
                std::cout <<"[layer instruciton generator] Operation is ELT_ADD, "
                             "but the number of groups of current layer is not 1."<<std::endl;
                throw;
            }
            numEffectiveGroups = numGroupsCurrentLayer;
            numOAGroupsCurrentLayer = numEffectiveGroups;
            numOAChannelsPerGroup = numOutputChannels;
            numNominalDramBlocksPerOutputStrip0 =
                   (t_ushort) DIVIDE_CEIL(numOAChannelsPerGroup, ACTIVATION_BURST_SIZE_BYTE);
            numIAMoverInputChannelsPerGroup0 = numInputChannels0;
            numIAMoverInputChannelsPerGroup1 = numInputChannels1;
            numIAMoverGroup0 = 1;
            numIAMoverGroup1 = 0;;
            numActiveElementsInFullComputeFold = ACTIVATION_BURST_SIZE_BYTE;
            numOAGroupsCurrentLayer = 1;
            numOutputChannelsBlob0MK = numInputChannels0;
            numOutputChannelsBlob1MK = 0;
            numOutputBlocksBlob0PerStripMK = DIVIDE_CEIL(numOutputChannelsBlob0MK, ACTIVATION_BURST_SIZE_BYTE);
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
            if (numGroupsCurrentLayer != 1)
            {
                std::cout <<"[layer instruciton generator] Operation is AVG_POOL, "
                             "but the number of groups of current layer is not 1."<<std::endl;
                throw;
            }
            numOAChannelsPerGroup = ACTIVATION_BURST_SIZE_BYTE;
            numNominalDramBlocksPerOutputStrip0 = 1;
            numIAMoverInputChannelsPerGroup0 = ACTIVATION_BURST_SIZE_BYTE;
            numIAMoverInputChannelsPerGroup1 = 0;
            numEffectiveGroups = DIVIDE_CEIL(numOutputChannels, numOAChannelsPerGroup);
            numOAGroupsCurrentLayer = numEffectiveGroups;
            numIAMoverGroup0 = numEffectiveGroups;
            numIAMoverGroup1 = 0;
            numActiveElementsInFullComputeFold = ACTIVATION_BURST_SIZE_BYTE;
            numOAGroupsCurrentLayer = 1;
            numOutputChannelsBlob0MK = numInputChannels0;
            numOutputChannelsBlob1MK = 0;
            numOutputBlocksBlob0PerStripMK = DIVIDE_CEIL(numOutputChannelsBlob0MK, ACTIVATION_BURST_SIZE_BYTE);
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

    unsigned int numComputeFoldPerGroup = DIVIDE_CEIL(numOAChannelsPerGroup, numActiveElementsInFullComputeFold);
    unsigned int numFullComputeFoldPerGroup = numOAChannelsPerGroup / numActiveElementsInFullComputeFold;
    unsigned int numActiveElementsInPartialComputeFold = numOAChannelsPerGroup % numActiveElementsInFullComputeFold;

    for (unsigned int iterGroup=0; iterGroup < numEffectiveGroups; iterGroup++)
    {
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
                    if (oaCacheRequirement > OA_CACHE_DEPTH)
                    {
                        std::cout << "OA tile size is too big to fit inside the oa cache. "
                                  << "output height per tile, output width per col: "
                                  << maxTP <<" "<<maxTQPerCol<<std::endl;
                        throw;
                    }

                    int numIATBInCachePerStrip =
                            DIVIDE_CEIL(numIAMoverInputChannelsPerGroup0, PE_ACTIVATION_BLOCK_SIZE_IN_WORD);
                    int numIABurstBlocksPerStrip =
                            DIVIDE_CEIL(numIATBInCachePerStrip, ACTIVATION_WIDE_SIZE);
                    int iaCacheRequirementInDramBlock = ia_cache_boundary_check(
                                //heightTile
                                maxTM,
                                //widthTile
                                maxTNPerCol,
                                //numDramBlockPerDenseStrip,
                                numIABurstBlocksPerStrip
                                );
                    if (iaCacheRequirementInDramBlock > IA_CACHE_DEPTH)
                    {
                        std::cout <<"IA tile size is too big to fit inside the IA buffer cache"<<std::endl;
                        std::cout <<"input tile height, input tile width per col, numIABurstBlocksPerStrip: "
                                 <<" "<<maxTM<<" "<<maxTNPerCol<<" "<<numIABurstBlocksPerStrip<<std::endl;
                        throw;
                    }
                }

                /*! Generate the first output OA instruction */
                {
                    int numOAInstructionPerTile = (op == CONCATENATION) ? 2 : 1;
                    for (int iInst=0; iInst<numOAInstructionPerTile; iInst++)
                    {
                        t_oa_mover_instruction instructionOA;
                        bool isFirstTile =
                                ( iterGroup == 0)
                                && (iterPTile == 0)
                                && (iterQTile == 0)
                                && (iInst == 0);
                        unsigned char actualFlagOutputSync = isFirstTile ? flagTensorSync : 0x0;
                        instructionOA.memSelectCatSparseFlagCatSyncFlagCatNumActiveCols =
                                ((t_uchar) numActiveCols & 0x0F)
                                | ((((t_uchar) actualFlagOutputSync) & 0x01) << 0x04);
                        instructionOA.memOAStart = (t_int)
                                (memOADramBlockStartIndex * ACTIVATION_BURST_SIZE_BYTE
                                  + (   (unsigned int) iterPGlobal*outputWidth +
                                        (unsigned int) iterQGlobal
                                    ) * memOAColStride
                                  + numOAChannelsPerGroup * iterGroup);

                        instructionOA.memOAPEColStride = (t_uint)((unsigned int) maxTQPerCol * memOAColStride);
                        instructionOA.memOARowStride = (t_ushort)((unsigned int) outputWidth * memOAColStride);
                        instructionOA.memOAColStride = (t_ushort) memOAColStride;
                        instructionOA.numNominalDramBlocksPerStrip = (t_ushort) numNominalDramBlocksPerOutputStrip0;
                        instructionOA.tileHeight = (t_uchar) maxTP;
                        instructionOA.columnTileWidth = (t_uchar) maxTQPerCol;

                        //Fix-up the memOAStart field when we are dealing with conatentation
                        if (iInst == 1)
                        {
                            instructionOA.memOAStart = (t_int)
                                    (memOADramBlockStartIndex * ACTIVATION_BURST_SIZE_BYTE
                                      + (   (unsigned int) iterPGlobal*outputWidth +
                                            (unsigned int) iterQGlobal
                                        ) * memOAColStride
                                      + numInputChannels0);
                            instructionOA.numNominalDramBlocksPerStrip = (t_ushort) numNominalDramBlocksPerOutputStrip1;
                        }

                        vecOAMoverInstruction.push_back(instructionOA);
                    }
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
                  Transfer the IAs. Two blobs if necessary
                */
                {
                    int numIAInstructionsPerTile = (op == CONCATENATION) ? 2 : 1;
                    for (int iInst=0; iInst < numIAInstructionsPerTile; iInst++)
                    {
                        bool isFirstTile =
                                (iterGroup == 0)
                                && (iterPTile == 0)
                                && (iterQTile == 0)
                                && (iInst == 0);

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
                                    memIA0DramBlockStartIndex * ACTIVATION_BURST_SIZE_BYTE
                                    + memIA0RowStride * iterMDense
                                    + memIA0ColStride * iterNDense
                                    + iterGroup * numIAMoverInputChannelsPerGroup0);

                        instructionIA.memBlockStart1 = (t_int) (
                                    memIA1DramBlockStartIndex * ACTIVATION_BURST_SIZE_BYTE
                                    + memIA1RowStride * iterMDense
                                    + memIA1ColStride * iterNDense
                                    + iterGroup * numIAMoverInputChannelsPerGroup1);

                        instructionIA.memBlockColStripStride = (t_ushort)memIA0ColStride;
                        instructionIA.memBlockRowStripStride = (t_ushort)memIA0RowStride;


                        //TODO: DOUBLE CHECK THE SETTING OF NUM TB PER STRIP?
//                        instructionIA.numTBPerStrip = (op == CONVOLUTION) ?
//                                    (t_ushort) DIVIDE_CEIL(numIAMoverInputChannelsPerGroup0, PE_ACTIVATION_BLOCK_SIZE_IN_WORD)
//                                  : ACTIVATION_WIDE_SIZE;
                        instructionIA.numTBPerStrip = ((op == CONVOLUTION) || (op == ELT_ADD) || (op == CONCATENATION)) ?
                                    (t_ushort) DIVIDE_CEIL(numIAMoverInputChannelsPerGroup0, PE_ACTIVATION_BLOCK_SIZE_IN_WORD)
                                  : ACTIVATION_WIDE_SIZE;
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

                        //Concatentation fix-up
                        if (iInst == 1)
                        {
                            instructionIA.memBlockStart0 = (t_int) (
                                    memIA1DramBlockStartIndex * ACTIVATION_BURST_SIZE_BYTE
                                    + memIA1RowStride * iterMDense
                                    + memIA1ColStride * iterNDense
                                    + iterGroup * numIAMoverInputChannelsPerGroup1);
                            instructionIA.numTBPerStrip
                                    = (t_ushort) DIVIDE_CEIL(numIAMoverInputChannelsPerGroup1, PE_ACTIVATION_BLOCK_SIZE_IN_WORD);
                            instructionIA.memBlockColStripStride = (t_ushort)memIA1ColStride;
                            instructionIA.memBlockRowStripStride = (t_ushort)memIA1RowStride;
                        }

                        vecIAMoverInstruction.push_back(instructionIA);
                    }

                } //Block. Transfer the IAs. Two blobs if necessary.

                //Instruction that only matters for convolution
                //IA tile controller instruction
                //OA tile controller instruction
                //Weight controller instruction
                //TODO: consider the IA tile contoller instruction more carefully
                if (op == CONVOLUTION)
                {
                    /*! Generate the output tile controller instruction.  */
                    {
                        t_oa_tile_controller_instruction instructionOAControl;

                        instructionOAControl.numLocalTilePerColHxW = (t_uchar)(maxTQPerCol*maxTP);
                        instructionOAControl.numBurstAlignedChannelsPerCurrentGroup =
                                (t_ushort) (
                                    DIVIDE_CEIL(numOAChannelsPerGroup, ACTIVATION_BURST_SIZE_BYTE)
                                    * ACTIVATION_BURST_SIZE_BYTE);
                        instructionOAControl.numDrainInstructions = (t_ushort) numComputeFoldPerGroup;
                        instructionOAControl.numFoldsInGroupCurrentLayer = (t_ushort) numComputeFoldPerGroup;
                        instructionOAControl.numFullFoldsInCurrentLayer = (t_ushort) numFullComputeFoldPerGroup;
                        instructionOAControl.numActiveElementsInFullFold = (t_ushort) numActiveElementsInFullComputeFold;
                        instructionOAControl.numActiveElementsInPartialFold = (t_ushort) numActiveElementsInPartialComputeFold;
                        instructionOAControl.numActiveCols = (t_uchar) numActiveCols;
                        instructionOAControl.numNominalDramBlocksPerStrip = numNominalDramBlocksPerOutputStrip0;

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
                                    | (t_uchar)((sourceIsMisc & 0x01) << 0x6) //drain source is convolution
                                );
                        vecOATileControlInstruction.push_back(instructionOAControl);
                    }

                     /*! Send the input IA controller instruction */
                    {
                        t_ia_tile_controller_instruction instructionIAControl;
                        instructionIAControl.localTileWidth = (t_uchar) maxTNPerCol;
                        instructionIAControl.localTileHeight = (t_uchar) maxTM;
                        instructionIAControl.kernelStride = (t_uchar) kernelStride;
                        instructionIAControl.kernelSize = (t_uchar) kernelSize;
                        instructionIAControl.numOutputInstructions = (t_uint)
                                ((t_uint) maxTP * (t_uint) maxTQPerCol * numComputeFoldPerGroup * (t_uint) kernelSize);
                        //Number of TBs in a strip seen by the IA buffer
                        //(not the same as an IA strip seen by the IA mover when performing multi-grouped convolution
                        unsigned short cacheIAStripColStrideTBCount = DIVIDE_CEIL(numIAMoverInputChannelsPerGroup0, PE_ACTIVATION_BLOCK_SIZE_IN_WORD);
                        instructionIAControl.cacheIAStripColStride = DIVIDE_CEIL(cacheIAStripColStrideTBCount, ACTIVATION_WIDE_SIZE);
                        instructionIAControl.numOutputChannelsInGroup = (t_ushort) numOAChannelsPerGroup;
                        //unsigned char inputNeedsBitmaskPadding = (flagSparseInput == 0x00) ? 0x80 : 0x00;
//                            instructionIAControl.flagPadBitmaskCatNumActiveCols = (t_uchar)
//                                    (inputNeedsBitmaskPadding | (0x7F & numActiveCols));
                        instructionIAControl.flagPadBitmaskCatNumActiveCols = (t_uchar) (0x7F & numActiveCols);

                        vecIATileControlInstruction.push_back(instructionIAControl);
                     } // generate the ia controller instruction

                    /*! Generate the weight mover instruction*/
                    {
                        unsigned int filterIndex = iterGroup * numOAChannelsPerGroup;
                        t_weight_mover_instruction instructionWMover;
                        instructionWMover.numFiltersInGroup = (t_ushort) numOAChannelsPerGroup;
                        instructionWMover.numFullFilterFold = (t_ushort) numFullComputeFoldPerGroup;
                        instructionWMover.numFiltersInPartialFold = (t_uchar) numActiveElementsInPartialComputeFold;
                        instructionWMover.filterReuse = (t_ushort) maxTQPerCol * (t_ushort) maxTP;
                        instructionWMover.numActivePeCols = (t_uchar) numActiveCols;
                        instructionWMover.memBiasStart = (t_int)(memBiasStartIndex + filterIndex);
                        instructionWMover.memWeightStart = (t_int) memWeightDramBlockStartIndex +(t_int) filterIndex * memWeightDramBlockFilterStride;
                        instructionWMover.memWeightFilterStride = (t_int) memWeightDramBlockFilterStride;
                        //TODO: double check the following calculation of the number of weight TB per strip
                        //is in agreement with the layout of weights in the off-chip memory
                        #if defined(SPW_SYSTEM)
                             unsigned int numTBPerInputStrip =
                                     DIVIDE_CEIL(numIAMoverInputChannelsPerGroup0,
                                                 PE_SIMD_SIZE * CLUSTER_SIZE * PRUNE_RANGE_IN_CLUSTER)
                                     * numNZClustersInPruningRange;
                        #else
                            unsigned int numTBPerInputStrip =
                                    DIVIDE_CEIL(
                                            numIAMoverInputChannelsPerGroup0,
                                            PE_SIMD_SIZE * CLUSTER_SIZE
                                        );
                        #endif
                        instructionWMover.numTBPerFilter = (t_uint) numTBPerInputStrip*kernelSize*kernelSize;

                        #if defined(SPW_SYSTEM)
                            instructionWMover.numNZClustersPerPruneRange = numNZClustersInPruningRange;
                        #endif
                        vecWeightMoverInstruction.push_back(instructionWMover);
                    } //Generate the weight mover instruction
                }   //Instruction that only matters for convolution

                //Generate the MISC instructions for misc operations
                if (op != CONVOLUTION)
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
                    else if (op == CONCATENATION)
                    {
                        opCodeField = 0x20;
                    }

                    t_misc_instruction instructionMisc;
                    instructionMisc.controlBits = (t_uchar) (opCodeField |( numActiveCols & 0x0F));
                    instructionMisc.numDramBlocksToReduce = (cl_ushort) numDramBlocksToReduceMK;
                    instructionMisc.numOutputBlocksPerUnit =
                            (cl_ushort) maxTP * (cl_ushort) maxTQPerCol * (cl_ushort) numOutputBlocksBlob0PerStripMK;

                    //Fix up for concatenation
                    if (op == CONCATENATION)
                    {
                        instructionMisc.numOutputBlocksPerUnit =
                                (cl_ushort) maxTP * (cl_ushort) maxTQPerCol
                                * (cl_ushort) (numOutputBlocksBlob0PerStripMK + numOutputBlocksBlob1PerStripMK);
                    }
                    vecMiscInstruction.push_back(instructionMisc);
                } //Non-convolution stuff

                iterQGlobal += maxTQ;
                iterNGlobal += ((unsigned int)kernelStride)*maxTQ;
            } //for iterQTile
            iterPGlobal += maxTP;
            iterMGlobal += ((unsigned int)kernelStride)*maxTP;
        } //for iterPTile
    } //for iterGroup

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

//int ia_tbcount_cache_boundary_check(
//      int heightTile,
//      int widthTile
//        )
//{
//    return (heightTile * widthTile);
//}


int oa_cache_boundary_check(
      int heightTile,
      int widthTile,
      int numChannels
        )
{
    int numBlocksPerStrip = DIVIDE_CEIL(numChannels, ACTIVATION_BURST_SIZE_BYTE);
    return (heightTile*widthTile* numBlocksPerStrip);
}

//TODO: CHANGE THE SIGNATURE TO CONSIDER NUMBER OF NZ BLOCKS
int filter_cache_boundary_check(
      int kernelSize,
      int inputChannelSize,
        int peBlockSize,
        int numClustersInPruningRange,
        int numNZClustersInPruningRange
        )
{
    //Nubmer of dram blocks required to store one filter
    int numTBPerStrip = DIVIDE_CEIL(inputChannelSize, peBlockSize * numClustersInPruningRange)
            * numNZClustersInPruningRange;
    int requirement = DIVIDE_CEIL(numTBPerStrip * kernelSize * kernelSize, WEIGHT_WIDE_SIZE);
    return requirement;
}

t_graph_output_tile_info deriveConvOutputTileShape(
        unsigned int outputHeight,
        unsigned int outputWidth,
        unsigned int sizeOutputFullTileHeight,
        unsigned int sizeOutputFullTileWidthPerCol,
        bool isConv
        )
{
    t_graph_output_tile_info tileInfo;
    unsigned int fullCols = (isConv == true) ? PE_COLS : 1;
    tileInfo.sizeOutputTileFullHeight = sizeOutputFullTileHeight;
    tileInfo.sizeOutputTileFullWidthPerCol = sizeOutputFullTileWidthPerCol;
    tileInfo.numOutputTileAlongHeight =
            1 + (outputHeight-1) / sizeOutputFullTileHeight;
    tileInfo.numFullOutputTileAlongHeight =
            outputHeight / sizeOutputFullTileHeight;

    unsigned int sizeOutputTileFullWidth =
            sizeOutputFullTileWidthPerCol * fullCols;
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
        unsigned int numColsForPartialWidthTile = fullCols;
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

unsigned int deriveNumActivationDramBlockPerStrip(
        unsigned int _numInputChannelsPerGroup
    )
{
    unsigned int numTransferBlockPerChannelGroup = 1 + ( _numInputChannelsPerGroup -1) / (CLUSTER_SIZE*TRANSFER_SIZE);
    unsigned int numDramBlockPerStrip = 1 + (numTransferBlockPerChannelGroup-1) / WIDE_SIZE;

    return numDramBlockPerStrip;
}

/*!
 * Loop idle cycles.
 * Need to be verified against AOCL early estimation report
 */
#define NUM_IDLE_CYCLES_PER_FILTER_TRANSFER_FROM_W_MOVER 2
#define NUM_IDLE_CYCLES_PER_STRIP_TRANSFER_FROM_IA_MOVER 0
#define NUM_IDLE_CYCLES_PER_STRIP_TRANSFER_TO_OA_MOVER 0

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
    unsigned int numIdealTransfersPerConvWindow = numTranferBlockPerInputGroup * _sizeKernel * _sizeKernel;
    unsigned int numTransfersPerConvWindow =
            numIdealTransfersPerConvWindow > PE_ROWS ? numIdealTransfersPerConvWindow : PE_ROWS;
    unsigned int latency =
            _numGroups * numPERowFoldPerGroup * numTransfersPerConvWindow
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

    unsigned int numDramBlockPerStrip =
            deriveNumActivationDramBlockPerStrip(_numInputChannelsPerGroup);
    //TODO: adjust the number of idle cycles after HW changes.
    unsigned int latency =
            _numGroups * (numDramBlockPerStrip + NUM_IDLE_CYCLES_PER_STRIP_TRANSFER_FROM_IA_MOVER)
            * (
                numFullOutputTileAlongHeight*numFullOutputTileAlongWidth*sizeFullTileInputHeight*sizeFullTileInputWidth
                + numFullOutputTileAlongHeight*numPartialTileAlongWidth*sizeFullTileInputHeight*sizePartialTileInputWidth
                + numPartialTileAlongHeight*numFullOutputTileAlongWidth*sizePartialTileInputHeight*sizeFullTileInputWidth
                + numPartialTileAlongHeight*numPartialTileAlongWidth*sizePartialTileInputHeight*sizePartialTileInputWidth
              );
    return latency;
 }

unsigned int deriveOutputTransferLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _sizeOutputHeight,
        unsigned int _numOutputChannelsPerNextGroup,
        unsigned int _numNextGroups)
{
    //Adjust for the face that the output bandwidth is the number of compute columns
    unsigned int latency =
            _numOutputChannelsPerNextGroup
            * _numNextGroups
            * _sizeOutputHeight
           * (
                _outputTileInfo.numFullOutputTileAlongWidth * _outputTileInfo.sizeOutputTileFullWidthPerCol
                + (_outputTileInfo.numOutputTileAlongWidth - _outputTileInfo.numFullOutputTileAlongWidth)
                    * _outputTileInfo.sizeOutputTilePartialWidthPerCol
             );
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
            1 + (_sizeKernel*_sizeKernel*numTBPerStrip - 1) / WEIGHT_WIDE_SIZE;
    unsigned int latency =
            _numGroups * _numOutputChannelsPerGroup * numTileAlongHeight * numTileAlongWidth
            * (numDramBlocksInFilter + NUM_IDLE_CYCLES_PER_FILTER_TRANSFER_FROM_W_MOVER);

    return latency;
}

unsigned int deriveFirstTileConvInputTransferLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _sizeKernel,
        unsigned _sizeStride
        )
{
    unsigned int sizeFirstTileOutputHeight =
        (_outputTileInfo.numFullOutputTileAlongHeight==0) ?
                 _outputTileInfo.sizeOutputTilePartialHeight
              : _outputTileInfo.sizeOutputTileFullHeight;

    unsigned int sizeFirstTileOuputWidth =
         (_outputTileInfo.numFullOutputTileAlongWidth==0) ?
                _outputTileInfo.sizeOutputTilePartialWidthPerCol
                    * _outputTileInfo.numActiveColsForPartialWidthTile
               : _outputTileInfo.sizeOutputTileFullWidthPerCol
                    * PE_COLS;

    //Convert to input height and width
    unsigned int sizeFirstTileInputHeight =
            deriveConvInputDimension1D(
                sizeFirstTileOutputHeight,
                _sizeKernel,
                _sizeStride
                );
    unsigned int sizeFirstTileInputWidth =
            deriveConvInputDimension1D(
                sizeFirstTileOuputWidth,
                _sizeKernel,
                _sizeStride
                );

    unsigned int numDramBlockPerStrip =
            deriveNumActivationDramBlockPerStrip(_numInputChannelsPerGroup);

    unsigned int latency =
            sizeFirstTileInputHeight
            * sizeFirstTileInputWidth
            * (numDramBlockPerStrip + NUM_IDLE_CYCLES_PER_STRIP_TRANSFER_FROM_IA_MOVER);

    return latency;
}

unsigned int deriveFirstTileConvComputationLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numOutputChannelsPerGroup,
        unsigned int _numInputChannelsPerGroup,
        unsigned int _sizeKernel
        )
{
    unsigned int numPERowFoldPerGroup =
            1 + (_numOutputChannelsPerGroup-1) / PE_ROWS;
    unsigned int numTranferBlockPerInputGroup =
            1 + (_numInputChannelsPerGroup-1) / (TRANSFER_SIZE * CLUSTER_SIZE);
    unsigned int numIdealTransfersPerConvWindow = numTranferBlockPerInputGroup * _sizeKernel * _sizeKernel;
    unsigned int numTransfersPerConvWindow =
            numIdealTransfersPerConvWindow > PE_ROWS ? numIdealTransfersPerConvWindow : PE_ROWS;

    unsigned int sizeFirstTileOutputHeight =
        (_outputTileInfo.numFullOutputTileAlongHeight==0) ?
                 _outputTileInfo.sizeOutputTilePartialHeight
              : _outputTileInfo.sizeOutputTileFullHeight;

    unsigned int sizeFirstTileOuputWidthPerCol =
         (_outputTileInfo.numFullOutputTileAlongWidth==0) ?
                _outputTileInfo.sizeOutputTilePartialWidthPerCol
               : _outputTileInfo.sizeOutputTileFullWidthPerCol;

    unsigned int latency =
            numPERowFoldPerGroup
            * numTransfersPerConvWindow
            * sizeFirstTileOutputHeight
            * sizeFirstTileOuputWidthPerCol;
    return latency;
}

unsigned int deriveLastTileOutputTransferLatency(
        t_graph_output_tile_info _outputTileInfo,
        unsigned int _numOutputChannelsPerNextGroup
        )
{
    unsigned int sizeLastTileOutputHeight =
            (_outputTileInfo.numFullOutputTileAlongHeight == _outputTileInfo.numOutputTileAlongHeight)?
                _outputTileInfo.sizeOutputTileFullHeight :
                _outputTileInfo.sizeOutputTilePartialHeight;

    unsigned int sizeLastTileOutputWidthPerCol =
            (_outputTileInfo.numFullOutputTileAlongWidth == _outputTileInfo.numOutputTileAlongWidth)?
                _outputTileInfo.sizeOutputTileFullWidthPerCol
              : _outputTileInfo.sizeOutputTilePartialWidthPerCol;

    unsigned int latency =
            sizeLastTileOutputHeight * sizeLastTileOutputWidthPerCol * _numOutputChannelsPerNextGroup;

    return latency;
}

