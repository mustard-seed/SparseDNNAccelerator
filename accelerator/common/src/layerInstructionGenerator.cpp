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
        //Whether the IA mover kernel should wait for the output to commit before moving
        //on to the next tensor
        //unsigned char flagInputSync,
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
    unsigned int sizeOutputTileFullHeight = (op == CONVOLUTION) ?
                _sizeOutputTileFullHeight : 1;
    unsigned int sizeOutputTileFullWidthPerCol = (op == CONVOLUTION) ?
                _sizeOutputTileFullWidthPerCol : 1;

    unsigned int outputHeight = ( ((unsigned int) inputSPHeight)
            + 2*((unsigned int) inputHeightPadding) - ((unsigned int) kernelSize))
            / ((unsigned int) kernelStride) + 1;

    unsigned int outputWidth = ( ((unsigned int) inputSPWidth)
            + 2* ((unsigned int) inputWidthPadding) - ((unsigned int) kernelSize))
            / ((unsigned int) kernelStride) + 1;

    unsigned char numActiveColsPartialOutputTile = (op == CONVOLUTION) ?
                _numActiveColsPartialOutputTile : (outputWidth % PE_COLS);

    //Input height and width before stretch and padding
    assert ((inputSPHeightUnit == 1) || ((inputSPHeight-1) % inputSPHeightUnit == 0));
    assert ((inputSPWidthUnit == 1) || ((inputSPWidth-1) % inputSPWidthUnit == 0));
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
            ( ((unsigned int) PE_COLS) * ((unsigned int) sizeOutputTileFullWidthPerCol) );

    unsigned int numOutputTileX =
            1 + ((unsigned int) (outputWidth-1) ) /
                ( ((unsigned int) PE_COLS) * ((unsigned int) sizeOutputTileFullWidthPerCol) );

    unsigned int sizeOutputTilePartialWidthPerCol = (numActiveColsPartialOutputTile != 0) ?
                ( ((unsigned int) outputWidth) %
                    ( ((unsigned int) PE_COLS) * ((unsigned int) sizeOutputTileFullWidthPerCol) )
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
    //Number of misc engine output blocks per *current layer* output strip
    unsigned int numOutputBlocksBlob0MK;
    unsigned int numOutputBlocksBlob1MK;
    //Number of dram block each MK should reduce in order to produce a block of output;
    unsigned int numDramBlocksToReduceMK;
    switch(op) {
        case CONVOLUTION : {
            numOAGroupsCurrentLayer = _numGroupsCurrentLayer;
            assert(numOutputChannels % _numGroupsCurrentLayer == 0);
            assert(numInputChannels0 % _numGroupsCurrentLayer == 0);
            numIAMoverInputChannelsPerGroup0 = numInputChannels0 / _numGroupsCurrentLayer;
            numIAMoverInputChannelsPerGroup1  = 0;
            numIAMoverGroup0 = _numGroupsCurrentLayer;
            numIAMoverGroup1 = 0;
            numActiveElementsInFullComputeFold = PE_ROWS;
            memIA0DramBlockGroupStride = _memIA0DramBlockGroupStride;
            numOutputChannelsBlob0MK = 0;
            numOutputChannelsBlob1MK = 0;
            numOutputBlocksBlob0MK = 0;
            numOutputBlocksBlob1MK = 0;
            numDramBlocksToReduceMK = 0;
        }
        break;
        case CONCATENATION : {
            assert(numOutputChannels == (numInputChannels0 + numInputChannels1));
            assert(flagSparseInput == FALSE);
            assert(kernelSize == 1);
            numIAMoverInputChannelsPerGroup0 = numInputChannels0;
            numIAMoverInputChannelsPerGroup1 = numInputChannels1;
            numIAMoverGroup0 = 1;
            numIAMoverGroup1 = 1;
            numActiveElementsInFullComputeFold = BURST_SIZE_BYTE;
            numOAGroupsCurrentLayer = 1;
            memIA0DramBlockGroupStride = _memIA0DramBlockGroupStride;
            memIA1DramBlockGroupStride = _memIA1DramBlockGroupStride;
            numOutputChannelsBlob0MK = numInputChannels0;
            numOutputChannelsBlob1MK = numInputChannels1;
            numOutputBlocksBlob0MK = 1+ (numOutputChannelsBlob0MK-1) / BURST_SIZE_BYTE;
            numOutputBlocksBlob1MK = 1+ (numOutputChannelsBlob1MK-1) / BURST_SIZE_BYTE;
            numDramBlocksToReduceMK = 1;
        }
        break;
        case MAX_POOL : {
            assert(numOutputChannels == numInputChannels0);
            assert(flagSparseInput == FALSE);
            numIAMoverInputChannelsPerGroup0 = BURST_SIZE_BYTE;
            numIAMoverInputChannelsPerGroup1 = 0;
            numIAMoverGroup0 = (1 + (numInputChannels0-1) / BURST_SIZE_BYTE);
            numIAMoverGroup1 = 0;
            numActiveElementsInFullComputeFold = BURST_SIZE_BYTE;
            numOAGroupsCurrentLayer = 1;
            memIA0DramBlockGroupStride = 1;
            numOutputChannelsBlob0MK = numInputChannels0;
            numOutputChannelsBlob1MK = 0;
            numOutputBlocksBlob0MK = 1+ (numOutputChannelsBlob0MK-1) / BURST_SIZE_BYTE;
            numOutputBlocksBlob1MK = 0;
            numDramBlocksToReduceMK = kernelSize * kernelSize;
        }
        break;
        case ELT_ADD : {
            assert(numInputChannels0==numInputChannels1);
            assert(numOutputChannels == numInputChannels0);
            assert(flagSparseInput == FALSE);
            assert(kernelSize == 1);
            numIAMoverInputChannelsPerGroup0 = BURST_SIZE_BYTE;
            numIAMoverInputChannelsPerGroup1 = BURST_SIZE_BYTE;
            numIAMoverGroup0 = (1 + (numInputChannels0-1) / BURST_SIZE_BYTE);
            numIAMoverGroup1 = (1 + (numInputChannels1-1) / BURST_SIZE_BYTE);
            numActiveElementsInFullComputeFold = BURST_SIZE_BYTE;
            numOAGroupsCurrentLayer = 1;
            memIA0DramBlockGroupStride = 1;
            memIA1DramBlockGroupStride = 1;
            numOutputChannelsBlob0MK = numInputChannels0;
            numOutputChannelsBlob1MK = 0;
            numOutputBlocksBlob0MK = 1+ (numOutputChannelsBlob0MK-1) / BURST_SIZE_BYTE;
            numOutputBlocksBlob1MK = 0;
            numDramBlocksToReduceMK = 2;
        }
        break;
        case AVG_POOL : {
            assert(numOutputChannels == numInputChannels0);
            assert(flagSparseInput == FALSE);
            numIAMoverInputChannelsPerGroup0 = BURST_SIZE_BYTE;
            numIAMoverInputChannelsPerGroup1 = 0;
            numIAMoverGroup0 = (1 + (numInputChannels0-1) / BURST_SIZE_BYTE);
            numIAMoverGroup1 = 0;
            numActiveElementsInFullComputeFold = BURST_SIZE_BYTE;
            numOAGroupsCurrentLayer = 1;
            memIA0DramBlockGroupStride = 1;
            numOutputChannelsBlob0MK = numInputChannels0;
            numOutputChannelsBlob1MK = 0;
            numOutputBlocksBlob0MK = 1+ (numOutputChannelsBlob0MK-1) / BURST_SIZE_BYTE;
            numOutputBlocksBlob1MK = 0;
            numDramBlocksToReduceMK = (unsigned int) kernelSize * (unsigned int) kernelSize;
        }
    break;
        default: {
            std::cout <<"Instruction generator: unsupported operation type: "<<op<<std::endl;
            assert(false);
        }
        break;
    }
    unsigned int numOutputChannelsPerGroupCurrentLayer =
            numOutputChannels / numOAGroupsCurrentLayer;

    assert(numOutputChannels % numGroupsNextLayer == 0);
    unsigned int numOutputChannelsPerGroupNextLayer = numOutputChannels / numGroupsNextLayer;

    unsigned int numComputeFoldPerGroup = (numOutputChannelsPerGroupCurrentLayer-1) / numActiveElementsInFullComputeFold + 1;
    unsigned int numFullComputeFoldPerGroup = numOutputChannelsPerGroupCurrentLayer / numActiveElementsInFullComputeFold;
    unsigned int numActiveElementsInPartialComputeFold = numOutputChannelsPerGroupCurrentLayer % numActiveElementsInFullComputeFold;

    /*
     * Cache limit check when performing convolution
    */
    if (op == CONVOLUTION)
    {
        int maxIATileHeight = sizeInputTileFullHeight > sizeInputTilePartialHeight ?
                    sizeInputTileFullHeight : sizeInputTilePartialHeight;
        int maxIATileWidth = sizeInputTileFullWidthPerCol > sizeInputTilePartialWidthPerCol ?
                    sizeInputTileFullWidthPerCol : sizeInputTilePartialWidthPerCol;
        int iaCacheRequirementInDramBlock = ia_cache_boundary_check(
                    //heightTile
                    maxIATileHeight,
                    //widthTile
                    maxIATileWidth,
                    //numDramBlockPerDenseStrip,
                    memIA0DramBlockColStride
                    );
        assert( iaCacheRequirementInDramBlock < IA_CACHE_DEPTH && "IA tile size is too big to fit inside the cache");

        #if defined(SPARSE_SYSTEM)
            int iaTBCountRequirement = ia_tbcount_cache_boundary_check(
                        maxIATileHeight,
                        maxIATileWidth
                        );
            assert(iaTBCountRequirement < IA_TBCOUNT_CACHE_SIZE && "Number if IA TB count too big to fit inside the cache");
        #endif

        int filterCacheRequirement = filter_cache_boundary_check(
                        kernelSize,
                        numInputChannels0
                    );
        assert(filterCacheRequirement < KERNEL_CACHE_DEPTH && "Individual fitler size is too big to fit inside the filter cache");
    }

    int maxOATileHeight = sizeOutputTileFullHeight > sizeOutputTilePartialHeight ?
                    sizeOutputTileFullHeight : sizeOutputTilePartialHeight;
    int maxOATileWidth = sizeOutputTileFullWidthPerCol > sizeOutputTilePartialWidthPerCol ?
                sizeOutputTileFullWidthPerCol : sizeOutputTilePartialWidthPerCol;
    int oaCacheRequirement = oa_cache_boundary_check(
                //heightTile
                maxOATileHeight,
                //widthTile
                maxOATileWidth,
                //numChannels,
                numOutputChannels
                );
    assert(oaCacheRequirement < (OA_CACHE_DEPTH * CLUSTER_SIZE) && "OA tile size is too big to fit inside the oa cache");


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
                        numActiveColsPartialOutputTile : PE_COLS;

            unsigned int maxTQPerCol = (iterQTile >= numFullOutputTileX) ?
                        sizeOutputTilePartialWidthPerCol : sizeOutputTileFullWidthPerCol;

            unsigned int maxTQ = numActiveCols * maxTQPerCol;

            unsigned int maxTNPerCol = (iterQTile >= numFullOutputTileX) ?
                        sizeInputTilePartialWidthPerCol : sizeInputTileFullWidthPerCol;

            unsigned int maxTN = maxTNPerCol * numActiveCols
                    - (((signed int) kernelSize) - ((signed int) kernelStride) ) * (numActiveCols-1);

            /*! Generate the output tile controller instruction */
            {
                t_oa_tile_controller_instruction instructionOAControl;

                instructionOAControl.numLocalTilePerColHxW = (t_uchar)(maxTQPerCol*maxTP);
                instructionOAControl.numLocalChannels = (t_uchar)(numOutputChannels);
                instructionOAControl.numDrainInstructions = (t_ushort) ( numComputeFoldPerGroup * numOAGroupsCurrentLayer);
                instructionOAControl.numMemInstructions = (t_uchar) numGroupsNextLayer;
                instructionOAControl.numFoldsInGroupCurrentLayer = (t_uchar) numComputeFoldPerGroup;
                instructionOAControl.numFullFoldsInCurrentLayer = (t_uchar) numFullComputeFoldPerGroup;
                instructionOAControl.numActiveElementsInFullFold = (t_uchar) numActiveElementsInFullComputeFold;
                instructionOAControl.numActiveElementsInPartialFold = (t_uchar) numActiveElementsInPartialComputeFold;
                instructionOAControl.numLocalChannelsPerCurrentGroup = (t_ushort) numOutputChannelsPerGroupCurrentLayer;
                instructionOAControl.numLocalChannelsPerNextGroup = (t_ushort) numOutputChannelsPerGroupNextLayer;
                instructionOAControl.numActiveCols = (t_uchar) numActiveCols;
                unsigned char leftShift = flagOutputShiftLeft;
                unsigned char scaleShift = outputShiftBits;
                assert ((((flagOutputShiftLeft == 0x00) && (outputShiftBits > 0))
                        || ((flagOutputShiftLeft == 0x01) && (outputShiftBits >= 0)))
                       && "If output shift direction is RIGHT, then the number of shift must be greater than 0");
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
                bool isLastOutputTile =
                        ( (iOutputGroup+1) == numGroupsNextLayer)
                        && ((iterPTile+1) == numOutputTileY)
                        && ((iterQTile+1) == numOutputTileX);
//                unsigned char actualFlagOutputSync = (flagOutputSync == 0x01) ?
//                            ((isLastOutputTile == true) ? 0x1 : 0x0)
//                            : 0x0;
                unsigned char actualFlagOutputSync = 0x0;
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

            //Transfer the first input blob, and convolution related items (if necessary)
            for (unsigned int iterInputGroup0=0; iterInputGroup0<numIAMoverGroup0; iterInputGroup0++)
            {
                 unsigned int inputStripStartIndex =
                        iterInputGroup0 * inputDenseHeight * inputDenseWidth
                        + iterMDense * inputDenseWidth + iterNDense;

                /*! Send the input IA instruction */
                {
                    t_ia_mover_instruction instructionIA;
//                    bool isLastInputTile =
//                            (op == CONVOLUTION || op == MAX_POOL)
//                            && ( iterInputGroup0 == (numIAMoverGroup0-1))
//                            && ((iterPTile+1) == numOutputTileY)
//                            && ((iterQTile+1) == numOutputTileX);
//                    unsigned char actualFlagInputSync = (flagInputSync == 0x01) ?
//                                ((isLastInputTile == true) ? 0x1 : 0x0)
//                                : 0x0;
                    unsigned char actualFlagInputSync = 0x0;
                    //Set the transport target. 0x0 means convolution, 0x1 means MISC
                    unsigned char flagTarget= (op == CONVOLUTION) ?  0x00 : 0x01;
                    instructionIA.memRegionCatSparseFlagCatDestinationCatSyncCatNumActiveCols = (t_uchar)
                            ( ( ((t_uchar) numActiveCols)& 0x0F)
                             | ((((t_uchar) actualFlagInputSync) & 0x01) << 0x04)
                             | ((((t_uchar) flagTarget) & 0x01) << 0x05)
                             | ((((t_uchar) flagSparseInput) & 0x01) << 0x06) //Sparse flag for the input tensor
                            );
                    assert ((((flagIA0ShiftLeft == false) && (numIA0ShiftAmount > 0))
                            || ((flagIA0ShiftLeft == true) && (numIA0ShiftAmount >= 0)))
                           && "If input shift direction is RIGHT, then the number of shift must be greater than 0");
                    t_uchar flagLeftShiftCatShiftAmount = flagIA0ShiftLeft ?
                                (0x10 | (0x0F & numIA0ShiftAmount))
                              : (0x00 | (0x0F & numIA0ShiftAmount));
                    instructionIA.flagLeftShiftCatShiftAmount = flagLeftShiftCatShiftAmount;
                    instructionIA.memBlockStart = (t_int) (
                                memIA0DramBlockStartIndex
                                + memIA0DramBlockGroupStride * iterInputGroup0
                                + memIA0DramBlockRowStride * iterMDense
                                + memIA0DramBlockColStride * iterNDense);
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
                } // generate t_ia_mover_instruction

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
                        instructionIAControl.numOutputChannelsInGroup = (t_ushort) numOutputChannelsPerGroupCurrentLayer;
                        unsigned char inputNeedsBitmaskPadding = (flagSparseInput == 0x00) ? 0x80 : 0x00;
                        instructionIAControl.flagPadBitmaskCatNumActiveCols = (t_uchar)
                                (inputNeedsBitmaskPadding | (0x7F & numActiveCols));
#if defined(SPARSE_SYSTEM)
                        assert(COMPRESSION_WINDOW_SIZE % 8 == 0);
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

            //Non-convolution stuff
            if (op != CONVOLUTION)
            {
                //Transfer the SECOND input blob if necessary
                for (unsigned int iterInputGroup1=0; iterInputGroup1<numIAMoverGroup1; iterInputGroup1++)
                {
                    /*! Send the input IA instruction */
                    {
                        t_ia_mover_instruction instructionIA;

//                        bool isLastInputTile =
//                                    (op != CONVOLUTION)
//                                && (op != MAX_POOL)
//                                && ( iterInputGroup1 == (numIAMoverGroup1-1))
//                                && ((iterPTile+1) == numOutputTileY)
//                                && ((iterQTile+1) == numOutputTileX);

//                        unsigned char actualFlagInputSync = (flagInputSync == 0x01) ?
//                                    ((isLastInputTile == true) ? 0x1 : 0x0)
//                                    : 0x0;
                        unsigned char actualFlagInputSync = 0x0;

                        instructionIA.memRegionCatSparseFlagCatDestinationCatSyncCatNumActiveCols = (t_uchar)
                                ( ( ((t_uchar) numActiveCols)& 0x0F)
                                 | ((((t_uchar) actualFlagInputSync) & 0x01) << 0x04)
                                 | ((((t_uchar) 0x01) & 0x01) << 0x05) //Compute engine is MISC
                                 | ((((t_uchar) 0x0) & 0x01) << 0x06) //Sparse flag for the input tensor. Will be dense
                                );
                        assert ((((flagIA1ShiftLeft == false) && (numIA1ShiftAmount > 0))
                                || ((flagIA1ShiftLeft == true) && (numIA1ShiftAmount >= 0)))
                               && "If input shift direction is RIGHT, then the number of shift must be greater than 0");
                        t_uchar flagLeftShiftCatShiftAmount = flagIA1ShiftLeft ?
                                    (0x10 | (0x0F & numIA1ShiftAmount))
                                  : (0x00 | (0x0F & numIA1ShiftAmount));
                        instructionIA.flagLeftShiftCatShiftAmount = flagLeftShiftCatShiftAmount;
                        instructionIA.memBlockStart = (t_int) (
                                    memIA1DramBlockStartIndex
                                    + memIA1DramBlockGroupStride * iterInputGroup1
                                    + memIA1DramBlockRowStride * iterMDense
                                    + memIA1DramBlockColStride * iterNDense);
                        instructionIA.memBlockColStripStride = (t_ushort)(memIA1DramBlockColStride);
                        instructionIA.memBlockRowStripStride = (t_ushort)(memIA1DramBlockRowStride);

                        #if defined(SPARSE_SYSTEM)
                            instructionIA.numCWOrTBInGroup = (t_ushort) (1 + (numIAMoverInputChannelsPerGroup1-1) / TRANSFER_SIZE / CLUSTER_SIZE);
                        #else
                            instructionIA.numTBPerStrip = (t_ushort) (1 + (numIAMoverInputChannelsPerGroup1-1) / TRANSFER_SIZE / CLUSTER_SIZE);
                        #endif
                            instructionIA.tileSPHeight = (t_uchar) maxTM;
                            instructionIA.tileSPWidth = (t_uchar) maxTN;
                            unsigned char inputTileLeftPadding = (iterNGlobal < inputWidthPadding) ?
                                        inputWidthPadding : 0;
                            unsigned char inputTileRightPadding = ((iterNGlobal + maxTQ) >= (inputWidthPadding + inputDenseWidth)) ?
                                        inputWidthPadding : 0;
                            unsigned char inputTileTopPadding = (iterMGlobal < inputHeightPadding) ?
                                        inputHeightPadding : 0;
                            unsigned char inputTileBottomPadding = ((iterMGlobal + maxTP) >= (inputHeightPadding + inputDenseHeight)) ?
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
                    } // generate t_ia_mover_instruction
                }  //For. Transfer the SECOND input blob, and convolution related items (if necessary)

                //Transfer the MISC instruction
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

                    if (numOutputBlocksBlob0MK > 0)
                    {
                        t_misc_instruction instructionMisc;
                        instructionMisc.controlBits = (t_uchar) (opCodeField |( numActiveCols & 0x0F));
                        instructionMisc.numDramBlocksToReduce = (unsigned short) numDramBlocksToReduceMK;
                        instructionMisc.numOutputBlocks = (unsigned char) numOutputBlocksBlob0MK;
                        instructionMisc.numEffectiveValuesInLastStrip = (t_uchar) (
                                    numOutputChannelsBlob0MK - (numOutputBlocksBlob0MK-1)*BURST_SIZE_BYTE);

                        vecMiscInstruction.push_back(instructionMisc);
                    } //if fold 0 iMiscFold

                    if (numOutputBlocksBlob1MK > 0)
                    {
                        t_misc_instruction instructionMisc;
                        instructionMisc.controlBits = (t_uchar) (opCodeField |( numActiveCols & 0x0F));
                        instructionMisc.numDramBlocksToReduce = (unsigned short) numDramBlocksToReduceMK;
                        instructionMisc.numOutputBlocks = (unsigned char) numOutputBlocksBlob1MK;
                        instructionMisc.numEffectiveValuesInLastStrip = (t_uchar) (
                                    numOutputChannelsBlob1MK - (numOutputBlocksBlob1MK-1)*BURST_SIZE_BYTE);

                        vecMiscInstruction.push_back(instructionMisc);
                    } //if fold 0 iMiscFold
                } //Transfer the MISC instruction
            } //Non-convolution stuff
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


