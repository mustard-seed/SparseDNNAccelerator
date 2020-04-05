#include "layerInstructionGenerator.hpp"

void convolution_instruction_generator(
        t_aligned_ia_mover_instruction_vector & vecIAMoverInstruction,
        t_aligned_oa_mover_instruction_vector & vecOAMoverInstruction,
        t_aligned_ia_tile_controller_instruction_vector & vecIATileControlInstruction,
        t_aligned_oa_tile_controller_instruction_vector & vecOATileControlInstruction,
        t_aligned_weight_mover_instruction_vector & vecWeightMoverInstruction,

        signed int memIADramBlockStartIndex,
        signed int memOADramBlockStartIndex,
        signed int memWeightDramBlockStartIndex,
        signed int memBiasStartIndex,

        signed int memIADramBlockStripStride,

        signed int memOADramBlockStripStride,

        signed int memWeightDramBlockFilterStride,

        #if defined(SPARSE_SYSTEM)
            signed int memIATBCountStart,
            unsigned int memIATBCountStripStride,

            signed int memOATBCountStart,
            unsigned int memOATBCountStripStride,

            signed int memWeightTBCountStart,
            unsigned int memWeightTBCountFilterStride,
        #else
            unsigned int numTBPerIAStrip,
            unsigned int numTBPerOAStrip,
            unsigned int numTBPerWeightFilter,
        #endif

        unsigned char pingPongRegionIA,
        unsigned char pingPongRegionOA,
        unsigned char flagSparseOutput,
        unsigned char flagInputSync,
        unsigned char flagOutputSync,
        unsigned char flagRelu,
        unsigned char inputFracBits,
        unsigned char weightFracBits,
        unsigned char outputFracBits,

        unsigned short inputSPWidth,
        unsigned short inputSPHeight,
        unsigned char inputSPWidthUnit,
        unsigned char inputSPHeightUnit,
        unsigned char inputWidthPadding,
        unsigned char inputHeightPadding,
        unsigned short numGroupsCurrentLayer,
        unsigned short inputChannels,

        unsigned short outputChannels,
        unsigned short numGroupsNextLayer,

        unsigned char kernelSize,
        unsigned char kernelStride,

        unsigned char sizeOutputTileFullHeight,
        unsigned char sizeOutputTileFullWidthPerCol,
        unsigned char numActiveColsPartialOutputTile
        )
{
    unsigned int outputHeight = ( (unsigned int) inputSPHeight
            + 2*(unsigned int) inputHeightPadding - (unsigned int) kernelSize)
            / (unsigned int) kernelStride + 1;

    unsigned int outputWidth = ( (unsigned int) inputSPWidth
            + 2*(unsigned int) inputWidthPadding - (unsigned int) kernelSize)
            / (unsigned int) kernelStride + 1;

    unsigned int inputDenseHeight = inputSPHeight / inputSPHeightUnit;
    unsigned int inputDenseWidth = inputSPWidth / inputSPWidthUnit;

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

    unsigned int sizeOutputTilePartialWidthPerCol =
            ( ((unsigned int) outputWidth) %
                ( ((unsigned int) PE_COLS) * ((unsigned int) sizeOutputTileFullWidthPerCol)
            ) / (unsigned int) numActiveColsPartialOutputTile;

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

    unsigned int numOutputChannelsPerGroupCurrentLayer = outputChannels / numGroupsCurrentLayer;

    unsigned int numOutputChannelsPerGroupNextLayer = outputChannels / numGroupsNextLayer;

    unsigned int numRowFoldPerGroup = (numOutputChannelsPerGroupCurrentLayer-1) / PE_ROWS + 1;
    unsigned int numFullRowFoldPerGroup = numOutputChannelsPerGroupCurrentLayer / PE_ROWS;
    unsigned int numFiltersInPartialRowFold = numOutputChannelsPerGroupCurrentLayer % PE_ROWS;

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
                        PE_COLS : numActiveColsPartialOutputTile;

            unsigned int maxTQPerCol = (iterQTile >= numFullOutputTileX) ?
                        sizeOutputTilePartialWidthPerCol : sizeOutputTileFullWidthPerCol;

            unsigned int maxTQ = numActiveCols * maxTQPerCol;

            unsigned int maxTNPerCol = (iterQTile >= numFullOutputTileX) ?
                        sizeInputTilePartialWidthPerCol : sizeInputTileFullWidthPerCol;

            unsigned int maxTN = maxTNPerCol * numActiveCols
                    - (((signed int) kernelSize) - ((signed int) kernelStride) ) * (numActiveCols-1);

            /*! Generate the output OA instruction */
            {
                t_oa_mover_instruction instructionOA;
                instructionOA.memSelectCatSparseFlagCatSyncFlagCatNumActiveCols =
                        ((t_uchar) numActiveCols & 0x0F)
                        | ((((t_uchar) flagOutputSync) & 0x01) << 0x04)
                        | ((((t_uchar) flagSparseOutput) & 0x01) << 0x06)
                        | ((((t_uchar) pingPongRegionOA) & 0x01) << 0x07);

                instructionOA.memOAStart = (t_int)
                        (memOADramBlockStartIndex
                          + ((unsigned int) iterPGlobal*outputWidth + (unsigned int) iterQGlobal)
                            * numGroupsNextLayer * memOADramBlockStripStride);

                instructionOA.memOAGroupStride = (t_uint)((unsigned int) outputWidth * outputHeight * memOADramBlockStripStride);
                instructionOA.memOATileStride = (t_uint)((unsigned int) outputWidth * outputHeight * numGroupsNextLayer * memOADramBlockStripStride);
                instructionOA.memOARowStride = (t_ushort)((unsigned int) outputWidth * memOADramBlockStripStride);
                instructionOA.memOAColStride = (t_ushort) memOADramBlockStripStride;

    #if defined(SPARSE_SYSTEM)
                instructionOA.memTBStart = (t_int)
                        (memOATBCountStart
                            + ( (unsigned int) iterPGlobal*outputWidth + (unsigned int) iterQGlobal)
                                * numGroupsNextLayer * memOATBCountStripStride);

                instructionOA.memTBGroupStride = (t_uint)((unsigned int) outputWidth * outputHeight * memOATBCountStripStride);
                instructionOA.memTBTileStride = (t_uint)((unsigned int) outputWidth * outputHeight * numGroupsNextLayer * memOATBCountStripStride);
                instructionOA.memTBRowStride = (t_ushort)((unsigned int) outputWidth * memOATBCountStripStride);
                instructionOA.memTBColStride = (t_ushort) memOATBCountStripStride;
    #else
                instructionOA.numTBPerStrip = (t_uint) numTBPerOAStrip;
    #endif
                instructionOA.numOAGroup = (t_uchar) numGroupsNextLayer;
                instructionOA.tileHeight = (t_uchar) maxTP;
                instructionOA.columnTileWidth = (t_uchar) maxTQPerCol;
                instructionOA.numOAGroupxColumnTileWidthxTileHeightxNumActiveCols
                        = (t_ushort) (numGroupsNextLayer * maxTQPerCol * maxTP * numActiveCols);

                vecOAMoverInstruction.push_back(instructionOA);
            }

            /*! Generate the output tile controller instruction */
            {
                t_oa_tile_controller_instruction instructionOAControl;

                instructionOAControl.numLocalTilePerColHxW = (t_uchar)(maxTQPerCol*maxTP);
                instructionOAControl.numLocalChannels = (t_uchar)(outputChannels);
                instructionOAControl.numDrainInstructions = (t_ushort)
                        ( numRowFoldPerGroup * numGroupsCurrentLayer * maxTQPerCol * maxTP);
                instructionOAControl.numMemInstructions = (t_ushort)
                        ( numGroupsNextLayer * maxTQPerCol * maxTP);
                instructionOAControl.numFoldsInGroupCurrentLayer = (t_uchar) numRowFoldPerGroup;
                instructionOAControl.numFullFoldsInCurrentLayer = (t_uchar) numFullRowFoldPerGroup;
                instructionOAControl.numActiveElementsInPartialFold = (t_uchar) numFiltersInPartialRowFold;
                instructionOAControl.numLocalChannelsPerNextGroup = (t_ushort) numOutputChannelsPerGroupNextLayer;
                instructionOAControl.numActiveCols = (t_uchar) numActiveCols;
                instructionOAControl.flagSparseCatFlagReluCatFlagSourceCatRShift = (t_uchar)
                        (
                            ((t_uchar)(inputFracBits+weightFracBits-outputFracBits) & 0x0F)
                            | (t_uchar)(0x01 << 0x4)
                            | (t_uchar)((flagRelu & 0x01) << 0x5)
                            | (t_uchar)((flagSparseOutput & 0x01) << 0x06)
                        );
                vecOATileControlInstruction.push_back(instructionOAControl);
            }
            for (unsigned int iterInputGroup=0; iterInputGroup<numGroupsCurrentLayer; iterInputGroup++)
            {
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

                unsigned int inputStripStartIndex =
                        iterInputGroup * inputDenseHeight * inputDenseWidth
                        + iterMDense * inputDenseWidth + iterNDense;

                /*! Send the input IA instruction */
                {
                    t_ia_mover_instruction instructionIA;
                    instructionIA.memRegionCatSparseFlagCatDestinationCatSyncCatNumActiveCols = (t_uchar)
                            ( ( ((t_uchar) numActiveCols)& 0x0F)
                             | ((((t_uchar) flagInputSync) & 0x01) << 0x04)
                             | ((((t_uchar) 0x00) & 0x01) << 0x05)
                             | ((((t_uchar) 0x01) & 0x01) << 0x06)
                             | ((((t_uchar) pingPongRegionIA) & 0x01) << 0x07)
                            );
                    instructionIA.memBlockStart = (t_int) (
                                memIADramBlockStartIndex
                                + inputStripStartIndex * memIADramBlockStripStride);
                    instructionIA.memBlockColStripStride = (t_ushort)(memIADramBlockStripStride);
                    instructionIA.memBlockRowStripStride = (t_ushort)(memIADramBlockStripStride * inputDenseWidth);

                    #if defined(SPARSE_SYSTEM)
                        instructionIA.memTBCountStart = (t_int)
                                (memIATBCountStart
                                 + inputStripStartIndex * memIATBCountStripStride);
                        instructionIA.memTBCountColStride = (t_ushort) memIATBCountStripStride;
                        instructionIA.memTBCountRowStride = (t_ushort) (memIATBCountStripStride * inputDenseWidth);
                    #else
                        instructionIA.numTBPerStrip = (t_uint) numTBPerIAStrip;
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
                        instructionIA.numCWInGroup = (t_uchar) (
                                        1 + (inputChannels-1) / COMPRESSION_WINDOW_SIZE / CLUSTER_SIZE
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

                /*! Send the input IA controller instruction */
                {
                    t_ia_tile_controller_instruction instructionIAControl;
                    instructionIAControl.localTileWidth = (t_uchar) maxTNPerCol;
                    instructionIAControl.localTileHeight = (t_uchar) maxTM;
                    instructionIAControl.kernelStride = (t_uchar) kernelStride;
                    instructionIAControl.kernelSize = (t_uchar) kernelSize;
                    instructionIAControl.numOutputInstructions = (t_ushort)
                            ((t_ushort) maxTP * (t_ushort) maxTQPerCol * (t_ushort) kernelSize * numRowFoldPerGroup);
                    instructionIAControl.cacheStripStride = (t_ushort) memIADramBlockStripStride;
#if !defined(SPARSE_SYSTEM)
                    instructionIAControl.numTBPerStrip = (t_ushort) numTBPerIAStrip;
#endif
                    vecIATileControlInstruction.push_back(instructionIAControl);
                } // generate the ia controller instruction

                /*! Generate the weight mover instruction*/
                {
                    unsigned int filterIndex = iterInputGroup * numOutputChannelsPerGroupCurrentLayer;
                    t_weight_mover_instruction instructionWMover;
                    instructionWMover.numFilterFold = (t_ushort) numRowFoldPerGroup;
                    instructionWMover.numFullFilterFold = (t_ushort) numFullRowFoldPerGroup;
                    instructionWMover.numFiltersInPartialFold = (t_uchar) numFiltersInPartialRowFold;
                    instructionWMover.filterReuse = (t_ushort) maxTNPerCol * (t_ushort) maxTM;
                    instructionWMover.memBiasStart = (t_int)(memBiasStartIndex + filterIndex);
                    instructionWMover.memWeightStart = (t_int) memWeightDramBlockStartIndex +(t_int) filterIndex * memWeightDramBlockFilterStride;
                    instructionWMover.memWeightFilterStride = (t_int) memWeightDramBlockFilterStride;
#if defined(SPARSE_SYSTEM)
                    instructionWMover.memTBCountStart = (t_int) memWeightTBCountStart + (t_int) filterIndex * memWeightTBCountFilterStride;
                    instructionWMover.memTBCountFilterStride = (t_int) memWeightTBCountFilterStride;
#else
                    instructionWMover.numTBPerFilter = (t_uint) numTBPerWeightFilter;
#endif
                    vecWeightMoverInstruction.push_back(instructionWMover);
                }


            }
            iterQGlobal += maxTQ;
            iterNGlobal += maxTN;
        } //for iterQTile
        iterPGlobal += maxTP;
        iterMGlobal += maxTM;
    } //for iterPTile

}
