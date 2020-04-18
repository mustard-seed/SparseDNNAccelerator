#include "params.hpp"
#include "device_structures.hpp"
#include "channels.hpp"
#include "device_utils.hpp"
#include "ihc_apint.h"
#include "rtl_lib.hpp"

/** t_conv_input_index
 * \brief Structure that annotates an index in the stretched-padded domain
 * \param isPad bool. Indicate whether the index in the stretched-padded domain is a padding
 * \aram index. The corresponding index in the dense domain. Meaningless if isPad is TRUE;
 */
typedef struct __attribute__((packed)) {
	bool isPad; //Flag for whether the element in the stretched-padded domain correspond to a padding;
	unsigned short index;
} t_conv_input_index; 


/**
 * @brief      Finds the corresponding index in the dense domain for an index in the streched-padded domain
 *
 * @param[in]  stridedPaddingShift          The amount the offset-adjusted index in the strechted-padded domain should be right shifted to recover the dense index
 * @param[in]  stridedPaddingRemainderMask  The mask to recover the remainder of the offset-adjusted index divided by 2^stridedPaddingShift
 * @param[in]  borderPadding                The border padding
 * @param[in]  numDenseInput                The number of dense inputs along the dimension
 * @param[in]  sPIndex                      The stretched-padded index
 *
 * @return     Whether the SP index corresponds to a padding, and (if not) the corresponding dense index
 */
t_conv_input_index sPIndex2RegularIndex (
		unsigned char stridedPaddingShift,
		unsigned char stridedPaddingRemainderMask,
		unsigned char borderPadding, //Number of paddings on the boarder
		unsigned short numDenseInput,
		unsigned short sPIndex //Index in the strided padded domain
	)
{
	//unsigned short unitSize = (stridedPadding + 1);
	unsigned short sPIndexZeroed = sPIndex - borderPadding;
	unsigned short regularIndex = sPIndexZeroed >> stridedPaddingShift;
	unsigned short mod = sPIndexZeroed & stridedPaddingRemainderMask;
	
	t_conv_input_index result;
	result.isPad = ((sPIndex < borderPadding)
		|| (sPIndex >= (borderPadding + ( (numDenseInput-1) << stridedPaddingShift) + 1))
		|| (mod != 0x0)); //Need to becareful when the shift is 0;
	result.index = regularIndex;

	return result;
}

#ifdef MEMORY_READER
/*! kernelSimpleWeightStreamer
Important assumption: The address cache only stores the BRAM address of the first streaming block in each strip.
*/
__attribute__((max_global_work_dim(0)))
__kernel void kernelMemoryReader (
	/*
	Pointers to external memory regions
	*/

	//Pointer to the filter weights in external memory
	 volatile __global t_dram_block* restrict pDramWeights,

	 #if defined(SPARSE_SYSTEM)
		 //Pointer to filter transfer block count
		 volatile __global t_streamblock_address* restrict pFilterStreamBlockAddress,
	 #else
		 // Number of transfer blocks inside a filter. Used for dense system only
		 unsigned short numTBCountPerFilter,
	 #endif //SPARSE_SYSTEM
	 //Pointer to input activations
	 volatile __global t_dram_block* restrict pInputActivation,

	 #if defined(SPARSE_SYSTEM)
		 //Pointer to input activation transfer block count
		 volatile __global t_streamblock_address* restrict pIAStreamBlockAddress,
	 #else //SPARSE_SYSTEM
		 // Number of transfer blocks inside a IA strip. Used for dense system only
		 unsigned short numTBCountPerIAStrip,
	 #endif
	 //Pointer to bias
	 volatile __global t_accumulator* restrict pBias,

	 /*
	 //Distance between the start of successive X-Y strip/filters in DRAM in terms of transfer blocks
	 */
	unsigned int strideExternalMemoryWeights,
	unsigned int strideExternalMemoryIA,

	/*
	Output width tiling parameters
	*/
	unsigned short outputWidth, //Q
	unsigned char sizeOutputTileWidthPerColumnFull, //TQ_A
	unsigned short sizeOutputTileWidthFull, //TQ_A * PE_COLS
	unsigned char sizeOutputTileWidthPerColumnPartial, //Output tile width per column for the final few columns
	unsigned short sizeOutputTileWidthPartial, //partialTQ_A * APartial
	unsigned char numPartialColumns, //APartial 
	unsigned char numOutputWidthTile, //ceil (Q / (TQ_A * PE_COLS)) 
	unsigned char numOutputWidthFullTile, // floor (Q / (TQ_A * PE_COLS)) 
	unsigned short sizeInputTileWidthFull, // (sizeOutputTileWidthFull - 1)*stride + kernelSize
	unsigned short sizeInputTileWidthPartial, // (sizeOutputTileWidthPartial - 1)*stride + kernelSize
	unsigned char sizeInputTileWidthPerColumnFull, // (sizeOutputTileWidthPerColumnFull - 1)*stride + kernelSize
	unsigned char sizeInputTileWidthPerColumnPartial, // (sizeOutputTileWidthPerColumnPartial - 1)*stride + kernelSize
	unsigned short strideInputTileWidthFull, //sizeOutputTileWidthFull * stride
	unsigned short strideInputTileWidthPartial, //sizeOutputTileWidthPartial * stride 

	/*
	Output height tiling parameters
	*/
	unsigned short outputHeight, //P
	unsigned char sizeOutputHeightTileFull, //TP
	unsigned char sizeOutputHeightTilePartial, //P mod TP
	unsigned char numOutputHightTile, //ceil (P / TP)
	unsigned char numOutputHeightFullTile, // floor (P / TP)
	unsigned short sizeInputTileHeightFull, // (sizeOutputTileHeightFull - 1)*stride + kernelSize
	unsigned short sizeInputTileHeightPartial, // (sizeOutputTileHeightPartial - 1)*stride + kernelSize
	unsigned short strideInputTileHeightFull, //sizeOutputHeightTileFull * stride
	unsigned short strideInputTileHeightPartial, //sizeOutputHeightTilePartial * stride

	unsigned short numOutputTiles, //numOutputHeightTile * numOutputWidthTile

	/*
	Input X-Y dimensions
	Without padding around or between input elements
	*/
	unsigned short inputWidth,
	unsigned short inputHeight,

	//Stride between successive strips of IA in terms of dram block
	unsigned short strideStripIACache, //Stride in terms of dram block

	/*
	Paddings. 
	Assume border paddings are symmetrical
	Stride padding is for transpose convolution
	index_in_zero_padded_tensor - padding >> stridePaddingShift = actual index
	index_in_zero_padded_tensor - padding & stridePaddingRemainderMask == 0x0 => is actual index
	*/
	unsigned char horizontalBorderPadding,
	unsigned char verticalBorderPadding,
	unsigned char horizontalStridedPaddingShift,
	unsigned char horizontalStridedPaddingRemainderMask,
	unsigned char verticalStridedPaddingShift,
	unsigned char verticalStridedPaddingRemainderMask,

	/*
	Stride and kernel sizes.
	For transpose convolution, the stride is 1
	*/
	unsigned char kernelSize,
	unsigned char stride,

	/*
	Input and output channels
	*/
	unsigned short numFiltersInKernel, //L

	unsigned char numGroups, // L / G
	unsigned short numFiltersInGroup, // G
	unsigned short numFilterFoldsInGroup, //ceil(numFiltersInGroup / PE_ROWS)
	unsigned short numFullFilterFoldsInGroup, 
	unsigned char numActiveRowsPartialFold,
	//unsigned short numFoldInGroup, // ceil (G / F)

	//TODO: What are these for?
	unsigned short numCompressionWindowsInputGroup,
	unsigned short kernelSizexNumFilterFoldsInGroup
	) 
{

	/*
	 Input and weight count cache
	 Bias cache 
	*/

	#if defined(SPARSE_SYSTEM)
		t_streamblock_address cacheFilterStreamBlockAddress [4096] __attribute__((numbanks(1)));
		t_streamblock_address cacheIAStreamBlockAddress [4096] __attribute__((numbanks(1)));
	#endif
	t_accumulator cacheBias [4096] __attribute__((numbanks(1)));

	//=============================================================
	//Read all the filter counts and biases into BRAM as soon as possible
	//==============================================================
	
	EMULATOR_PRINT(("[kernelMemoryReader] Reading the filter biases and counts\n\n"));
	{
		#if defined(SPARSE_SYSTEM)
			for (unsigned short countFilters=0; countFilters < numFiltersInKernel; countFilters++) {
				cacheFilterStreamBlockAddress[countFilters] = pFilterStreamBlockAddress[countFilters];
			}
		#endif

		for (unsigned short countFilters=0; countFilters < numFiltersInKernel; countFilters++) {
			cacheBias[countFilters] = pBias[countFilters];
		}
	}

	//Tile counters
	unsigned char iterPTile=0;
	unsigned char iterQTile=0;

	//Element counters for the upper-left corner of an input tile
	unsigned short iterMElementBase = 0;
	unsigned short iterNElementBase = 0;

	//Iterate over all output tiles
	for (unsigned char iterPxQTile =0; iterPxQTile < numOutputTiles; iterPxQTile++)
	{
		/*
			X-Y Size of the output tile
		*/
		unsigned char sizeOutputHeightTileLocal = (iterPTile < numOutputHeightFullTile) ? 
			sizeOutputHeightTileFull : sizeOutputHeightTilePartial;

		unsigned short sizeOutputWidthTileLocal = (iterQTile < numOutputWidthFullTile) ?
			sizeOutputTileWidthFull : sizeOutputTileWidthPartial;

		unsigned char sizeOutputWidthTilePerColLocal = (iterQTile < numOutputWidthFullTile) ?
			sizeOutputTileWidthPerColumnFull : sizeOutputTileWidthPerColumnPartial;

		unsigned char numActivePeCols = (iterQTile < numOutputWidthFullTile) ?
			PE_COLS : numPartialColumns;

		/*
		Size of the tiles in the padded stretched input domain
		*/
		unsigned short sizeInputHeightTileLocal = (iterPTile < numOutputHeightFullTile) ? sizeInputTileHeightFull : sizeInputTileHeightPartial;
		unsigned short sizeInputWidthTileLocal = (iterQTile < numOutputWidthFullTile) ? sizeInputTileWidthFull : sizeInputTileWidthPartial;
		unsigned char sizeInputWidthTilePerColLocal = (iterQTile < numOutputWidthFullTile) ? sizeInputTileWidthPerColumnFull : sizeInputTileWidthPerColumnPartial;

		unsigned short strideInputTileHeightLocal = (iterPTile < numOutputHeightFullTile) ? strideInputTileHeightFull : strideInputTileHeightPartial;
		unsigned short strideInputTileWidthLocal = (iterQTile < numOutputWidthFullTile) ? strideInputTileWidthFull : strideInputTileWidthPartial;
		
		EMULATOR_PRINT(("[kernelMemoryReader] Start on tY=%d, tX=%d, Toy=%d, Tox=%d, Tox per column=%d, num cols=%d\n\n",
			iterPTile, iterQTile, sizeOutputHeightTileLocal, sizeOutputWidthTileLocal, sizeOutputWidthTilePerColLocal, numActivePeCols));


		#if defined(SPARSE_SYSTEM)
		/*
		Load the streaming block addresses for all the INPUT activation strips in the X-Y region
		Assume the addresses are stored in Group-H-W-C layout.
		*/
		{
			unsigned char iAddressGroup = 0;
			//Iterators of x-y position on the input plane
			unsigned short iterMAddressElement = iterMElementBase;
			unsigned short iterNAddressElement = iterNElementBase;
			//Iterators of x-y positions in the INUPT tile
			unsigned short iterMInTile = 0;
			unsigned short iterNInTile = 0;
			//Index of the address cache array
			unsigned short iterAddressCache=0;

			unsigned short numGroupsxInputTileWxInputTileH = ((unsigned short) numGroups) * sizeInputHeightTileLocal * sizeInputWidthTileLocal;

			EMULATOR_PRINT(("[kernelMemoryReader] tY=%d, tX=%d. START loading the input counts.\n\n", iterPTile, iterQTile));
			//Index over X in tile -> Y in tile -> Group
			for (unsigned short i=0; i<numGroupsxInputTileWxInputTileH; i++)
			{

				/*
					Determine the indices of the said input element in the dense input domain
				*/
				t_conv_input_index idxMDense = sPIndex2RegularIndex(
						verticalStridedPaddingShift,
						verticalStridedPaddingRemainderMask,
						verticalBorderPadding,
						inputHeight,
						iterMAddressElement
					);

				t_conv_input_index idxNDense = sPIndex2RegularIndex(
						horizontalStridedPaddingShift,
						horizontalStridedPaddingRemainderMask,
						horizontalBorderPadding,
						inputWidth,
						iterNAddressElement
					);	

				/*
				Get the input TB count;
				*/
				bool isPad = (idxMDense.isPad || idxNDense.isPad);

				unsigned short iterAddressDDR = isPad ? 0 : (iAddressGroup*inputHeight + idxMDense.index) *inputWidth + idxNDense.index;
				if (isPad == true)
				{
					cacheIAStreamBlockAddress[iterAddressCache] = numCompressionWindowsInputGroup;
				}
				else
				{
					cacheIAStreamBlockAddress[iterAddressCache] = pIAStreamBlockAddress[iterAddressDDR];
				}

				/*
				Update intra tile loop carrier variables
				*/
				if ((iterNInTile+1) == sizeInputWidthTileLocal)
				{
					iterNInTile=0;
					iterNAddressElement = iterNElementBase;

					if ((iterMInTile+1) == sizeInputHeightTileLocal)
					{
						iterMInTile = 0;
						iterMAddressElement = iterMElementBase;

						iAddressGroup++;
					}
					else
					{
						iterMInTile++;
						iterMAddressElement++;
					}
				}
				else
				{
					iterNInTile++;
					iterNAddressElement++;
				}

				iterAddressCache++;
			} // end of loop for loading the input transfer block counts for the current tile
			EMULATOR_PRINT(("[kernelMemoryReader] tY=%d, tX=%d. FINISHED loading the input counts.\n\n", iterPTile, iterQTile));
		} // Load input counts
		#endif //SPARSE_SYSTEM

		/*
		Send control packet to the tile controller
		*/
		{
			EMULATOR_PRINT(("[kernelMemoryReader] tY=%d, tX=%d. START sending the input tile control packet.\n\n", iterPTile, iterQTile));
			t_input_buffer_tile_controller_packet tileControllerPacket;

			tileControllerPacket.inputTileWidth = sizeInputWidthTilePerColLocal;
			tileControllerPacket.inputTileHeight = sizeInputHeightTileLocal;
			//tileControllerPacket.stride = stride;
			//tileControllerPacket.kernelSize = kernelSize;
			tileControllerPacket.strideConcatKernelSize = ((stride & 0xF) << 0x4) | (kernelSize & 0xF);
			tileControllerPacket.numOutputInstructions = kernelSizexNumFilterFoldsInGroup * sizeOutputHeightTileLocal * sizeOutputWidthTilePerColLocal;
			tileControllerPacket.numActivePeColsConcatNumOutputChannelsInGroup = (((unsigned short) numActivePeCols) << 12) | (numFiltersInGroup & 0xFFF);
			//tileControllerPacket.numOutputChannelsInGroup = numFiltersInGroup;
			tileControllerPacket.strideStripIACache = strideStripIACache;
			#ifndef SPARSE_SYSTEM
				tileControllerPacket.numTBCountPerStrip = numTBCountPerIAStrip;
			#endif

			write_channel_intel(channel_to_ia_tile_controller, tileControllerPacket);
			EMULATOR_PRINT(("[kernelMemoryReader] tY=%d, tX=%d. Finished sending the input tile control packet.\n\n", iterPTile, iterQTile));

		}

		/*
		Stream IA strips to the buffers
		The stride between successive strip being sent is sizeInputWidthTilePerColLocal
		*/
		{
			unsigned char iIAGroup = 0;
			unsigned char iterMInTile = 0;
			unsigned char iterNInPerColTile = 0;
			unsigned short iterNInTile = 0;
			unsigned char iterPeCol = 0;
			unsigned char strideInputWidthTilePerColLocal = stride * sizeOutputWidthTilePerColLocal;

			unsigned short iFilterGlobal = 0;
			int iTransferBlockFilterBaseDDR = 0;

			//Iterate over groups
			while (iIAGroup < numGroups)
			{

				unsigned short sizeInputTileWidthPerColxSizeInputTileHeightxSizeActivePeCols =
					(unsigned short) sizeInputWidthTilePerColLocal* (unsigned short) sizeInputHeightTileLocal* (unsigned short) numActivePeCols;
				EMULATOR_PRINT(("[kernelMemoryReader] tY=%d, tX=%d, iGroup=%d. START sending the input tile from memory to buffers.\n\n", iterPTile, iterQTile, iIAGroup));
				//iterate over input within the groups
				//while (iterMInTile < sizeInputHeightTileLocal)
				for (unsigned short iter=0; iter<sizeInputTileWidthPerColxSizeInputTileHeightxSizeActivePeCols; iter++)
				{
					unsigned short iterMStretchedPaddedGlobal = iterMInTile + iterMElementBase;
					unsigned short iterNStretchedPaddedGlobal = iterNInTile + iterNElementBase;

					t_conv_input_index denseMIndex = sPIndex2RegularIndex (
						verticalStridedPaddingShift,
						verticalStridedPaddingRemainderMask,
						verticalBorderPadding, //Number of paddings on the boarder
						inputHeight,
						iterMStretchedPaddedGlobal //Index in the strided padded domain
					);

					t_conv_input_index denseNIndex = sPIndex2RegularIndex (
						horizontalStridedPaddingShift,
						horizontalStridedPaddingRemainderMask,
						horizontalBorderPadding, //Number of paddings on the boarder
						inputWidth,
						iterNStretchedPaddedGlobal //Index in the strided padded domain
					);

					int stripIndexGlobal = 
						((int) iIAGroup * (int) inputHeight + denseMIndex.index)*(int) inputWidth + denseNIndex.index;

					bool isPad = denseMIndex.isPad || denseNIndex.isPad;

					//Address in terms of t_transfer_block
					int iterIADDR = isPad ? 0 : stripIndexGlobal * strideExternalMemoryIA;

					int stripSPIndexLocal =
						((unsigned short) iIAGroup * (unsigned short) sizeInputHeightTileLocal + (unsigned short) iterMInTile) * (unsigned short) sizeInputWidthTileLocal + (unsigned short) iterNInTile;

					#if defined(SPARSE_SYSTEM)
						unsigned short numIATransferBlocks = cacheIAStreamBlockAddress[stripSPIndexLocal];
					#else
						unsigned short numIATransferBlocks = numTBCountPerIAStrip;
					#endif
					unsigned short dramBlockCount = 1+ ( (numIATransferBlocks-1) >> WIDE_SIZE_OFFSET );

					//Transfer the strip to the input buffer
					
					#if defined(SPARSE_SYSTEM)
						unsigned short numTransferActions = dramBlockCount + 1;
					#else
						unsigned short numTransferActions = dramBlockCount;
					#endif

					for (unsigned short iterTransfer=0; iterTransfer<numTransferActions; iterTransfer++)
					{
						t_dram_block dramBlock;

						#if defined(SPARSE_SYSTEM)
							if (iterTransfer==0)
							{
								dramBlock = transferBlockCount2DramBlock(numIATransferBlocks);
							}
							else
							{
								if (isPad)
								{
									//Prepare a DRAM block with 0 bitmasks
									#pragma unroll
									for (unsigned char i=0; i<WIDE_SIZE; i++)
									{
										#pragma unroll
										for (unsigned char j=0; j<(TRANSFER_SIZE*CLUSTER_SIZE); j++)
										{
											dramBlock.transferBlocks[i].values[j]=0;
										}
									}
								}
								else
								{
									dramBlock = pInputActivation[iterIADDR >> WIDE_SIZE_OFFSET];
									iterIADDR += WIDE_SIZE;
								}
							}
						#else //SPARSE_SYSTEM
							if (isPad)
							{
								//Prepare a DRAM block with 0 bitmasks
								#pragma unroll
								for (unsigned char i=0; i<WIDE_SIZE; i++)
								{
									#pragma unroll
									for (unsigned char j=0; j<(TRANSFER_SIZE*CLUSTER_SIZE); j++)
									{
										dramBlock.transferBlocks[i].values[j]=0;
									}
								}
							}
							else
							{
								dramBlock = pInputActivation[iterIADDR >> WIDE_SIZE_OFFSET];
								iterIADDR += WIDE_SIZE;
							}
						#endif //SPARSE_SYSTEM

						t_dram_block_ia_tagged iaBlock;
						iaBlock.dramBlock = dramBlock;
						iaBlock.destinationCol = iterPeCol;

						write_channel_intel(channel_ia_wide[0], iaBlock);
					}


					/*
					Parameter updates
					*/
					if ((iterPeCol+1) == numActivePeCols)
					{
						iterPeCol = 0;
						if ((iterNInPerColTile+1) == sizeInputWidthTilePerColLocal)
						{
							iterNInPerColTile = 0;
							iterMInTile++;
						}
						else
						{
							iterNInPerColTile++;
						}
						iterNInTile = iterNInPerColTile;

					}
					else
					{
						iterPeCol++;
						iterNInTile += strideInputWidthTilePerColLocal;
					}
				} // while over input tiles

				EMULATOR_PRINT(("[kernelMemoryReader] tY=%d, tX=%d, iGroup=%d. FINISHED sending the input tile from memory to buffers.\n\n", iterPTile, iterQTile, iIAGroup));

				//unsigned short iFilterInGroup = 0; //gf * F

				unsigned short iFilterFold = 0;
				unsigned char iFilterInFold = 0;
				//while (iFilterFold < numFilterFoldsInGroup)
				for (unsigned short iFilterInGroup=0; iFilterInGroup<numFiltersInGroup; iFilterInGroup++)
				{
					EMULATOR_PRINT(("[kernelMemoryReader] tY=%d, tX=%d, iGroup=%d, iFilterGlobal=%d. START sending the weights\n\n",
						iterPTile, iterQTile, iIAGroup, iFilterGlobal));

					unsigned char numFiltersInFold = (iFilterFold < numFullFilterFoldsInGroup) ?
						PE_ROWS : numActiveRowsPartialFold;

					#if defined(SPARSE_SYSTEM)
						unsigned short maxTransferBlockInFilter = cacheFilterStreamBlockAddress[iFilterGlobal];
					#else
						unsigned short maxTransferBlockInFilter = numTBCountPerFilter;
					#endif`
					
					t_accumulator bias = cacheBias[iFilterGlobal];

					unsigned short maxDramBlockInFilter = ((maxTransferBlockInFilter-1) >> WIDE_SIZE_OFFSET) + 1;
					//unsigned short maxTransmitCount = maxDramBlockInFilter+1; //one extra for filter stream control;
					
					t_filter_streamer_control control;
					control.numOutputs = (unsigned short) sizeOutputHeightTileLocal * (unsigned short) sizeOutputWidthTilePerColLocal;
					control.bias = bias;
					control.numTransferBlocks = maxTransferBlockInFilter;
					control.maxPeCols = (numActivePeCols - 1);

					t_dram_block dramControl = filterStreamerControl2dramBlock(control);

					unsigned int iTransferBlockDDR = iTransferBlockFilterBaseDDR;

					//one extra for filter stream control
					for (unsigned short iTransmitCount=0; iTransmitCount<=maxDramBlockInFilter; iTransmitCount++)
					{
						t_dram_block block;
						if (iTransmitCount == 0) 
						{
							block = dramControl;
						}
						else
						{
							block = pDramWeights[iTransferBlockDDR >> WIDE_SIZE_OFFSET];
							iTransferBlockDDR += WIDE_SIZE;
						}

						t_dram_block_w_tagged taggedBlock;
						taggedBlock.dramBlock = block;
						taggedBlock.destinationRow = iFilterInFold;

						write_channel_intel(channel_weight_wide[0], taggedBlock);
					} // iTransmitCount


					EMULATOR_PRINT(("[kernelMemoryReader] tY=%d, tX=%d, iGroup=%d, iFilterGlobal=%d. FINISHED sending the weights\n\n",
						iterPTile, iterQTile, iIAGroup, iFilterGlobal));

					iTransferBlockFilterBaseDDR += strideExternalMemoryWeights;
					iFilterGlobal++;

					/*
					Parameter updates
					*/
					if ((iFilterInFold+1) == numFiltersInFold)
					{
						iFilterInFold = 0;
						iFilterFold++;
					}
					else
					{
						iFilterInFold++;
					}
				} // while loop over the filter folds

				iIAGroup++;
				
			} //while over groups

		} // end of IA and weight transfer for the 2D tile

		/*
		Update loop-carried parameters
		*/
		if ((iterQTile + 1) == numOutputWidthTile)
		{
			iterQTile = 0;
			iterPTile++;

			iterNElementBase = 0;

			iterMElementBase += strideInputTileHeightLocal;
		}
		else
		{
			iterQTile++;

			iterNElementBase += strideInputTileWidthLocal;
		}
	} // end of the loop over the output width and height tiles
}

__attribute__((max_global_work_dim(0)))
__kernel void kernelIAMover (
		volatile __global t_dram_block* restrict pIA1,
		volatile __global t_dram_block* restrict pIA2,

		#if defined(SPARSE_SYSTEM)
			volatile __global t_streamblock_address* restrict pTBCount1,
			volatile __global t_streamblock_address* restrict pTBCount2,
		#endif

		volatile __global t_ia_mover_instruction* restrict pInstruction,
		unsigned int numInstruction
	)
{
	for (unsigned int iInst=0; iInst<numInstruction; iInst++)
	{
		//Read the instruction
		t_ia_mover_instruction inst = pInstruction[iInst];

		/*! Unpacked the concatenated fields of the instruction */
		unsigned char numActiveCols = inst.memRegionCatSparseFlagCatDestinationCatSyncCatNumActiveCols & 0x0F;
		t_flag syncWithOA = (inst.memRegionCatSparseFlagCatDestinationCatSyncCatNumActiveCols >> 0x04) & 0x01;
		t_flag destinationMisc.memRegionCatSparseFlagCatDestinationCatSyncCatNumActiveCols = (inst >> 0x05) & 0x01;
		t_flag sparseInput = (inst.memRegionCatSparseFlagCatDestinationCatSyncCatNumActiveCols >> 0x06) & 0x01;
		t_flag memRegion = (inst.memRegionCatSparseFlagCatDestinationCatSyncCatNumActiveCols >> 0x07) & 0x01;
		uint2_t tileLeftPadding = inst.concatPadding & 0x03;
		uint2_t tileRightPadding = (inst.concatPadding >> 0x02) & 0x03;
		uint2_t tileTopPadding = (inst.concatPadding >> 0x04) & 0x03;
		uint2_t tileBottomPadding = (inst.concatPadding >> 0x06) & 0x03;
		uint4_t hInitSPIndex = inst.concatInitSPIndices & 0x0F;
		uint4_t vInitSPIndex = (inst.concatInitSPIndices >> 0x04) & 0x0F;
		uint4_t colSPSize = inst.concatSPSize & 0x0F;
		uint4_t rowSPSize = (inst.concatSPSize >> 0x04) & 0x0F;

		/*! Setup the iterators for the tile transfer*/
		uint4_t iColSPUnitIndex = hInitSPIndex;
		uint4_t iRowSPUnitIndex = vInitSPIndex;

		//Iterator of the column and row index of the strip inside the tile
		signed char iColInSPTile = 0;
		signed char iRowInSPTile = 0;

		//Address offset contribution from tile and column in the tile
		signed int offsetIADramBlockRow = 0;
		signed int offsetIADramBlockCol = 0;
		#if defined(SPARSE_SYSTEM)
			signed int offsetTBCountRow = 0;
			signed int offsetTBCountCol = 0;
		#endif

		//Wait for the wait for the synchronous signal from the OA buffer
		if (syncWithOA == TRUE)
		{
			unsigned char token = read_channel_intel(channel_activation_sync);
			mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
		}

		//iterate over all IA strips in tile
		for (unsigned short iter=0; iter < inst.tileSPWidthxTileSPHeight; iter++)
		{
			//Setup the strip transfer parameters
			bool colIsDense = (iColInSPTile >= ((signed char) tileLeftPadding))
				&& (iColInSPTile < (inst.tileSPWidth - ((unsigned char) tileRightPadding)) )
				&& (iColSPUnitIndex == 0);
			bool rowIsDense = (iRowInSPTile >= ((signed char) tileTopPadding))
				&& (iRowInSPTile < (inst.tileSPHeight - ((unsigned char) tileBottomPadding)) )
				&& (iRowSPUnitIndex == 0);

			bool realStrip = colsIsDense && rowIsDense;

			int addressIADramBlockDDR = ((t_int) inst.memBlockStart) + offsetIADramBlockCol + offsetIADramBlockRow;

			#if defined(SPARSE_SYSTEM)
				//For the sparse case, we need to consider whether the input is actually sparse,
				//whether the input is padding
				int addressTBCountDDR = ((t_int) inst.memTBCountStart) + offsetTBCountCol + offsetTBCountRow;
				unsigned short numTBInStrip;
				if (realStrip == true)
				{
					if (sparseInput == 0x1)
					{
						if (memRegion == 0x0)
						{
							numTBInStrip = pTBCount1[addressTBCountDDR];
						}
						else
						{
							numTBInStrip = pTBCount2[addressTBCountDDR];
						}
					}
					else
					{
						numTBInStrip = (t_ushort) inst.numCWOrTBInGroup
					}
				}
				else
				{
					numTBInStrip = (t_ushort) inst.numCWOrTBInGroup;
				}
			#else
				unsigned short numTBInStrip = (t_ushort) inst.numCWOrTBInGroup;
			#endif

			//dramBlockCount = ceil(numTBInStrip / WIDE_SIZE)
			unsigned short dramBlockCount = 1 + ( (numTBInStrip-1) >> WIDE_SIZE_OFFSET );
			unsigned short numTransferActions = dramBlockCount + 1;

			for (unsigned short iterTransfer=0; iterTransfer<numTransferActions; iterTransfer++)
			{
				t_dram_block_ia_tagged iaBlock;
				if (iterTransfer==0)
				{
					iaBlock.dramBlock = iaMetadata2DramBlock(
							numTBInStrip, //tbCount
							inst.columnSPWidth, //colSPWidth
							inst.columnWidthStride, //colSPStride	
							iColInSPTile //iColInSPTile
						);
				}
				else
				{
					if (realStrip == true)
					{
						iaBlock.dramBlock = (memRegion == 0x0) ? 
							pIA1[addressIADramBlockDDR] : pIA2[addressIADramBlockDDR];
						addressIADramBlockDDR++;
					}
					else
					{
						//Prepare a DRAM block with 0
						#pragma unroll
						for (unsigned char i=0; i<WIDE_SIZE; i++)
						{
							#pragma unroll
							for (unsigned char j=0; j<(TRANSFER_SIZE*CLUSTER_SIZE); j++)
							{
								iaBlock.dramBlock.transferBlocks[i].values[j]=0;
							}
						}
					}
				}

				unsigned char isLastField = ((iterTransfer+1) == numTransferActions) ?
					0x80 : 0x00;
				unsigned char isMiscField = (destinationMisc == 0x1) ?
					0x40 : 0x00;
				iaBlock.route = isLastField | isMiscField | ((numActiveCols-1) & 0x3F);
				write_channel_intel(channel_ia_wide[0], iaBlock);
			}

			/*! Loop carried variable updates*/
			iColInSPTile++;
			if ( (iColInSPTile >= ((signed char) tileLeftPadding)) 
				&& (iColInSPTile < (inst.tileSPWidth - ((unsigned char) tileRightPadding))) )
			{
				iColSPUnitIndex++;
				if (iColSPUnitIndex >= ((unsigned char) colSPSize))
				{
					iColSPUnitIndex = 0;
					offsetIADramBlockCol += ((t_int) inst.memBlockColStripStride);
					#if defined(SPARSE_SYSTEM)
						offsetTBCountCol +=(t_int) inst.memTBCountColStride;
					#endif
				}
			}
			
			if (iColInSPTile == ((signed char) inst.tileSPWidth))
			{
				iColInSPTile = 0;
				iColSPUnitIndex = hInitSPIndex;
				offsetIADramBlockCol = 0;
				offsetTBCountCol = 0;

				iRowInSPTile++;
				if ( (iRowInSPTile >= ((signed char) tileTopPadding)) 
					&& (iRowInSPTile < (inst.tileSPHeight - ((unsigned char) tileBottomPadding)) ) )
				{
					iRowSPUnitIndex++;
					if (iRowSPUnitIndex >= ((unsigned char) rowSPSize))
					{
						iRowSPUnitIndex = 0;
						offsetIADramBlockRow += ((t_int) inst.memBlockRowStripStride);
						#if defined(SPARSE_SYSTEM)
							offsetTBCountRow += (t_int) inst.memTBCountRowStride;
						#endif
					}
				}
			}
		} //for. iter
	} // for. iInst
}

__attribute__((max_global_work_dim(0)))
__kernel void kernelWMover (
		volatile __global t_weight_mover_instruction* restrict pInst,
		volatile __global t_dram_block* restrict pW,
		volatile __global t_accumulator* restrict pBias,
		#if defined(SPARSE_SYSTEM)
		 //Pointer to filter transfer block count
		 volatile __global t_streamblock_address* restrict pFilterTBCount,
		#else
		 // Number of transfer blocks inside a filter. Used for dense system only
		 unsigned short numTBCountPerFilter,
		#endif //SPARSE_SYSTEM
		int numInstruction
	)
{
	for (int iInst=0; iInst<numInstruction; iInst++)
	{
		t_weight_mover_instruction inst = pInst[iInst];

		unsigned short iFilterFold = 0;
		unsigned char iFilterInFold = 0;

		signed int addrWeightFilterBase = inst.memWeightStart;
		signed int addrBias = inst.memBiasStart;
		#if defined(SPARSE_SYSTEM)
			signed int addrWeightTB = inst.memTBCountStart;
		#endif
		for (unsigned short iFilterInGroup=0; iFilterInGroup<inst.numFiltersInGroup; iFilterInGroup++)
		{

			unsigned char numFiltersInFold = (iFilterFold < inst.numFullFilterFold) ?
				PE_ROWS : inst.numFiltersInPartialFold;

			#if defined(SPARSE_SYSTEM)
				unsigned short maxTransferBlockInFilter = pFilterTBCount[addrWeightTB];
			#else
				unsigned short maxTransferBlockInFilter = inst.numTBPerFilter;
			#endif
			
			t_accumulator bias = pBias[addrBias];

			unsigned short maxDramBlockInFilter = ((maxTransferBlockInFilter-1) >> WIDE_SIZE_OFFSET) + 1;
			//unsigned short maxTransmitCount = maxDramBlockInFilter+1; //one extra for filter stream control;
			
			t_filter_streamer_control control;
			control.numOutputs = inst.filterReuse;
			control.bias = bias;
			control.numTransferBlocks = maxTransferBlockInFilter;
			control.maxPeCols = (inst.numActivePeCols - 1);

			t_dram_block dramControl = filterStreamerControl2dramBlock(control);

			int iDramBlock = addrWeightFilterBase;

			//one extra for filter stream control
			for (unsigned short iTransmitCount=0; iTransmitCount<=maxDramBlockInFilter; iTransmitCount++)
			{
				t_dram_block block;
				if (iTransmitCount == 0) 
				{
					block = dramControl;
				}
				else
				{
					block = pW[iDramBlock];
					iDramBlock++;
				}

				t_dram_block_w_tagged taggedBlock;
				taggedBlock.dramBlock = block;
				taggedBlock.destinationRow = iFilterInFold;

				write_channel_intel(channel_weight_wide[0], taggedBlock);
			} // iTransmitCount

			addrWeightFilterBase += inst.memWeightFilterStride;

			/*
			Parameter updates
			*/
			if ((iFilterInFold+1) == numFiltersInFold)
			{
				iFilterInFold = 0;
				iFilterFold++;
			}
			else
			{
				iFilterInFold++;
			}
			addrBias++;
			#if defined (SPARSE_SYSTEM)
				addrWeightTB++;
			#endif
		} //for loop over the filters
	}  //for loop over instructions
}

#endif //MEMORY_READER

#ifdef IA_MEMORY
//TODO: Change this to handle the new IA data packet
#define IA_BUFFER_STATE_DECODE 0x1
#define IA_BUFFER_STATE_COMPUTE_NUM_ACCESS 0x2
#define IA_BUFFER_STATE_ACCESS 0x4
#define IA_BUFFER_STATE_PADD 0x8 //NOP operation
#define IA_BUFFER_STATE_UPDATE_STRIP 0x10
#define IA_BUFFER_PADD_COUNT 2 
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelIABuffer ()
{
	typedef uint5_t t_state;
	typedef uint6_t t_strip;
	int colID = get_compute_id(0);

	t_dram_block cacheIABlocks [IA_CACHE_DEPTH] __attribute__((bankwidth(BURST_SIZE_BYTE)));

	#if defined(SPARSE_SYSTEM)
		t_streamblock_address cacheIAStreamBlockAddress [256] __attribute__((numbanks(1)));
	#endif

	/*
		Loop carried-variables
	*/

	uint1_t isLoad = FALSE;
	uint1_t isLast = FALSE;
	unsigned short iaDramBlockAddressBase = 0;
	unsigned short iaDramBlockColStride = 0;
	unsigned short iaDramBlockRowStride = 0;
	unsigned short iaDramBlockColContribution = 0;
	unsigned short iaDramBlockRowContribution = 0;
	#if defined(SPARSE_SYSTEM)
		unsigned char tbAddressBase = 0;
		unsigned char tbAddressRowStride = 0;
		unsigned char tbAddressColContribution = 0;
		unsigned char tbAddressRowContribution = 0;
		uint1_t flagPadBitmask = FALSE;
		unsigned char iTBInCW = 0; //Only useful for dense input
	#else
		unsigned short numTBPerStrip = 0;
	#endif
	unsigned char maxPeRowID = 0;
	unsigned short numIAAccess = 0;
	unsigned short iterAccess = 0;


	//Counters and states
	t_strip iStripInRow = 0;
	t_strip iStripInCol = 0;
	t_strip numStripsRow = 0;
	t_strip numStripsCol = 0;

	t_state currentState = IA_BUFFER_STATE_DECODE;

	while (true)
	{
		t_state nextState = currentState;
		t_dram_block dramBlock;
		bool dataReadSuccess = false;

		//Handle channel read separately
        if (((currentState == IA_BUFFER_STATE_COMPUTE_NUM_ACCESS) && (isLoad == TRUE))
                || ((currentState == IA_BUFFER_STATE_ACCESS) && (isLoad == TRUE)))
		{
			dramBlock = read_channel_nb_intel(channel_ia_wide_local[colID], &dataReadSuccess);
		}
		/*
		First, read the instruction from the tile controller
		*/
		if (currentState == IA_BUFFER_STATE_DECODE)
		{
			bool success = false;
			t_input_buffer_tile_buffer_packet controlPacketReceived = read_channel_nb_intel(channel_control_to_ia_buffer_local[colID], &success);

			if (success)
			{
				isLoad = ((controlPacketReceived.controlBits & 0x3) == 0x1) ? TRUE: FALSE;
				iaDramBlockAddressBase = controlPacketReceived.iaDramBlockAddressBase;
				iaDramBlockColStride = controlPacketReceived.iaDramBlockColStride;
				iaDramBlockRowStride = controlPacketReceived.iaDramBlockRowStride;
				iaDramBlockColContribution = 0;
				iaDramBlockRowContribution = 0;
				maxPeRowID = controlPacketReceived.maxPeRowID;
				numStripInRow = controlPacketReceived.numStripInRow;
				#if defined(SPARSE_SYSTEM)
					flagPadBitmask = (controlPacketReceived.controlBits >> 2) & 0x01;
					tbAddressBase = controlPacketReceived.tbAddressBase;
					tbAddressRowStride = controlPacketReceived.tbAddressRowStride;
					tbAddressColContribution = 0;
					tbAddressRowContribution = 0;
				#endif
				numStripsRow = controlPacketReceived.numStripsRow;
				numStripsCol = controlPacketReceived.numStripsCol;

				//Initialize the counters
				iterAccess = 0;
				iStripInRow = 0;
				iStripInCol = 0;

				nextState = IA_BUFFER_STATE_COMPUTE_NUM_ACCESS;

					EMULATOR_PRINT(("[kernelIABuffer %d] START processing instruction. isLoad=%d, isLast=%d, iActivationDramBlockAddressBase=%d, numStrips=%d, maxPeRowID=%d\n\n",
				colID, (unsigned char) isLoad, (unsigned char) isLast, iActivationDramBlockAddressBase, (unsigned char) numStripInRow, maxPeRowID));
			}
		} //IA_BUFFER_STATE_DECODE
		else if (currentState == IA_BUFFER_STATE_COMPUTE_NUM_ACCESS)
		{
			if (isLoad == TRUE)
			{
				if (dataReadSuccess == true)
				{
					t_streamblock_address numIATransferBlocks = getTBCount(dramBlock);
					numIAAccess = 1 + ((numIATransferBlocks-1) >> WIDE_SIZE_OFFSET);
					#if defined(SPARSE_SYSTEM)
						cacheIAStreamBlockAddress[tbAddressBase + tbAddressColContribution + tbAddressRowContribution] = numIATransferBlocks;
					#else
						numTBPerStrip = numIATransferBlocks;
					#endif

					nextState = IA_BUFFER_STATE_ACCESS;
				}
			}
			else
			{
				#if defined(SPARSE_SYSTEM)
					numIAAccess = cacheIAStreamBlockAddress[tbAddressBase + tbAddressColContribution + tbAddressRowContribution];
					iAddressCache++;
					iTBInCW = 0;
					nextState = IA_BUFFER_STATE_ACCESS;
				#else
					numIAAccess = numTBPerStrip;
					nextState = IA_BUFFER_STATE_ACCESS;
				#endif
			}
		} //IA_BUFFER_STATE_COMPUTE_NUM_ACCESS
		else if (currentState == IA_BUFFER_STATE_PADD)
		{
			iterAccess++;
			if (iterAccess >= IA_BUFFER_PADD_COUNT)
			{
				nextState = IA_BUFFER_STATE_DECODE;
			}
		} //IA_BUFFER_STATE_PADD
		else if (currentState == IA_BUFFER_STATE_ACCESS)
		{
			if (isLoad == TRUE)
			{
				if (dataReadSuccess)
				{
					cacheIABlocks[iterAccess + iaDramBlockAddressBase + iaDramBlockColContribution + iaDramBlockRowContribution] = dramBlock;
					iterAccess++;
				}
			}
			else
			{
				t_dram_block dramBlock = cacheIABlocks[iaDramBlockAddressBase + iaDramBlockColContribution + iaDramBlockRowContribution +  ((unsigned short)(iterAccess >> WIDE_SIZE_OFFSET))];	

				t_transferblock_tagged taggedBlock;

				unsigned char isLastTemp =  (((iterAccess + 1) == numIAAccess) && ((iStripInRow+1) == numStripsRow) && ((iStripInCol+1) == numStripsCol)) ?
						TRUE : FALSE;

				#if defined(SPARSE_SYSTEM)
					//Insert the bitmask
					if ((iTBInCW == 0) && (flagPadBitmask == TRUE))
					{
						#pragma unroll
						for (int i=0; i<TRANSFER_SIZE*CLUSTER_SIZE; i++)
						{
							taggedBlock.values.values[i] = 0xFF;
						}
						isLastTemp = FALSE;
					}
					else
					{
						taggedBlock.values = dramBlock.transferBlocks[iterAccess & WIDE_SIZE_REMAINDER_MASK];
					}
				#else
					taggedBlock.values = dramBlock.transferBlocks[iterAccess & WIDE_SIZE_REMAINDER_MASK];
				#endif

				setMaxTransferID(&taggedBlock, maxPeRowID);
				setIsLast(&taggedBlock, isLastTemp);
				
				bool success = write_channel_nb_intel(channel_activation[0][colID], taggedBlock);
				if (success)
				{
					#if defined(SPARSE_SYSTEM)
						if ((iTBInCW > 0) || (flagPadBitmask == FALSE))
						{
							iterAccess++
						}
						iTBInCW++;
						if ( iTBInCW == (COMPRESSION_WINDOW_SIZE / TRANSFER_SIZE) )
						{
							iTBInCW = 0;
						}
					else
						iterAccess++;
					#endif
					EMULATOR_PRINT(("[kernelIABuffer %d] Sent TB %d / %d. TB[0-3]: %#04x %#04x %#04x %#04x \n\n",
					colID, iterAccess, numIAAccess
					,taggedBlock.values.values[0]
					,taggedBlock.values.values[1]
					,taggedBlock.values.values[2]
					,taggedBlock.values.values[3]
					));
				}
			}

			if (iterAccess == numIAAccess)
			{
				nextState = IA_BUFFER_STATE_UPDATE_STRIP;
			}
		} //IA_BUFFER_STATE_ACCESS
		else if (currentState == IA_BUFFER_STATE_UPDATE_STRIP)
		{
			#if defined(SPARSE_SYSTEM)
				nextState = IA_BUFFER_STATE_COMPUTE_NUM_ACCESS;
			#else
				nextState = (isLoad == TRUE) ?  IA_BUFFER_STATE_COMPUTE_NUM_ACCESS : IA_BUFFER_STATE_ACCESS;
			#endif
			iterAccess = 0;

			iStripInCol++;
			iaDramBlockColContribution += iaDramBlockColStride;
			#if defined(SPARSE_SYSTEM)
				tbAddressColContribution += 1;
			#endif
			if (iStripInCol == numStripsCol)
			{
				iStripInCol = 0;
				iStripInRow++;
				iaDramBlockColContribution = 0;
				iaDramBlockRowContribution += iaDramBlockRowStride;
				#if defined(SPARSE_SYSTEM)
					tbAddressColContribution = 0;
					tbAddressRowContribution += tbAddressRowStride;
				#endif

				if (iStripInRow == numStripsRow)
				{
					nextState = IA_BUFFER_STATE_PADD;
					EMULATOR_PRINT(("[kernelIABuffer %d] FINISHED processing instruction. isLoad=%d, isLast=%d, iActivationDramBlockAddressBase=%d, maxPeRowID=%d\n\n",
						colID, (unsigned char) isLoad, (unsigned char) isLast, iActivationDramBlockAddressBase, maxPeRowID));
						nextState = IA_BUFFER_STATE_DECODE;
				}
			}
		}
		currentState = nextState;
	}
}

//TODO: imcomplete
__attribute__((max_global_work_dim(0)))
__kernel void kernelIATileController (
	__global t_ia_tile_controller_instruction* restrict pInstruction,
	int numInstructions
	)
{
	for (int iInstruction=0; iInstruction < numInstructions; iInstruction++)
	{
		/*
		1. Read the instruction of the tile from the memory reader
		*/
		t_ia_tile_controller_instruction instruction = pInstruction[iInstruction];

		unsigned char inputTileWidth = instruction.localTileWidth;
	    unsigned char inputTileHeight = instruction.localTileHeight;
	    unsigned char stride = instruction.kernelStride;
	    unsigned char kernelSize = instruction.kernelSize;
        unsigned int numOutputInstructions = instruction.numOutputInstructions;
	    unsigned char numActivePeCols = instruction.flagPadBitmaskCatNumActiveCols & 0x7F;
	    unsigned short numOutputChannelsInGroup = instruction.numOutputChannelsInGroup;
	    unsigned short iaCacheColStride = instruction.cacheIAStripColStride;

	    #if defined(SPARSE_SYSTEM)
	    	uint1_t flagPadBitmask = ((instruction.flagPadBitmaskCatNumActiveCols & 0x80) >> 7);
	    	unsigned char iStripInTile = 0;
	    #endif
		/*
		2. Send load instructions to the tile buffer
		*/
		unsigned short iaCacheRowStride = iaCacheColStride * ((unsigned short)(inputTileWidth));
		#if defined(SPARSE_SYSTEM)
			loadControlBits |= ((unsigned char) flagPadBitmask) << 2;
		#endif
		EMULATOR_PRINT(("[kernelIATileController] START sending the buffer refresh comomand for instruction=%d .\n\n", iInstruction));
		{
			t_input_buffer_tile_buffer_packet tileBufferControlPacket;
			tileBufferControlPacket.iaDramBlockAddressBase = 0;

			tileBufferControlPacket.iaDramBlockColStride = iaCacheColStride;
			tileBufferControlPacket.iaDramBlockRowStride = iaCacheRowStride;
			
			tileBufferControlPacket.controlBits = ((numActivePeCols-1) << 0x3) | 0x1;

			#if defined(SPARSE_SYSTEM)
                tileBufferControlPacket.tbAddressBase = 0;
                tileBufferControlPacket.tbAddressRowStride = inputTileWidth;
		    #endif

		    tileBufferControlPacket.numStripsCol = inputTileWidth;
		    tileBufferControlPacket.numStripsRow = inputTileHeight;

			write_channel_intel(channel_control_to_ia_buffer[0], tileBufferControlPacket);
		}
		EMULATOR_PRINT(("[kernelIATileController] FINISHED sending the buffer refresh instructions for iInstruction=%d .\n\n", iInstruction));
			
		//End of sending load instructions

		/*
		3. Send the streaming instructions to the tile buffer
		*/
		unsigned short iFilterInGroup = 0;
		unsigned char iInputTileWidth = 0;
		unsigned char iInputTileHeight = 0;
		
		//while (iFilterInGroup < numOutputChannelsInGroup)
        for (unsigned int i=0; i<numOutputInstructions; i++)
		{
			unsigned char numActivePeRows = ((numOutputChannelsInGroup - iFilterInGroup) < (unsigned short) (PE_ROWS)) ?
				(unsigned char) (numOutputChannelsInGroup - iFilterInGroup) : PE_ROWS;

			unsigned char iStripInTile = iInputTileHeight * inputTileWidth + iInputTileWidth;

			t_input_buffer_tile_buffer_packet tileBufferControlPacket;
			tileBufferControlPacket.iaDramBlockAddressBase = ((unsigned short) iStripInTile) * ((unsigned short) strideStripIACache);
			tileBufferControlPacket.maxPeRowID = (numActivePeRows - 1);
			
			tileBufferControlPacket.iaDramBlockColStride = iaCacheColStride;
			tileBufferControlPacket.iaDramBlockRowStride = iaCacheRowStride;

			#if defined(SPARSE_SYSTEM)
		    	tileBufferControlPacket.tbAddressBase = iStripInTile;
		    	tileBufferControlPacket.tbAddressRowStride = inputTileWidth;
		    #endif
			unsigned char sendInstructionType = 0x2; //Stream from the buffer
			tileBufferControlPacket.controlBits =
				(sendInstructionType & 0x3)
				| ((numActivePeCols-1) << 0x3);
			#if defined(SPARSE_SYSTEM)
				tileBufferControlPacket.controlBits |= ((unsigned char) flagPadBitmask) << 2;
			#endif
			tileBufferControlPacket.numStripsCol = kernelSize;
			tileBufferControlPacket.numStripsRow = kernelSize;

			//bool success = write_channel_nb_intel(channel_control_to_ia_buffer[0], tileBufferControlPacket);
			write_channel_intel(channel_control_to_ia_buffer[0], tileBufferControlPacket);	
			/*
				Parameters update
			*/
			//if (success)
			//{
				EMULATOR_PRINT(("[kernelIATileController] FINISHED sending the buffer stream instruction for iInstruction=%d, iFilterInGroup=%d, iInputTileHeight=%d, iInputTileWidth=%d. \n\n", 
				iInstruction, iFilterInGroup, iInputTileHeight, iInputTileWidth));

				if ((iInputTileWidth + kernelSize) >= inputTileWidth)
				{
					iInputTileWidth = 0;

					if ((iInputTileHeight + kernelSize) >= inputTileHeight)
					{
						iInputTileHeight = 0;
						iFilterInGroup += numActivePeRows;
					}
					else
					{
						iInputTileHeight += stride;
					}
				}
				else
				{
					iInputTileWidth += stride;
				}
		}
		//End of sending streaming instructions
	}
}



//TODO: Maybe this kernel can be eliminiated
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelIAControlTee ()
{
	int colID = get_compute_id(0);

	while (true)
	{
		t_input_buffer_tile_buffer_packet controlPacket = read_channel_intel(channel_control_to_ia_buffer[colID]);

		unsigned char maxColID = (controlPacket.controlBits) >> 0x2;

		write_channel_intel(channel_control_to_ia_buffer_local[colID], controlPacket);

		if (colID < (PE_COLS - 1) )
		{
			if (maxColID > (unsigned char) colID )
			{
				write_channel_intel(channel_control_to_ia_buffer[colID+1], controlPacket);
			}
		}
	}
}

#define IA_TEE_COMMAND_READ_STRIP_HEADER 0X0
#define IA_TEE_COMMAND_TRANSFER 0x1
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelIATee ()
{
	int colID = get_compute_id(0);
	typedef uint1_t t_state;
	t_state regState = IA_TEE_COMMAND_READ_STRIP_HEADER;
	uint1_t regFlagRoute2Misc = FALSE;
	uint1_t regFlagRoute2Conv = FALSE;

	while (true)
	{
		uint1_t nextFlagRoute2Misc = regFlagRoute2Misc;
		uint1_t nextFlagRoute2Conv = regFlagRoute2Conv;
		t_state nextState = regState;

		bool readSuccess = false;

		t_dram_block_ia_tagged taggedBlock = read_channel_nb_intel(channel_ia_wide[colID], &readSuccess);
		t_dram_block dramBlock = taggedBlock.dramBlock;

		int destinationCol = (int) (taggedBlock.route & 0x3F);
		uint1_t flag2Misc = (taggedBlock.route >> 0x6) & 0x01;
		uint1_t flagIsLastInStrip = (taggedBlock.route >> 0x7) & 0x01;

		if (readSuccess == true)
		{
			switch (regState) {
				case (IA_TEE_COMMAND_READ_STRIP_HEADER) : 
				{
					signed char actualColIndex = getColSPIndex(dramBlock);
					unsigned char colSPStride = getColSPStride(dramBlock);
					unsigned char colSPWidth = getColSPWidth(dramBlock);

					if ( (((signed) colSPWidth) > actualColIndex)
							&& (actualColIndex >= 0)
						)
					{
						nextFlagRoute2Misc = flag2Misc;
						nextFlagRoute2Conv = ~flag2Misc;
					}

					//Adjust the col index seen by the next compute column
					taggedBlock.dramBlock.transferBlocks[1].values[0] = actualColIndex - ((signed char) colSPStride);

					nextState = IA_TEE_COMMAND_TRANSFER;
				} //IA_TEE_COMMAND_READ_STRIP_HEADER
				break;
				case (IA_TEE_COMMAND_TRANSFER) :
				{
					if (flagIsLastInStrip == TRUE)
					{
						nextState = IA_TEE_COMMAND_READ_STRIP_HEADER;
					}

				} //IA_TEE_COMMAND_TRANSFER
				break;
				default:
				break;
			} //switch

			//Forward to the next column
			if (colID < (PE_COLS - 1))
			{
				if (destinationCol > colID)
				{
					write_channel_intel(channel_ia_wide[colID+1], taggedBlock);
				}
			}

			bool write2Conv = false;;
			if ( ((regState == IA_TEE_COMMAND_READ_STRIP_HEADER) && (nextFlagRoute2Conv == TRUE))
					|| ((regState == IA_TEE_COMMAND_TRANSFER) && (regFlagRoute2Conv == TRUE))
				)
			{
				write2Conv = true;
			}

			if (write2Conv == true)
			{
				write_channel_intel(channel_ia_wide_local[colID], dramBlock);
			}

			//TODO: Add logic to handle the write to MISC unit			
		}

		regState = nextState;
		regFlagRoute2Misc = nextFlagRoute2Misc;
		regFlagRoute2Conv = nextFlagRoute2Conv;
	}
}
#endif //IA_MEMORY

#if defined(MISC_ENGINE)
__attribute__((max_global_work_dim(0)))
__kernel void kernelMiscControlMover (
		__global t_misc_instruction* restrict pInstruction,

	)
{

}
#endif

#ifdef MEMORY_WRITER
__attribute__((max_global_work_dim(0)))
__kernel void kernelOutputWriter (
	//Pointer to the output activation
	volatile __global t_output_dram_block* restrict pOutputActivation,

	#if defined(SPARSE_SYSTEM)
	//Pointer to the output activation transfer block count
	volatile __global t_streamblock_address* restrict pOAStreamBlockAddress,
	#else
	unsigned short numWideCountPerOAStrip,
	#endif

	unsigned int strideExternalMemoryOA, //In terms of output dram block

	/*
	Output width tiling parameters
	*/
	unsigned short outputWidth, //Q
	unsigned char sizeOutputTileWidthPerColumnFull, //TQ_A
	unsigned short sizeOutputTileWidthFull, //TQ_A * PE_COLS
	unsigned char sizeOutputTileWidthPerColumnPartial, //Output tile width per column for the final few columns
	unsigned short sizeOutputTileWidthPartial, //partialTQ_A * APartial
	unsigned char numPartialColumns, //APartial 
	unsigned char numOutputWidthTile, //ceil (Q / (TQ_A * PE_COLS)) 
	unsigned char numOutputWidthFullTile, // floor (Q / (TQ_A * PE_COLS)) 

	/*
	Output height tiling parameters
	*/
	unsigned short outputHeight, //P
	unsigned char sizeOutputHeightTileFull, //TP
	unsigned char sizeOutputHeightTilePartial, //P mod TP
	unsigned char numOutputHightTile, //ceil (P / TP)
	unsigned char numOutputHeightFullTile, // floor (P / TP)

	/*
	Auxillary
	*/
	unsigned short numOutputHxWTiles, //numOutputHeightTile * numOutputWidthTile
	unsigned short numOutputHxW,

	//Number of groups in the output activations
	unsigned short numOutputChannels,
	unsigned short numGroupsCurrentLayer,
	unsigned short numChannelsPerGroupCurrentLayer,
	unsigned short numGroupsNextLayer,
	unsigned short numChannelsPerGroupNextLayer,
	unsigned short numFoldsInGroupCurrentLayer,
    unsigned short numFullFoldsInGroupCurrentLayer,
    unsigned char numActiveRowsInPartialFolds,

	/*
	Output modification
	*/
	unsigned char numAccumulatorBitsToRightShift,
	unsigned char enableOutputRelu, //argument cannot be bool
	unsigned char enableSparsification //argument cannot be bool
	)
{
	#if defined(SPARSE_SYSTEM)
	//Cache of output activation stream block address in one tile
	t_streamblock_address cacheOAStreamBlockAddress [4096] __attribute__((numbanks(1)));
	#endif

	//Auxillary variables
	unsigned short iterP = 0;
	unsigned short iterQ = 0;

	//Loop control variables
	unsigned short iterHeightTile=0;
	unsigned short iterWidthTile=0;
	unsigned short iterHxWTile=0;

	//Iterate over the tiles
	while (iterHxWTile < numOutputHxWTiles)
	{
		//Calculate the effectual output tile height
		unsigned char maxTP = (iterHeightTile < numOutputHeightFullTile) ?
			sizeOutputHeightTileFull : sizeOutputHeightTilePartial;

		/*
		Input activation tile parameters
		*/
		unsigned char maxTQ_A = (iterWidthTile < numOutputWidthFullTile) ?
			sizeOutputTileWidthPerColumnFull : sizeOutputTileWidthPerColumnPartial;

		unsigned char maxPeCols = (iterWidthTile < numOutputWidthFullTile) ?
			PE_COLS : numPartialColumns;

		unsigned short maxTQ = (iterWidthTile < numOutputWidthFullTile) ?
			sizeOutputTileWidthFull : sizeOutputTileWidthPartial;

		unsigned short numGroupNextLayerxTileHeightxTileWidthPerCol = (unsigned short) numGroupsNextLayer* (unsigned short) maxTP* (unsigned short) maxTQ_A;
		//Send the output control
		//Both to the tile controller, and to the Tees.
		{
			EMULATOR_PRINT(("[kernelOutputWriter] START sending the output tile instruction and drainage tee instruction for iterHeightTile=%d, iterWidthTile=%d, out of %d tiles \n\n", 
				iterHeightTile, iterWidthTile, numOutputHxWTiles));
			t_output_tile_controller_packet outputControl;
			outputControl.numOutputTileHeightxWidth = maxTP*maxTQ_A;
			outputControl.numFoldsInGroupCurrentLayer = numFoldsInGroupCurrentLayer;
			outputControl.numFullFoldsInGroupCurrentLayer = numFullFoldsInGroupCurrentLayer;
			outputControl.numActiveRowsInPartialFolds = numActiveRowsInPartialFolds;
			outputControl.numActivePeCols = maxPeCols;
			
			outputControl.numGroupsNextLayer = numGroupsNextLayer;
			outputControl.numChannelsInGroupCurrentLayer = numChannelsPerGroupCurrentLayer;
			outputControl.numChannelsInGroupNextLayer = numChannelsPerGroupNextLayer;
			outputControl.outputModifierBits = generateOutputModifier(numAccumulatorBitsToRightShift, enableOutputRelu, enableSparsification);

			write_channel_intel(channel_output_writer_to_oa_controller, outputControl);

			t_output_tile_tee_packet teePacket;
			teePacket.numOutputGroupxTileHeightxTileWidth = numGroupNextLayerxTileHeightxTileWidthPerCol;
			teePacket.maxColID = (maxPeCols - 1);

			write_channel_intel(channel_output_writer_to_tee[0], teePacket);

			EMULATOR_PRINT(("[kernelOutputWriter] FINISHED sending the output tile instruction and drainage tee instruction for iterHeightTile=%d, iterWidthTile=%d \n\n", 
				iterHeightTile, iterWidthTile));
		}	

		//Drain the outputs
		unsigned short iOutputGroup = 0;
		unsigned char iOutputHeightInTile = 0;
		unsigned char iOutputWidthInTile = 0;
		unsigned char iCol = 0; //iterator for the PE columns
		unsigned short numGroupNextLayerxTileHeightxTileWidth = numGroupNextLayerxTileHeightxTileWidthPerCol * (unsigned short) maxPeCols;

		for (unsigned short iter=0; iter<numGroupNextLayerxTileHeightxTileWidth; iter++)
		{
			//Global indices for the sub-tile from the first PE_COL
			unsigned short iHeightGlobal = iterP + (unsigned short) iOutputHeightInTile;
			unsigned short iWidthGlobal = iterQ + (unsigned short) iOutputWidthInTile;

			unsigned short iActivationGlobal =
				(unsigned int) (iOutputGroup*numOutputHxW + iHeightGlobal*outputWidth + iWidthGlobal + iCol*maxTQ_A)
				* (unsigned int) strideExternalMemoryOA; //iCol*maxTQ_A is zero

			unsigned short iAddressCache = 
				(iOutputGroup*maxTP + iOutputHeightInTile)*maxTQ + (unsigned short) iOutputWidthInTile + (unsigned short) iCol * (unsigned short) maxTQ_A;

			unsigned short clusterCount;

			EMULATOR_PRINT(("[kernelOutputWriter] START draining the output strip from Col=%d, at iOutputHeightInTile=%d, iOutputWidthInTile=%d, iOutputGroup=%d for tile %d / %d\n\n", 
				iCol, iOutputHeightInTile, iOutputWidthInTile, iOutputGroup, iterHxWTile, numOutputHxWTiles));

			#if defined(SPARSE_SYSTEM)
				bool proceed = true;
				while (proceed)
				{
					t_output_dram_block_tagged receivedBlock = read_channel_intel(channel_output_wide[0]);
					if ((receivedBlock.isLastFlag & 0x1) == 0x1)
					{
						proceed = false;
						clusterCount = outputDramBlock2ClusterCount(receivedBlock.block);
					}
					else 
					{
						//Store the dram count
						pOutputActivation[iActivationGlobal++] = receivedBlock.block;
					}
				} //while
			#else
				for (unsigned short i=0; i<numWideCountPerOAStrip; i++)
				{
					t_output_dram_block_tagged receivedBlock = read_channel_intel(channel_output_wide[0]);
					pOutputActivation[iActivationGlobal++] = receivedBlock.block;
				}
			#endif

			#if defined(SPARSE_SYSTEM)
			t_streamblock_address tbBlockCount = clusterCount >> CLUSTER_TO_TRANSFER_BLOCK_SHIFT;
			//Store the cluster count
			cacheOAStreamBlockAddress[iAddressCache] = tbBlockCount;
			#endif

			EMULATOR_PRINT(("[kernelOutputWriter] FINISHED draining the output strip from Col=%d, at iOutputHeightInTile=%d, iOutputWidthInTile=%d, iOutputGroup=%d for tile %d / %d\n\n", 
				iCol, iOutputHeightInTile, iOutputWidthInTile, iOutputGroup, iterHxWTile, numOutputHxWTiles));

			/*
			Parameters update
			*/
			if ((iCol+1)==maxPeCols)
			{
				iCol = 0;
				if ((iOutputWidthInTile+1)==maxTQ_A)
				{
					iOutputWidthInTile = 0;
					if ((iOutputHeightInTile+1)==maxTP)
					{
						iOutputHeightInTile = 0;
						iOutputGroup++;
					}
					else
					{
						iOutputHeightInTile++;
					}
				}
				else
				{
					iOutputWidthInTile++;
				}
			}
			else
			{
				iCol++;
			}
		} //for-loop for draining output

		/*
		Drain the transfer block counts for the tile
		*/
		#if defined(SPARSE_SYSTEM)
			EMULATOR_PRINT(("[kernelOutputWriter] START draining the output count for tile %d / %d\n\n", 
				iterHxWTile, numOutputHxWTiles));
			unsigned char iAddressGroup = 0;
			unsigned char iAddressHeightInTile = 0;
			unsigned char iAddressWidthInTile = 0;

			for (unsigned short iter=0; iter<numGroupNextLayerxTileHeightxTileWidth; iter++)
			{
				unsigned int dramAddress = (iAddressGroup*outputHeight + iterP + iAddressHeightInTile) * outputWidth + iterQ + iAddressWidthInTile;
				unsigned short cacheAddress = (iAddressGroup*maxTP + iAddressHeightInTile) * maxTQ + iAddressWidthInTile;

				pOAStreamBlockAddress[dramAddress] = cacheOAStreamBlockAddress[cacheAddress];
				/*
				Parameter updates
				*/
				if ((iAddressWidthInTile+1)==maxTQ)
				{
					iAddressWidthInTile = 0;
					if ((iAddressHeightInTile+1)==maxTP)
					{
						iAddressHeightInTile = 0;
						iAddressGroup++;
					}
					else
					{
						iAddressHeightInTile++;
					}
				}
				else
				{
					iAddressWidthInTile++;
				}
			}

			EMULATOR_PRINT(("[kernelOutputWriter] FINISHED draining the output count for tile %d / %d\n\n", 
					iterHxWTile, numOutputHxWTiles));
		#endif //SPARSE_SYSTEM
		
		/*
		Parameter updates
		*/
		if ((iterWidthTile+1)==numOutputWidthTile)
		{
			iterWidthTile = 0;
			iterHeightTile++;

			iterQ = 0;
			iterP += maxTP;
		}
		else
		{
			iterWidthTile++;

			iterQ += maxTQ;
		}
		iterHxWTile++;
	} //while loop over iterHxWTile
} //kernelMemoryWriter

__attribute__((max_global_work_dim(0)))
__kernel void kernelOAMover (
		volatile __global t_output_dram_block* restrict pOA0,
		volatile __global t_output_dram_block* restrict pOA1,

		#if defined(SPARSE_SYSTEM)
			volatile __global t_streamblock_address* restrict pTBCount0,
			volatile __global t_streamblock_address* restrict pTBCount1,
		#endif

		volatile __global t_oa_mover_instruction* restrict pInstruction,
		unsigned int numInstruction
	)
{
	for (unsigned int iInst=0; iInst<numInstruction; iInst++)
	{
		/*! Read the instruction and decode the packed field*/
		t_oa_mover_instruction inst = pInstruction[iInst];
		uint1_t outputMemSelect = (inst.memSelectCatSparseFlagCatSyncFlagCatNumActiveCols >> 7) & 0x01;
		uint1_t enableSparsification = (inst.memSelectCatSparseFlagCatSyncFlagCatNumActiveCols >> 6) & 0x01;
		uint1_t enableSendSync = (inst.memSelectCatSparseFlagCatSyncFlagCatNumActiveCols >> 4) & 0x01;
		unsigned char numActivePeCols = inst.memSelectCatSparseFlagCatSyncFlagCatNumActiveCols & 0x0F;
		//Select the memory region
		__global t_output_dram_block* pOA;
		pOA = (outputMemSelect == 0x01) ? pOA1 : pOA0;
		#if defined(SPARSE_SYSTEM)
			__global t_streamblock_address* pTB;
			pTB = (outputMemSelect == 0x01) ? pTBCount1 : pTBCount0;
		#endif

		//Control variables
		unsigned short iOutputGroup = 0;
		unsigned char iOutputHeightInColTile = 0;
		unsigned char iOutputWidthInColTile = 0;
		unsigned char iCol = 0; //iterator for the PE columns

		//Memory pointer contribution
		signed int addrOAPeColContribution = 0;
		signed int addrOAColContribution = 0;
		signed int addrOARowContribution = 0;
		//signed int addrOAGroupContribution = 0;
		#if defined(SPARSE_SYSTEM)
			signed int addrTBPeColContribution = 0;
			signed int addrTBColContribution = 0;
			signed int addrTBRowContribution = 0;
			//signed int addrTBGroupContribution = 0;
		#endif
		for (unsigned short iter=0; iter<inst.numColumnTileWidthxTileHeightxNumActiveCols; iter++)
		{

			//int addrOA = inst.memOAStart + addrOAGroupContribution + addrOARowContribution + addrOAColContribution + addrOAPeColContribution;
			int addrOA = inst.memOAStart + addrOARowContribution + addrOAColContribution + addrOAPeColContribution;
			#if defined(SPARSE_SYSTEM)
				//int addrTB = inst.memTBStart + addrTBGroupContribution + addrTBRowContribution + addrTBColContribution + addrTBPeColContribution;
				int addrTB = inst.memTBStart + addrTBRowContribution + addrTBColContribution + addrTBPeColContribution;
				bool proceed = true;
				unsigned short clusterCount = 0;
				while (proceed)
				{
					t_output_dram_block_tagged receivedBlock = read_channel_intel(channel_output_wide[0]);
					if ((receivedBlock.isLastFlag & 0x1) == TRUE)
					{
						proceed = false;
					}
						clusterCount = outputDramBlock2ClusterCount(receivedBlock.block);
					
					if (((receivedBlock.isLastFlag & 0x1) == FALSE) || (enableSparsification == FALSE))
					{
						//Store the dram block
						pOA[addrOA++] = receivedBlock.block;
					}
				} //while

				t_streamblock_address tbBlockCount = clusterCount >> CLUSTER_TO_TRANSFER_BLOCK_SHIFT;
				
				if (enableSparsification == TRUE)
				{
					//Store the cluster count
					pTB[addrTB++] = tbBlockCount;
				}
			#else
				for (unsigned int i=0; i<inst.numDramBlockPerStrip; i++)
				{
					t_output_dram_block_tagged receivedBlock = read_channel_intel(channel_output_wide[0]);
					pOA[addrOA++] = receivedBlock.block;
				}
			#endif

			#if defined(SPARSE_SYSTEM)
				
			#endif

			/*
			Loop parameters update
			*/
			iCol++;
			addrOAPeColContribution += (signed int) inst.memOAPEColStride;
			#if defined(SPARSE_SYSTEM)
				addrTBPeColContribution += (signed int) inst.memTBPEColStride;
			#endif
			if (iCol==numActivePeCols)
			{
				iCol = 0;
				addrOAPeColContribution = 0;
				#if defined(SPARSE_SYSTEM)
					addrTBPeColContribution = 0;
				#endif

				iOutputWidthInColTile++;
				addrOAColContribution += (signed int) inst.memOAColStride;
				#if defined(SPARSE_SYSTEM)
					addrTBColContribution += (signed int) inst.memTBPEColStride;
				#endif
				if (iOutputWidthInColTile==inst.columnTileWidth)
				{
					iOutputWidthInColTile = 0;
					addrOAColContribution = 0;
					#if defined(SPARSE_SYSTEM)
						addrTBColContribution = 0;
					#endif

					iOutputHeightInColTile++;
					addrOARowContribution += (signed int) inst.memOARowStride;
					#if defined(SPARSE_SYSTEM)
						addrTBRowContribution += (signed int) inst.memTBRowStride;
					#endif
					// if (iOutputHeightInColTile==inst.tileHeight)
					// {
					// 	iOutputHeightInColTile = 0;
					// 	addrOARowContribution = 0;
					// 	#if defined(SPARSE_SYSTEM)
					// 		addrTBRowContribution = 0;
					// 	#endif

					// 	iOutputGroup++;
					// 	addrOAGroupContribution += (signed int) inst.memOAGroupStride;
					// 	#if defined(SPARSE_SYSTEM)
					// 		addrTBGroupContribution += (signed int) inst.memTBGroupStride;
					// 	#endif
					// }
				}
			}
		} //for. strip inside the OA tile

		//Send the activation sync if needed
		if (enableSendSync == TRUE)
		{
			mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
			write_channel_intel(0x0, channel_activation_sync);
		}
	} //for. over instruction
} //kernelOAMover
#endif //MEMORY_WRITER 

#ifdef OA_MEMORY
//TODO: HANDLE MULTI-BYTE BITMASK
#define OA_BUFFER_STATE_DECODE 0x1
#define OA_BUFFER_STATE_NUM_ACCESS 0x2
#define OA_BUFFER_UPDATE_STRIP 0x3
#define OA_BUFFER_STATE_PADD 0x4 //NOP instructions
#define OA_BUFFER_STATE_ACCESS 0x5
#define OA_BUFFER_PAD_COUNT	 2
//#define OA_BUFFER_PAD_COUNT	 20
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelOABuffer ()
{
	typedef uint3_t t_state;

	int colID = get_compute_id(0);
	char cacheOutputActivations[OA_CACHE_SIZE] __attribute__((numbanks(1)));

	/*
	 *Loop carried variables
	*/
	t_state currentState = OA_BUFFER_STATE_DECODE;

	unsigned short startOutputIndex = 0X0;
	unsigned short numOutputsPerStrip = 0X0;
	unsigned short numStripsToAccess = 0x0;
	unsigned short iaStridePerCol = 0x0;
	uint1_t isDrainBuffer = FALSE;

	//Information relevant for loading the cache only
	unsigned char numAccumulatorBitsToRightShift = 0x0;
	uint1_t enableRelu = FALSE;
	uint1_t enableSparsification = FALSE;
	unsigned short numClustersToDrain = 0;
	unsigned short numWindowsToDrain = 0;

	//Loop-carried variables 
	unsigned char countSurvivingClustersInWindow = 0;
	unsigned char iClustersInWindowFetched = 0;
	unsigned short iOutputChannelFetched = 0;
	unsigned short iClustersFetched = 0;
	t_bitmask mask;
	#pragma unroll
	for (int i=0; i<NUM_BITMASK_BYTES; i++)
	{
		mask.bytes[i] = 0x0;
	}

	unsigned short iStrip = 0;
	unsigned short numLoopsPerStip = 0;

	unsigned char delayCount = 0;
	unsigned short iLoopPerStip = 0;
	unsigned short indexOutput = 0;


	//#pragma ivdep
	while (true)
	{
		t_state nextState = currentState;

		if (currentState == OA_BUFFER_STATE_DECODE)
		{
			bool readSuccess = false;

			t_output_tile_buffer_packet controlPacket =
				read_channel_nb_intel(channel_control_to_oa_buffer_local[colID], &readSuccess);

			if (readSuccess)
			{
				startOutputIndex = controlPacket.startOutputIndex;
				numOutputsPerStrip = controlPacket.numOutputsPerStrip;
				numStripsToAccess = controlPacket.numStripsToAccess;
				iaStridePerCol = controlPacket.iaStridePerCol;
                isDrainBuffer = (controlPacket.controlBits >> 7) & 0x1;

				//Information relevant for loading the cache only
                numAccumulatorBitsToRightShift = controlPacket.controlBits & 0xF;
                enableRelu = (controlPacket.controlBits >> 4) & 0x1;
                enableSparsification = ( controlPacket.controlBits >> 5) & 0x1;

                iStrip = 0;
				
				nextState = OA_BUFFER_STATE_NUM_ACCESS;
			}

		}
		else if (currentState == OA_BUFFER_STATE_NUM_ACCESS)
		{
			numClustersToDrain = 1 + ((numOutputsPerStrip - 1) >> VALUE_TO_CLUSTER_SHIFT);
			numWindowsToDrain = 1 + ((numClustersToDrain - 1) >> CLUSTER_TO_WINDOW_SHIFT);

			//Loop-carried variables 
			countSurvivingClustersInWindow = 0;
			iClustersInWindowFetched = 0;
			iOutputChannelFetched = 0;
			iClustersFetched = 0;
			#pragma unroll
			for (int i=0; i<NUM_BITMASK_BYTES; i++)
			{
				mask.bytes[i] = 0x0;
			}

			//Loop control
			#if defined(SPARSE_SYSTEM)
				numLoopsPerStip = (isDrainBuffer == TRUE) ?
					(numClustersToDrain + numWindowsToDrain) 
					: numOutputsPerStrip;
			#else
                numLoopsPerStip = (isDrainBuffer == TRUE) ?
                    (numClustersToDrain)
                    : numOutputsPerStrip;
			#endif

			iLoopPerStip = 0;
			indexOutput = startOutputIndex;

			nextState = OA_BUFFER_STATE_ACCESS;

			EMULATOR_PRINT(("[kernelOABuffer %d] START processing instruction. Type=%d, startOutputIndex=%d, numOutputsPerStrip %d\n\n", 
			colID, (unsigned char) isDrainBuffer, startOutputIndex, numOutputsPerStrip));
		}
		else if (currentState == OA_BUFFER_STATE_PADD)
		{
			delayCount++;
			if (delayCount >= OA_BUFFER_PAD_COUNT)
			{
				nextState = OA_BUFFER_STATE_DECODE;
			}
		}
		else if (currentState == OA_BUFFER_STATE_ACCESS)
		{
			if (isDrainBuffer == FALSE) //Case: draining the array
			{
				bool readSuccess = false;
				t_conv_drain_tagged wideOutputTagged;

				wideOutputTagged = = read_channel_nb_intel(channel_drain_conv[0][colID], &readSuccess);

				t_accumulator wideOutput = wideOutputTagged.value;
				
				if (readSuccess == true) {
					t_operand shortOutput = modifyOutput(wideOutput, numAccumulatorBitsToRightShift, enableRelu);
					cacheOutputActivations[indexOutput] = shortOutput;

					EMULATOR_PRINT(("[kernelOABuffer %d] Read and processed value from PE. Value: %#04x, %d out of %d values read.\n\n", 
					colID, shortOutput, indexOutput, numOutputsPerStrip));
					//Loop variable updates
					indexOutput++;
					iLoopPerStip++;

				}
			} //Case: draining the array

			else //Case: Stream the buffered output to the cache
			{
				#if defined(SPARSE_SYSTEM)
					//If we haven't finished streaming a window or haven't drained the current group
					if ((iClustersInWindowFetched < COMPRESSION_WINDOW_SIZE) && (iClustersFetched < numClustersToDrain))
					{
						bool keep = (enableSparsification == FALSE);
						t_cluster cluster;
						//bool writeSuccess = true;

						#pragma unroll
						for (unsigned char i=0; i<CLUSTER_SIZE; i++)
						{
							unsigned short tempOC = iOutputChannelFetched + i;
							char tempValue = (tempOC >= numOutputsPerStrip) ?
								0x0 : cacheOutputActivations[indexOutput+i];
							cluster.cluster_values[i] = tempValue;
							keep = keep || (tempValue != 0x0);
						}

						if (keep == true)
						{
							write_channel_intel(channel_output_buffer_to_compressor_data[colID], cluster);
							//mask |= ((unsigned char) 1) << iClustersInWindowFetched;
							mask.bytes[iClustersInWindowFetched >> 0x3] |= ((unsigned char) 1) << (iClustersInWindowFetched & 0x07);
							countSurvivingClustersInWindow++;
						}

						iClustersFetched++;
						iClustersInWindowFetched++;

						//Gotcha
						iOutputChannelFetched += CLUSTER_SIZE;
						indexOutput += CLUSTER_SIZE;
						iLoopPerStip++;

					}
					else //Send mask along with other informatin
					{
						//bool writeSuccess = false;
						t_output_cluster_info info;
						#pragma unroll
						for (int i=0; i<NUM_BITMASK_BYTES; i++)
						{
							info.bitmask.bytes[i] = mask.bytes[i];
							mask.bytes[i] = 0x0;
						}
						unsigned char isLastInStrip = (iClustersFetched == numClustersToDrain) ? 0x1 : 0x0;
						info.statusBits = (countSurvivingClustersInWindow & 0x3F)
							| ((isLastInStrip & 0x1) << 0x6)
							| ( (((unsigned char) enableSparsification) & 0x1) << 0x7);
						
						write_channel_intel(channel_output_buffer_to_compressor_info[colID], info);
						countSurvivingClustersInWindow = 0;
						iClustersInWindowFetched = 0;
						iLoopPerStip++;
					}
				#else //SPARSE_SYSTEM
					t_output_cluster_tagged taggedCluster;
					#pragma unroll
					for (unsigned char i=0; i<CLUSTER_SIZE; i++)
					{
						unsigned short tempOC = iOutputChannelFetched + i;
						char tempValue = (tempOC >= numOutputsPerStrip) ?
							0x0 : cacheOutputActivations[indexOutput+i];
						taggedCluster.cluster.cluster_values[i] = tempValue;
					}

					unsigned short tempIClustersFetched = iClustersFetched+1;
					bool isLastInStrip = (tempIClustersFetched == numClustersToDrain) ? true : false;
					taggedCluster.isLastInStrip = isLastInStrip;

					write_channel_intel(channel_oa_buffer_to_oa_tee[colID], taggedCluster);

					iClustersFetched++;

					iOutputChannelFetched += CLUSTER_SIZE;
					indexOutput += CLUSTER_SIZE;
					iLoopPerStip++;

				#endif //SPARSE_SYSTEM
			} //Case: Stream the buffered output to the cache

            if (iLoopPerStip == numLoopsPerStip)
            {
                nextState = OA_BUFFER_UPDATE_STRIP;
            }
		}
		else if (currentState == OA_BUFFER_UPDATE_STRIP)
		{
			nextState = OA_BUFFER_STATE_NUM_ACCESS;
			iStrip++;
			startOutputIndex += iaStridePerCol;
			if (iStrip == numStripsToAccess)
			{
				delayCount = 0;
				nextState = OA_BUFFER_STATE_PADD;
			}
		}

		currentState = nextState;
	} //end while
} //kernelOABuffer

__attribute__((max_global_work_dim(0)))
__kernel void kernelOATileController (
	__global t_oa_tile_controller_instruction* restrict pInst,
	unsigned int numInstructions
	)
{
	for (unsigned int iInst=0; iInst<numInstructions; iInst++)
	{
		/*
		1. Read the instruction and decode it
		*/
		t_oa_tile_controller_instruction inst = pInst[iInst];

		//t_output_tile_controller_packet controlPacket = read_channel_intel(channel_output_writer_to_oa_controller);
		unsigned char numOutputTileHeightxWidth = inst.numLocalTilePerColHxW;
		unsigned char numFoldsInGroupCurrentLayer = inst.numFoldsInGroupCurrentLayer;
	    unsigned char numFullFoldsInGroupCurrentLayer = inst.numFullFoldsInCurrentLayer;
	    unsigned char numActiveRowsInPartialFolds = inst.numActiveElementsInPartialFold;

	    unsigned short numChannelsInGroupCurrentLayer = inst.numLocalChannelsPerCurrentGroup;
	    unsigned short numChannelsInGroupNextLayer = inst.numLocalChannelsPerNextGroup;
	    unsigned char outputModifierBits = inst.flagSparseCatFlagReluCatFlagSourceCatRShift;
	    unsigned char numActivePeCols = inst.numActiveCols;

	    unsigned short numOutputChannels = inst.numLocalChannels;

	    /*
	    2. Send instruction to drain from the PE array
	    */
	    unsigned short iChannelCurrentLayer = 0;
	    unsigned short iChannelInGroup = 0;
	    unsigned short iFoldInGroup = 0;
	    //unsigned short iOutputTileHxWDrain = 0;

	   	EMULATOR_PRINT(("[kernelOATileController] START sending the drain-from-array instruction for tile %d\n\n", 
				iInst));
	   	//TODO: Make sure to chagne how numDrainInstructions is calculated on the host
	    for  (unsigned short i=0; i < inst.numDrainInstructions; i++)
	    {
	    	unsigned char numActivePeRows = (iFoldInGroup < numFullFoldsInGroupCurrentLayer) ?
	    		PE_ROWS : numActiveRowsInPartialFolds;
	    	unsigned short startOutputIndex = iChannelCurrentLayer+iChannelInGroup;

	    	t_output_tile_buffer_packet_tagged bufferPacketTagged;
	    	bufferPacketTagged.bufferPacket.startOutputIndex = startOutputIndex;
	    	bufferPacketTagged.bufferPacket.numOutputsPerStrip = numActivePeRows;
	    	bufferPacketTagged.bufferPacket.numStripsToAccess = numOutputTileHeightxWidth;
	    	bufferPacketTagged.bufferPacket.iaStridePerCol = numOutputChannels;
	    	bufferPacketTagged.bufferPacket.controlBits = outputModifierBits & 0x7F;
	    	bufferPacketTagged.maxColID = (numActivePeCols - 1);

	    	write_channel_intel(channel_oa_noc_control[0], bufferPacketTagged);

	    	/*
	    	Parameter updates
	    	*/
    		iOutputTileHxWDrain = 0;

    		iFoldInGroup++;
    		iChannelInGroup += numActivePeRows;
    		if (iFoldInGroup == numFoldsInGroupCurrentLayer)
    		{
    			iFoldInGroup = 0;
    			iChannelCurrentLayer += numChannelsInGroupCurrentLayer; 
    			iChannelInGroup = 0;
    		}
	    } //while-loop.  Send instruction to drain from the PE array
	    EMULATOR_PRINT(("[kernelOATileController] FINISHED sending the drain-from-array instruction for tile%d\n\n", 
				iInst));

	    /*
	    3. Send instructions to stream cached data
	    */
	   	EMULATOR_PRINT(("[kernelOATileController] START sending the write-to-memory instruction for tile %d\n\n", 
				iInst));
	    unsigned short iChannelNextLayer = 0;
	    //TODO: make sure to change how numMemInstructions are calculated on the host.
	    for  (unsigned short i=0; i<inst.numMemInstructions; i++)
	    {
	    	unsigned short startOutputIndex = iChannelNextLayer;
	    	t_output_tile_buffer_packet_tagged bufferPacketTagged;
	    	bufferPacketTagged.bufferPacket.startOutputIndex = startOutputIndex;
			bufferPacketTagged.bufferPacket.numOutputsPerStrip = numChannelsInGroupNextLayer;
	    	bufferPacketTagged.bufferPacket.numStripsToAccess = numOutputTileHeightxWidth;
	    	bufferPacketTagged.bufferPacket.iaStridePerCol = numOutputChannels;
	    	bufferPacketTagged.bufferPacket.controlBits = (((unsigned char) 0x1) << 0x7 ) | (outputModifierBits & 0x7F);
	    	bufferPacketTagged.maxColID = (numActivePeCols - 1);

	    	write_channel_intel(channel_oa_noc_control[0], bufferPacketTagged);

	    	iChannelNextLayer += numChannelsInGroupNextLayer;
	    } //while-loop. Send instructions to stream cached data
	    EMULATOR_PRINT(("[kernelOATileController] FINISH sending the write-to-memory instruction for tile %d\n\n", 
				iInst));

	} // iterate over tiles
}


#if defined(SPARSE_SYSTEM)
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
//TODO: HANDLE MULTI-BYTE BITMASK
__kernel void kernelCompressorOranizer()
{
	int colID = get_compute_id(0);
	while (true)
	{
		//Read the information first
		t_output_cluster_info info = read_channel_intel(channel_output_buffer_to_compressor_info[colID]);
		t_bitmask bitmask = info.bitmask;
		//TODO: What if the compression window size is greater than 32?
		unsigned char numSurvivingClusters = info.statusBits & 0x3F;
		bool isLastWindowInStrip = (((info.statusBits >> 0x6) & 0x1) == 0x1);
		bool enableSparsification = (((info.statusBits >> 0x7) & 0x1) == 0x1);

		//Depending on whether sparsification is enabled, we may or may not to add an extra transfer block to encode the bitmask/surviving cluster count
		unsigned char numClustersToSend = enableSparsification ?
			 numSurvivingClusters + TRANSFER_SIZE : numSurvivingClusters;

		//For every weindow, we want to sent a number of clusters that is a multiple of transfer size
		//The number of clusters that we actually need to send is the number of surviving clusters, the bitmask/surviving cluster count
		unsigned char numLoops = 
			(1 + ((numClustersToSend - 1) >> CLUSTER_TO_TRANSFER_SIZE_SHIFT)) << CLUSTER_TO_TRANSFER_SIZE_SHIFT; //This controls the amount of padding we have to do. Add extra one to send mask
		unsigned char iClusterSent = 0;
		unsigned char iBitmaskByte = 0;
		for (unsigned char iLoop=0; iLoop<numLoops; iLoop++)
		{
			bool sendCluster = false;
			t_output_cluster_tagged clusterTagged;
			if (iClusterSent < numClustersToSend)
			{
				//TODO: Change this if the number of bitmask grows
				if ((iClusterSent < TRANSFER_SIZE) && enableSparsification)
				{
					#pragma unroll
					for (int i=0; i<CLUSTER_SIZE; i++)
					{
						if ((iBitmaskByte+i) < NUM_BITMASK_BYTES)
						{
							clusterTagged.cluster.cluster_values[i] = bitmask.bytes[iBitmaskByte+i];
						}
						else
						{
							clusterTagged.cluster.cluster_values[i] = 0x0;
						}
					}
					iBitmaskByte += CLUSTER_SIZE;
				}
				else
				{
					clusterTagged.cluster = read_channel_intel(channel_output_buffer_to_compressor_data[colID]);
				}
				iClusterSent++;
			}
			else
			{
				#pragma unroll
				for (unsigned char i=0; i<CLUSTER_SIZE; i++)
				{
					clusterTagged.cluster.cluster_values[i] = 0x0;
				}
			}	

			clusterTagged.isLastInStrip = (isLastWindowInStrip && (iLoop == (numLoops - 1))) ? true : false;

			write_channel_intel(channel_compressor_to_tee[colID], clusterTagged);
		}
	}
}
#endif //SPARSE_SYSTEM

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelOAControlTee ()
{
	int colID = get_compute_id(0);

	while (true)
	{
		t_output_tile_buffer_packet_tagged controlPacketTagged = read_channel_intel(channel_oa_noc_control[colID]);

		unsigned char maxColID = controlPacketTagged.maxColID;

		write_channel_intel(channel_control_to_oa_buffer_local[colID], controlPacketTagged.bufferPacket);

		//Send instruction to the OA tee, if this is a drain command
		t_output_tile_tee_packet teePacket;
		teePacket.numLocalTileHeightxLocalTileWidth = controlPacketTagged.bufferPacket.numStripsToAccess;
		teePacket.flagSourceCatFlagSparseFlagMaxColID = (
				((unsigned char) (maxColID & 0x0F))
				| (controlPacketTagged.bufferPacket.controlBits & 0x30));
		uint1_t drainBuffer = (controlPacketTagged.bufferPacket.controlBits & 0x80) >> 7;
		if (drainBuffer == TRUE)
		{
			write_channel_intel(channel_oa_tee_local[colID], teePacket);
		}

		if (colID < (PE_COLS - 1) )
		{
			if (maxColID > (unsigned char) colID )
			{
				write_channel_intel(channel_oa_noc_control[colID+1], controlPacketTagged);
			}
		}
	}
}


#define OA_TEE_INSTRUCTION_DRAIN_CONV 0X0
#define OA_TEE_INSTRUCTION_DRAIN_PADD 0X1
#define OA_TEE_INSTRUCTION_SEND_SELF 0x2
#define OA_TEE_INSTRUCTION_DRAIN_OTHERS 0X3
#define OA_TEE_INSTRUCTION_DECODE_COMMAND 0X4
#define OA_TEE_INSTRUCTION_SEND_COUNT 0x5
#define OA_TEE_INSTRUCTION_LOOP_UPDATE 0x6
#define OA_TEE_INSTRUCTION_DRAIN_MISC 0x7
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelOATee ()
{
	typedef uint3_t t_instruction;
	int colID = get_compute_id(0);

	//Registers
	//Register of the number of strips that we need to drain from this computing column
	unsigned short regStripsInTile = 0;
	//Iterator that keeps track of the number of strips that we have drained from the computing column
	unsigned short iStripsInTile = 0;
	//Flag that indicates whether this computing column is the right-most in this drain transfer
	uint1_t regIsLastTee = FALSE;
	//State
	t_instruction regInstruction = OA_TEE_INSTRUCTION_DECODE_COMMAND;

	//Iterator that keeps track of the numbers of clusters that we have drained from a strip
	unsigned short iClustersInStrip = 0;
	//Iterator that keeps track of the number of clusters that we have drained from a dram block 
	unsigned char iClusterInDramBlock = 0;

	//Flag that indicates whether the tee should drain from the MISC kernel or from the convolution engine
	uint1_t regFlagDrainMisc = FALSE;
	//Flag that indicates whether the clusters drained from the CONV kernel are sparsified.
	uint1_t regFlagSparse = FALSE;

	//Shift register that keeps track of the convolution clusters within the same dram block
	t_output_dram_block regConvDramBlock;
	//Flag the keeps track of whether the dram block is the last in the convolution strip that is currently being drained.
	uint1_t regFlagLastDramBlockInStrip = FALSE;


	while (true)
	{
		bool sendDramBlockPreviousEnable = false;

		bool shiftInNewCluster = false;

		t_instruction tempInstruction = regInstruction;
		t_output_cluster_tagged tempClusterTagged;
		t_output_tile_tee_packet tempTeeControl;
		t_output_dram_block_tagged tempDramBlockTagged;

		//uint1_t transferIsLast = regDramBlockTagged.isLastFlag & 0x01;

		bool readSuccess = false;

		//Select the output instruction to read
		if (regInstruction == OA_TEE_INSTRUCTION_DECODE_COMMAND)
		{
			tempTeeControl = read_channel_nb_intel(channel_oa_tee_local[colID], &readSuccess);
		}

		//Select the output cluster to read
		if (regInstruction == OA_TEE_INSTRUCTION_DRAIN_CONV)
		{
			#if defined(SPARSE_SYSTEM)
				tempClusterTagged = read_channel_nb_intel(channel_compressor_to_tee[colID], &readSuccess);
			#else
				tempClusterTagged = read_channel_nb_intel(channel_oa_buffer_to_oa_tee[colID], &readSuccess);
			#endif
		}
		else
		{
			tempClusterTagged.isLastInStrip = true;
			#pragma unroll
			for (int i=0; i<CLUSTER_SIZE; i++)
			{
				tempClusterTagged.cluster.cluster_values[i] = 0x0;
			}
		}

		//Select the output dram block to be sent back
		if (regInstruction == OA_TEE_INSTRUCTION_DRAIN_OTHERS)
		{
			if ( (colID < (PE_COLS-1)))
			{
				tempDramBlockTagged = read_channel_nb_intel(channel_output_wide[colID+1], &readSuccess);
				if (readSuccess == true)
				{
					sendDramBlockPreviousEnable = true;
				}
			}
		}
		else if (regInstruction == OA_TEE_INSTRUCTION_SEND_SELF)
		{
			//transferIsLast = (regDramBlockTagged.isLastFlag & 0x1);
            #if defined(SPARSE_SYSTEM)
                tempDramBlockTagged.isLastFlag = (regFlagSparse == TRUE) ?
                	 ( ( (unsigned char) regIsLastTee) << 0x1 ):
                	 ( ( (unsigned char) regIsLastTee) << 0x1 ) | ((unsigned char) regFlagLastDramBlockInStrip);
            #else
				tempDramBlockTagged.isLastFlag = ( ( (unsigned char) regIsLastTee) << 0x1 ) | ((unsigned char) regFlagLastDramBlockInStrip);
            #endif
			tempDramBlockTagged.block = regConvDramBlock;
			sendDramBlockPreviousEnable = true;
		}
		else if (regInstruction == OA_TEE_INSTRUCTION_DRAIN_MISC)
		{
			t_output_dram_block miscOutput = read_chanel_nb_intel(channel_drain_misc[colID], &readSuccess);
			if (readSuccess)
			{
				tempDramBlockTagged.block = miscOutput;
				tempDramBlockTagged.isLastFlag = ((regIsLastTee & 0x1) << 0x1) | 0x1;
				sendDramBlockPreviousEnable = true;
			}
		}
		#if defined(SPARSE_SYSTEM)
			else if (regInstruction == OA_TEE_INSTRUCTION_SEND_COUNT)
			{
				tempDramBlockTagged.isLastFlag = ((regIsLastTee & 0x1) << 0x1) | 0x01;
				t_output_dram_block countDramBlock = clusterCount2OutputDramBlock(iClustersInStrip);
				tempDramBlockTagged.block = countDramBlock;
				sendDramBlockPreviousEnable = true;
            }
		#endif


		//Write channels: output dram block passing
		if ( sendDramBlockPreviousEnable == true)
		{
			write_channel_intel(channel_output_wide[colID], tempDramBlockTagged);
		}

		//State and register update (excluding the shift register)

		switch (regInstruction) {
			case (OA_TEE_INSTRUCTION_DRAIN_CONV) :
			{
				if (readSuccess == true)
				{
					shiftInNewCluster = true;
					iClustersInStrip++;
					iClusterInDramBlock++;
					regFlagLastDramBlockInStrip = tempClusterTagged.isLastInStrip ? TRUE : FALSE;

					if ( iClusterInDramBlock == NUM_CLUSTER_IN_DRAM_SIZE )
					{
						tempInstruction = OA_TEE_INSTRUCTION_SEND_SELF;
					}
	                else if ( tempClusterTagged.isLastInStrip == true )
					{
						tempInstruction = OA_TEE_INSTRUCTION_DRAIN_PADD;
					}
				}
			} //OA_TEE_INSTRUCTION_DRAIN_CONV
			break;
			case (OA_TEE_INSTRUCTION_DRAIN_PADD) :
			{
				shiftInNewCluster = true;	
				iClusterInDramBlock++;

				if ( iClusterInDramBlock == NUM_CLUSTER_IN_DRAM_SIZE )
				{
					tempInstruction = OA_TEE_INSTRUCTION_SEND_SELF;
				}
			} //OA_TEE_INSTRUCTION_DRAIN_PADD
			break;
			case (OA_TEE_INSTRUCTION_SEND_SELF) :
			{
				iClusterInDramBlock = 0;
				if (regFlagLastDramBlockInStrip == TRUE)
				{
					#if defined(SPARSE_SYSTEM)
						if (regFlagSparse == TRUE)
						{
							tempInstruction = OA_TEE_INSTRUCTION_SEND_COUNT;
						}
						else
						{
							tempInstruction = OA_TEE_INSTRUCTION_DRAIN_OTHERS;
						}
					#else
						tempInstruction = OA_TEE_INSTRUCTION_DRAIN_OTHERS;
					#endif
				}
				else 
				{
					tempInstruction = OA_TEE_INSTRUCTION_DRAIN_CONV;
				}
			} //OA_TEE_INSTRUCTION_SEND_SELF
			break;
			case (OA_TEE_INSTRUCTION_DRAIN_OTHERS) :
			{
				if (regIsLastTee == TRUE)
				{
					tempInstruction = OA_TEE_INSTRUCTION_LOOP_UPDATE;
				}
				else if (readSuccess == true)
				{
					if ((tempDramBlockTagged.isLastFlag & 0x3) == 0x3)
					{
						tempInstruction = OA_TEE_INSTRUCTION_LOOP_UPDATE;
					}
				}
			} //OA_TEE_INSTRUCTION_DRAIN_OTHERS
			// break;
			case (OA_TEE_INSTRUCTION_DRAIN_MISC) :
			{
				if (readSuccess == true)
				{
					tempInstruction = OA_TEE_INSTRUCTION_DRAIN_OTHERS;
				}
			} //OA_TEE_INSTRUCTION_DRAIN_MISC
			break;
			case (OA_TEE_INSTRUCTION_DECODE_COMMAND) :
			{
				if (readSuccess == true)
				{
					regIsLastTee = (tempTeeControl.maxColID > colID) ? FALSE: TRUE;
					regStripsInTile = tempTeeControl.numLocalTileHeightxLocalTileWidth;
					iStripsInTile = 0;
					regFlagSparse = tempTeeControl.flagSourceCatFlagSparseFlagMaxColID >> 4;
					uint1_t tempFlagDrainMisc = tempTeeControl.flagSourceCatFlagSparseFlagMaxColID >> 5;

					if (tempFlagDrainMisc == TRUE)
					{
						tempInstruction = OA_TEE_INSTRUCTION_DRAIN_MISC;
					}
					else
					{
						tempInstruction = OA_TEE_INSTRUCTION_DRAIN_CONV;
					}
					regFlagDrainMisc = tempFlagDrainMisc;
				}
			} //OA_TEE_INSTRUCTION_DECODE_COMMAND
			break;
			case (OA_TEE_INSTRUCTION_LOOP_UPDATE) :
			{
				iClustersInStrip = 0;
				//iColDrained = 0;
				iStripsInTile++;
				if (iStripsInTile == regStripsInTile)
				{
					tempInstruction = OA_TEE_INSTRUCTION_DECODE_COMMAND;
				}
				else
				{

					if (regFlagDrainMisc == TRUE)
					{
						tempInstruction = OA_TEE_INSTRUCTION_DRAIN_MISC;
					}
					else
					{
						tempInstruction = OA_TEE_INSTRUCTION_DRAIN_CONV;
					}
				}
				
			} //OA_TEE_INSTRUCTION_LOOP_UPDATE
			break;
			#if defined(SPARSE_SYSTEM)
				case (OA_TEE_INSTRUCTION_SEND_COUNT) :
				{
					if (regIsLastTee == TRUE)
					{
						tempInstruction = OA_TEE_INSTRUCTION_LOOP_UPDATE;
						//iColDrained++;
					}
					else
					{
						if ((tempDramBlockTagged.isLastFlag & 0x3) == 0x3)
						{
							tempInstruction = OA_TEE_INSTRUCTION_LOOP_UPDATE;
						}
					}
				}
				break;
			#endif
			default:
			break;
		}

		regInstruction = tempInstruction;

		#pragma unroll
		for (int i=0; i<NUM_CLUSTER_IN_DRAM_SIZE-1; i++)
		{
			if (shiftInNewCluster == true)
			{
				regDramBlockTagged.block.clusters[i] = regDramBlockTagged.block.clusters[i+1];
			}
		}
		if (shiftInNewCluster == true)
		{
			regDramBlockTagged.block.clusters[NUM_CLUSTER_IN_DRAM_SIZE-1] = tempClusterTagged.cluster;
		}

	}//while
}
#endif  //OA_MEMORY


#ifdef WEIGHT_MEMORY

#define STATE_FILTER_TEE_HEADER 0X0
#define STATE_FILTER_TEE_PAYLOAD 0X1

#define SWITCH_FILTER_TEE_STREAMER 0X0
#define SWITCH_FILTER_TEE_PASS 0x1
/*! kernelFilterTee
	\brief Pass a dram block to the assigned filter streamer or to the next filter tee
*/
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_ROWS)))
__kernel void kernelFilterTee ()
{
	int rowID = get_compute_id(0);
	//uint1_t state = STATE_FILTER_TEE_HEADER;
	//uint1_t direction = SWITCH_FILTER_TEE_PASS;
	//unsigned short transferBlockCount;
	//unsigned short maxTransferBlockSize;
	while (true)
	{
		t_dram_block_w_tagged taggedBlock = read_channel_intel(channel_weight_wide[rowID]);

		int destinationRow = (int) taggedBlock.destinationRow;

		if (destinationRow == rowID)
		{
			write_channel_intel(channel_weight_wide_local[rowID], taggedBlock.dramBlock);
		}
		else
		{
			if (rowID < (PE_ROWS - 1) )
			{
				write_channel_intel(channel_weight_wide[rowID+1], taggedBlock);
			}
		}
	}
}

#define STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL 0X1
#define STATE_FILTER_STREAMER_WRITE_CACHE_WRITE 0X2
#define STATE_FILTER_STREAMER_WRITE_CACHE_WAIT 0X4

#define STATE_FILTER_STREAMER_READ_CACHE_SETUP 0X1
#define STATE_FILTER_STREAMER_READ_CACHE_READ 0X2
#define STATE_FILTER_STREAMER_READ_CACHE_WAIT 0X4

/*! kernelFilterStreamer
	\brief Stream filter values to the PE array
*/
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_ROWS)))
__kernel void kernelFilterBuffer ()
{
	int rowID = get_compute_id(0);

	typedef uint3_t t_state;
	//important to size the bankwidth, otherwise the default 32 bit will be used, resulting in complex store logic
	t_dram_block cacheNzBlocks [2][KERNEL_CACHE_DEPTH] __attribute__((bankwidth(BURST_SIZE_BYTE))); 
	uint1_t regWriteSide = 0x0;
	unsigned short maxOutputCount[2];
	//unsigned char maxOutputHeightTileSize[2]; //maxTP
	//unsigned char maxOutputWidthTileSize[2]; //maxTQ 
	unsigned short maxTransferBlockInFilter[2]; //maxCg
	unsigned char maxPeCols[2];
	t_accumulator cacheBias[2];

	//=================Write into cache variables=================
	t_state stateWriteCache = STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL;
	unsigned short iTransferBlockInFilterWrite; //iCg

	//=================Read from cache variables=================
	t_state stateReadCache = STATE_FILTER_STREAMER_READ_CACHE_WAIT;
	unsigned short iTransferBlockInFilterRead = 0; //iCg
	//unsigned char iWidthInOutputTileRead; //pq*A
	//unsigned char iHeightInOutputTileRead; //p
	unsigned short iOutputRead = 0;


	//TODO: Delete this;
	//unsigned short iterCount= 0;

	//#pragma ivdep array(cacheNzBlocks)
	#pragma ivdep
	while (true)
	{
		//===============Write side====================
		t_state nextStateWriteCache = stateWriteCache;
		{
			bool success = false;
			t_dram_block writeBlock;
			if ( (stateWriteCache == STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL)
				|| (stateWriteCache == STATE_FILTER_STREAMER_WRITE_CACHE_WRITE) )
			{
				writeBlock = read_channel_nb_intel(channel_weight_wide_local[rowID], &success);
			}
			
			if (stateWriteCache == STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL)
			{
				if (success)
				{
					t_filter_streamer_control control = 
						dramBlock2FilterStreamerControl(writeBlock);

					//maxOutputHeightTileSize[regWriteSide] = control.maxOutputHeightTileSize;
					//maxOutputWidthTileSize[regWriteSide] = control.maxOutputWidthTileSize;
					maxOutputCount[regWriteSide] = control.numOutputs;
					maxPeCols[regWriteSide] = control.maxPeCols;
					maxTransferBlockInFilter[regWriteSide] = control.numTransferBlocks;
					cacheBias[regWriteSide] = control.bias;
					
					iTransferBlockInFilterWrite = 0;


					EMULATOR_PRINT(("[kernelFilterBuffer %d] Received setup packet for a new filter. Number of transfer blocks to follow: %d\n\n", rowID, control.numTransferBlocks));

					nextStateWriteCache = STATE_FILTER_STREAMER_WRITE_CACHE_WRITE;
				}
			} // STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL
			else if (stateWriteCache == STATE_FILTER_STREAMER_WRITE_CACHE_WRITE)
			{
				if (success)
				{
					unsigned short dramBlockIndex = (iTransferBlockInFilterWrite >> WIDE_SIZE_OFFSET);
					cacheNzBlocks[regWriteSide][dramBlockIndex] = writeBlock;
					iTransferBlockInFilterWrite += WIDE_SIZE;
					if (iTransferBlockInFilterWrite >= maxTransferBlockInFilter[regWriteSide])
					{
						nextStateWriteCache = STATE_FILTER_STREAMER_WRITE_CACHE_WAIT;
					}
				}
			} // STATE_FILTER_STREAMER_WRITE_CACHE_WRITE
		} // WRITE

		t_state nextStateReadCache = stateReadCache;
		//t_transferblock_tagged weightBlockTagged;
		
		if (stateReadCache == STATE_FILTER_STREAMER_READ_CACHE_SETUP)
		{
			iTransferBlockInFilterRead = 0;
			if (iOutputRead == maxOutputCount[(~regWriteSide) & 0x1])
			{
				nextStateReadCache = STATE_FILTER_STREAMER_READ_CACHE_WAIT;
				iOutputRead = 0;
				EMULATOR_PRINT(("[kernelFilterBuffer %d] FINISHED stream all the weights in the buffer for the tile.\n\n", rowID));
			}
			else
			{
				nextStateReadCache = STATE_FILTER_STREAMER_READ_CACHE_READ;
				//iOutputRead++;
			}
		} // STATE_FILTER_STREAMER_READ_CACHE_SETUP
		// Send bias, then followed by the clusters
		else if ( stateReadCache == STATE_FILTER_STREAMER_READ_CACHE_READ)
		{
			t_transferblock_tagged weightBlockTagged;

			unsigned char tempIsLast = FALSE;

			if (iTransferBlockInFilterRead > 0)
			{
				unsigned short dramIndex = (iTransferBlockInFilterRead - 1) >> WIDE_SIZE_OFFSET;
				unsigned short indexInDramBlock = (iTransferBlockInFilterRead - 1) & WIDE_SIZE_REMAINDER_MASK;
				t_dram_block dramBlock = cacheNzBlocks[(~regWriteSide) & 0x1][dramIndex];
				t_transfer_block tblock = dramBlock.transferBlocks[indexInDramBlock];
				weightBlockTagged.values = tblock;
				tempIsLast = ((iTransferBlockInFilterRead) >= maxTransferBlockInFilter[(~regWriteSide) & 0x1]) ?
					TRUE : FALSE;
			}
			else
			{
				t_accumulator bias = cacheBias[(~regWriteSide) & 0x1];
				t_transfer_block tblock = bias2TransferBlock(bias);
				weightBlockTagged.values = tblock;
				//weightBlockTagged.isLast = false;
			}
			
			//weightBlockTagged.maxTransportID = maxPeCols[(~regWriteSide) & 0x1];

			setIsLast(&weightBlockTagged, tempIsLast);
			setMaxTransferID(&weightBlockTagged, maxPeCols[(~regWriteSide) & 0x1]);
			//weightBlockTagged.isLastConcatMaxTransportID = ((rowID+1) << 0x6) | (iterCount & 0x3F);
			// EMULATOR_PRINT(("[kernelFilterBuffer %d] Attempt to send transfer block %d / %d, in the %d / %d time.\n\n", 
			// 		rowID, iTransferBlockInFilterRead, maxTransferBlockInFilter[(~regWriteSide) & 0x1], iOutputRead, maxOutputCount[(~regWriteSide) & 0x1]));
			bool success = false;
			success = write_channel_nb_intel(channel_weight[rowID][0], weightBlockTagged);
			if (success)
			{
				EMULATOR_PRINT(("[kernelFilterBuffer %d] Sent transfer block %d / %d with tag %d in the %d / %d time.\n\n", 
					rowID, iTransferBlockInFilterRead, maxTransferBlockInFilter[(~regWriteSide) & 0x1], weightBlockTagged.isLastConcatMaxTransportID, iOutputRead, maxOutputCount[(~regWriteSide) & 0x1]));

                EMULATOR_PRINT(("[kernelFilterStreamer %d] Sent tb %d: %#04x %#04x %#04x %#04x\n",
					rowID, 
					iTransferBlockInFilterRead,
                    weightBlockTagged.values.values[0],
                    weightBlockTagged.values.values[1],
                    weightBlockTagged.values.values[2],
                    weightBlockTagged.values.values[3]));

				//Omit plus 1 to send the bias
				if ((iTransferBlockInFilterRead) >= maxTransferBlockInFilter[(~regWriteSide) & 0x1])
				{
					nextStateReadCache = STATE_FILTER_STREAMER_READ_CACHE_SETUP;
					iOutputRead++;
				}
				else
				{
					iTransferBlockInFilterRead++;
				}

				//TODO: Delete this
				//iterCount++;
			}
		} // STATE_FILTER_STREAMER_READ_CACHE_READ

		if ( (stateWriteCache == STATE_FILTER_STREAMER_WRITE_CACHE_WAIT) 
			&& (stateReadCache == STATE_FILTER_STREAMER_READ_CACHE_WAIT) )
		{
			nextStateReadCache = STATE_FILTER_STREAMER_READ_CACHE_READ;
			nextStateWriteCache = STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL;
			regWriteSide = (~regWriteSide) & 0x1; 
			EMULATOR_PRINT(("[kernelFilterBuffer %d] Swap\n\n", rowID));

		}

		stateReadCache = nextStateReadCache;
		stateWriteCache = nextStateWriteCache;

	} // while
}
#endif //WEIGHT_MEMORY

#ifdef PE_SYSTEM


//__attribute__((task))
__attribute__((max_global_work_dim(0)))
#ifdef FULL_SYSTEM
__attribute__((num_compute_units(PE_ROWS, PE_COLS)))
#endif
__attribute__ ((autorun))
__kernel void kernelWeightTransport (
	)
{
	
#ifdef FULL_SYSTEM
	int idx = get_compute_id(1);
	int idy = get_compute_id(0);
#else
	int idx = 0;
	int idy = 0;
#endif

	while (true)
	{

			EMULATOR_PRINT(("[WEIGHT TRANSPORT (%d, %d)] Waiting weight/bias transfer block.\n", idy, idx));

			t_transferblock_tagged block;
			block = read_channel_intel(channel_weight[idy][idx]);

			EMULATOR_PRINT(("[WEIGHT TRANSPORT (%d, %d)] Read weight/bias transfer block. Tag is %#04x\n", idy, idx, block.isLastConcatMaxTransportID));

			unsigned char maxTransportID = getMaxTransferID(block);

			if (idx < (PE_COLS - 1)){
				if ( idx < maxTransportID ) {
					//EMULATOR_PRINT ( ("[kernelWeightTransport]: Waiting to pass a weight block to the output\n") );
					write_channel_intel(channel_weight[idy][idx+1], block);

					EMULATOR_PRINT(("[WEIGHT TRANSPORT (%d, %d)] Passed on weight/bias transfer block.\n", idy, idx));
				}
			}
			write_channel_intel(channel_dpWeightInput[idy][idx], block); 
	}
}

__attribute__((max_global_work_dim(0)))
#ifdef FULL_SYSTEM
__attribute__((num_compute_units(PE_ROWS, PE_COLS)))
#endif
__attribute__ ((autorun))
__kernel void kernelActivationTransport ()
{
	#ifdef FULL_SYSTEM
		int idx = get_compute_id(1);
		int idy = get_compute_id(0);
	#else
		int idx = 0;
		int idy = 0;
	#endif

	while (true)
	{
		t_transferblock_tagged block;

		//Read incoming activaiton transfer blocks
		#ifdef FULL_SYSTEM
			block = read_channel_intel(channel_activation[idy][idx]);
		#else
			block = read_channel_intel(channel_activation[0][0]);
		#endif


		//Determine whether the block should be passed to the next PE on the column
		unsigned char maxTransportID = getMaxTransferID(block);

		if (idy < (PE_ROWS - 1)){
			if ( idy < maxTransportID ) {
				//EMULATOR_PRINT ( ("[kernelWeightTransport]: Waiting to pass an activation block to the output\n") );
				write_channel_intel(channel_activation[idy+1][idx], block);

			}
		}

		write_channel_intel(channel_dpActivationInput[idy][idx], block);


		unsigned char isLastTemp = getIsLast(block);

		if (isLastTemp == TRUE)
		{

			EMULATOR_PRINT(("[ACTIVATION TRANSPORT (%d, %d)] End of activation compression window detected.\n\n", idy, idx));
#if defined(FULL_SYSTEM)
			unsigned char isLastDrain = (maxTransportID == idy) ? TRUE : FALSE;
#else
			unsigned char isLastDrain = TRUE;
#endif

			//write_channel_intel(channel_drain_token[idy][idx], isLastDrain);
		}
	} //while
}

// //MAC Operands
typedef struct __attribute__((packed)) {
	char values [NUM_SIMD_WORDS];
} t_simd_operand;

t_accumulator madd (t_simd_operand activations, t_simd_operand weights) {
	t_accumulator output = 0x0;

	//#ifdef DIRECT_COMPRESSION_SIMD
		#pragma unroll
		for(int i=0; i<TRANSFER_SIZE*CLUSTER_SIZE/4; i++){
			//output += input.data[i]*weights.data[i];
			// use packed DSP blocks to improve efficiency
			#if defined (ARRIA10)
				output += (t_accumulator) a10_mac_8bitx4_input_registered(
					activations.values[i*4],
					weights.values[i*4],
					activations.values[i*4+1],
					weights.values[i*4+1],
					activations.values[i*4+2],
					weights.values[i*4+2],
					activations.values[i*4+3],
					weights.values[i*4+3]
					);
			#elif defined (C5SOC)
				output += (t_accumulator) c5_mac_8bitx4_input_registered(
						activations.values[i*4],
						weights.values[i*4],
						activations.values[i*4+1],
						weights.values[i*4+1],
						activations.values[i*4+2],
						weights.values[i*4+2],
						activations.values[i*4+3],
						weights.values[i*4+3]
						);
			#else
			#error Unsupported FPGA type!
			#endif
		}

	return output;
}

#if defined (SPARSE_SYSTEM)


#define OPERAND_FILTER_READ_BIAS 0x0
#define OPERAND_FILTER_ACCEPT_MASK 0x1
#define OPERAND_FILTER_MASK_SYNC 0x2
#define OPERAND_FILTER_FILTER 0x3
#define OPERAND_FILTER_MAC_SYNC 0x4
#define OPERAND_FILTER_FILTER_SYNC 0x5
#define OPERAND_FILTER_COMMIT 0x6

#define STATE_DRAIN_TRANSPORT_SYNC 0x0
#define STATE_DRAIN_TRANSPORT_DRAIN_SELF 0X1
#define STATE_DRAIN_TRANSPORT_DRAIN_OTHERS 0x2
typedef uint2_t t_drain_instruction;

#ifndef SPARSE_UTILITY
#define SPARSE_UTILITY
	//TODO: Change these if the compression configuration changes
	//Define the instruction type
	typedef uint3_t t_instruction;
	//typedef unsigned char t_bitmask;
	typedef uint6_t t_start;
	typedef uint2_t t_buffer_size;
	typedef int7_t t_num_tb;
	typedef uint1_t t_flag;

	typedef struct {
		unsigned char bytes[NUM_ACCUM_BITMASK_BYTES];
	} t_accum_bitmask;

	typedef struct {
		char values [NUM_SIMD_WORDS];
	} t_pe_buffer;
	

	/**
	 * @brief      Extract a bitmask from an input transfer block, and store the bitmask's bytes into the array that is passed in
	 * 			   TODO: The number of bytes in the bitmask array and the indexing of the tagged block  need to be adjusted if TRANSFER_SIZE or COMPRESSION_WINDOW_SIZE change
	 *
	 * @param      bitmaskBytes  Array for storing the bitmask bytes
	 * @param      taggedBlock   The input transfer block. Type: t_transferblock_tagged
	 */
	void convertTransferBlock2PEBitmask(t_bitmask* bitmaskBytes, t_transferblock_tagged* taggedBlock)
	{
		(*bitmaskBytes).bytes[0] = (*taggedBlock).values.values[0];
		#if (NUM_BITMASK_BYTES > 1)
			(*bitmaskBytes).bytes[1] = (*taggedBlock).values.values[1];
		#endif
		#if (NUM_BITMASK_BYTES > 2)
			(*bitmaskBytes).bytes[2] = (*taggedBlock).values.values[2];
		#endif
		#if (NUM_BITMASK_BYTES > 3)
			(*bitmaskBytes).bytes[3] = (*taggedBlock).values.values[3];
		#endif
		#if (NUM_BITMASK_BYTES > 4)
			(*bitmaskBytes).bytes[4] = (*taggedBlock).values.values[4];
		#endif
		#if (NUM_BITMASK_BYTES > 5)
			(*bitmaskBytes).bytes[5] = (*taggedBlock).values.values[5];
		#endif
		#if (NUM_BITMASK_BYTES > 6)
			(*bitmaskBytes).bytes[6] = (*taggedBlock).values.values[6];
		#endif
		#if (NUM_BITMASK_BYTES > 7)
			(*bitmaskBytes).bytes[7] = (*taggedBlock).values.values[7];
		#endif
	}

	/**
	 * @brief      Wrapper to the call to the bitmask accumulation HDL function smallBufferMaskAccumulator
	 * 			   TODO: The number of bytes in both arrays need to be adjusted if TRANSFER_SIZE or COMPRESSION_WINDOW_SIZE change
	 *
	 * @param      accumulatedBitmaskBytes  Array that stores the the accumulated bitmask bytes
	 * @param      bitmaskBytes             Array that stores the plain bitmask bytes
	 */
	void peAccumulateBitmask(t_accum_bitmask* accumulatedBitmaskBytes, t_bitmask* bitmaskBytes)
	{
		ulong4 accumulatedBitmask = smallBufferMaskAccumulator (
				(*bitmaskBytes).bytes[0],//unsigned char bitmask0,

				#if (NUM_BITMASK_BYTES > 1)
					(*bitmaskBytes).bytes[1],
				#else
					0,
				#endif

				#if (NUM_BITMASK_BYTES > 2)
					(*bitmaskBytes).bytes[2],
				#else
					0,
				#endif

				#if (NUM_BITMASK_BYTES > 3)
					(*bitmaskBytes).bytes[3],
				#else
					0,
				#endif

				#if (NUM_BITMASK_BYTES > 4)
					(*bitmaskBytes).bytes[4],
				#else
					0,
				#endif

				#if (NUM_BITMASK_BYTES > 5)
					(*bitmaskBytes).bytes[5],
				#else
					0,
				#endif

				#if (NUM_BITMASK_BYTES > 6)
					(*bitmaskBytes).bytes[6],
				#else
					0,
				#endif

				#if (NUM_BITMASK_BYTES > 7)
					(*bitmaskBytes).bytes[7]
				#else
					0
				#endif
			);

		{
			(*accumulatedBitmaskBytes).bytes[0] = (unsigned char) (accumulatedBitmask.s0);
			#if (NUM_ACCUM_BITMASK_BYTES > 1)
				(*accumulatedBitmaskBytes).bytes[1] = (unsigned char) (accumulatedBitmask.s0 >> 8);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 2)
				(*accumulatedBitmaskBytes).bytes[2] = (unsigned char) (accumulatedBitmask.s0 >> 16);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 3)
				(*accumulatedBitmaskBytes).bytes[3] = (unsigned char) (accumulatedBitmask.s0 >> 24);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 4)
				(*accumulatedBitmaskBytes).bytes[4] = (unsigned char) (accumulatedBitmask.s0 >> 32);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 5)
				(*accumulatedBitmaskBytes).bytes[5] = (unsigned char) (accumulatedBitmask.s0 >> 40);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 6)
				(*accumulatedBitmaskBytes).bytes[6] = (unsigned char) (accumulatedBitmask.s0 >> 48);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 7)
				(*accumulatedBitmaskBytes).bytes[7] = (unsigned char) (accumulatedBitmask.s0 >> 56);
			#endif

			#if (NUM_ACCUM_BITMASK_BYTES > 8)
				(*accumulatedBitmaskBytes).bytes[8] = (unsigned char) (accumulatedBitmask.s1 >> 0);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 9)
				(*accumulatedBitmaskBytes).bytes[9] = (unsigned char) (accumulatedBitmask.s1 >> 8);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 10)
				(*accumulatedBitmaskBytes).bytes[10] = (unsigned char) (accumulatedBitmask.s1 >> 16);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 11)
				(*accumulatedBitmaskBytes).bytes[11] = (unsigned char) (accumulatedBitmask.s1 >> 24);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 12)
				(*accumulatedBitmaskBytes).bytes[12] = (unsigned char) (accumulatedBitmask.s1 >> 32);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 13)
				(*accumulatedBitmaskBytes).bytes[13] = (unsigned char) (accumulatedBitmask.s1 >> 40);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 14)
				(*accumulatedBitmaskBytes).bytes[14] = (unsigned char) (accumulatedBitmask.s1 >> 48);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 15)
				(*accumulatedBitmaskBytes).bytes[15] = (unsigned char) (accumulatedBitmask.s1 >> 56);
			#endif

			#if (NUM_ACCUM_BITMASK_BYTES > 16)
				(*accumulatedBitmaskBytes).bytes[16] = (unsigned char) (accumulatedBitmask.s2 >> 0);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 17)
				(*accumulatedBitmaskBytes).bytes[17] = (unsigned char) (accumulatedBitmask.s2 >> 8);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 18)
				(*accumulatedBitmaskBytes).bytes[18] = (unsigned char) (accumulatedBitmask.s2 >> 16);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 19)
				(*accumulatedBitmaskBytes).bytes[19] = (unsigned char) (accumulatedBitmask.s2 >> 24);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 20)
				(*accumulatedBitmaskBytes).bytes[20] = (unsigned char) (accumulatedBitmask.s2 >> 32);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 21)
				(*accumulatedBitmaskBytes).bytes[21] = (unsigned char) (accumulatedBitmask.s2 >> 40);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 22)
				(*accumulatedBitmaskBytes).bytes[22] = (unsigned char) (accumulatedBitmask.s2 >> 48);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 23)
				(*accumulatedBitmaskBytes).bytes[23] = (unsigned char) (accumulatedBitmask.s2 >> 56);
			#endif

			#if (NUM_ACCUM_BITMASK_BYTES > 24)
				(*accumulatedBitmaskBytes).bytes[24] = (unsigned char) (accumulatedBitmask.s3 >> 0);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 25)
				(*accumulatedBitmaskBytes).bytes[25] = (unsigned char) (accumulatedBitmask.s3 >> 8);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 26)
				(*accumulatedBitmaskBytes).bytes[26] = (unsigned char) (accumulatedBitmask.s3 >> 16);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 27)
				(*accumulatedBitmaskBytes).bytes[27] = (unsigned char) (accumulatedBitmask.s3 >> 24);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 28)
				(*accumulatedBitmaskBytes).bytes[28] = (unsigned char) (accumulatedBitmask.s3 >> 32);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 29)
				(*accumulatedBitmaskBytes).bytes[29] = (unsigned char) (accumulatedBitmask.s3 >> 40);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 30)
				(*accumulatedBitmaskBytes).bytes[30] = (unsigned char) (accumulatedBitmask.s3 >> 48);
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 31)
				(*accumulatedBitmaskBytes).bytes[31] = (unsigned char) (accumulatedBitmask.s3 >> 56);
			#endif
		}
	}

	/**
	 * @brief      Wrapper to the call to the bitmask 1s counting HDL function smallBufferPopCounter
	 *
	 * @param      bitmaskBytes  Array that stores the bitmask bytes
	 *
	 * @return     Number of ones in the bitmask
	 */
	unsigned char pePopCounter (t_bitmask* bitmaskBytes)
	{
		unsigned char result = 0;

		result = smallBufferPopCounter (
			(*bitmaskBytes).bytes[0],//unsigned char bitmask0,

			#if (NUM_BITMASK_BYTES > 1)
				(*bitmaskBytes).bytes[1],
			#else
				0,
			#endif

			#if (NUM_BITMASK_BYTES > 2)
				(*bitmaskBytes).bytes[2],
			#else
				0,
			#endif

			#if (NUM_BITMASK_BYTES > 3)
				(*bitmaskBytes).bytes[3],
			#else
				0,
			#endif

			#if (NUM_BITMASK_BYTES > 4)
				(*bitmaskBytes).bytes[4],
			#else
				0,
			#endif

			#if (NUM_BITMASK_BYTES > 5)
				(*bitmaskBytes).bytes[5],
			#else
				0,
			#endif

			#if (NUM_BITMASK_BYTES > 6)
				(*bitmaskBytes).bytes[6],
			#else
				0,
			#endif

			#if (NUM_BITMASK_BYTES > 7)
				(*bitmaskBytes).bytes[7]
			#else
				0
			#endif
		);

		return result;
	}
	

	/**
	 * @brief      Helpfer function for update the instruction/state of the operand filter
	 *
	 * @param[in]  currentInstruction  The current instruction
	 * @param[in]  thisTBAvailable     Flag for this filter's new TB availability
	 * @param[in]  otherTBVailable     Flag for the other filter's new TB availability 
	 * @param[in]  thisNumTBLeft       Number of TB left in the compression window for this filter to process
	 * @param[in]  otherWindowDone     Flag that indicates that the other filter has finished processing one compression window
	 * @param[in]  thisLastTB          Flag for indicating whether the filter has encountered the last TB in the kernel
	 *
	 * @return     The t instruction.
	 */

	t_instruction sparseOperandFilterStateUpdate (
			t_instruction currentInstruction,
			t_flag thisTBAvailable,
			t_flag otherTBAvailable,
			t_flag thisMaskAvailable,
			t_flag otherMaskAvailable,
			t_flag thisWindowDone,
			t_flag otherWindowDone,
			t_flag thisLastTB,
			t_flag otherIsLast,
			t_flag thisMacAvailable,
			t_flag otherMacAvailable,
			t_flag swap
		)
	{
		t_instruction nextInstruction = currentInstruction;

		switch (currentInstruction) {
			case (OPERAND_FILTER_READ_BIAS) :{
				if ((thisTBAvailable == TRUE) && (otherTBAvailable == TRUE)) {
					nextInstruction = OPERAND_FILTER_ACCEPT_MASK;
				}
			}
			break; //OPERAND_FILTER_READ_BIAS

			case (OPERAND_FILTER_ACCEPT_MASK) :{

				if (thisMaskAvailable == TRUE) {
					nextInstruction = OPERAND_FILTER_MASK_SYNC;

					if (otherMaskAvailable == TRUE) {
						nextInstruction = OPERAND_FILTER_FILTER;

						if (thisWindowDone == TRUE)
						{
							nextInstruction = OPERAND_FILTER_ACCEPT_MASK;

							if (thisLastTB == TRUE)
							{
								nextInstruction = OPERAND_FILTER_FILTER_SYNC;
							}
						}
					}
				}
			}
			break; //OPERAND_FILTER_ACCEPT_MASK

			case (OPERAND_FILTER_MASK_SYNC) :{

				if (otherMaskAvailable == TRUE)
				{
					nextInstruction = OPERAND_FILTER_FILTER;

					if (thisWindowDone == TRUE)
					{
						nextInstruction = OPERAND_FILTER_ACCEPT_MASK;

						if (thisLastTB == TRUE)
						{
							nextInstruction = OPERAND_FILTER_FILTER_SYNC;
						}
					}
				}
			}
			break; //OPERAND_FILTER_MASK_SYNC

			case (OPERAND_FILTER_FILTER) :{
				if (thisMacAvailable == TRUE && (otherMacAvailable == FALSE))
				{
					nextInstruction = OPERAND_FILTER_MAC_SYNC;
				}
				else
				{
					if (thisTBAvailable == TRUE && (thisWindowDone == TRUE)) {
						nextInstruction = OPERAND_FILTER_ACCEPT_MASK;
						if (thisLastTB == TRUE)
						{
							nextInstruction = OPERAND_FILTER_FILTER_SYNC;
						}
					}
				}

			}
			break; //OPERAND_FILTER_FILTER

			case (OPERAND_FILTER_MAC_SYNC) : {
				if (otherMacAvailable == TRUE)
				{
					nextInstruction = OPERAND_FILTER_FILTER;
					if (thisWindowDone == TRUE)
					{
						nextInstruction = OPERAND_FILTER_ACCEPT_MASK;
						if (thisLastTB == TRUE)
						{
							nextInstruction = OPERAND_FILTER_FILTER_SYNC;
						}
					}
				}
			} 
			break;//OPERAND_FILTER_MAC_SYNC
			case (OPERAND_FILTER_FILTER_SYNC) :{
				if (otherIsLast == TRUE)
				{
					nextInstruction = OPERAND_FILTER_COMMIT;
				}
			}
			break; //OPERAND_FILTER_FILTER_SYNC
			case (OPERAND_FILTER_COMMIT) :{
				if (swap == TRUE)
				{
					nextInstruction = OPERAND_FILTER_READ_BIAS;
				}
			}
			break; //OPERAND_FILTER_COMMIT

			default:
			break;
		} //end of switch. weight FilterInstruction

		return nextInstruction;
	}

	/**
	 * @brief      Helper function for matching sparse oeprands
	 *
	 * @param[in]  accumulatedBitmaskBytes           Bytes of the accumulated sparse bitgmask of this filter.
	 * @param[in]  mutualBitmaskBytes      Bytes of the mutual bitmask
	 * @param[in]  currentStartIndex  The current start index for scanning this filter's bitmask
	 * @param[in]  currentBufferSize  The current buffer size
	 * @param      pCurrentBuffer     Pointer to the current buffer
	 * @param[out]      pNewBlock          Pointer to the content of the new TB block
	 * @param[out]      pNextBuffer        Pointer to the current next buffer
	 * @param[out]     pMacOutput         Pointer to the content of MacOutput block
	 * @param[out]      pMacValid          Pointer to flag that indicates whether the flag is valid
	 * @param[out]      pNextStartIndex    Pointer to the scan start index for the bitmask
	 * @param[out]      pNextBufferSize    Pointer to the new buffer size
	 */
	void filterSparseOperand (
			t_accum_bitmask *accumulatedBitmaskBytes,
			t_bitmask *mutualBitmaskBytes,
			t_start currentStartIndex,
			t_buffer_size currentBufferSize,
			t_pe_buffer* pCurrentBuffer,
			t_transfer_block* pNewBlock,

			t_pe_buffer* pNextBuffer,
			t_simd_operand* pMacOutput,
			t_flag* pMacValid,
			t_start* pNextStartIndex,
			t_buffer_size* pNextBufferSize
		)
	{
		unsigned short maskFilterOutput = smallBufferMaskFilter (
			//Bytes of the mutual mask
			(*mutualBitmaskBytes).bytes[0],
			#if (NUM_BITMASK_BYTES > 1)
				(*mutualBitmaskBytes).bytes[1],
			#else
				0,
			#endif
			#if (NUM_BITMASK_BYTES > 2)
				(*mutualBitmaskBytes).bytes[2],
			#else
				0,
			#endif
			#if (NUM_BITMASK_BYTES > 3)
				(*mutualBitmaskBytes).bytes[3],
			#else
				0,
			#endif
			#if (NUM_BITMASK_BYTES > 4)
				(*mutualBitmaskBytes).bytes[4],
			#else
				0,
			#endif
			#if (NUM_BITMASK_BYTES > 5)
				(*mutualBitmaskBytes).bytes[5],
			#else
				0,
			#endif
			#if (NUM_BITMASK_BYTES > 6)
				(*mutualBitmaskBytes).bytes[6],
			#else
				0,
			#endif
			#if (NUM_BITMASK_BYTES > 7)
				(*mutualBitmaskBytes).bytes[7],
			#else
				0,
			#endif

			//Bytes of the accumulated bitmask
			//Might not need all of them
			(*accumulatedBitmaskBytes).bytes[0],
			#if (NUM_ACCUM_BITMASK_BYTES > 1)
				(*accumulatedBitmaskBytes).bytes[1],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 2)
				(*accumulatedBitmaskBytes).bytes[2],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 3)
				(*accumulatedBitmaskBytes).bytes[3],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 4)
				(*accumulatedBitmaskBytes).bytes[4],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 5)
				(*accumulatedBitmaskBytes).bytes[5],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 6)
				(*accumulatedBitmaskBytes).bytes[6],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 7)
				(*accumulatedBitmaskBytes).bytes[7],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 8)
				(*accumulatedBitmaskBytes).bytes[8],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 9)
				(*accumulatedBitmaskBytes).bytes[9],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 10)
				(*accumulatedBitmaskBytes).bytes[10],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 11)
				(*accumulatedBitmaskBytes).bytes[11],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 12)
				(*accumulatedBitmaskBytes).bytes[12],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 13)
				(*accumulatedBitmaskBytes).bytes[13],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 14)
				(*accumulatedBitmaskBytes).bytes[14],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 15)
				(*accumulatedBitmaskBytes).bytes[15],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 16)
				(*accumulatedBitmaskBytes).bytes[16],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 17)
				(*accumulatedBitmaskBytes).bytes[17],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 18)
				(*accumulatedBitmaskBytes).bytes[18],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 19)
				(*accumulatedBitmaskBytes).bytes[19],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 20)
				(*accumulatedBitmaskBytes).bytes[20],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 21)
				(*accumulatedBitmaskBytes).bytes[21],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 22)
				(*accumulatedBitmaskBytes).bytes[22],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 23)
				(*accumulatedBitmaskBytes).bytes[23],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 24)
				(*accumulatedBitmaskBytes).bytes[24],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 25)
				(*accumulatedBitmaskBytes).bytes[25],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 26)
				(*accumulatedBitmaskBytes).bytes[26],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 27)
				(*accumulatedBitmaskBytes).bytes[27],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 28)
				(*accumulatedBitmaskBytes).bytes[28],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 29)
				(*accumulatedBitmaskBytes).bytes[29],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 30)
				(*accumulatedBitmaskBytes).bytes[30],
			#else
				0,
			#endif
			#if (NUM_ACCUM_BITMASK_BYTES > 31)
				(*accumulatedBitmaskBytes).bytes[31],
			#else
				0,
			#endif

			currentStartIndex//unsigned char startIndex
			);
	
		#if defined(ARRIA10)
			unsigned short tempRegMaskFilterOutput = __fpga_reg(maskFilterOutput);
		#else
			unsigned short tempRegMaskFilterOutput = maskFilterOutput;
		#endif

		//TODO: Change this if the smallBufferMask implementation changes
		unsigned char operandSelectMask = tempRegMaskFilterOutput & 0x0FF;
		*pNextStartIndex = ((tempRegMaskFilterOutput >> 8) & 0x0FF);

		ulong4 bufferUpdateBus = smallBufferMacBufferUpdate(
				operandSelectMask, //inputSelectBitmask

				//Bytes of the input buffer
				//unsigned char inputTransferBlock0,
				(*pNewBlock).values[0],
				//unsigned char inputTransferBlock1,
				#if (NUM_SIMD_WORDS > 1)
					(*pNewBlock).values[1],
				#else
					0,
				#endif
				//unsigned char inputTransferBlock2,
				#if (NUM_SIMD_WORDS > 2)
					(*pNewBlock).values[2],
				#else
					0,
				#endif
				//unsigned char inputTransferBlock3,
				#if (NUM_SIMD_WORDS > 3)
					(*pNewBlock).values[3],
				#else
					0,
				#endif
				//unsigned char inputTransferBlock4,
				#if (NUM_SIMD_WORDS > 4)
					(*pNewBlock).values[4],
				#else
					0,
				#endif
				//unsigned char inputTransferBlock3,
				#if (NUM_SIMD_WORDS > 5)
					(*pNewBlock).values[5],
				#else
					0,
				#endif
				//unsigned char inputTransferBlock3,
				#if (NUM_SIMD_WORDS > 6)
					(*pNewBlock).values[6],
				#else
					0,
				#endif
				//unsigned char inputTransferBlock7,
				#if (NUM_SIMD_WORDS > 7)
					(*pNewBlock).values[7],
				#else
					0,
				#endif

				//Bytes of the buffer
				//unsigned char currentBuffer0,
				(*pCurrentBuffer).values[0],
				//unsigned char currentBuffer1,
				#if (NUM_SIMD_WORDS > 1)
					(*pCurrentBuffer).values[1],
				#else
					0,
				#endif
				//unsigned char currentBuffer2,
				#if (NUM_SIMD_WORDS > 2)
					(*pCurrentBuffer).values[2],
				#else
					0,
				#endif
				//unsigned char currentBuffer3,
				#if (NUM_SIMD_WORDS > 3)
					(*pCurrentBuffer).values[3],
				#else
					0,
				#endif
				//unsigned char currentBuffer4,
				#if (NUM_SIMD_WORDS > 4)
					(*pCurrentBuffer).values[4],
				#else
					0,
				#endif
				//unsigned char currentBuffer5,
				#if (NUM_SIMD_WORDS > 5)
					(*pCurrentBuffer).values[5],
				#else
					0,
				#endif
				//unsigned char currentBuffer6,
				#if (NUM_SIMD_WORDS > 6)
					(*pCurrentBuffer).values[6],
				#else
					0,
				#endif
				//unsigned char currentBuffer7,
				#if (NUM_SIMD_WORDS > 7)
					(*pCurrentBuffer).values[7],
				#else
					0,
				#endif

				(unsigned char) currentBufferSize
			);

		//TODO: Change the loop boundaries below if the TRANSFER_SIZE of CLUSTER_SIZE changes
		#pragma unroll
		for (unsigned char j=0; j<(TRANSFER_SIZE*CLUSTER_SIZE); j++)
		{
			(*pMacOutput).values[j] = (bufferUpdateBus.s0 >> (j*8)) & 0x0FF;
			(*pNextBuffer).values[j] = (bufferUpdateBus.s1 >> (j*8)) & 0x0FF;
		}

		//Define the following as MASkS!!!!
		*pMacValid = ((unsigned char) (bufferUpdateBus.s2 >> 8) ) & 0x01;
		*pNextBufferSize = (unsigned char) bufferUpdateBus.s2;

	}
#endif //SPARSE_UTILITY

__attribute__((task))
__attribute__((max_global_work_dim(0)))
#ifdef FULL_SYSTEM
__attribute__((num_compute_units(PE_ROWS, PE_COLS)))
#endif
__attribute__((autorun))
__kernel void kernelOperandFilter ()
{
	//Obtain kernel location
	#ifdef FULL_SYSTEM
		int idx = get_compute_id(1);
		int idy = get_compute_id(0);
	#else
		int idx = 0;
		int idy = 0;
	#endif	

	//Psum and drain parameters
	t_accumulator pSum[2];
	unsigned char regMaxTransportID[2];
	#pragma unroll
	for (int i=0; i<2; i++)
	{
		pSum[i] = 0;
		regMaxTransportID[i] = 0;
	}
	uint1_t drainSide = 0;
	t_drain_instruction drainInstruction = STATE_DRAIN_TRANSPORT_SYNC;

	//========Weight filter states=========
	t_instruction weightFilterInstruction = OPERAND_FILTER_READ_BIAS;
	//TODO: make the weight buffer size parametrizable
	t_pe_buffer weightBuffer;
	t_simd_operand macWeightBuffer;
	t_accum_bitmask regWeightAccumulatedBitMaskBytes;
	#pragma unroll
	for (int i=0; i<NUM_ACCUM_BITMASK_BYTES; i++)
	{
		regWeightAccumulatedBitMaskBytes.bytes[i] = 0x0;
	}
	t_bitmask regWeightBitmaskBytes;
	#pragma unroll
	for (int i=0; i<NUM_BITMASK_BYTES; i++)
	{
		regWeightBitmaskBytes.bytes[i] = 0x0;
	}
	t_start regWeightWindowStartIndex = 0;
	t_buffer_size regWeightBufferSize = 0;
	t_num_tb regNumWeightClusterLeft = 0;
	t_flag regWeightIsLast = FALSE;

	//=====================================
	

	//=========Activation filter states===
	t_instruction activationFilterInstruction = OPERAND_FILTER_READ_BIAS;
	t_pe_buffer activationBuffer;
	t_simd_operand macActivationBuffer;
	t_accum_bitmask regActivationAccumulatedBitMaskBytes;
	#pragma unroll
	for (int i=0; i<NUM_ACCUM_BITMASK_BYTES; i++)
	{
		regActivationAccumulatedBitMaskBytes.bytes[i] = 0x0;
	}
	t_start regActivationWindowStartIndex = 0;
	t_buffer_size regActivationBufferSize = 0;
	t_num_tb regNumActivationClusterLeft = 0;
	t_flag regActivationIsLast = FALSE;
	//====================================
	
	//Mutual bitmask
	t_bitmask regMutualBitmaskBytes;
	#pragma unroll
	for (int i=0; i<NUM_BITMASK_BYTES; i++)
	{
		regMutualBitmaskBytes.bytes[i] = 0x0;
	}
	t_bitmask regActivationBitmaskBytes;
	#pragma unroll
	for (int i=0; i<NUM_BITMASK_BYTES; i++)
	{
		regActivationBitmaskBytes.bytes[i] = 0x0;
	}

	//State logic
	while (1) {
		//========Signal declaration and state actions=============
		t_instruction nextWeightFilterInstruction = weightFilterInstruction;
		t_transferblock_tagged weightBlock;
		t_flag validWeightMac = FALSE;
		t_flag weightTBAvailable = FALSE;
		t_flag weightWindowDone = FALSE;
		t_flag weightFilterDone = FALSE; 
		t_flag weightMaskNew = FALSE;

		t_pe_buffer nextWeightBuffer;
		t_simd_operand nextMacWeightBuffer;
		//GOTCHAA!!!!!!
		#pragma unroll
		for (unsigned char i=0; i<NUM_SIMD_WORDS; i++)
		{
			nextWeightBuffer.values[i] = weightBuffer.values[i];
			nextMacWeightBuffer.values[i] = macWeightBuffer.values[i];
		}


		t_flag nextWeightIsLast = regWeightIsLast;
		t_num_tb nextNumWeightClusterLeft = regNumWeightClusterLeft;
	    t_bitmask nextWeightBitmaskBytes;
	    #pragma unroll
		for (int i=0; i<NUM_BITMASK_BYTES; i++)
		{
			nextWeightBitmaskBytes.bytes[i] = regWeightBitmaskBytes.bytes[i];
		}
		t_start nextWeightWindowIndex = regWeightWindowStartIndex;
		t_buffer_size nextWeightBufferSize = regWeightBufferSize;

		t_instruction nextActivationFilterInstruction = activationFilterInstruction;
		t_transferblock_tagged activationBlock;
		t_flag validActivationMac = FALSE;
		t_flag activationTBAvailable = FALSE;
		t_flag activationWindowDone = FALSE;
		t_flag activationFilterDone = FALSE;
		t_flag activationMaskNew = FALSE;

		t_pe_buffer nextActivationBuffer;
		t_simd_operand nextMacActivationBuffer;
		//GOTCHAA!!!!!!
		#pragma unroll
		for (unsigned char i=0; i<NUM_SIMD_WORDS; i++)
		{
			nextActivationBuffer.values[i] = activationBuffer.values[i];
			nextMacActivationBuffer.values[i] = macActivationBuffer.values[i];
		}

		t_flag nextActivationIsLast = regActivationIsLast;
		t_num_tb nextNumActivationClusterLeft = regNumActivationClusterLeft;
		t_bitmask nextActivationBitmaskBytes;
		#pragma unroll
		for (int i=0; i<NUM_BITMASK_BYTES; i++)
		{
			nextActivationBitmaskBytes.bytes[i] = regActivationBitmaskBytes.bytes[i];
		}
		t_start nextActivationWindowIndex = regActivationWindowStartIndex;
		t_buffer_size nextActivationBufferSize = regActivationBufferSize;

		t_flag swap = FALSE;
		//GOTTCHA: MUST APPLY THE AND MASK AFTER NEGATION !!!
		unsigned char nextMaxTransportID = regMaxTransportID[(~drainSide) & 0x01];
		t_drain_instruction nextDrainInstruction = drainInstruction;

		//t_bitmask nextMutualBitmask = regMutualBitmask;

		// #ifdef FULL_SYSTEM
		// 	EMULATOR_PRINT(("[Op Filter WEIGHT (%d, %d)] LIVE. Current instruction: %#04x \n", idy, idx, (unsigned char) weightFilterInstruction));
		// #else
		// 	EMULATOR_PRINT(("[Op Filter WEIGHT] LIVE. Current instruction: %#04x \n", (unsigned char) weightFilterInstruction));
		// #endif

		//Weight: Read the input channel
		if ((weightFilterInstruction == OPERAND_FILTER_ACCEPT_MASK) 
			|| (weightFilterInstruction == OPERAND_FILTER_READ_BIAS) 
			|| (weightFilterInstruction == OPERAND_FILTER_FILTER))
		{
			bool readSuccess = false;
			#if defined (FULL_SYSTEM)
				weightBlock = read_channel_nb_intel(
					channel_dpWeightInput[idy][idx],
					&readSuccess);
			#else
				weightBlock = read_channel_nb_intel(
					channel_dpWeightInput[0][0],
					&readSuccess);
			#endif
			weightTBAvailable = (readSuccess == true) ? TRUE : FALSE;
			nextWeightIsLast = FALSE;

			if (readSuccess == true)
			{
				nextWeightIsLast = getIsLast(weightBlock);
				// if (nextWeightIsLast == TRUE)
				// {
				// 	weightFilterDone = TRUE;
				// }

					EMULATOR_PRINT(("[Op Filter WEIGHT (%d, %d)] Read new weight block. IsLast: %#04x. [0-3]: %#04x %#04x %#04x %#04x Current instruction: %#04x \n\n"
						,idy, idx, (unsigned char) nextWeightIsLast, 
						weightBlock.values.values[0],
						weightBlock.values.values[1],
						weightBlock.values.values[2],
						weightBlock.values.values[3],
						(unsigned char) weightFilterInstruction));
			}
		}

		//Activation: read the input channel
		if ((activationFilterInstruction == OPERAND_FILTER_ACCEPT_MASK)  
			|| (activationFilterInstruction == OPERAND_FILTER_FILTER))
		{
			bool readSuccess = false;
			#if defined (FULL_SYSTEM)
				activationBlock = read_channel_nb_intel(
					channel_dpActivationInput[idy][idx],
					&readSuccess);
			#else
				activationBlock = read_channel_nb_intel(
					channel_dpActivationInput[0][0],
					&readSuccess);
			#endif
			activationTBAvailable = (readSuccess == true) ? TRUE : FALSE;
			nextActivationIsLast = FALSE;

			if (readSuccess == true)
			{
				nextActivationIsLast = getIsLast(activationBlock);
				nextMaxTransportID = getMaxTransferID(activationBlock);
				// if (nextActivationIsLast == TRUE)
				// {
				// 	activationFilterDone = TRUE;
				// }

					EMULATOR_PRINT(("[Op Filter ACTIVATION (%d, %d)] Read new activation block. IsLast: %#04x. [0-3]: %#04x %#04x %#04x %#04x Current instruction: %#04x \n\n"
						,idy, idx, (unsigned char) nextActivationIsLast, 
						activationBlock.values.values[0],
						activationBlock.values.values[1],
						activationBlock.values.values[2],
						activationBlock.values.values[3],
						(unsigned char) activationFilterInstruction));
			}

		}

		/*
			Weight: new signal update
		*/
		if (weightFilterInstruction == OPERAND_FILTER_READ_BIAS)
		{
			nextWeightBufferSize = 0x0;
		}
		else if (weightFilterInstruction == OPERAND_FILTER_ACCEPT_MASK)
		{
			convertTransferBlock2PEBitmask(&nextWeightBitmaskBytes, &weightBlock);
			nextNumWeightClusterLeft = pePopCounter(&nextWeightBitmaskBytes);
			peAccumulateBitmask(&regWeightAccumulatedBitMaskBytes, &nextWeightBitmaskBytes);

			nextWeightWindowIndex = 0x0;

			if (weightTBAvailable == TRUE)
			{
				if (nextNumWeightClusterLeft <= 0x0)
				{
					weightWindowDone = TRUE;
				}
				weightMaskNew = TRUE;
			}
		}
		else if (weightFilterInstruction == OPERAND_FILTER_MASK_SYNC)
		{
			weightMaskNew = TRUE;
			weightTBAvailable = TRUE;
			if (nextNumWeightClusterLeft <= 0x0)
			{
				weightWindowDone = TRUE;
			}
		}
		else if (weightFilterInstruction == OPERAND_FILTER_FILTER)
		{
			if (weightTBAvailable == TRUE)
			{
				filterSparseOperand (
						&regWeightAccumulatedBitMaskBytes, //bitmask
						&regMutualBitmaskBytes, //regMutualBitmask
						regWeightWindowStartIndex,
						regWeightBufferSize,
						&weightBuffer,
						&(weightBlock.values),

						&nextWeightBuffer,
						&nextMacWeightBuffer,
						&validWeightMac,
						&nextWeightWindowIndex,
						&nextWeightBufferSize
					);

				nextNumWeightClusterLeft -= TRANSFER_SIZE;

				if (nextNumWeightClusterLeft <= 0x0)
				{
					weightWindowDone = TRUE;
				}

			}
		}
		else if (weightFilterInstruction == OPERAND_FILTER_MAC_SYNC)
		{
			if (nextNumWeightClusterLeft <= 0x0)
			{
				weightWindowDone = TRUE;
			}

			validWeightMac = TRUE;
		}
		else if (weightFilterInstruction == OPERAND_FILTER_FILTER_SYNC)
		{
			weightFilterDone = TRUE;

			validWeightMac = TRUE;
		}

		/*
			Activation: new signal update
		*/
		if (activationFilterInstruction == OPERAND_FILTER_READ_BIAS)
		{
			//hack
			activationTBAvailable = TRUE;
			nextActivationBufferSize = 0x0;
		}
		else if (activationFilterInstruction == OPERAND_FILTER_ACCEPT_MASK)
		{

			convertTransferBlock2PEBitmask(&nextActivationBitmaskBytes, &activationBlock);
			nextNumActivationClusterLeft = pePopCounter(&nextActivationBitmaskBytes);
			peAccumulateBitmask(&regActivationAccumulatedBitMaskBytes, &nextActivationBitmaskBytes);

			nextActivationWindowIndex = 0x0;

			if (activationTBAvailable == TRUE)
			{
				if (nextNumActivationClusterLeft <= 0x0)
				{
					activationWindowDone = TRUE;
				}
				activationMaskNew = TRUE;
			}
		}
		else if (activationFilterInstruction == OPERAND_FILTER_MASK_SYNC)
		{
			activationMaskNew = TRUE;
			activationTBAvailable = TRUE;

			if (nextNumActivationClusterLeft <= 0x0)
			{
				activationWindowDone = TRUE;
			}
		}
		else if (activationFilterInstruction == OPERAND_FILTER_FILTER)
		{
			if (activationTBAvailable == TRUE)
			{
				filterSparseOperand (
						&regActivationAccumulatedBitMaskBytes, //bitmask
						&regMutualBitmaskBytes, //regMutualBitmask
						regActivationWindowStartIndex,
						regActivationBufferSize,
						&activationBuffer,
						&(activationBlock.values),

						&nextActivationBuffer,
						&nextMacActivationBuffer,
						&validActivationMac,
						&nextActivationWindowIndex,
						&nextActivationBufferSize
					);

				nextNumActivationClusterLeft -= TRANSFER_SIZE;

				if (nextNumActivationClusterLeft <= 0x0)
				{
					activationWindowDone = TRUE;
				}
			}
		}
		else if (activationFilterInstruction == OPERAND_FILTER_MAC_SYNC)
		{
			if (nextNumActivationClusterLeft <= 0x0)
			{
				activationWindowDone = TRUE;
			}

			validActivationMac = TRUE;
		}
		else if (activationFilterInstruction == OPERAND_FILTER_FILTER_SYNC)
		{
			activationFilterDone = TRUE;

			validActivationMac = TRUE;
		}

		/*
		 * Mutual bitmask update
		*/
		if ((activationMaskNew == TRUE) && (weightMaskNew == TRUE))
		{
			#pragma unroll
			for (int i=0; i<NUM_BITMASK_BYTES; i++)
			{
				regMutualBitmaskBytes.bytes[i] = nextActivationBitmaskBytes.bytes[i] & nextWeightBitmaskBytes.bytes[i];
				EMULATOR_PRINT(("[Op Filter BITMASK(%d, %d)] Byte %d. Activation bitmask byte: %#04x; Weight Bitmask byte: %#04x; Mutual bitmask byte: %#04x, Current ACTIVAION instruction: %#04x \n"
				,idy, idx, i, (unsigned char) nextActivationBitmaskBytes.bytes[i], (unsigned char) nextWeightBitmaskBytes.bytes[i], (unsigned char) regMutualBitmaskBytes.bytes[i], (unsigned char) activationFilterInstruction));
			}

		}

		/*
		 * Performas MAC
		*/
		if ( (weightFilterInstruction == OPERAND_FILTER_READ_BIAS) && (weightTBAvailable == TRUE) )
		{
			//GOTTCHA: MUST APPLY THE AND MASK AFTER NEGATION !!!
			pSum[(~drainSide) & 0x01] = transferBlock2Bias(weightBlock.values);
		}
		else if ( (validActivationMac == TRUE) && (validWeightMac == TRUE))
		{

			EMULATOR_PRINT(("[Op Filter (%d,%d)] MAC: weight [0-3]: %#04x %#04x %#04x %#04x. act [0-3]: %#04x %#04x %#04x %#04x \n",
					idy, idx,
					nextMacWeightBuffer.values[0] & 0xFF, 
					nextMacWeightBuffer.values[1] & 0xFF,
					nextMacWeightBuffer.values[2] & 0xFF,
					nextMacWeightBuffer.values[3] & 0xFF,
					nextMacActivationBuffer.values[0] & 0xFF, 
					nextMacActivationBuffer.values[1] & 0xFF,
					nextMacActivationBuffer.values[2] & 0xFF,
					nextMacActivationBuffer.values[3] & 0xFF));

			t_accumulator tempPSum = madd(nextMacActivationBuffer, nextMacWeightBuffer);
			//GOTTCHA: MUST APPLY THE AND MASK AFTER NEGATION !!!
			pSum[(~drainSide) & 0x01] += tempPSum;
		}

		//=====================================
		
		/*
			Drain output
		*/
		if ( (drainInstruction == STATE_DRAIN_TRANSPORT_DRAIN_SELF) 
				||
			 (drainInstruction == STATE_DRAIN_TRANSPORT_DRAIN_OTHERS) 
		   )
		{
			#if defined(FULL_SYSTEM)
				//GOTTCHA: MUST APPLY THE AND MASK AFTER NEGATION !!!
				t_flag drainedLast = (regMaxTransportID[drainSide & 0x01] == idy) ? TRUE : FALSE;
			#else
				t_flag drainedLast = TRUE;
			#endif
			t_conv_drain_tagged drainTransportBlock;
			//GOTTCHA: MUST APPLY THE AND MASK AFTER NEGATION !!!
			drainTransportBlock.value = pSum[drainSide & 0x01];
			drainTransportBlock.isLast = (unsigned char) drainedLast;
			bool writePSum = true;
			if (drainInstruction == STATE_DRAIN_TRANSPORT_DRAIN_OTHERS)
			{
				if (idy < (PE_ROWS - 1))
				{
					drainTransportBlock = read_channel_nb_intel(channel_drain_conv[idy+1][idx], &writePSum);
					if (writePSum == true)
					{
						EMULATOR_PRINT(("[Op Filter (%d, %d)] Drained others.\n", idy, idx));
						if (drainTransportBlock.isLast == TRUE)
						{
							nextDrainInstruction = STATE_DRAIN_TRANSPORT_SYNC;
						}
					}
				}
			}
			else
			{
				EMULATOR_PRINT(("[Op Filter (%d, %d)] Commit. pSum value: %#04x \n", idy, idx, pSum));
				if (drainTransportBlock.isLast == TRUE)
				{
					nextDrainInstruction = STATE_DRAIN_TRANSPORT_SYNC;
				}
				else
				{
					nextDrainInstruction = STATE_DRAIN_TRANSPORT_DRAIN_OTHERS;
				}
			}
			
			if (writePSum == true)
			{
				write_channel_intel(channel_drain_conv[idy][idx], drainTransportBlock);
			}
		}

		//GOTTCHA: MUST APPLY THE AND MASK AFTER NEGATION !!!
		regMaxTransportID[(~drainSide) & 0x01] = nextMaxTransportID;

		if ( (weightFilterInstruction == OPERAND_FILTER_COMMIT) && (activationFilterInstruction == OPERAND_FILTER_COMMIT)
				&& (drainInstruction == STATE_DRAIN_TRANSPORT_SYNC) 
			)
		{
			swap = TRUE;
			//GOTTCHA: MUST APPLY THE AND MASK AFTER NEGATION !!!
			drainSide = (~drainSide) & 0x01;	
			nextDrainInstruction = STATE_DRAIN_TRANSPORT_DRAIN_SELF;	
		}

		//=========Next state update==============
		nextWeightFilterInstruction = sparseOperandFilterStateUpdate (
				weightFilterInstruction, //current instruction
				weightTBAvailable, //thisTBAvailable,
				activationTBAvailable, //otherTBAvailable,
				weightMaskNew, //thisMaskAvailable
				activationMaskNew, //otherMaskAvailabe,
				weightWindowDone, //this window done
				activationWindowDone, //other window done
				nextWeightIsLast, //thisLastTB
				activationFilterDone,
				validWeightMac,
				validActivationMac,
				swap
			);

		nextActivationFilterInstruction = sparseOperandFilterStateUpdate (
				activationFilterInstruction, //current instruction
				activationTBAvailable, //thisTBAvailable,
				weightTBAvailable, //otherTBAvailable,
				activationMaskNew, //thisMaskAvaialble
				weightMaskNew, //otherMaskAvailable
				activationWindowDone, //thisWindowDone
				weightWindowDone, //otherWindowDone,
				nextActivationIsLast, //thisLastTB
				weightFilterDone,
				validActivationMac,
				validWeightMac,
				swap
			);
		//========================================
		

		//==============Update the loop dependent variables====
		//Weight filter update
        //EMULATOR_PRINT(("[Op Filter WEIGHT (%d, %d)] Current instruction: %#04x. Next instruction: %#04x\n", idy, idx, (unsigned char) weightFilterInstruction, (unsigned char) nextWeightFilterInstruction));
		//regWeightBitmask = nextWeightBitmask;
		regWeightWindowStartIndex = nextWeightWindowIndex;
		regWeightBufferSize = nextWeightBufferSize;
		regWeightIsLast = nextWeightIsLast;
		regNumWeightClusterLeft = nextNumWeightClusterLeft;

		weightFilterInstruction = nextWeightFilterInstruction;
		#pragma unroll
		for (unsigned char i=0; i<NUM_SIMD_WORDS; i++)
		{
			weightBuffer.values[i] = nextWeightBuffer.values[i];
			macWeightBuffer.values[i] = nextMacWeightBuffer.values[i];
		}

		//Activation filter update
        //EMULATOR_PRINT(("[Op Filter ACTIVATION (%d, %d)] Current instruction: %#04x. Next instruction: %#04x\n", idy, idx, (unsigned char) activationFilterInstruction, (unsigned char) nextActivationFilterInstruction));
		//regActivationBitmask = nextActivationBitmask;
		regActivationWindowStartIndex = nextActivationWindowIndex;
		regActivationBufferSize = nextActivationBufferSize;
		regActivationIsLast = nextActivationIsLast;
		regNumActivationClusterLeft = nextNumActivationClusterLeft;

		activationFilterInstruction = nextActivationFilterInstruction;
		#pragma unroll
		for (unsigned char i=0; i<NUM_SIMD_WORDS; i++)
		{
			activationBuffer.values[i] = nextActivationBuffer.values[i];
			macActivationBuffer.values[i] = nextMacActivationBuffer.values[i];
		}

		#pragma unroll
		for (int i=0; i<NUM_BITMASK_BYTES; i++)
		{
			regWeightBitmaskBytes.bytes[i] = nextWeightBitmaskBytes.bytes[i];
			regActivationBitmaskBytes.bytes[i] = nextActivationBitmaskBytes.bytes[i];
		}

		drainInstruction = nextDrainInstruction;

		//regMutualBitmask = nextMutualBitmask;
		
		//=====================================================
	} //while

}
#else //SPARSE_SYSTEM

#define DENSE_PE_INSTRUCTION_READ_BIAS 0X0
#define DENSE_PE_INSTRUCTION_MAC 0X1
#define DENSE_PE_INSTRUCTION_MAC_SYNC 0X2
#define DENSE_PE_INSTRUCTION_COMMIT 0x3

#define STATE_DRAIN_TRANSPORT_SYNC 0x0
#define STATE_DRAIN_TRANSPORT_DRAIN_SELF 0X1
#define STATE_DRAIN_TRANSPORT_DRAIN_OTHERS 0x2

typedef uint2_t t_drain_instruction;
typedef uint2_t t_dense_pe_instruction;
typedef uint1_t t_dense_pe_flag;

t_dense_pe_instruction densePEInstructionUpdate (
		t_dense_pe_instruction currentInstruction,
		t_dense_pe_flag thisTBAvailable,
		t_dense_pe_flag otherTBAvailable,
		t_dense_pe_flag thisIsLast,
		t_dense_pe_flag otherIsLast,
		t_dense_pe_flag swap
	)
{
	t_dense_pe_instruction nextInstruction = currentInstruction;

	switch (currentInstruction) {
		case (DENSE_PE_INSTRUCTION_READ_BIAS): {
			if ((thisTBAvailable == TRUE) && (otherTBAvailable == TRUE)) {
					nextInstruction = DENSE_PE_INSTRUCTION_MAC;
			}
		}
		break;

		case (DENSE_PE_INSTRUCTION_MAC): {
			if (thisTBAvailable == TRUE && (otherTBAvailable == FALSE))
			{
				nextInstruction = DENSE_PE_INSTRUCTION_MAC_SYNC;
			}
			else
			{
				if (thisTBAvailable == TRUE && (thisIsLast == TRUE)) {
					nextInstruction = DENSE_PE_INSTRUCTION_COMMIT;
				}
			}
		}
		break;

		case (DENSE_PE_INSTRUCTION_MAC_SYNC): {
			if (otherTBAvailable == TRUE)
			{
				nextInstruction = DENSE_PE_INSTRUCTION_MAC;
				if (thisIsLast == TRUE)
				{
					nextInstruction = DENSE_PE_INSTRUCTION_COMMIT;
				}
			}
		}
		break;

		case (DENSE_PE_INSTRUCTION_COMMIT): {
			if (swap == TRUE)
			{
				nextInstruction = DENSE_PE_INSTRUCTION_READ_BIAS;
			}
		}
		break;

		default:
		break;
	}
	return nextInstruction;
}


__attribute__((task))
__attribute__((max_global_work_dim(0)))
#ifdef FULL_SYSTEM
__attribute__((num_compute_units(PE_ROWS, PE_COLS)))
#endif
__attribute__((autorun))
__kernel void kernelDensePE ()
{
	
#ifdef FULL_SYSTEM
	int idx = get_compute_id(1);
	int idy = get_compute_id(0);
#else
	int idx = 0;
	int idy = 0;
#endif
	//====================registers===============
	//Psum and drain parameters
	t_accumulator pSum[2];
	unsigned char regMaxTransportID[2];
	#pragma unroll
	for (int i=0; i<2; i++)
	{
		pSum[i] = 0;
		regMaxTransportID[i] = 0;
	}
	uint1_t drainSide = 0;
	t_drain_instruction drainInstruction = STATE_DRAIN_TRANSPORT_SYNC;

	//=============Weight side instruction============
	t_dense_pe_instruction regWeightInstruction= DENSE_PE_INSTRUCTION_READ_BIAS;
	t_transfer_block regWeightTB;
	t_dense_pe_flag regWeightIsLast = FALSE;

	//============Activation side instruction================
	t_dense_pe_instruction regActivationInstruction = DENSE_PE_INSTRUCTION_READ_BIAS;
	t_transfer_block regActivationTB;
	t_dense_pe_flag regActivationIsLast = FALSE;

	while (1) {
		//Declare temp variables
		t_dense_pe_instruction nextWeightInstruction = regWeightInstruction;
		t_transfer_block nextWeightTB = regWeightTB;
		t_dense_pe_flag weightTBAvailable = FALSE;
		t_dense_pe_flag nextWeightIsLast = regWeightIsLast;
		t_dense_pe_flag validWeightMac = FALSE;

		t_dense_pe_instruction nextActivationInstruction = regActivationInstruction;
		t_transfer_block nextActivationTB = regActivationTB;
		t_dense_pe_flag activationTBAvailable = FALSE;
		t_dense_pe_flag nextActivationIsLast = regActivationIsLast;
		t_dense_pe_flag validActivationMac = FALSE;

		t_dense_pe_flag swap = FALSE;
		unsigned char nextMaxTransportID = regMaxTransportID[(~drainSide) & 0x01];
		t_drain_instruction nextDrainInstruction = drainInstruction;

		//Handling reading from the W channel
		if ( (regWeightInstruction == DENSE_PE_INSTRUCTION_READ_BIAS)
			|| (regWeightInstruction == DENSE_PE_INSTRUCTION_MAC) )
		{
			bool readSuccess = false;
			t_transferblock_tagged taggedBlock;
			#if defined (FULL_SYSTEM)
				taggedBlock = read_channel_nb_intel(
					channel_dpWeightInput[idy][idx],
					&readSuccess);
			#else
				taggedBlock = read_channel_nb_intel(
					channel_dpWeightInput[0][0],
					&readSuccess);
			#endif
            
            weightTBAvailable = (readSuccess == true) ? TRUE : FALSE;
            nextWeightIsLast = FALSE;

            if (readSuccess == true)
            {
            	nextWeightIsLast = getIsLast(taggedBlock);

            	nextWeightTB = taggedBlock.values;

            	EMULATOR_PRINT(("[DENSE PE WEIGHT (%d, %d)] Read new weight block. IsLast: %#04x. [0-3]: %#04x %#04x %#04x %#04x Current instruction: %#04x \n\n"
						,idy, idx, (unsigned char) nextWeightIsLast, 
						taggedBlock.values.values[0],
						taggedBlock.values.values[1],
						taggedBlock.values.values[2],
						taggedBlock.values.values[3],
						(unsigned char) regWeightInstruction));

            }
		}

		//Handling reading from the A channel
		if (regActivationInstruction == DENSE_PE_INSTRUCTION_MAC)
		{
			bool readSuccess = false;
			t_transferblock_tagged taggedBlock;
			#if defined (FULL_SYSTEM)
				taggedBlock = read_channel_nb_intel(
					channel_dpActivationInput[idy][idx],
					&readSuccess);
			#else
				taggedBlock = read_channel_nb_intel(
					channel_dpActivationInput[0][0],
					&readSuccess);
			#endif
            
            activationTBAvailable = (readSuccess == true) ? TRUE : FALSE;
            nextActivationIsLast = FALSE;

            if (readSuccess == true)
            {
            	nextActivationIsLast = getIsLast(taggedBlock);
            	nextMaxTransportID = getMaxTransferID(taggedBlock);

            	nextActivationTB = taggedBlock.values;

            	EMULATOR_PRINT(("[DENSE PE ACTIVATION (%d, %d)] Read new activation block. IsLast: %#04x. [0-3]: %#04x %#04x %#04x %#04x Current instruction: %#04x \n\n"
						,idy, idx, (unsigned char) nextActivationIsLast, 
						taggedBlock.values.values[0],
						taggedBlock.values.values[1],
						taggedBlock.values.values[2],
						taggedBlock.values.values[3],
						(unsigned char) regActivationInstruction));

            }
		}

		if (regWeightInstruction == DENSE_PE_INSTRUCTION_MAC)
		{
			if (weightTBAvailable == TRUE)
			{
				validWeightMac = TRUE;
			}
		}
		else if (regWeightInstruction == DENSE_PE_INSTRUCTION_MAC_SYNC)
		{
			weightTBAvailable = TRUE;
			validWeightMac = TRUE;
		}


		if (regActivationInstruction == DENSE_PE_INSTRUCTION_READ_BIAS)
		{
			activationTBAvailable = TRUE;
		}
		else if (regActivationInstruction == DENSE_PE_INSTRUCTION_MAC)
		{
			if (activationTBAvailable == TRUE)
			{
				validActivationMac = TRUE;
			}
		}
		else if (regActivationInstruction == DENSE_PE_INSTRUCTION_MAC_SYNC)
		{
			activationTBAvailable = TRUE;
			validActivationMac = TRUE;
		}

		/**
		 * PSum update
		 */
		if ( (regWeightInstruction == DENSE_PE_INSTRUCTION_READ_BIAS) 
			 && (weightTBAvailable == TRUE))
		{
			//GOTTCHA: MUST APPLY THE AND MASK AFTER NEGATION !!!
			pSum[(~drainSide) & 0x01] = transferBlock2Bias(nextWeightTB);
		}
		else if ( (validActivationMac == TRUE) && (validWeightMac == TRUE))
		{

			EMULATOR_PRINT(("[DENSE PE (%d,%d)] MAC: weight [0-3]: %#04x %#04x %#04x %#04x. act [0-3]: %#04x %#04x %#04x %#04x \n",
					idy, idx,
					nextWeightTB.values[0] & 0xFF, 
					nextWeightTB.values[1] & 0xFF,
					nextWeightTB.values[2] & 0xFF,
					nextWeightTB.values[3] & 0xFF,
					nextActivationTB.values[0] & 0xFF, 
					nextActivationTB.values[1] & 0xFF,
					nextActivationTB.values[2] & 0xFF,
					nextActivationTB.values[3] & 0xFF));

			t_simd_operand simdActivation, simdWeight;

			#pragma unroll
			for (unsigned int i=0; i<NUM_SIMD_WORDS; i++)
			{
				simdActivation.values[i] = nextActivationTB.values[i];
				simdWeight.values[i] = nextWeightTB.values[i];
			}

			t_accumulator tempPSum = madd(simdActivation, simdWeight);
			//GOTTCHA: MUST APPLY THE AND MASK AFTER NEGATION !!!
			pSum[(~drainSide) & 0x01] += tempPSum;
		}

		/**
		 * Drain output
		 */
		if ( (drainInstruction == STATE_DRAIN_TRANSPORT_DRAIN_SELF) 
				||
		 		(drainInstruction == STATE_DRAIN_TRANSPORT_DRAIN_OTHERS) 
		   )
		{
			#if defined(FULL_SYSTEM)
				//GOTTCHA: MUST APPLY THE AND MASK AFTER NEGATION !!!
				t_dense_pe_flag drainedLast = (regMaxTransportID[drainSide & 0x01] == idy) ? TRUE : FALSE;
			#else
				t_dense_pe_flag drainedLast = TRUE;
			#endif
			t_conv_drain_tagged drainTransportBlock;
			//GOTTCHA: MUST APPLY THE AND MASK AFTER NEGATION !!!
			drainTransportBlock.value = pSum[drainSide & 0x01];
			drainTransportBlock.isLast = (unsigned char) drainedLast;
			bool writePSum = true;
			if (drainInstruction == STATE_DRAIN_TRANSPORT_DRAIN_OTHERS)
			{
				if (idy < (PE_ROWS - 1))
				{
					drainTransportBlock = read_channel_nb_intel(channel_drain_conv[idy+1][idx], &writePSum);
					if (writePSum == true)
					{
						EMULATOR_PRINT(("[DENSE PE(%d, %d)] Drained others.\n", idy, idx));
						if (drainTransportBlock.isLast == TRUE)
						{
							nextDrainInstruction = STATE_DRAIN_TRANSPORT_SYNC;
						}
					}
				}
			}
			else
			{
				EMULATOR_PRINT(("[DENSE PE (%d, %d)] Commit. pSum value: %#04x \n", idy, idx, pSum));
				if (drainTransportBlock.isLast == TRUE)
				{
					nextDrainInstruction = STATE_DRAIN_TRANSPORT_SYNC;
				}
				else
				{
					nextDrainInstruction = STATE_DRAIN_TRANSPORT_DRAIN_OTHERS;
				}
			}
			
			if (writePSum == true)
			{
				write_channel_intel(channel_drain_conv[idy][idx], drainTransportBlock);
			}
		}

		//Update registers
		regMaxTransportID[(~drainSide) & 0x01] = nextMaxTransportID;

		if ( (regWeightInstruction == DENSE_PE_INSTRUCTION_COMMIT) && (regActivationInstruction == DENSE_PE_INSTRUCTION_COMMIT)
				&& (drainInstruction == STATE_DRAIN_TRANSPORT_SYNC) 
			)
		{
			swap = TRUE;
			//GOTTCHA: MUST APPLY THE AND MASK AFTER NEGATION !!!
			drainSide = (~drainSide) & 0x01;	
			nextDrainInstruction = STATE_DRAIN_TRANSPORT_DRAIN_SELF;	
		}

		nextWeightInstruction = densePEInstructionUpdate (
				regWeightInstruction, //current Instruction
				weightTBAvailable, //thisTBAvailable,
				activationTBAvailable, //otherTBAvailable,
				nextWeightIsLast, //thisIsLast
				nextActivationIsLast, //otherIsLast
				swap
			);

		nextActivationInstruction = densePEInstructionUpdate (
				regActivationInstruction, //current Instruction
				activationTBAvailable, //thisTBAvailable,
				weightTBAvailable, //otherTBAvailable,
				nextActivationIsLast, //thisIsLast
				nextWeightIsLast, //otherIsLast
				swap
			);

		regWeightInstruction = nextWeightInstruction;
		regWeightIsLast = nextWeightIsLast;
		regWeightTB = nextWeightTB;

		regActivationInstruction = nextActivationInstruction;
		regActivationIsLast = nextActivationIsLast;
		regActivationTB = nextActivationTB;

		drainInstruction = nextDrainInstruction;		
	} // while-loop

}
#endif //SPARSE SYSTEM

#endif //PE_SYSTEM

