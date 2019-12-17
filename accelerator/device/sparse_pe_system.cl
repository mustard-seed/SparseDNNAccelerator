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
	unsigned short kernelSizexkernelSizexNumFilterFoldsInGroup
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
			tileControllerPacket.numOutputPerCol = kernelSizexkernelSizexNumFilterFoldsInGroup * sizeOutputHeightTileLocal * sizeOutputWidthTilePerColLocal;
			tileControllerPacket.numActivePeColsConcatNumOutputChannelsInGroup = (((unsigned short) numActivePeCols) << 12) | (numFiltersInGroup & 0xFFF);
			//tileControllerPacket.numOutputChannelsInGroup = numFiltersInGroup;
			tileControllerPacket.strideStripIACache = strideStripIACache;
			tileControllerPacket.numTBCountPerStrip = numTBCountPerIAStrip;

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
										for (unsigned char j=0; j<TRANSFER_SIZE; j++)
										{
											#pragma unroll
											for (unsigned char k=0; k<CLUSTER_SIZE; k++)
											{
												dramBlock.transferBlocks[i].values[j].cluster_values[k]=0;
											}
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
									for (unsigned char j=0; j<TRANSFER_SIZE; j++)
									{
										#pragma unroll
										for (unsigned char k=0; k<CLUSTER_SIZE; k++)
										{
											dramBlock.transferBlocks[i].values[j].cluster_values[k]=0;
										}
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
					#endif
					
					t_accumulator bias = cacheBias[iFilterGlobal];

					unsigned short maxDramBlockInFilter = ((maxTransferBlockInFilter-1) >> WIDE_SIZE_OFFSET) + 1;
					unsigned short maxTransmitCount = maxDramBlockInFilter+1; //one extra for filter stream control;
					
					t_filter_streamer_control control;
					control.numOutputs = (unsigned short) sizeOutputHeightTileLocal * (unsigned short) sizeOutputWidthTilePerColLocal;
					control.bias = bias;
					control.numTransferBlocks = maxTransferBlockInFilter;
					control.maxPeCols = (numActivePeCols - 1);

					t_dram_block dramControl = filterStreamerControl2dramBlock(control);

					unsigned int iTransferBlockDDR = iTransferBlockFilterBaseDDR;

					for (unsigned short iTransmitCount=0; iTransmitCount<maxTransmitCount; iTransmitCount++)
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


#endif //MEMORY_READER

#ifdef IA_MEMORY
#define IA_BUFFER_STATE_DECODE 0x0
#define IA_BUFFER_STATE_COMPUTE_NUM_ACCESS 0x1
#define IA_BUFFER_STATE_ACCESS 0x2
#define IA_BUFFER_STATE_PADD 0x3 //NOP operation
#define IA_BUFFER_PADD_COUNT 2 
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelIABuffer ()
{
	typedef uint2_t t_state;
	int colID = get_compute_id(0);

	t_dram_block cacheIABlocks [IA_CACHE_DEPTH] __attribute__((bankwidth(BURST_SIZE_BYTE)));

	#if defined(SPARSE_SYSTEM)
		t_streamblock_address cacheIAStreamBlockAddress [256] __attribute__((numbanks(1)));
	#endif

	/*
		Loop carried-variables
	*/

	bool isLoad = false;
	bool isLast = false;
	unsigned short iActivationDramBlockAddressBase = 0;
	unsigned char iAddressCache = 0;
	unsigned char maxPeRowID = 0;
	unsigned short numIAAccess = 0;
	unsigned short iterAccess = 0;

	//Used as a delay counter, after the access run
	unsigned char paddCount = 0;

	t_state currentState = IA_BUFFER_STATE_DECODE;

//	#pragma ivdep array(cacheIABlocks)
//	#pragma ivdep array(cacheIAStreamBlockAddress)
//	#pragma ivdep safelen(5)
	#pragma ivdep
	while (true)
	{
		t_state nextState = currentState;
		t_dram_block dramBlock;
		bool dataReadSuccess = false;

		//Handle channel read separately
        if (((currentState == IA_BUFFER_STATE_COMPUTE_NUM_ACCESS) && (isLoad == true))
                || ((currentState == IA_BUFFER_STATE_ACCESS) && (isLoad == true)))
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
				isLoad = ((controlPacketReceived.controlBits >> 0x1) & 0x1) == 0x0;
				isLast = (controlPacketReceived.controlBits & 0x1) == 0x1;
				iActivationDramBlockAddressBase = controlPacketReceived.iActivationDramBlockAddressBase;
				iAddressCache = controlPacketReceived.iAddressCache;
				maxPeRowID = controlPacketReceived.maxPeRowID;

				#if defined(SPARSE_SYSTEM)
					nextState = IA_BUFFER_STATE_COMPUTE_NUM_ACCESS;
				#else
					nextState = IA_BUFFER_STATE_PADD;
					numIAAccess = controlPacketReceived.numTBCountPerStrip;
					iterAccess = 0;
					paddCount = 0;
				#endif

				EMULATOR_PRINT(("[kernelIABuffer %d] START processing instruction. isLoad=%d, isLast=%d, iAddressCache=%d, iActivationDramBlockAddressBase=%d, maxPeRowID=%d\n\n",
				colID, isLoad, isLast, iAddressCache, iActivationDramBlockAddressBase, maxPeRowID));

			}
		} //IA_BUFFER_STATE_DECODE
		#if defined(SPARSE_SYSTEM)
			else if (currentState == IA_BUFFER_STATE_COMPUTE_NUM_ACCESS)
			{
				if (isLoad)
				{
					if (dataReadSuccess == true)
					{
						t_streamblock_address numIATransferBlocks = dramBlock2TransferBlockCount(dramBlock);

						cacheIAStreamBlockAddress[iAddressCache] = numIATransferBlocks;
						numIAAccess = ((numIATransferBlocks & WIDE_SIZE_REMAINDER_MASK) != 0X0)
							? (numIATransferBlocks >> WIDE_SIZE_OFFSET) + 1 : (numIATransferBlocks >> WIDE_SIZE_OFFSET);

						nextState = IA_BUFFER_STATE_ACCESS;
					}
				}
				else
				{
					numIAAccess = cacheIAStreamBlockAddress[iAddressCache];
					nextState = IA_BUFFER_STATE_ACCESS;
				}

				iterAccess = 0;
				paddCount = 0;
			} //IA_BUFFER_STATE_COMPUTE_NUM_ACCESS
		#endif
		else if (currentState == IA_BUFFER_STATE_PADD)
		{
			paddCount++;
			if (paddCount >= IA_BUFFER_PADD_COUNT)
			{
				nextState = IA_BUFFER_STATE_ACCESS;
			}
		} //IA_BUFFER_STATE_PADD
		else if (currentState == IA_BUFFER_STATE_ACCESS)
		{
			if (isLoad)
			{
				if (dataReadSuccess)
				{
					cacheIABlocks[iterAccess + iActivationDramBlockAddressBase] = dramBlock;
					iterAccess++;
				}
			}
			else
			{
				t_dram_block dramBlock = cacheIABlocks[iActivationDramBlockAddressBase+ ((unsigned short)(iterAccess >> WIDE_SIZE_OFFSET))];	

				t_transferblock_tagged taggedBlock;
				taggedBlock.values = dramBlock.transferBlocks[iterAccess & WIDE_SIZE_REMAINDER_MASK];
				taggedBlock.maxTransportID = maxPeRowID;
				taggedBlock.isLast = isLast && ((iterAccess + 1) == numIAAccess);
				bool success = write_channel_nb_intel(channel_activation[0][colID], taggedBlock);
				if (success)
				{
					iterAccess++;
				}
			}

			if (iterAccess == numIAAccess)
			{
				EMULATOR_PRINT(("[kernelIABuffer %d] FINISHED processing instruction. isLoad=%d, isLast=%d, iAddressCache=%d, iActivationDramBlockAddressBase=%d, maxPeRowID=%d\n\n",
					colID, isLoad, isLast, iAddressCache, iActivationDramBlockAddressBase, maxPeRowID));
				nextState = IA_BUFFER_STATE_DECODE;
			}
		} //IA_BUFFER_STATE_ACCESS

		currentState = nextState;
	}
}

//Can't use autorun, as the compiler withh crash crash
__attribute__((max_global_work_dim(0)))
//__attribute__((autorun))
__kernel void kernelIATileController (unsigned short numGroupxTiles)
{
	for (unsigned short iTile=0; iTile < numGroupxTiles; iTile++)
	{
		/*
		1. Read the instruction of the tile from the memory reader
		*/
		t_input_buffer_tile_controller_packet tileControlPacketReceived
			= read_channel_intel(channel_to_ia_tile_controller);

		unsigned char inputTileWidth = tileControlPacketReceived.inputTileWidth;
	    unsigned char inputTileHeight = tileControlPacketReceived.inputTileHeight;
	    //unsigned char stride = tileControlPacketReceived.stride;
	    //unsigned char kernelSize = tileControlPacketReceived.kernelSize;
	    unsigned char stride = (tileControlPacketReceived.strideConcatKernelSize >> 0x4) & 0xF;
	    unsigned char kernelSize = tileControlPacketReceived.strideConcatKernelSize & 0xF;
        unsigned int numOutputPerCol = tileControlPacketReceived.numOutputPerCol;
	    unsigned char numActivePeCols = (tileControlPacketReceived.numActivePeColsConcatNumOutputChannelsInGroup >> 12) & 0xF;
	    unsigned short numOutputChannelsInGroup = (tileControlPacketReceived.numActivePeColsConcatNumOutputChannelsInGroup) & 0xFFF;
	    unsigned short strideStripIACache = tileControlPacketReceived.strideStripIACache; //S

	    #if !defined(SPARSE_SYSTEM)
	    unsigned short numTBCountPerIAStrip = tileControlPacketReceived.numTBCountPerStrip;
	    #endif
		/*
		2. Send load instructions to the tile buffer
		*/
		unsigned char numStripsInTile = inputTileWidth * inputTileHeight;
		unsigned char loadControlBits = (numActivePeCols-1) << 0x2;
		//unsigned char iStripInTile = 0;
		unsigned short iActivationDramBlockAddressBaseLoad = 0;
		EMULATOR_PRINT(("[kernelIATileController] START sending the buffer refresh instructions for iTile=%d .\n\n", iTile));
		for (unsigned char iStripInTile = 0; iStripInTile<numStripsInTile; iStripInTile++)
		{
			t_input_buffer_tile_buffer_packet tileBufferControlPacket;
			tileBufferControlPacket.iActivationDramBlockAddressBase = iActivationDramBlockAddressBaseLoad;
			tileBufferControlPacket.iAddressCache = iStripInTile;
			tileBufferControlPacket.controlBits = loadControlBits;

			#if !defined(SPARSE_SYSTEM)
		    tileBufferControlPacket.numTBCountPerStrip = numTBCountPerIAStrip;
		    #endif


			write_channel_intel(channel_control_to_ia_buffer[0], tileBufferControlPacket);
			iActivationDramBlockAddressBaseLoad += strideStripIACache;
			// bool success = write_channel_nb_intel(channel_control_to_ia_buffer[0], tileBufferControlPacket);
			// if (success)
			// {
			// 	iStripInTile++;
			// 	iActivationDramBlockAddressBaseLoad += strideStripIACache;
			// }
		}
		EMULATOR_PRINT(("[kernelIATileController] FINISHED sending the buffer refresh instructions for iTile=%d .\n\n", iTile));
			
		//End of sending load instructions

		/*
		3. Send the streaming instructions to the tile buffer
		*/
		unsigned short iFilterInGroup = 0;
		unsigned char iInputTileWidth = 0;
		unsigned char iInputTileHeight = 0;
		unsigned char iKernelWidth = 0;
		unsigned char iKernelHeight = 0;
		unsigned char kernelSizeFlat = kernelSize * kernelSize;
		//unsigned short numOutputs = (unsigned short) kernelSizeFlat * (unsigned short) numOutputTileHxW;
		unsigned char iKernelSizeFlat = 0;
		
		//while (iFilterInGroup < numOutputChannelsInGroup)
        for (unsigned int i=0; i<numOutputPerCol; i++)
		{
			unsigned char numActivePeRows = ((numOutputChannelsInGroup - iFilterInGroup) < (unsigned short) (PE_ROWS)) ?
				(unsigned char) (numOutputChannelsInGroup - iFilterInGroup) : PE_ROWS;

			unsigned char iStripInTile = (iInputTileHeight + iKernelHeight) * inputTileWidth + iInputTileWidth + iKernelWidth;

			t_input_buffer_tile_buffer_packet tileBufferControlPacket;
			tileBufferControlPacket.iActivationDramBlockAddressBase = ((unsigned short) iStripInTile) * ((unsigned short) strideStripIACache);
			tileBufferControlPacket.iAddressCache = iStripInTile;
			tileBufferControlPacket.maxPeRowID = (numActivePeRows - 1);
			#if !defined(SPARSE_SYSTEM)
		    tileBufferControlPacket.numTBCountPerStrip = numTBCountPerIAStrip;
		    #endif
			unsigned char isLastBit = ((iKernelSizeFlat+1) == kernelSizeFlat) ? 0x1 : 0x0;
			tileBufferControlPacket.controlBits =
				isLastBit
				| (0x1 << 0x1)
				| ((numActivePeCols-1) << 0x2);

			//bool success = write_channel_nb_intel(channel_control_to_ia_buffer[0], tileBufferControlPacket);
			write_channel_intel(channel_control_to_ia_buffer[0], tileBufferControlPacket);	
			/*
				Parameters update
			*/
			//if (success)
			//{
				EMULATOR_PRINT(("[kernelIATileController] FINISHED sending the buffer stream instruction for iTile=%d, iFilterInGroup=%d, iInputTileHeight=%d, iInputTileWidth=%d, iKernelSizeFlat=%d, iStripInTile=%d. \n\n", 
				iTile, iFilterInGroup, iInputTileHeight, iInputTileWidth, iKernelSizeFlat, iStripInTile));
				if ((iKernelSizeFlat+1) == kernelSizeFlat)
				{
					iKernelWidth = 0;
					iKernelHeight = 0;
					iKernelSizeFlat = 0;

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
				else
				{
					iKernelSizeFlat++;
					if ((iKernelWidth+1)==kernelSize)
					{
						iKernelWidth=0;
						iKernelHeight++;
					}
					else
					{
						iKernelWidth++;
					}
				}
			//}
		}
		//End of sending streaming instructions
	}
}


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


__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelIATee ()
{
	int colID = get_compute_id(0);

	while (true)
	{
		t_dram_block_ia_tagged taggedBlock = read_channel_intel(channel_ia_wide[colID]);

		int destinationCol = (int) taggedBlock.destinationCol;

		if (destinationCol == colID)
		{
			write_channel_intel(channel_ia_wide_local[colID], taggedBlock.dramBlock);
		}
		else
		{
			if (colID < (PE_COLS - 1) )
			{
				write_channel_intel(channel_ia_wide[colID+1], taggedBlock);
			}
		}
	}
}
#endif //IA_MEMORY

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

	unsigned int strideExterrnalMemoryOA, //In terms of output dram block

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
				* (unsigned int) strideExterrnalMemoryOA; //iCol*maxTQ_A is zero

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
					if (receivedBlock.isLast)
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
#endif //MEMORY_WRITER 

#ifdef OA_MEMORY
#define OA_BUFFER_STATE_DECODE 0x0
#define OA_BUFFER_STATE_NUM_ACCESS 0x1
#define OA_BUFFER_STATE_PADD 0x2 //NOP instructions
#define OA_BUFFER_STATE_ACCESS 0x3
#define OA_BUFFER_PAD_COUNT	 2
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelOABuffer ()
{
	typedef uint2_t t_state;

	int colID = get_compute_id(0);
	char cacheOutputActivations[OA_CACHE_SIZE] __attribute__((numbanks(1)));

	/*
	 *Loop carried variables
	*/
	t_state currentState = OA_BUFFER_STATE_DECODE;

	unsigned short startOutputIndex = 0X0;
	unsigned short numOutputToAccess = 0X0;
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
	unsigned char mask = 0;

	unsigned short numLoops = 0;

	unsigned char delayCount = 0;
	unsigned short iLoop = 0;
	unsigned short indexOutput = 0;


	#pragma ivdep
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
				numOutputToAccess = controlPacket.numOutputToAccess;
                isDrainBuffer = (controlPacket.controlBits >> 6) & 0x1;

				//Information relevant for loading the cache only
                numAccumulatorBitsToRightShift = controlPacket.controlBits & 0xF;
                enableRelu = (controlPacket.controlBits >> 4) & 0x1;
                enableSparsification = ( controlPacket.controlBits >> 5) & 0x1;
				
				nextState = OA_BUFFER_STATE_NUM_ACCESS;
			}

		}
		else if (currentState == OA_BUFFER_STATE_NUM_ACCESS)
		{
			numClustersToDrain = 1 + ((numOutputToAccess - 1) >> VALUE_TO_CLUSTER_SHIFT);
			numWindowsToDrain = 1 + ((numClustersToDrain - 1) >> CLUSTER_TO_WINDOW_SHIFT);

			//Loop-carried variables 
			countSurvivingClustersInWindow = 0;
			iClustersInWindowFetched = 0;
			iOutputChannelFetched = 0;
			iClustersFetched = 0;
			mask = 0;

			//Loop control
			#if defined(SPARSE_SYSTEM)
				numLoops = (isDrainBuffer == TRUE) ?
					(numClustersToDrain + numWindowsToDrain) 
					: numOutputToAccess;
			#else
                numLoops = (isDrainBuffer == TRUE) ?
                    (numClustersToDrain)
                    : numOutputToAccess;
			#endif

			delayCount = 0;
			iLoop = 0;
			indexOutput = startOutputIndex;

			nextState = OA_BUFFER_STATE_PADD;

			EMULATOR_PRINT(("[kernelOABuffer %d] START processing instruction. Type=%d, startOutputIndex=%d, numOutputToAccess %d\n\n", 
			colID, isDrainBuffer, startOutputIndex, numOutputToAccess));
		}
		else if (currentState == OA_BUFFER_STATE_PADD)
		{
			delayCount++;
			if (delayCount >= OA_BUFFER_PAD_COUNT)
			{
				nextState = OA_BUFFER_STATE_ACCESS;
			}
		}
		else if (currentState == OA_BUFFER_STATE_ACCESS)
		{
			if (isDrainBuffer == FALSE) //Case: draining the array
			{
				bool readSuccess = false;
				t_accumulator wideOutput = read_channel_nb_intel(channel_drain[0][colID], &readSuccess);
				
				if (readSuccess == true) {
					t_operand shortOutput = modifyOutput(wideOutput, numAccumulatorBitsToRightShift, enableRelu);
					cacheOutputActivations[indexOutput] = shortOutput;

					EMULATOR_PRINT(("[kernelOABuffer %d] Read and processed value from PE. Value: %#04x, %d out of %d values read.\n\n", 
					colID, shortOutput, indexOutput, numOutputToAccess));
					//Loop variable updates
					indexOutput++;
					iLoop++;

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
							char tempValue = (tempOC >= numOutputToAccess) ?
								0x0 : cacheOutputActivations[indexOutput+i];
							cluster.cluster_values[i] = tempValue;
							keep = keep || (tempValue != 0x0);
						}

						if (keep == true)
						{
							write_channel_intel(channel_output_buffer_to_compressor_data[colID], cluster);
							mask |= ((unsigned char) 1) << iClustersInWindowFetched;
							countSurvivingClustersInWindow++;
						}

						iClustersFetched++;
						iClustersInWindowFetched++;

						//Gotcha
						iOutputChannelFetched += CLUSTER_SIZE;
						indexOutput += CLUSTER_SIZE;
						iLoop++;

					}
					else //Send mask along with other informatin
					{
						//bool writeSuccess = false;
						t_output_cluster_info info;
						info.bitmask = mask;
						unsigned char isLastInStrip = (iClustersFetched == numClustersToDrain) ? 0x1 : 0x0;
						info.statusBits = (countSurvivingClustersInWindow & 0x3F)
							| ((isLastInStrip & 0x1) << 0x6)
							| ( (((unsigned char) enableSparsification) & 0x1) << 0x7);
						
						write_channel_intel(channel_output_buffer_to_compressor_info[colID], info);
						mask = 0;
						countSurvivingClustersInWindow = 0;
						iClustersInWindowFetched = 0;
						iLoop++;
					}
				#else //SPARSE_SYSTEM
					t_output_cluster_tagged taggedCluster;
					#pragma unroll
					for (unsigned char i=0; i<CLUSTER_SIZE; i++)
					{
						unsigned short tempOC = iOutputChannelFetched + i;
						char tempValue = (tempOC >= numOutputToAccess) ?
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
					iLoop++;

				#endif //SPARSE_SYSTEM
			} //Case: Stream the buffered output to the cache

            if (iLoop == numLoops)
            {
                nextState = OA_BUFFER_STATE_DECODE;
                    EMULATOR_PRINT(("[kernelOABuffer %d] FINISHED processing instruction. Type=%d, startOutputIndex=%d, numOutputToAccess %d\n\n",
                colID, isDrainBuffer, startOutputIndex, numOutputToAccess));
            }
		}

		currentState = nextState;
	} //end while
}

__attribute__((max_global_work_dim(0)))
__kernel void kernelOATileController (unsigned short numGroupxTiles)
{
	for (unsigned short i=0; i<numGroupxTiles; i++)
	{
		/*
		1. Read the instruction and decode it
		*/
		t_output_tile_controller_packet controlPacket = read_channel_intel(channel_output_writer_to_oa_controller);
		unsigned char numOutputTileHeightxWidth = controlPacket.numOutputTileHeightxWidth;
		unsigned char numFoldsInGroupCurrentLayer = controlPacket.numFoldsInGroupCurrentLayer ;
	    unsigned char numFullFoldsInGroupCurrentLayer = controlPacket.numFullFoldsInGroupCurrentLayer;
	    unsigned char numActiveRowsInPartialFolds = controlPacket.numActiveRowsInPartialFolds;
	    unsigned short numGroupsNextLayer = controlPacket.numGroupsNextLayer;
	    unsigned short numChannelsInGroupCurrentLayer = controlPacket.numChannelsInGroupCurrentLayer;
	    unsigned short numChannelsInGroupNextLayer = controlPacket.numChannelsInGroupNextLayer;
	    unsigned char outputModifierBits = controlPacket.outputModifierBits;
	    unsigned char numActivePeCols = controlPacket.numActivePeCols;

	    unsigned short numOutputChannels = numGroupsNextLayer * numChannelsInGroupNextLayer;

	    /*
	    2. Send instruction to drain from the PE array
	    */
	    unsigned short iChannelCurrentLayer = 0;
	    unsigned short iChannelInGroup = 0;
	    unsigned short iFoldInGroup = 0;
	    unsigned short iOutputTileHxWDrain = 0;
	    unsigned short numDrainInstruction = (unsigned short) numFoldsInGroupCurrentLayer * (unsigned short) numOutputTileHeightxWidth;

	   	EMULATOR_PRINT(("[kernelOATileController] START sending the drain-from-array instruction for tile %d\n\n", 
				i));
	    for  (unsigned short i=0; i < numDrainInstruction; i++)
	    {
	    	unsigned char numActivePeRows = (iFoldInGroup < numFullFoldsInGroupCurrentLayer) ?
	    		PE_ROWS : numActiveRowsInPartialFolds;
	    	unsigned short startOutputIndex = iOutputTileHxWDrain*numOutputChannels + iChannelCurrentLayer+iChannelInGroup;

	    	t_output_tile_buffer_packet_tagged bufferPacketTagged;
	    	bufferPacketTagged.bufferPacket.startOutputIndex = startOutputIndex;
	    	bufferPacketTagged.bufferPacket.numOutputToAccess = numActivePeRows;
	    	bufferPacketTagged.bufferPacket.controlBits = outputModifierBits & 0x3F;
	    	bufferPacketTagged.maxColID = (numActivePeCols - 1);

	    	write_channel_intel(channel_control_to_oa_buffer[0], bufferPacketTagged);

	    	/*
	    	Parameter updates
	    	*/
	    	if ((iOutputTileHxWDrain+1) == numOutputTileHeightxWidth)
	    	{
	    		iOutputTileHxWDrain = 0;
	    		if ((iFoldInGroup+1) == numFoldsInGroupCurrentLayer)
	    		{
	    			iFoldInGroup = 0;
	    			iChannelCurrentLayer += numChannelsInGroupCurrentLayer; 
	    			iChannelInGroup = 0;
	    		}
	    		else
	    		{
	    			iFoldInGroup++;
	    			iChannelInGroup += numActivePeRows;
	    		}
	    	}
	    	else
	    	{
	    		iOutputTileHxWDrain++;
	    	}
	    } //while-loop.  Send instruction to drain from the PE array
	    EMULATOR_PRINT(("[kernelOATileController] FINISHED sending the drain-from-array instruction for tile %d\n\n", 
				i));

	    /*
	    3. Send instructions to stream cached data
	    */
	   	EMULATOR_PRINT(("[kernelOATileController] START sending the write-to-memory instruction for tile %d\n\n", 
				i));
	    unsigned short iChannelNextLayer = 0;
	    unsigned short iOutputTileHxWSend = 0;
	    unsigned short numOutputTileHeightxWidthxNumGroupsNextLayer = (unsigned short) numOutputTileHeightxWidth * (unsigned short) numGroupsNextLayer;
	    for  (unsigned short i=0; i<numOutputTileHeightxWidthxNumGroupsNextLayer; i++)
	    {
	    	t_output_tile_buffer_packet_tagged bufferPacketTagged;
	    	bufferPacketTagged.bufferPacket.startOutputIndex = iOutputTileHxWSend*numOutputChannels + iChannelNextLayer;
	    	bufferPacketTagged.bufferPacket.numOutputToAccess = numChannelsInGroupNextLayer;
	    	bufferPacketTagged.bufferPacket.controlBits = (((unsigned char) 0x1) << 0x6 ) | (outputModifierBits & 0x3F);
	    	bufferPacketTagged.maxColID = (numActivePeCols - 1);

	    	write_channel_intel(channel_control_to_oa_buffer[0], bufferPacketTagged);

	    	if ((iOutputTileHxWSend+1) == numOutputTileHeightxWidth)
	    	{
	    		iOutputTileHxWSend = 0;
	    		iChannelNextLayer += numChannelsInGroupNextLayer;
	    	}
	    	else
	    	{
	    		iOutputTileHxWSend++;
	    	}
	    } //while-loop. Send instructions to stream cached data
	    EMULATOR_PRINT(("[kernelOATileController] FINISH sending the write-to-memory instruction for tile %d\n\n", 
				i));

	} // iterate over tiles
}


#if defined(SPARSE_SYSTEM)
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelCompressorOranizer()
{
	int colID = get_compute_id(0);
	while (true)
	{
		//Read the information first
		t_output_cluster_info info = read_channel_intel(channel_output_buffer_to_compressor_info[colID]);
		unsigned char bitmask = info.bitmask;
		unsigned char numSurvivingClusters = info.statusBits & 0x3F;
		bool isLastWindowInStrip = (((info.statusBits >> 0x6) & 0x1) == 0x1);
		bool enableSparsification = (((info.statusBits >> 0x7) & 0x1) == 0x1);

		//Depending on whether sparsification is enabled, we may or may not to add an extra cluster to encode the bitmask/surviving cluster count
		unsigned char numClustersToSend = enableSparsification ?
			 numSurvivingClusters + 1 : numSurvivingClusters;

		//For every weindow, we want to sent a number of clusters that is a multiple of transfer size
		//The number of clusters that we actually need to send is the number of surviving clusters, the bitmask/surviving cluster count
		unsigned char numLoops = 
			(1 + ((numClustersToSend - 1) >> CLUSTER_TO_TRANSFER_SIZE_SHIFT)) << CLUSTER_TO_TRANSFER_SIZE_SHIFT; //This controls the amount of padding we have to do. Add extra one to send mask
		unsigned char iClusterSent = 0;
		for (unsigned char iLoop=0; iLoop<numLoops; iLoop++)
		{
			bool sendCluster = false;
			t_output_cluster_tagged clusterTagged;
			if (iClusterSent < numClustersToSend)
			{
				if ((iClusterSent == 0) && enableSparsification)
				{
					//TODO: Account for case that doesn't require sparsification
					clusterTagged.cluster.cluster_values[0] = bitmask;
					clusterTagged.cluster.cluster_values[1] = numSurvivingClusters;
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
		t_output_tile_buffer_packet_tagged controlPacketTagged = read_channel_intel(channel_control_to_oa_buffer[colID]);

		unsigned char maxColID = controlPacketTagged.maxColID;

		write_channel_intel(channel_control_to_oa_buffer_local[colID], controlPacketTagged.bufferPacket);

		if (colID < (PE_COLS - 1) )
		{
			if (maxColID > (unsigned char) colID )
			{
				write_channel_intel(channel_control_to_oa_buffer[colID+1], controlPacketTagged);
			}
		}
	}
}


#define STATE_OA_TEE_DRAIN_SELF 0x0
#define STATE_OA_TEE_DRAIN_SELF_SEND_COUNT 0X1
#define STATE_OA_TEE_DRAIN_OTHERS 0x2
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelOATee ()
{
	typedef uint2_t t_state;
	int colID = get_compute_id(0);

	while (true)
	{
		/*
		Read the control
		*/
		t_output_tile_tee_packet teeControl = read_channel_intel(channel_output_writer_to_tee[colID]);


		/*
		Decode the control
		*/
		unsigned short numOutputGroupxTileHeightxTileWidth = teeControl.numOutputGroupxTileHeightxTileWidth;
		unsigned char maxColID = teeControl.maxColID;
		unsigned char numOtherColToCollect = maxColID - colID;


		/*
		Pass on the control to the right if needed
		*/
		if (colID < (PE_COLS - 1))
		{
			if (colID < maxColID)
			{
				write_channel_intel(channel_output_writer_to_tee[colID+1], teeControl);
			}
		}

		/*
		Drain the outputs
		*/
		for (unsigned char iterOutput=0; iterOutput<numOutputGroupxTileHeightxTileWidth; iterOutput++)
		{
			t_state state = STATE_OA_TEE_DRAIN_SELF;
			unsigned short iClusters = 0;
			unsigned short iClusterInDram = 0;
			unsigned char iColDrained = 0;
			unsigned char numColToDrain = numOtherColToCollect + 1;
			t_output_dram_block dramBlock;

			while (iColDrained < numColToDrain)
			{
				t_output_dram_block_tagged dramBlockTagged;
				bool writeChannel = false;
				t_state nextState = state;

				if (state == STATE_OA_TEE_DRAIN_SELF)
				{
					#if defined(SPARSE_SYSTEM)
						t_output_cluster_tagged clusterTagged = read_channel_intel(channel_compressor_to_tee[colID]);
					#else
						t_output_cluster_tagged clusterTagged = read_channel_intel(channel_oa_buffer_to_oa_tee[colID]);
					#endif
					dramBlock.clusters[iClusterInDram] = clusterTagged.cluster;
					iClusters++;

					if ( ( (iClusterInDram + 1) == NUM_CLUSTER_IN_DRAM_SIZE) || (clusterTagged.isLastInStrip) )
					{
						writeChannel = true;

						dramBlockTagged.block = dramBlock;
						#if defined(SPARSE_SYSTEM)
							dramBlockTagged.isLast = false;
						#else
							dramBlockTagged.isLast = clusterTagged.isLastInStrip;
						#endif

						iClusterInDram = 0;
					}
					else
					{
						iClusterInDram++;
					}
					
					if (clusterTagged.isLastInStrip)
					{
						#if defined(SPARSE_SYSTEM)
							nextState = STATE_OA_TEE_DRAIN_SELF_SEND_COUNT;
						#else
							nextState = STATE_OA_TEE_DRAIN_OTHERS;
							iColDrained++;
						#endif
					}
				} //STATE_OA_TEE_DRAIN_SELF
				#if defined(SPARSE_SYSTEM)
					else if (state == STATE_OA_TEE_DRAIN_SELF_SEND_COUNT)
					{
						writeChannel = true;

						t_output_dram_block countDramBlock = clusterCount2OutputDramBlock(iClusters);

						dramBlockTagged.block = countDramBlock;
						dramBlockTagged.isLast = true;

						nextState = STATE_OA_TEE_DRAIN_OTHERS;
						iColDrained++;
					} //STATE_OA_TEE_DRAIN_SELF_SEND_COUNT
				#endif
				else if (state == STATE_OA_TEE_DRAIN_OTHERS)
				{

					if (colID < (PE_COLS - 1))
					{	
						writeChannel = true;
						t_output_dram_block_tagged receivedBlock = read_channel_intel(channel_output_wide[colID+1]);
						dramBlockTagged = receivedBlock;
						if (receivedBlock.isLast)
						{
							iColDrained++;
						}
					}

				} //STATE_OA_TEE_DRAIN_OTHERS

				if (writeChannel)
				{
					write_channel_intel(channel_output_wide[colID], dramBlockTagged);
				}

				state = nextState;
			} //while

		} //for

	} //while

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

#define STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL 0X0
#define STATE_FILTER_STREAMER_WRITE_CACHE_WRITE 0X1
#define STATE_FILTER_STREAMER_WRITE_CACHE_WAIT 0X2

#define STATE_FILTER_STREAMER_READ_CACHE_SETUP 0X0
#define STATE_FILTER_STREAMER_READ_CACHE_READ 0X1
#define STATE_FILTER_STREAMER_READ_CACHE_WAIT 0X2

/*! kernelFilterStreamer
	\brief Stream filter values to the PE array
*/
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_ROWS)))
__kernel void kernelFilterBuffer ()
{
	int rowID = get_compute_id(0);

	typedef uint2_t t_state;
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
		t_transferblock_tagged weightBlockTagged;
		
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

			if (iTransferBlockInFilterRead > 0)
			{
				unsigned short dramIndex = (iTransferBlockInFilterRead - 1) >> WIDE_SIZE_OFFSET;
				unsigned short indexInDramBlock = (iTransferBlockInFilterRead - 1) & WIDE_SIZE_REMAINDER_MASK;
				t_dram_block dramBlock = cacheNzBlocks[(~regWriteSide) & 0x1][dramIndex];
				t_transfer_block tblock = dramBlock.transferBlocks[indexInDramBlock];
				weightBlockTagged.values = tblock;
				weightBlockTagged.isLast = ((iTransferBlockInFilterRead) >= maxTransferBlockInFilter[(~regWriteSide) & 0x1]) ?
					TRUE : FALSE;
			}
			else
			{
				t_accumulator bias = cacheBias[(~regWriteSide) & 0x1];
				t_transfer_block tblock = bias2TransferBlock(bias);
				weightBlockTagged.values = tblock;
				weightBlockTagged.isLast = false;
			}
			
			weightBlockTagged.maxTransportID = maxPeCols[(~regWriteSide) & 0x1];

			// EMULATOR_PRINT(("[kernelFilterBuffer %d] Attempt to send transfer block %d / %d, in the %d / %d time.\n\n", 
			// 		rowID, iTransferBlockInFilterRead, maxTransferBlockInFilter[(~regWriteSide) & 0x1], iOutputRead, maxOutputCount[(~regWriteSide) & 0x1]));
			bool success = false;
			success = write_channel_nb_intel(channel_weight[rowID][0], weightBlockTagged);
			if (success)
			{
				EMULATOR_PRINT(("[kernelFilterBuffer %d] Sent transfer block %d / %d with lastTag %d in the %d / %d time.\n\n", 
					rowID, iTransferBlockInFilterRead, maxTransferBlockInFilter[(~regWriteSide) & 0x1], weightBlockTagged.isLast, iOutputRead, maxOutputCount[(~regWriteSide) & 0x1]));

                EMULATOR_PRINT(("[kernelFilterStreamer %d] Sent tb %d: %#04x %#04x %#04x %#04x\n",
					rowID, 
					iTransferBlockInFilterRead,
                    weightBlockTagged.values.values[0].cluster_values[0],
                    weightBlockTagged.values.values[0].cluster_values[1],
                    weightBlockTagged.values.values[1].cluster_values[0],
                    weightBlockTagged.values.values[1].cluster_values[1]));

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

//MAC Operands
typedef struct __attribute__((packed)) {
	char values [SIMD_SIZE*CLUSTER_SIZE];
} t_simd_operand;


__attribute__((task))
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
	int idx = IDX;
	int idy = IDY;
#endif

	while (true)
	{
		//t_simdblock_di_tagged block = read_channel_intel(channel_weightInput);
		//t_simdblock_di peBlock;
		#ifdef DIRECT_COMPRESSION_SIMD
		t_simdblock_bitmask_tagged block;
		t_simdblock_bitmask peBlock;
		#endif

		#ifdef FLEXIBLE_BITMASK_COMPRESSION
		t_transferblock_tagged block;
		t_transferblock_local peBlock;
		#endif
	// #ifdef FULL_SYSTEM
	// 			EMULATOR_PRINT(("[WEIGHT TRANSPORT (%d, %d)] Waiting to read weight/bias transfer block.\n\n", idy, idx));
	// #else
	// 			EMULATOR_PRINT(("[WEIGHT TRANSPORT] Waiting to read weight/bias transfer block.\n\n"));
	// #endif
	#ifdef FULL_SYSTEM
		block = read_channel_intel(channel_weight[idy][idx]);
	#else
		block = read_channel_intel(channel_weight[0][0]);
	#endif

	#ifdef FULL_SYSTEM
				EMULATOR_PRINT(("[WEIGHT TRANSPORT (%d, %d)] Read weight/bias transfer block. Is Last tag is %d\n\n", idy, idx, block.isLast));
	#else
				EMULATOR_PRINT(("[WEIGHT TRANSPORT] Read weight/bias transfer block.\n\n"));
	#endif

	// #ifdef FULL_SYSTEM
	// 			EMULATOR_PRINT(("[WEIGHT TRANSPORT (%d, %d)] Waiting to pass on weight/bias transfer block.\n\n", idy, idx));
	// #else
	// 			EMULATOR_PRINT(("[WEIGHT TRANSPORT] Waiting to pass on weight/bias transfer block.\n\n"));
	// #endif

		#pragma unroll
		for (unsigned char i=0; i<SIMD_SIZE; i++) {
			#ifdef DIRECT_COMPRESSION_SIMD
				peBlock.values.values[i] = block.values[i];
			#endif
			#ifdef FLEXIBLE_BITMASK_COMPRESSION
				peBlock.values.values[i] = block.values.values[i];
			#endif
		}
		//peBlock.streamingBlockIndex = block.streamingBlockIndex;
		peBlock.isLast = block.isLast;

		if (idx < (PE_COLS - 1)){
			if ( idx < block.maxTransportID ) {
				//EMULATOR_PRINT ( ("[kernelWeightTransport]: Waiting to pass a weight block to the output\n") );
	#ifdef FULL_SYSTEM
				write_channel_intel(channel_weight[idy][idx+1], block);
	#else
				write_channel_intel(channel_weight[0][1], block);
	#endif
			}
		}
	#ifdef FULL_SYSTEM
		write_channel_intel(channel_dpWeightInput[idy][idx], peBlock); 
	#else
		write_channel_intel(channel_dpWeightInput[0][0], peBlock); 
	#endif

	#ifdef FULL_SYSTEM
				EMULATOR_PRINT(("[WEIGHT TRANSPORT (%d, %d)] Passed on weight/bias transfer block.\n\n", idy, idx));
	#else
				EMULATOR_PRINT(("[WEIGHT TRANSPORT] Passed on weight/bias transfer block.\n\n"));
	#endif
		}
}

#define STATE_ACTIVATION_TRANSPORT_READ 0X0
#define STATE_ACTIVATION_TRANSPORT_DRAIN_SELF 0x1
#define STATE_ACTIVATION_TRANSPORT_DRAIN_OTHERS 0x2

__attribute__((task))
__attribute__((max_global_work_dim(0)))
#ifdef FULL_SYSTEM
__attribute__((num_compute_units(PE_ROWS, PE_COLS)))
#endif
__attribute__ ((autorun))
__kernel void kernelActivationTransport ()
{
	typedef uint2_t t_state;

#ifdef FULL_SYSTEM
	int idx = get_compute_id(1);
	int idy = get_compute_id(0);
#else
	int idx = IDX;
	int idy = IDY;
#endif

	t_state state = STATE_ACTIVATION_TRANSPORT_READ;
	unsigned char numOtherPSumToDrain;
	unsigned char countOtherPSum;

	while (true)
	{
		t_state nextState = state;
		t_accumulator pSum;
		if (state == STATE_ACTIVATION_TRANSPORT_READ)
		{
			#ifdef DIRECT_COMPRESSION_SIMD
			t_simdblock_bitmask_tagged block;
			t_simdblock_bitmask peBlock;
			#endif

			#ifdef FLEXIBLE_BITMASK_COMPRESSION
			t_transferblock_tagged block;
			t_transferblock_local peBlock;
			#endif
#ifdef FULL_SYSTEM
			block = read_channel_intel(channel_activation[idy][idx]);
#else
			block = read_channel_intel(channel_activation[0][0]);
#endif
			#pragma unroll
			for (unsigned char i=0; i<SIMD_SIZE; i++) {
				#ifdef DIRECT_COMPRESSION_SIMD
					peBlock.values.values[i] = block.values[i];
				#endif
				#ifdef FLEXIBLE_BITMASK_COMPRESSION
					peBlock.values.values[i] = block.values.values[i];
				#endif
			}
			//peBlock.streamingBlockIndex = block.streamingBlockIndex;
			peBlock.isLast = block.isLast;

			if (idy < (PE_ROWS - 1)){
				if ( idy < block.maxTransportID ) {
					//EMULATOR_PRINT ( ("[kernelWeightTransport]: Waiting to pass an activation block to the output\n") );
#ifdef FULL_SYSTEM
			write_channel_intel(channel_activation[idy+1][idx], block);
#else
			write_channel_intel(channel_activation[0][1], block);
#endif
				}
			}

			if (block.isLast == TRUE)
			{
#ifdef FULL_SYSTEM
			EMULATOR_PRINT(("[ACTIVATION TRANSPORT (%d, %d)] End of activation compression window detected.\n\n", idy, idx));
#else
			EMULATOR_PRINT(("[ACTIVATION TRANSPORT] End of activation compression window detected.\n\n"));
#endif
				nextState = STATE_ACTIVATION_TRANSPORT_DRAIN_SELF;	
				numOtherPSumToDrain = block.maxTransportID - idy;
				countOtherPSum = 0;
			}

#ifdef FULL_SYSTEM
			write_channel_intel(channel_dpActivationInput[idy][idx], peBlock);
#else
			write_channel_intel(channel_dpActivationInput[0][0], peBlock);
#endif
			 

		} //STATE_ACTIVATION_TRANSPORT_READ
		else if (state == STATE_ACTIVATION_TRANSPORT_DRAIN_SELF)
		{
#ifdef FULL_SYSTEM
			pSum = read_channel_intel(channel_peDrainOutput[idy][idx]);
			EMULATOR_PRINT(("[ACTIVATION TRANSPORT (%d, %d)] Drained from PE\n\n", idy, idx));
#else
			pSum = read_channel_intel(channel_peDrainOutput[0][0]);
			EMULATOR_PRINT(("[ACTIVATION TRANSPORT] Drained from PE\n\n"));
#endif
			if (countOtherPSum == numOtherPSumToDrain)
			{
				nextState = STATE_ACTIVATION_TRANSPORT_READ;
			}
			else
			{
				nextState = STATE_ACTIVATION_TRANSPORT_DRAIN_OTHERS;
			}
		} //STATE_ACTIVATION_TRANSPORT_DRAIN_SELF
		else if (state == STATE_ACTIVATION_TRANSPORT_DRAIN_OTHERS)
		{
			//TODO: change the following in deply
#ifdef FULL_SYSTEM
			if (idy < PE_ROWS - 1)
			{
				pSum = read_channel_intel(channel_drain[idy+1][idx]);
				EMULATOR_PRINT(("[ACTIVATION TRANSPORT (%d, %d)] Drained from others. %d more to drain\n\n", idy, idx, numOtherPSumToDrain-countOtherPSum-1));
			}
#else
				pSum = read_channel_intel(channel_drain[1][0]);
				EMULATOR_PRINT(("[ACTIVATION TRANSPORT] Drained from others. %d more to drain\n\n", idy, idx, numOtherPSumToDrain-countOtherPSum-1));
#endif
			countOtherPSum++;
			if (countOtherPSum == numOtherPSumToDrain)
			{
				nextState = STATE_ACTIVATION_TRANSPORT_READ;
			} 
		} //STATE_ACTIVATION_TRANSPORT_DRAIN_OTHERS

		if ((state == STATE_ACTIVATION_TRANSPORT_DRAIN_OTHERS) 
			|| 
			(state == STATE_ACTIVATION_TRANSPORT_DRAIN_SELF))
		{
#ifdef FULL_SYSTEM
			write_channel_intel(channel_drain[idy][idx], pSum);
#else
			write_channel_intel(channel_drain[0][0], pSum);
#endif
		}

		state = nextState;
	}
}

t_accumulator madd (t_simd_operand activations, t_simd_operand weights) {
	t_accumulator output = 0x0;

	//#ifdef DIRECT_COMPRESSION_SIMD
		#pragma unroll
		for(int i=0; i<SIMD_SIZE*CLUSTER_SIZE/4; i++){
			//output += input.data[i]*weights.data[i];
			// use packed DSP blocks to improve efficiency
			#if defined (ARRIA10)
				output += (t_accumulator) a10_mac_8bitx4(
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
				output += (t_accumulator) c5_mac_8bitx4(
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
	//#endif
	//#ifdef FLEXIBLE_BITMASK_COMPRESSION
	/*
		#pragma unroll
		for(int i=0; i<SIMD_SIZE/2; i++){
			//output += input.data[i]*weights.data[i];
			// use packed DSP blocks to improve efficiency
			#if defined (ARRIA10)
				output += a10_mac_8bitx2(
					activations.values[i*2],
					weights.values[i*2],
					activations.values[i*2+1],
					weights.values[i*2+1]
					);
			#elif defined (C5SOC)
				output += c5_mac_8bitx2(
						activations.values[i*2],
						weights.values[i*2],
						activations.values[i*2+1],
						weights.values[i*2+1]
					);
			#else
			#error Unsupported FPGA type!
			#endif
		}
		*/
	//#endif

	return output;
}

#if defined (SPARSE_SYSTEM)
#define ASSEMBLER_STATE_LOAD_BITMASK 0X0
#define ASSEMBLER_STATE_LOAD_VALUE 0X1
//#define ASSEMBLER_STATE_ALIGN 0x2
#define ASSEMBLER_STATE_WAIT 0x2
#define ASSEMBLER_STATE_LOAD_BIAS 0x3

#define BITWIDTH_COMPRESSION_WINDOW_INDEX 3
#define MASK_COMPRESSION_WINDOW_INDEX 0x7

#define MAC_STATE_WAIT 0x0
#define MAC_STATE_ALIGN 0x1
#define MAC_STATE_PROCESS_WINDOW 0x2
#define MAC_STATE_WRITE_PSUM 0x3
#define MAC_STATE_LOAD_BIAS 0x4

//DENSE TODO; Replace it with a dense PE
__attribute__((task))
__attribute__((max_global_work_dim(0)))
#ifdef FULL_SYSTEM
__attribute__((num_compute_units(PE_ROWS, PE_COLS)))
#endif
__attribute__((autorun))
__kernel void kernelPE ()
{
	
#ifdef FULL_SYSTEM
	int idx = get_compute_id(1);
	int idy = get_compute_id(0);
#endif

	//================Ping-ponged registers========================
	//BRAM for storing the compression windows
	// t_cluster activationWindow[COMPRESSION_WINDOW_SIZE+1][2]  __attribute__((numbanks(1))); 
	// t_cluster weightWindow[COMPRESSION_WINDOW_SIZE+1][2]  __attribute__((numbanks(1))); 
	t_cluster activationWindow[COMPRESSION_WINDOW_SIZE+1][2]; 
	t_cluster weightWindow[COMPRESSION_WINDOW_SIZE+1][2]; 

	//Flags that indicates whether we are at the last window
	uint1_t isLast[2] = {FALSE, TRUE};
	unsigned char bitmaskA[2];
	unsigned char bitmaskW[2];
	t_accumulator bias[2];

	uint1_t regLoadSide = 0x0;

	//========Assembler side registers====================
	unsigned char countActivation;
	unsigned char countWeight;
	unsigned char numActivation;
	unsigned char numWeight;
	uint2_t stateActivation = ASSEMBLER_STATE_LOAD_BIAS;
	uint2_t stateWeight = ASSEMBLER_STATE_LOAD_BIAS;
	//unsigned long alignmentData;


	//=========MAC side logic========================
	uint3_t stateMac = MAC_STATE_WAIT;
	t_accumulator pSum = 0;
	unsigned char countOperands;
	unsigned char numOperands;
	unsigned int indicesW;
	unsigned int indicesA;

	//================Debug====================
	//unsigned short debugCount = 0;

	//#pragma ivdep array(activationWindow)
	//#pragma ivdep array(weightWindow)
	//#pragma ivdep safelen(7)
	#pragma ivdep
	while (true)
	{

		//================ACTIVATION========================
		
		uint2_t nextStateActivation = stateActivation;
		{ 
			if (stateActivation == ASSEMBLER_STATE_LOAD_BITMASK
				|| stateActivation == ASSEMBLER_STATE_LOAD_VALUE)
			{
				t_transferblock_local activationTransferBlock;
				bool activationReadSuccess;

#ifdef FULL_SYSTEM
				activationTransferBlock = read_channel_nb_intel (
							channel_dpActivationInput[idy][idx],
							&activationReadSuccess
						);
#else
				activationTransferBlock = read_channel_nb_intel (
							channel_dpActivationInput[0][0],
							&activationReadSuccess
						);
#endif
				if (activationReadSuccess)
				{
					//isLastActivation = activationTransferBlock.isLast;
					//DEBUG_PRINT(("[Assembler] Activation read!\n"));

					if (stateActivation == ASSEMBLER_STATE_LOAD_BITMASK)
					{
						unsigned char bitmask = activationTransferBlock.values.values[0].cluster_values[0];
						bitmaskA[regLoadSide & 0x01] = bitmask;
						numActivation = popCounter(bitmask);
						countActivation = 0;
#ifdef FULL_SYSTEM
							//EMULATOR_PRINT(("[PE (%d, %d)] bitmaskA: %#04x \n", idy, idx, bitmask));
#else
							//EMULATOR_PRINT(("[PE] bitmaskA: %#04x \n", bitmask));
#endif
					}
					//else
					//{

						//uint3_t offset = (stateActivation == ASSEMBLER_STATE_LOAD_BITMASK) ?
						//	0X1 : 0X0; 

						#pragma unroll
						for (uint3_t i=0; i<TRANSFER_SIZE; i++)
						{
							//if (i >= offset)
							//{
								activationWindow[countActivation+i][regLoadSide & 0x01]
									= activationTransferBlock.values.values[i];
								//EMULATOR_PRINT(("[assembler] activation value: %#04x %#04x \n"
								//	, activationTransferBlock.values.values[i].cluster_values[0] & 0xFF
								//	, activationTransferBlock.values.values[i].cluster_values[1] & 0xFF));
								//EMULATOR_PRINT(("[assembler] activation offset, countActivation: %#04x %#04x\n"
								//	, offset, countActivation));
							//}
						} // for. Transfer the values in the transfer block to the compression window

						//if (debugCount < maxDebugCount)
						//{
// #ifdef FULL_SYSTEM
// 							EMULATOR_PRINT(("[PE (%d %d)] ActivationTransferBlock [0-4]: %#04x %#04x %#04x %#04x\n",
// 								idy, idx,
// 								activationTransferBlock.values.values[0].cluster_values[0] & 0xFF, 
// 								activationTransferBlock.values.values[0].cluster_values[1] & 0xFF,
// 								activationTransferBlock.values.values[1].cluster_values[0] & 0xFF,
// 								activationTransferBlock.values.values[1].cluster_values[1] & 0xFF));
// #else
// 							EMULATOR_PRINT(("[PE] ActivationTransferBlock [0-4]: %#04x %#04x %#04x %#04x\n",
// 								activationTransferBlock.values.values[0].cluster_values[0] & 0xFF, 
// 								activationTransferBlock.values.values[0].cluster_values[1] & 0xFF,
// 								activationTransferBlock.values.values[1].cluster_values[0] & 0xFF,
// 								activationTransferBlock.values.values[1].cluster_values[1] & 0xFF));
// #endif
						//}

						countActivation += (unsigned char)(TRANSFER_SIZE);
					//}

					//State update
					if (countActivation > numActivation) //countActivation needs to be strictly larger than numActivation
					{
						nextStateActivation = ASSEMBLER_STATE_WAIT;
					}
					else {
						nextStateActivation = ASSEMBLER_STATE_LOAD_VALUE;
					}

				} // if activationReadSuccess
			} // ASSEMBLER_STATE_LOAD_BITMASK || ASSEMBLER_STATE_LOAD_VALUE 
			else if (stateActivation == ASSEMBLER_STATE_LOAD_BIAS)
			{
#ifdef FULL_SYSTEM
				//EMULATOR_PRINT(("[PE (%d %d)] Wait for bias\n", idy, idx));
#else
				//EMULATOR_PRINT(("[PE] Wait for bias\n"));
#endif
				
				nextStateActivation = ASSEMBLER_STATE_WAIT;
			}
		}
		//===================================================

		//================WEIGHT========================
		
		uint2_t nextStateWeight = stateWeight;
		{
			bool weightReadSuccess;
			t_transferblock_local weightTransferBlock;

			if (stateWeight == ASSEMBLER_STATE_LOAD_BITMASK
				|| stateWeight == ASSEMBLER_STATE_LOAD_VALUE
				|| stateWeight == ASSEMBLER_STATE_LOAD_BIAS) 
			{
#ifdef FULL_SYSTEM
				weightTransferBlock = read_channel_nb_intel (
							channel_dpWeightInput[idy][idx],
							&weightReadSuccess
						);
#else
				weightTransferBlock = read_channel_nb_intel (
							channel_dpWeightInput[0][0],
							&weightReadSuccess
						);
#endif
			}

			if (stateWeight == ASSEMBLER_STATE_LOAD_BITMASK
				|| stateWeight == ASSEMBLER_STATE_LOAD_VALUE)
			{
				if (weightReadSuccess)
				{
					isLast[regLoadSide & 0x01] = weightTransferBlock.isLast;
					//DEBUG_PRINT(("[Assembler] Weight read!\n"));

					if (stateWeight == ASSEMBLER_STATE_LOAD_BITMASK)
					{
						unsigned char bitmask =  weightTransferBlock.values.values[0].cluster_values[0];
						bitmaskW[regLoadSide & 0x01] = bitmask; 
						numWeight = popCounter(bitmask);
						countWeight = 0;
#ifdef FULL_SYSTEM
						//EMULATOR_PRINT(("[PE (%d %d)] bitmaskW: %#04x \n", idy, idx, bitmask));
#else
						//EMULATOR_PRINT(("[PE] bitmaskW: %#04x \n", bitmask));
#endif
					}
					//else
					//{

						//uint3_t offset = (stateWeight == ASSEMBLER_STATE_LOAD_BITMASK) ?
						//	0X1 : 0X0; 

						#pragma unroll
						for (uint3_t i=0; i<TRANSFER_SIZE; i++)
						{
							//if (i >= offset)
							//{
								weightWindow[countWeight+i][regLoadSide & 0x01]
									= weightTransferBlock.values.values[i];
								//EMULATOR_PRINT(("[assembler] weight value: %#04x %#04x \n"
								//	, weightTransferBlock.values.values[i].cluster_values[0] & 0xFF
								//	, weightTransferBlock.values.values[i].cluster_values[1] & 0xFF));
							//}
						} // for. Transfer the values in the transfer block to the compression window

						//if (debugCount < maxDebugCount)
						//{
// #ifdef FULL_SYSTEM
// 						EMULATOR_PRINT(("[PE (%d %d)] weightTransferBlock [0-4]: %#04x %#04x %#04x %#04x\n",
// 								idy, idx,
// 								weightTransferBlock.values.values[0].cluster_values[0] & 0xFF, 
// 								weightTransferBlock.values.values[0].cluster_values[1] & 0xFF,
// 								weightTransferBlock.values.values[1].cluster_values[0] & 0xFF,
// 								weightTransferBlock.values.values[1].cluster_values[1] & 0xFF));
// #else
// 						EMULATOR_PRINT(("[PE] weightTransferBlock [0-4]: %#04x %#04x %#04x %#04x\n",
// 								weightTransferBlock.values.values[0].cluster_values[0] & 0xFF, 
// 								weightTransferBlock.values.values[0].cluster_values[1] & 0xFF,
// 								weightTransferBlock.values.values[1].cluster_values[0] & 0xFF,
// 								weightTransferBlock.values.values[1].cluster_values[1] & 0xFF));
// #endif
						//}

						countWeight += (unsigned char)(TRANSFER_SIZE);
					//}

					//State update
					if (countWeight > numWeight) //countWeight needs to be strictly larger than numWeight
					{
						nextStateWeight = ASSEMBLER_STATE_WAIT;
					}
					else 
					{
						nextStateWeight = ASSEMBLER_STATE_LOAD_VALUE;
					}

				} // if weightReadSuccess
			} //ASSEMBLER_STATE_LOAD_BITMASK || ASSEMBLER_STATE_LOAD_VALUE
			else if (stateWeight == ASSEMBLER_STATE_LOAD_BIAS)
			{
				if (weightReadSuccess)
				{
#ifdef FULL_SYSTEM
					//EMULATOR_PRINT(("[PE (%d %d)] Wait for bias\n", idy, idx));
#else
					//EMULATOR_PRINT(("[PE] Wait for bias\n"));
#endif
					bias[regLoadSide & 0x01] = transferBlock2Bias(weightTransferBlock.values);
					nextStateWeight = ASSEMBLER_STATE_WAIT;
				}
			}
		}
		//===================================================

		//==================MAC states===================
		uint3_t nextStateMac = stateMac;

		if (stateMac == MAC_STATE_ALIGN)
		{
			
			unsigned long alignmentData = operandMatcher8(
				bitmaskW [(~regLoadSide) & 0x1],
				bitmaskA [(~regLoadSide) & 0x1]
			);
			
			
			//unsigned long alignmentData = 0;
			numOperands = (alignmentData >> 48) & 0xFF;
			indicesW = (alignmentData >> 24) & 0xFFFFFF;
			indicesA = (alignmentData) & 0xFFFFFF;
			countOperands = 0; 
			//EMULATOR_PRINT ( ("[aligner]: indicesW: %#06x indicesA: %#06x numOperands: %#04x \n"
			//		, indicesW, indicesA,  numOperands) );

			/*
			if (countOperands >= numOperands)
			{
				if (isLast[(~regLoadSide) & 0x1])
				{
					nextStateMac = MAC_STATE_WRITE_PSUM;
				}
				else
				{
					nextStateMac = MAC_STATE_WAIT;
				}
			}
			else
			{
				nextStateMac = MAC_STATE_PROCESS_WINDOW;
			}
			*/
			nextStateMac = MAC_STATE_PROCESS_WINDOW;
		}
		else if (stateMac == MAC_STATE_PROCESS_WINDOW)
		{

			t_simd_operand simdActivations;
			t_simd_operand simdWeights;
			t_cluster zeros;
			#pragma unroll
			for (int i=0; i<CLUSTER_SIZE; i++)
			{
				zeros.cluster_values[i] = 0x0;
			}


			#pragma unroll
			for (unsigned char i=0; i<SIMD_SIZE; i++)
			{
				unsigned char indexW = 
					(indicesW >> (i*BITWIDTH_COMPRESSION_WINDOW_INDEX))
					& MASK_COMPRESSION_WINDOW_INDEX;
				t_cluster w = ((countOperands + i) < numOperands) ?
					weightWindow[indexW+1][(~regLoadSide) & 0x01] : zeros;
				//char w = weightWindow[i][(~regLoadSide) & 0x1];
				//simdWeights.values[i] = w;

				unsigned char indexA = 
					(indicesA >> (i*BITWIDTH_COMPRESSION_WINDOW_INDEX))
					& MASK_COMPRESSION_WINDOW_INDEX;
				t_cluster a = ((countOperands + i) < numOperands) ?
					activationWindow[indexA+1][(~regLoadSide) & 0x01] : zeros;
				//char a = activationWindow[i][(~regLoadSide) & 0x1];

				#pragma unroll
				for (unsigned char j=0; j<CLUSTER_SIZE; j++)
				{
					simdActivations.values[CLUSTER_SIZE*i + j] = a.cluster_values[j];
					simdWeights.values[CLUSTER_SIZE*i + j] = w.cluster_values[j];
				}

				//EMULATOR_PRINT ( ("[dispatcher]: w0: %#04x w1: %#04x a0: %#04x a1: %#04x \n"
				//	, w.cluster_values[0] & 0xFF, w.cluster_values[1] & 0xFF,  a.cluster_values[0] & 0xFF, a.cluster_values[1] & 0xFF) );
				//EMULATOR_PRINT ( ("[dispatcher]: wIndex: %u aIndex :%u \n", (indexW) & 0xFF, (indexA) & 0xFF));
			}


			t_accumulator tempPSum = madd(simdActivations, simdWeights);
			pSum += tempPSum;
			//if (debugCount < maxDebugCount)
			//	{
			//		DEBUG_PRINT(("[PE Dispatcher] a0, a1, a1, a2: %#04x %#04x %#04x %#04x\n",
			//			simdActivations.values[0] & 0xFF, 
			//			simdActivations.values[1] & 0xFF,
			//			simdActivations.values[2] & 0xFF,
			//			simdActivations.values[3] & 0xFF));

			//		DEBUG_PRINT(("[PE Dispatcher] w0, w1, w2, w3: %#04x %#04x %#04x %#04x\n",
			//			simdWeights.values[0] & 0xFF, 
			//			simdWeights.values[1] & 0xFF,
			//			simdWeights.values[2] & 0xFF,
			//			simdWeights.values[3] & 0xFF));

			//		DEBUG_PRINT(("[PE Madd] Psum %#04x\n", pSum));

			//	}		
			countOperands += SIMD_SIZE;
			indicesW = indicesW >> (SIMD_SIZE*BITWIDTH_COMPRESSION_WINDOW_INDEX);
			indicesA = indicesA >> (SIMD_SIZE*BITWIDTH_COMPRESSION_WINDOW_INDEX);

			if (countOperands >= numOperands)
			{
				if (isLast[(~regLoadSide) & 0x1] == TRUE)
				{
					nextStateMac = MAC_STATE_WRITE_PSUM;
				}
				else
				{
					nextStateMac = MAC_STATE_WAIT;
				}
			}
		} // if state == MAC_STATE_PROCESS_WINDOW
		else if (stateMac == MAC_STATE_WRITE_PSUM)
		{
			bool writeSuccess;
#ifdef FULL_SYSTEM
			writeSuccess = write_channel_nb_intel(channel_peDrainOutput[idy][idx], pSum);
#else
			writeSuccess = write_channel_nb_intel(channel_peDrainOutput[0][0], pSum);
#endif
			

			//write_channel_intel(channel_peDrainOutput, pSum);
			if (writeSuccess)
			{
				//DEBUG_PRINT(("[MAC] Sending!\n"));
#ifdef FULL_SYSTEM
				EMULATOR_PRINT(("[PE (%d, %d)] Commit. pSum value: %#04x \n", idy, idx, pSum));
#else
				EMULATOR_PRINT(("[PE] Commit. pSum value: %#04x \n", pSum));
#endif
				//DEBUG_PRINT(("[PE Psum] Commit. %#04x\n", pSum));
				//pSum = 0;
				nextStateMac = MAC_STATE_WAIT;
				//pSum = 0;
			}
		}
		else if (stateMac == MAC_STATE_LOAD_BIAS)
		{
#ifdef FULL_SYSTEM
			EMULATOR_PRINT(("[PE (%d, %d)] Load Bias.\n", idy, idx));
#else
			EMULATOR_PRINT(("[PE] Load Bias\n"));
#endif
			pSum = bias[(~regLoadSide) & 0x1];
			nextStateMac = MAC_STATE_WAIT;
		}


	//===================SWAP===========================
	//Take an extra iteration for swapping, otherwise Fmax is low
		if ( (stateActivation == ASSEMBLER_STATE_WAIT)
			&& (stateWeight == ASSEMBLER_STATE_WAIT)
			&& (stateMac == MAC_STATE_WAIT) )
		{
			nextStateWeight = (isLast[(regLoadSide) & 0x1] == TRUE) ? 
				ASSEMBLER_STATE_LOAD_BIAS : ASSEMBLER_STATE_LOAD_BITMASK;
			nextStateActivation = (isLast[(regLoadSide) & 0x1] == TRUE) ? 
				ASSEMBLER_STATE_LOAD_BIAS : ASSEMBLER_STATE_LOAD_BITMASK;
			nextStateMac = (isLast[(~regLoadSide) & 0x1] == TRUE) ?
				MAC_STATE_LOAD_BIAS: MAC_STATE_ALIGN;

			regLoadSide = ~regLoadSide;

#ifdef FULL_SYSTEM
				EMULATOR_PRINT(("[PE (%d, %d)] Swap. \n", idy, idx));
#else
				EMULATOR_PRINT(("[PE] Swap. \n"));
#endif
			//countActivation = 0;
			//countWeight = 0;

		}

		//================DEBUG==============================
		//if (debugCount < maxDebugCount)
		//{
		//	DEBUG_PRINT(("[PE] countWeight, %#03x\n", countWeight));
		//	DEBUG_PRINT(("[PE] countActivation: %#03x\n", countActivation));
		//	DEBUG_PRINT(("[PE] countOperands: %#03x\n", countOperands));
		//	DEBUG_PRINT(("[PE] indicesW: %#03x\n", indicesW));
		//	DEBUG_PRINT(("[PE] indicesA: %#03x\n", indicesA));
		//	debugCount++;
		//}
		
		//===================================================

		//================Next state update==================
		stateWeight = nextStateWeight;
		stateActivation = nextStateActivation;
		stateMac = nextStateMac;
		//===================================================
	} // while true
} // end of kernel
#else //SPARSE_SYSTEM

#define DENSE_PE_INSTRUCTION_BIAS_FROM_CH 0x0
#define DENSE_PE_INSTRUCTION_W_FROM_CH_A_FROM_CH_MAC 0X1
#define DENSE_PE_INSTRUCTION_W_FROM_R_A_FROM_CH_MAC 0X2
#define DENSE_PE_INSTRUCTION_W_FROM_CH_A_FROM_R_MAC 0X3
#define DENSE_PE_INSTRUCTION_COMMIT 0X4

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
#endif
	typedef unsigned char instruction_t;
	//====================registers===============
	t_transfer_block regActivationTB;
	t_transfer_block regWeightTB;
	t_accumulator pSum;
	uint1_t regIsLast;

	instruction_t currentInstruction = DENSE_PE_INSTRUCTION_BIAS_FROM_CH;

	while (1) {
		//Declare temp variables
		instruction_t tempInstruction = currentInstruction;
		bool readASuccess = false;
		bool readWSuccess = false;
		t_transferblock_local tempATBLocal, tempWTBLocal;

		t_simd_operand simdActivations;
		t_simd_operand simdWeights;

		bool performMAC = false;
		bool loadAFromCh = false;
		bool loadWFromCh = false;
		bool updateRegA = false;
		bool updateRegW = false;
		bool updateRegIsLast = false;

		//Handling reading from the W channel
		if ( (currentInstruction == DENSE_PE_INSTRUCTION_BIAS_FROM_CH)
			|| (currentInstruction == DENSE_PE_INSTRUCTION_W_FROM_CH_A_FROM_R_MAC)
			|| (currentInstruction == DENSE_PE_INSTRUCTION_W_FROM_CH_A_FROM_CH_MAC))
		{
			#ifdef FULL_SYSTEM
                tempWTBLocal = read_channel_nb_intel (
							channel_dpWeightInput[idy][idx],
							&readWSuccess
						);
			#else
				t_transferblock_local tempWTBLocal = read_channel_nb_intel (
							channel_dpWeightInput[0][0],
							&readWSuccess
						);
			#endif
		}

		//Handling reading from the A channel
		if ( (currentInstruction == DENSE_PE_INSTRUCTION_W_FROM_R_A_FROM_CH_MAC)
			|| (currentInstruction == DENSE_PE_INSTRUCTION_W_FROM_CH_A_FROM_CH_MAC))
		{
			#ifdef FULL_SYSTEM
                tempATBLocal = read_channel_nb_intel (
							channel_dpActivationInput[idy][idx],
							&readASuccess
						);
			#else
				t_transferblock_local tempATBLocal = read_channel_nb_intel (
							channel_dpActivationInput[0][0],
							&readASuccess
						);
			#endif
		}

		//Control signal generation and next state update;
		switch (currentInstruction) {

			case (DENSE_PE_INSTRUCTION_BIAS_FROM_CH):{
				if (readWSuccess) {
					pSum = transferBlock2Bias(tempWTBLocal.values);

#ifdef FULL_SYSTEM
			EMULATOR_PRINT(("[PE (%d, %d)] Load Bias.\n", idy, idx));
#else
			EMULATOR_PRINT(("[PE] Load Bias\n"));
#endif

					tempInstruction = DENSE_PE_INSTRUCTION_W_FROM_CH_A_FROM_CH_MAC;
				}
			} //DENSE_PE_INSTRUCTION_BIAS_FROM_CH
			break;
			case (DENSE_PE_INSTRUCTION_W_FROM_CH_A_FROM_CH_MAC):{
				if (readWSuccess && readASuccess) {
					performMAC = true;
					updateRegA = true;
					updateRegW = true;
					updateRegIsLast = true;

					if (tempWTBLocal.isLast == TRUE) {
						tempInstruction = DENSE_PE_INSTRUCTION_COMMIT;
					}
				}
				else if (readWSuccess) {
					performMAC = false;
					updateRegW = true;
					updateRegIsLast = true;

					tempInstruction = DENSE_PE_INSTRUCTION_W_FROM_R_A_FROM_CH_MAC;
				}
				else if (readASuccess) {
					performMAC = false;
					updateRegA = true;

					tempInstruction = DENSE_PE_INSTRUCTION_W_FROM_CH_A_FROM_R_MAC;
				}
			} // DENSE_PE_INSTRUCTION_W_FROM_CH_A_FROM_CH_MAC
			break;
			case (DENSE_PE_INSTRUCTION_W_FROM_R_A_FROM_CH_MAC):{
				if (readASuccess)
				{
					performMAC = true;
					updateRegA = true;

					if (regIsLast == TRUE) {
						tempInstruction = DENSE_PE_INSTRUCTION_COMMIT;
					}
					else
					{
						tempInstruction = DENSE_PE_INSTRUCTION_W_FROM_CH_A_FROM_CH_MAC;
					}
				}
				
			} //DENSE_PE_INSTRUCTION_W_FROM_R_A_FROM_CH_MAC
			break;
			case (DENSE_PE_INSTRUCTION_W_FROM_CH_A_FROM_R_MAC):{
				if (readWSuccess) {
					performMAC = true;
					updateRegW = true;
					updateRegIsLast = true;

					if (tempWTBLocal.isLast == TRUE) {
						tempInstruction = DENSE_PE_INSTRUCTION_COMMIT;
					}
					else
					{
						tempInstruction = DENSE_PE_INSTRUCTION_W_FROM_CH_A_FROM_CH_MAC;
					}
				}
			} //DENSE_PE_INSTRUCTION_W_FROM_CH_A_FROM_R_MAC
			break;
			case (DENSE_PE_INSTRUCTION_COMMIT): {
			} //DENSE_PE_INSTRUCTION_COMMIT
			break;
			default:
			break;
		} ////Control signal generation and next state update;

		//Select MAC operands
		if (currentInstruction == DENSE_PE_INSTRUCTION_W_FROM_CH_A_FROM_CH_MAC)
		{
			#pragma unroll
			for (unsigned char i=0; i<SIMD_SIZE; i++) {
				#pragma unroll
				for (unsigned char j=0; j<CLUSTER_SIZE; j++)
				{
					simdActivations.values[CLUSTER_SIZE*i + j] = tempATBLocal.values.values[i].cluster_values[j];
					simdWeights.values[CLUSTER_SIZE*i + j] = tempWTBLocal.values.values[i].cluster_values[j];
				}
			}
			
		}
		else if (currentInstruction == DENSE_PE_INSTRUCTION_W_FROM_R_A_FROM_CH_MAC)
		{
			#pragma unroll
			for (unsigned char i=0; i<SIMD_SIZE; i++) {
				#pragma unroll
				for (unsigned char j=0; j<CLUSTER_SIZE; j++)
				{
					simdActivations.values[CLUSTER_SIZE*i + j] = tempATBLocal.values.values[i].cluster_values[j];
					simdWeights.values[CLUSTER_SIZE*i + j] = regWeightTB.values[i].cluster_values[j];
				}
			}
		}
		else if (currentInstruction == DENSE_PE_INSTRUCTION_W_FROM_CH_A_FROM_R_MAC)
		{
			#pragma unroll
			for (unsigned char i=0; i<SIMD_SIZE; i++) {
				#pragma unroll
				for (unsigned char j=0; j<CLUSTER_SIZE; j++)
				{
					simdActivations.values[CLUSTER_SIZE*i + j] = regActivationTB.values[i].cluster_values[j];
					simdWeights.values[CLUSTER_SIZE*i + j] = tempWTBLocal.values.values[i].cluster_values[j];
				}
			}
		} //Select MAC Operands

		//Regs
		if (updateRegIsLast == true)
		{
			regIsLast = tempWTBLocal.isLast;
			// EMULATOR_PRINT(("[PE (%d, %d)] Update regIsLast to %d.\n", idy, idx, regIsLast));
		}

		if (updateRegA)
		{
			regActivationTB = tempATBLocal.values;
			// EMULATOR_PRINT(("[PE (%d, %d)] Read activation transfer block.\n\n", idy, idx));
		}

		if (updateRegW == true)
		{
			regWeightTB = tempWTBLocal.values;
// #ifdef FULL_SYSTEM
// 						EMULATOR_PRINT(("[PE (%d %d)] weightTransferBlock [0-4]: %#04x %#04x %#04x %#04x. IsLast=%d\n",
// 								idy, idx,
// 								tempWTBLocal.values.values[0].cluster_values[0] & 0xFF, 
// 								tempWTBLocal.values.values[0].cluster_values[1] & 0xFF,
// 								tempWTBLocal.values.values[1].cluster_values[0] & 0xFF,
// 								tempWTBLocal.values.values[1].cluster_values[1] & 0xFF,
// 								tempWTBLocal.isLast));
// #else
// 						EMULATOR_PRINT(("[PE] weightTransferBlock [0-4]: %#04x %#04x %#04x %#04x. IsLast=%d\n",
// 								tempWTBLocal.values.values[0].cluster_values[0] & 0xFF, 
// 								tempWTBLocal.values.values[0].cluster_values[1] & 0xFF,
// 								tempWTBLocal.values.values[1].cluster_values[0] & 0xFF,
// 								tempWTBLocal.values.values[1].cluster_values[1] & 0xFF,
// 								tempWTBLocal.isLast));
// #endif
		}

		//MAC
		if (performMAC == true)
		{
			t_accumulator tempPSum = madd(simdActivations, simdWeights);
			pSum += tempPSum;
		}

		//Write output
		if (currentInstruction == DENSE_PE_INSTRUCTION_COMMIT)
		{
			bool writeSuccess = false;
			#ifdef FULL_SYSTEM
				writeSuccess = write_channel_nb_intel(channel_peDrainOutput[idy][idx], pSum);
			#else
				writeSuccess = write_channel_nb_intel(channel_peDrainOutput[0][0], pSum);
			#endif

			if (writeSuccess)
			{
				//DEBUG_PRINT(("[MAC] Sending!\n"));
			#ifdef FULL_SYSTEM
				EMULATOR_PRINT(("[PE (%d, %d)] Commit. pSum value: %#04x \n", idy, idx, pSum));
			#else
				EMULATOR_PRINT(("[PE] Commit. pSum value: %#04x \n", pSum));
			#endif
				//DEBUG_PRINT(("[PE Psum] Commit. %#04x\n", pSum));
				//pSum = 0;
				tempInstruction = DENSE_PE_INSTRUCTION_BIAS_FROM_CH;
				//pSum = 0;
			}
		}

		//EMULATOR_PRINT(("[PE (%d, %d)] Instruction: %d\n", idy, idx, currentInstruction));
		currentInstruction = tempInstruction;
	} // while-loop

}
#endif //SPARSE_SYSTEM
#endif //PE_SYSTEM

