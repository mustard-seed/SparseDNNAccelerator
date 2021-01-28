#include "params.hpp"
#include "device_structures.hpp"
#include "channels.hpp"
#include "device_utils.hpp"
#include "ihc_apint.h"
#include "rtl_lib.hpp"
#include "prints.hpp"

#if defined(C5SOC)
#define VOLATILE volatile
#else
#define VOLATILE
#endif

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

__attribute__((max_global_work_dim(0)))
__kernel void kernelNoop ()
{
}

#ifdef MEMORY_READER
//TODO: reduce the number of activation/TB count input ports from 2 to 1
__attribute__((max_global_work_dim(0)))
__kernel void kernelIAMover (
		// Memory port for input activations
		VOLATILE __global const signed char* restrict pIA,

		//Memory port for transfer instructions
		VOLATILE __global const t_ia_mover_instruction* restrict pInstruction,
		//Number of transfer instructions
		unsigned int numInstruction,

		//Starting offset to read the instruction from
		unsigned int offsetInstruction
	)
{
	#pragma max_concurrency 1
	for (unsigned int iInst=0; iInst < numInstruction; iInst += 8)
	{
		t_ia_mover_instruction cacheInstruction[8];
		unsigned char numInstInTile = (iInst + 8) < numInstruction ? 
			8 : numInstruction - iInst;
		#pragma unroll 
		for (unsigned char i=0; i<8; i++)
		{
			cacheInstruction[i] = pInstruction[i+iInst+offsetInstruction];
		}

		unsigned char iInstTile = 0;
		while (iInstTile<numInstInTile)
		{
			//Read the instruction
			t_ia_mover_instruction inst = cacheInstruction[iInstTile];

			/*! Unpackethe concatenated fields of the instruction */
			//Number of compute columns that are active in this transfer
			unsigned char numActiveCols = inst.flagSyncCatInputArrangementCatSparseFlagCatDestinationCatNumActiveCols & 0x0F;
			//Flag for the transfer destignatiion. 1 for MISC channel, 0 for CONV PE array.
			t_flag destinationMisc = (inst.flagSyncCatInputArrangementCatSparseFlagCatDestinationCatNumActiveCols >> 0x04) & 0x01;
			//Flag for whether the input is sparse
			//t_flag sparseInput = (inst.flagSyncCatInputArrangementCatSparseFlagCatDestinationCatNumActiveCols >> 0x05) & 0x01;
			
			//Bit [6]: Input arrangment mode.
		    //  1'b0: One input tensor (e.g convolution, strided convolution)
	    	//	1'b1: Two input tensors, and interleave the two tensors per dramblock (e.g. eltwise addition)
			unsigned char inputArrangement = (inst.flagSyncCatInputArrangementCatSparseFlagCatDestinationCatNumActiveCols >> 0x06) & 0x01;
			
			t_flag flagWaitForSync = (inst.flagSyncCatInputArrangementCatSparseFlagCatDestinationCatNumActiveCols >> 0x07) & 0x01;

			unsigned char input0LeftShiftAmount = inst.inputShiftAmounts & 0x0F;
			unsigned char input1LeftShiftAmount = (inst.inputShiftAmounts >> 0x04) & 0x0F;
			//Amount of input padding needed on the four sides
			uint2_t tileLeftPadding = inst.concatPadding & 0x03;
			uint2_t tileRightPadding = (inst.concatPadding >> 0x02) & 0x03;
			uint2_t tileTopPadding = (inst.concatPadding >> 0x04) & 0x03;
			uint2_t tileBottomPadding = (inst.concatPadding >> 0x06) & 0x03;
			//Intial values for iterators that are created to handle strided convolution. SP: strided-padded
			uint4_t hInitSPIndex = inst.concatInitSPIndices & 0x0F;
			uint4_t vInitSPIndex = (inst.concatInitSPIndices >> 0x04) & 0x0F;

			//Width and height in the strided-padded domain
			uint4_t colSPSize = inst.concatSPSize & 0x0F;
			uint4_t rowSPSize = (inst.concatSPSize >> 0x04) & 0x0F;

			/*! Iterators for keeping track of current strip's position in the strided-padded unit.*/
			//uint4_t iColSPUnitIndex = hInitSPIndex;
			uint4_t iRowSPUnitIndex = vInitSPIndex;

			//Iterators of the column and row positions of the strip inside the tile
			//signed char iColInSPTile = 0;
			//signed char iRowInSPTile = 0;

			//Address offset contributions from row and column movements in the tile
			signed int offsetIADramBlockRow = 0;

			bool instructionProceed = true;
			//Synchornization with the OA mover
			if (flagWaitForSync == TRUE)
			{
				unsigned char token = read_channel_nb_intel(channel_activation_sync, &instructionProceed);
			}

			if (instructionProceed == true)
			{
				//iterate over IA tile height
				for (
						signed char iRowInSPTile=0;
						iRowInSPTile < ((unsigned char) inst.tileSPHeight);
						iRowInSPTile++
					)
				{
					uint4_t iColSPUnitIndex = hInitSPIndex;
					signed int offsetIADramBlockCol = 0;

					#pragma ii 1
					#pragma speculated_iterations 0
					for (
							signed char iColInSPTile=0;
							iColInSPTile < ((signed char) inst.tileSPWidth);
							iColInSPTile++
						)
					{
						/*!Setup the strip transfer parameters*/
						//Determine whether the strip consists of padding, or actual values from memory
						bool colIsDense = (iColInSPTile >= ((signed char) tileLeftPadding))
							&& (iColInSPTile < (inst.tileSPWidth - ((unsigned char) tileRightPadding)) )
							&& (iColSPUnitIndex == 0);
						bool rowIsDense = (iRowInSPTile >= ((signed char) tileTopPadding))
							&& (iRowInSPTile < (inst.tileSPHeight - ((unsigned char) tileBottomPadding)) )
							&& (iRowSPUnitIndex == 0);

						bool realStrip = colIsDense && rowIsDense;

						unsigned char numInputInterleavePerDramblock = (inputArrangement == 0x01) ? 0x02 : 0x01;

						{
							int addressIADramBlockDDR0 = 
								((t_int) inst.memBlockStart0) + offsetIADramBlockCol + offsetIADramBlockRow;

							//The second iter is special, used during elementwise addition only
							//hence the mixing of indices from 1 and 0
							int addressIADramBlockDDR1 = 
								((t_int) inst.memBlockStart1) + offsetIADramBlockCol + offsetIADramBlockRow;

							unsigned short numTBInStrip = (t_ushort) inst.numTBPerStrip;

							//dramBlockCount = ceil(numTBInStrip / WIDE_SIZE)
							//Compute the number of dram block transfers needed for the strip
							unsigned short dramBlockCount = (numInputInterleavePerDramblock == 0x01) ?
								 1 + ( (numTBInStrip-1) >> ACTIVATION_WIDE_SIZE_OFFSET )
								: (1 + ( (numTBInStrip-1) >> ACTIVATION_WIDE_SIZE_OFFSET )) << 1;
							//The actual number of transfer is one more than the number of DRAM block.
							//The extra block is in the beginning, and it contains routing information
							//as well as the number of TB count, which is required by the convolution PE array
							unsigned short numTransferActions = dramBlockCount;

							EMULATOR_PRINT(("[kernelIAMover] START strip transfer. "
										"offsetInstruction=%d, "
										"iInstTile+iInst=%d, "
										"tileSPWidthxTileSPHeight=%d, "
										"iRowInSPTile=%d, " 
				                        "iColInSPTile=%d, "
										"rowIsDense=%#03x, "
										"colIsDense=%#03x, "
										"inputArrangement=%#03x, "
										"numInputInterleavePerDramblock=%#03x, "
										"numTBInStrip=%d, "
										"numActiveCols=%d, "
										"addressIADramBlockDDR0=%#010x, "
										"addressIADramBlockDDR1=%#010x, "
										"leftShiftAmount=%#04x"
										"destinationMisc=%#03x\n",
										offsetInstruction,
										iInstTile+iInst,
										(unsigned int) inst.tileSPWidthxTileSPHeight, 
										iRowInSPTile,
										iColInSPTile,
										rowIsDense,
										colIsDense,
										inputArrangement,
										numInputInterleavePerDramblock,
										numTBInStrip,
										numActiveCols,
										(unsigned int) addressIADramBlockDDR0,
										(unsigned int) addressIADramBlockDDR1,
										(unsigned int) inst.inputShiftAmounts,
										(unsigned int) destinationMisc));

							unsigned char iterInputDramblockInterleave = 0x0;
							#pragma ii 1
							#pragma speculated_iterations 0
							for (unsigned short iterTransfer=0; iterTransfer<numTransferActions; iterTransfer++)
							{
								t_dram_block_ia_tagged iaBlock;

								iaBlock.colSPWidth = inst.columnSPWidth;
								iaBlock.colSPStride = inst.columnWidthStride;
								iaBlock.iColInSPTile = iColInSPTile;
								iaBlock.numTB = numTBInStrip;
								if (realStrip == true)
								{
									// iaBlock.dramBlock = (memRegion == 0x0) ? 
									// 	pIA1[addressIADramBlockDDR] : pIA2[addressIADramBlockDDR];
									
									int addressIADramBlockDDR = (iterInputDramblockInterleave == 0x0) ?
										addressIADramBlockDDR0 : addressIADramBlockDDR1;

									//Burst coalesced access
									#pragma unroll
									for (unsigned int i=0; i<ACTIVATION_BURST_SIZE_BYTE; i++)
									{
										iaBlock.dramBlock.values[i] = 
											pIA[addressIADramBlockDDR+i];
									}

									if (iterInputDramblockInterleave == 0x0)
									{
										addressIADramBlockDDR0 += ACTIVATION_BURST_SIZE_BYTE;
									}
									else
									{
										addressIADramBlockDDR1 += ACTIVATION_BURST_SIZE_BYTE;
									}
								}
								else  //Strip is padding
								{
									//Prepare a DRAM block with 0
									#pragma unroll
									for (unsigned int i=0; i<ACTIVATION_BURST_SIZE_BYTE; i++)
									{
										iaBlock.dramBlock.values[i] = 0x0;
									}
								}

								unsigned char isLastField = ((iterTransfer+1) == numTransferActions) ?
									0x80 : 0x00;
								unsigned char isMiscField = (destinationMisc == 0x1) ?
									0x40 : 0x00;
								iaBlock.route = isLastField | isMiscField | ((numActiveCols-1) & 0x3F);
								iaBlock.miscLeftShiftAmount = input0LeftShiftAmount;
								if (inputArrangement == 0x01)
								{
									if (iterInputDramblockInterleave == 0x00)
									{
										iaBlock.miscLeftShiftAmount = input0LeftShiftAmount;
									}
									else
									{
										iaBlock.miscLeftShiftAmount = input1LeftShiftAmount;
									}
									//Toggle between 0 and 1
									iterInputDramblockInterleave = (iterInputDramblockInterleave + 0x01) & 0x01;
								}

								write_channel_intel(channel_ia_wide[0], iaBlock);
							}

							EMULATOR_PRINT(("[kernelIAMover] FINISHED strip transfer.\n\n"));
						}

						

						/*! Loop carried variable updates*/
						//TODO: Double check the iColSPUnitIndex and iRowSPUnitIndex update conditions
						if ( (iColInSPTile >= ((signed char) tileLeftPadding)) 
							&& (iColInSPTile < (inst.tileSPWidth - ((unsigned char) tileRightPadding) - (unsigned char ) 1)) )
						{
							iColSPUnitIndex++;
							if (iColSPUnitIndex >= ((unsigned char) colSPSize))
							{
								iColSPUnitIndex = 0;
								offsetIADramBlockCol += ((t_int) inst.memBlockColStripStride);
							}
						}

					} //for over iColInSPTile

					if ( (iRowInSPTile >= ((signed char) tileTopPadding)) 
							&& (iRowInSPTile < (inst.tileSPHeight - ((unsigned char) tileBottomPadding)) - (unsigned char ) 1 ) )
					{
						iRowSPUnitIndex++;
						if (iRowSPUnitIndex >= ((unsigned char) rowSPSize))
						{
							iRowSPUnitIndex = 0;
							offsetIADramBlockRow += ((t_int) inst.memBlockRowStripStride);
						}
					}
				} // for over iRowInSPTile

				iInstTile++;
			} // if proceed
		} //while over iInstTile
	} // for over iInst
}

__attribute__((max_global_work_dim(0)))
__kernel void kernelWMover (
		//Memory port for instructions
		__global const t_weight_mover_instruction* restrict pInst,
		__global const t_weight_dram_block* restrict pW,
		__global const t_bias* restrict pBias,
		unsigned int numInstruction
	)
{
	#if defined(WMOVER_STREAM_CACHE)
			t_bias cacheBias[WEIGHT_MOVER_BIAS_CACHE_SIZE];
	#endif //WMOVER_STREAM_CACHE

	#if defined(WMOVER_WEIGHT_COALESCE_CACHE)
	t_weight_dram_block cacheFilter[KERNEL_CACHE_DEPTH];
	#endif

	for (unsigned int iInst=0; iInst<numInstruction; iInst++)
	{
		t_weight_mover_instruction inst = pInst[iInst];

		signed int addrWeightFilterBase = inst.memWeightStart;

		#if defined(WMOVER_STREAM_CACHE)
			/*
			 * pre-fetch the bias
			*/
			{
				signed int addrBias = inst.memBiasStart;
				for (unsigned short iFilterInGroup=0; iFilterInGroup<inst.numFiltersInGroup; iFilterInGroup++)
				{
					cacheBias[iFilterInGroup] = pBias[addrBias];
					addrBias++;
				}
			}
		#else //WMOVER_STREAM_CACHE
			signed int addrBias = inst.memBiasStart;
		#endif

		//Number of filters in the group
		unsigned short numActualFiltersInGroup = inst.numFiltersInGroup;

		//Number of filters to be sent to the array
		//Include padding
		unsigned short numFiltersSentToArray =
			(1 + ((numActualFiltersInGroup - 1) >> DIVIDE_BY_PE_ROWS_PER_GROUP_SHIFT)) << DIVIDE_BY_PE_ROWS_PER_GROUP_SHIFT;


		//Number of weight blocks seen by the PEs
		//in each filter
		//For SpW weight block, this is proporational
		//to the number NZ clusters per pruning range.
		unsigned short numTransferBlockInFilter = inst.numTBPerFilter;

		unsigned short numDramBlockInActualFilter = 
			((numTransferBlockInFilter-1) >> WEIGHT_WIDE_SIZE_OFFSET) + 1;

		unsigned char iFilterRowDestination = 0;
		for (
				unsigned short iFiltersSentToArray=0;
				iFiltersSentToArray < numFiltersSentToArray;
				iFiltersSentToArray++
			)
		{
			//Test if this is a real filter
			t_flag isRealFilter = (iFiltersSentToArray < numActualFiltersInGroup) ?
				TRUE : FALSE;

			//Load the bias
			#if defined(WMOVER_STREAM_CACHE)
				t_bias bias = (isRealFilter == TRUE) ?
					cacheBias[iFiltersSentToArray] : 0x0;
			#else
				t_bias bias = (isRealFilter == TRUE) ?
					pBias[addrBias] : 0x0;
			#endif


			//Setup the control bias
			t_filter_streamer_control control;
			control.numOutputsXNumTransferBlocks = (unsigned int) inst.filterReuse * (unsigned int) numTransferBlockInFilter;
			control.bias = bias;
			control.numTransferBlocks = numTransferBlockInFilter;
			control.flagIsReal = (isRealFilter == TRUE) ? TRUE : FALSE;
			control.maxPeCols = (inst.numActivePeCols - 1);
			#if defined(SPW_SYSTEM)
			control.numNZClustersPerPruneRange = inst.numNZClustersPerPruneRange;
			#endif

			t_weight_dram_block dramControl = filterStreamerControl2dramBlock(control);


			//Transfer to the PE array
			signed int addrDramBlock = addrWeightFilterBase;

			unsigned short numDramBlockInFilter = (isRealFilter == TRUE) ?
				numDramBlockInActualFilter : 0x0;


			EMULATOR_PRINT(("[kernelWMover] START filter transfer. "
						"iInst=%d, "
						"iFiltersSentToArray=%d, " 
                        "numFiltersSentToArray=%d, "
						"numActualFiltersInGroup=%d, "
						"num. active PE cols=%d, "
						"num. filter reuse=%d, "
						"bias=%#04x, "
						"numTransferBlocks=%d\n\n",
						iInst, 
						(unsigned int) iFiltersSentToArray,
						(unsigned int) numFiltersSentToArray,
						(unsigned int) numActualFiltersInGroup,
						inst.numActivePeCols,
						inst.filterReuse,
						bias,
						(unsigned int) numTransferBlockInFilter));


			#if defined(WMOVER_WEIGHT_COALESCE_CACHE)
				/**
				 * Input dram block coalescing
				 */
				for (unsigned int iDramAccessCount=0; iDramAccessCount<numDramBlockInFilter; iDramAccessCount += WMOVER_FILTER_DRAM_BLOCK_ACCESS_UNROLL_FACTOR)
				{
					#pragma unroll
					for (unsigned int i=0; i<WMOVER_FILTER_DRAM_BLOCK_ACCESS_UNROLL_FACTOR; i++)
					{
						cacheFilter[iDramAccessCount+i] = pW[addrDramBlock + i];
					}
					addrDramBlock += WMOVER_FILTER_DRAM_BLOCK_ACCESS_UNROLL_FACTOR;
				}

				unsigned short iFilterCacheCount = 0;
			#endif //WMOVER_WEIGHT_COALESCE_CACHE

			//one extra for filter stream control
			#pragma ii 1
			#pragma speculated_iterations 0
			for (unsigned short iTransmitCount=0; iTransmitCount<=numDramBlockInFilter; iTransmitCount++)
			{
				t_weight_dram_block block;
				if (iTransmitCount == 0) 
				{
					block = dramControl;
				}
				else
				{
					#if !defined(WMOVER_WEIGHT_COALESCE_CACHE)
						block = pW[addrDramBlock];
						addrDramBlock++;
					#else
						block = cacheFilter[iFilterCacheCount];
						iFilterCacheCount++;
					#endif
				}

				t_dram_block_w_tagged taggedBlock;
				taggedBlock.dramBlock = block;
				taggedBlock.destinationRow = iFilterRowDestination;

				write_channel_intel(channel_weight_wide[0], taggedBlock);
			} // for over iTransmitCount


			
			EMULATOR_PRINT(("[kernelWMover] FINISHED filter transfer.\n"));

			if (isRealFilter == TRUE)
			{
				addrWeightFilterBase += inst.memWeightFilterStride;
				#if !defined(WMOVER_STREAM_CACHE)
					addrBias++;
				#endif
			}

			//Update the filter row destination tracker
			iFilterRowDestination++;
			if (iFilterRowDestination == PE_ROWS)
			{
				iFilterRowDestination = 0X0;
			}

		} //for over iFiltersSentToArray
	}  //for loop over instructions
}

#endif //MEMORY_READER

#ifdef IA_MEMORY
#define IA_BUFFER_WRITE_STATE_DECODE 0x0
#define IA_BUFFER_WRITE_STATE_ACCESS 0x1
#define IA_BUFFER_WRITE_STATE_UPDATE_STRIP 0x2

#define IA_BUFFER_READ_STATE_DECODE 0x0
#define IA_BUFFER_READ_STATE_ACCESS 0x1
// #define IA_BUFFER_READ_STATE_UPDATE_STRIP 0x2

#define IA_BUFFER_INSTRUCTION_STATE_DECODE 0x0
#define IA_BUFFER_INSTRUCTION_STATE_SEND_TO_READER 0x1
#define IA_BUFFER_INSTRUCTION_STATE_SEND_TO_WRITER 0x2
#define IA_BUFFER_INSTRUCTION_STATE_WRITER_STALL 0x3

#define IA_BUFFER_READ_STRIP_UPDATE_HORIZONTAL 0x0
#define IA_BUFFER_READ_STRIP_UPDATE_DONE 0x1

#if defined(EMULATOR) && defined(EMUPRINT)
typedef unsigned char t_ia_buffer_w_state;
typedef unsigned char t_ia_buffer_r_state;
typedef unsigned char t_ia_buffer_d_state;
#else
typedef uint2_t t_ia_buffer_w_state;
typedef uint1_t t_ia_buffer_r_state;
typedef uint2_t t_ia_buffer_d_state;
#endif

/**
 * Helper data bundle for accessing the dram_block cache in IA buffers
 */
typedef struct __attribute__((packed)) 
{
	unsigned short addressBase;
	unsigned short colStride;
	//unsigned short rowStride;
	unsigned short colContribution;
	//unsigned short rowContribution;
} t_ia_data_buffer_access_info;


/**
 * Helper data struct for accessing the position in tile
 */
typedef struct __attribute__((packed)) 
{
	//unsigned char iRow;
	unsigned char iCol;
	//unsigned char numStripsRow;
	unsigned char numStripsCol;
} t_ia_tile_access_info;

/**
 * IA Buffer writer state and variables encapsulation
 */
typedef struct __attribute__((packed)) {

	t_ia_data_buffer_access_info iaBlockInfo;
	t_ia_tile_access_info tileInfo;

	//Number of dram blocks in each strip during buffer loading, 
	unsigned short numIAAccess;

	//Iterator for the buffer access count
	unsigned short iterAccess;

	//Which cache bank to write to
	t_flag accessBank;
} t_ia_buffer_write_registers;

/**
 * @brief      Gets the ia buffer writer interface outputs
 *
 * @param[in]  currentState           The current state
 * @param      pOutAcceptInstruction  Pointer to the flag indicating whether the writer is ready to receive new instructions
 * @param      pOutAcceptData         Pointer to the flag indicating whether the writer is ready to accept new dram block
 * @param[in]  colId                  The col identifier
 */
void getIABufferWriterOutput (
		//Inputs
		t_ia_buffer_w_state currentState,

		//Outputs
		t_flag* pOutAcceptInstruction,
		t_flag* pOutAcceptData,

		//Auxillary,
		int colID
	);

/**
 * @brief      Update the ia buffer writer state and variable
 *
 * @param[in]  control                    Incoming instruction
 * @param[in]  validControl               Flag that indicates whether the incoming instruction is valid
 * @param[in]  dramBlock                  Incoming dram block
 * @param[in]  validDramBlock             Flag that indicates whether the incoming dram block is valid
 * @param[in]  numTBPerStrip              Number of TB count per strip. Available for dense architecture only
 * @param      cacheIABlocks              IA dram block cache
 * @param      pCurrentState              The current state
 * @param      pCurrentRegisters          The current registers
 * @param[in]  colID                      The col id
 */
void updateIABufferWriter (
		//Inputs
		t_input_buffer_tile_buffer_packet control,
		t_flag validControl,

		t_dram_block_ia_to_pe taggedDramBlock,
		t_flag validDramBlock,

		//Modified buffer and buffers
		unsigned short numTBPerStrip[2],
		t_activation_dram_block cacheIABlocks [2][IA_CACHE_DEPTH],
		t_ia_buffer_w_state* pCurrentState,
		t_ia_buffer_write_registers* pCurrentRegisters,

		//Auxillary
		int colID
	);

/**
 * IA Buffer reader state and variables encapsulation
 */
typedef struct __attribute__((packed)) {

	t_ia_data_buffer_access_info iaBlockInfo;
	t_ia_tile_access_info tileInfo;

	//Number of dram blocks in each strip during buffer loading, 
	unsigned short numTBPerStrip;

	//Iterator for the buffer access count (in each strip?)
	unsigned short iterAccess;

	//Maximum convolution PE row group that will be affected by the buffer read operation
	unsigned char maxPeRowGroupID;

	//Instruction on how the dram cache pointer should be updated in update_strip state
	//Also used to indicate whether the current strip is the last one to be streamed
	// uint1_t stripUpdateMode;

	//Whether the IA buffer reader is streaming the last row of a convolution window to the PEs
	uint1_t flagIsLastRow;
	

	//Which cache bank to read from
	t_flag accessBank;
} t_ia_buffer_read_registers;

/**
 * @brief      Gets the ia buffer reader interface outputs.
 *
 * @param[in]  currentState               The current state
 * @param[in]  currentRegisters           Current indices, etc
 * @param[in]  numTBPerStrip              Number of TB per strip (Dense architecture only)
 * @param      cacheIABlocks              Number of TB per strip (dense architecture only)
 * @param      pOutAcceptInstruction      Pointer to the flag indicating that this module can accept new instruciton
 * @param      pTaggedBlock               Pointer to the transfer block fetchted from the IA cache
 * @param      pSendTransferBlock         Pointer to the flag indicating that the transfer block is valid
 * @param[in]  colId                      The col identifier
 */
void getIABufferReaderOutput (
		//Inputs
		t_ia_buffer_r_state currentState,
		t_ia_buffer_read_registers currentRegisters,

		//Buffers to read from
		unsigned short numTBPerStrip[],
		t_activation_dram_block cacheIABlocks [2][IA_CACHE_DEPTH],

		//Outputs
		t_flag* pOutAcceptInstruction,
		
		t_pe_a_block* pTaggedBlock,
		t_flag* pSendTransferBlock,

		//Auxillary,
		int colID
	);

/**
 * @brief      Updates the state and variables of the IA buffer reader
 *
 * @param[in]  control                  Incoming control packet
 * @param[in]  validControl             Flag that indicates the incoming control packet is valid
 * @param[in]  writeSuccessTaggedBlock  Flag indicating whether the transfer block on the PE array interface has been successfully sent
 * @param      numTBPerStrip              The number tb per strip
 * @param      pCurrentState            Current state
 * @param      pCurrentRegisters        Current variable values
 * @param[in]  colID                    The col id
 */
void updateIABufferReader (
		//Inputs
		t_input_buffer_tile_buffer_packet control,
		t_flag validControl,

		t_flag writeSuccessTaggedBlock,

		//Buffers to read from
		unsigned short numTBPerStrip[],

		//Modified buffer and buffers
		t_ia_buffer_r_state* pCurrentState,
		t_ia_buffer_read_registers* pCurrentRegisters,

		//Auxillary
		int colID
	);

void getIABufferDispatcherOutput (
		//Current state
		t_ia_buffer_d_state currentState,
		//sync,
		bool unblockReader,
		bool unblockWriter,

		//Instruction buffer
		t_input_buffer_tile_buffer_packet controlBuffer,

		//instruction channel interface
		t_flag* pOutReadyForInstruction,

		//accessor interface
		t_flag* pOutValidReaderInstruction,
		t_flag* pOutValidWriterInstruction,
		t_input_buffer_tile_buffer_packet* pOutReaderControl,
		t_input_buffer_tile_buffer_packet* pOutWriterControl,

		//Auxillary
		int colID
	);

void updateIABufferDispatcher (
		//instruction channel interface
		t_flag inInstructionValid,
		t_input_buffer_tile_buffer_packet inControl,

		//accessor interface
		t_flag inReaderReady,
		t_flag inWriterReady,

		//context
		t_ia_buffer_d_state* pState,
		t_input_buffer_tile_buffer_packet* pControlBuffer,

		//sync,
		bool unblockReader,
		bool unblockWriter,

		//Auxillary
		int colID
	);

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelIABuffer ()
{
	#if defined(EMULATOR) && defined(EMUPRINT)
		int iReceived = 0;
		int iSent = 0;
	#endif
	int colID = get_compute_id(0);

	t_activation_dram_block cacheIABlocks [2][IA_CACHE_DEPTH] __attribute__((bankwidth(ACTIVATION_BURST_SIZE_BYTE)));

	//Number of activation transfer blocks per 1x1 strip
	t_streamblock_address numTBPerStrip [2];

	/**
	 * Writer state and registers
	 */
	t_ia_buffer_write_registers regWriterContext = {
			.iaBlockInfo={
				.addressBase = 0,
				.colStride = 0,
				.colContribution = 0
			},
			.tileInfo={
				.iCol = 0,
				.numStripsCol = 0
			},
			.numIAAccess=0,
			.iterAccess=0,
			.accessBank = 0
		};
	t_ia_buffer_w_state regWriterState = IA_BUFFER_WRITE_STATE_DECODE;

	/**
	 * Reader state and registers
	 */
	t_ia_buffer_read_registers regReaderContext = {
			.iaBlockInfo={
				.addressBase = 0,
				.colStride = 0,
				.colContribution = 0
			},
			.tileInfo={
				.iCol = 0,
				.numStripsCol = 0
			},
			.numTBPerStrip = 0,
			.iterAccess=0,
			.accessBank = 0,
			.maxPeRowGroupID = 0,
			// .stripUpdateMode = 0,
			.flagIsLastRow = FALSE
		};
	t_ia_buffer_r_state regReaderState = IA_BUFFER_READ_STATE_DECODE;


	/**
	 * Dispatcher state and registers
	 */
	t_input_buffer_tile_buffer_packet regDispatcherInstructionBuffer;
	t_ia_buffer_d_state regDispatcherState = IA_BUFFER_INSTRUCTION_STATE_DECODE;

	#pragma ii 1
	#pragma speculated_iterations 0
	while (true)
	{
		/**
		 * instruction channel <====> dispatcher interface
		 */
		t_flag dispatcherReadyForInstruction = FALSE;
		t_flag dispatcherInstructionValid = FALSE;
		t_input_buffer_tile_buffer_packet dispatcherNewInstruction;

		/**
		 * IA dram block channel <=====> writer interface
		 */
		t_flag writerReadyForBlock = FALSE;
		t_flag writerBlockValid = FALSE;
		t_dram_block_ia_to_pe writerNewBlock;

		/**
		 * PE transfer block channel <===> reader interface
		 */
		t_flag readerBlockValid = FALSE;
		t_flag readerBlockSent = FALSE;
		t_pe_a_block readerTB;

		/**
		 * dispatcher <===> writer interface
		 */
		t_flag writerReadyForInstruction = FALSE;
		t_flag writerInstructionValid = FALSE;
		t_input_buffer_tile_buffer_packet writerNewInstruction;

		/**
		 * dispatcher <===> reader interface
		 */
		t_flag readerReadyForInstruction = FALSE;
		t_flag readerInstructionValid = FALSE;
		t_input_buffer_tile_buffer_packet readerNewInstruction;

		/**
		 * Reader-writer synchornization
		 */
		bool unblockReader = 
				(regWriterState == IA_BUFFER_WRITE_STATE_DECODE) 
				|| (regWriterContext.accessBank != (regDispatcherInstructionBuffer.controlBits & 0x01));

		bool unblockWriter = 
				(regReaderState == IA_BUFFER_READ_STATE_DECODE) 
				|| (regReaderContext.accessBank != (regDispatcherInstructionBuffer.controlBits & 0x01));
		/**
		 * Derive current interface outputs from the modules
		 */
		getIABufferWriterOutput (
			regWriterState,

			&writerReadyForInstruction,
			&writerReadyForBlock,

			colID
			);

		getIABufferReaderOutput (
			regReaderState,
			regReaderContext,

			numTBPerStrip,
			cacheIABlocks,

			&readerReadyForInstruction,

			&readerTB,
			&readerBlockValid,

			colID
			);

		getIABufferDispatcherOutput (
			regDispatcherState,
			unblockReader,
			unblockWriter,

			regDispatcherInstructionBuffer,

			&dispatcherReadyForInstruction,

			&readerInstructionValid,
			&writerInstructionValid,
			&readerNewInstruction,
			&writerNewInstruction,

			colID
			);

		/**
		 * Perform channel access
		 */
		if (dispatcherReadyForInstruction == TRUE)
		{
			bool success = false;
			dispatcherNewInstruction = read_channel_nb_intel(channel_control_to_ia_buffer_local[colID], &success);
			if (success == true)
			{
				dispatcherInstructionValid = TRUE;
			}
		}

		if (writerReadyForBlock == TRUE)
		{
			bool success = false;
			writerNewBlock = read_channel_nb_intel(channel_ia_wide_local[colID], &success);
			if (success == true)
			{
				writerBlockValid = TRUE;
				#if defined(EMULATOR) && defined(EMUPRINT)
					EMULATOR_PRINT(("[kernelIABuffer %d] RECEIVED dram block %d. TB[0-3]: %#04x %#04x %#04x %#04x \n\n",
					colID,
					iReceived
					,(unsigned int) writerNewBlock.dramBlock.values[0]
					,(unsigned int) writerNewBlock.dramBlock.values[1]
					,(unsigned int) writerNewBlock.dramBlock.values[2]
					,(unsigned int) writerNewBlock.dramBlock.values[3]
					));
					iReceived++;
				#endif
			}
		}

		if (readerBlockValid == TRUE)
		{
			bool success = false;
			success = write_channel_nb_intel(channel_activation[0][colID], readerTB);
			if (success == true)
			{
				readerBlockSent = TRUE;

				EMULATOR_PRINT(("[kernelIABuffer %d] Sent TB %d / %d, "
							"bank=%d, "
							"addrBase=%d, "
							"colContribution=%d, "
							"TB index=%d\n"
							"data[0]=%#04x, "
							"data[1]=%#04x, "
							"data[2]=%#04x, "
							"data[3]=%#04x\n",
							colID, regReaderContext.iterAccess, regReaderContext.numTBPerStrip,
							(unsigned int)(regReaderContext.accessBank),
							(unsigned int)(regReaderContext.iaBlockInfo.addressBase),
							(unsigned int)(regReaderContext.iaBlockInfo.colContribution),
							(unsigned int)((regReaderContext.iterAccess) & ACTIVATION_WIDE_SIZE_REMAINDER_MASK)
							,readerTB.values[0]
							,readerTB.values[1]
							,readerTB.values[2]
							,readerTB.values[3]
							));

				#if defined(EMULATOR) && defined(EMUPRINT)
					EMULATOR_PRINT(("[kernelIABuffer %d] Sent iSent=%d\n",
					colID, iSent
					));
					// if ((regReaderContext.numTBPerStrip == 1) && (readerTB.values.values[0] != 0))
					// {
					// 	EMULATOR_PRINT(("[kernelIABuffer %d] ERROR. iSent=%d " 
					// 		"Number of TB in strip is 1, but bitmask is not 1.\n",
					// 		colID, iSent));
					// }
					// //TODO: the following condition only for shallow tensor input.
					// else if ((regReaderContext.numTBPerStrip > 1) && (regReaderContext.iterAccess == 0) && (readerTB.values.values[0] == 0))
					// {
					// 	EMULATOR_PRINT(("[kernelIABuffer %d] ERROR. iSent=%d " 
					// 		"Number of TB in strip is greater than 1, but bitmask is 0.\n",
					// 		colID, iSent));
					// }
					iSent++;
				#endif
			}
		}


		/**
		 * Update module states
		 */
		updateIABufferWriter (
			writerNewInstruction,
			writerInstructionValid,

			writerNewBlock,
			writerBlockValid,

			numTBPerStrip,	
			cacheIABlocks,
			&regWriterState,
			&regWriterContext,

			colID
			);

		updateIABufferReader (
			readerNewInstruction,
			readerInstructionValid,

			readerBlockSent,

			numTBPerStrip,

			&regReaderState,
			&regReaderContext,

			colID
			);

		updateIABufferDispatcher (
			dispatcherInstructionValid,
			dispatcherNewInstruction,

			readerReadyForInstruction,
			writerReadyForInstruction,

			&regDispatcherState,
			&regDispatcherInstructionBuffer,

			unblockReader,
			unblockWriter,

			colID
			);
	}
}

void getIABufferWriterOutput (
		//Inputs
		t_ia_buffer_w_state currentState,

		//Outputs
		t_flag* pOutAcceptInstruction,
		t_flag* pOutAcceptData,

		//Auxillary,
		int colID
	)
{
	//Default values:
	*pOutAcceptInstruction = FALSE;
	*pOutAcceptData = FALSE;


	//Driver for the instruction interface's READY signal
	if (currentState == IA_BUFFER_WRITE_STATE_DECODE)
	{
		*pOutAcceptInstruction = TRUE;
	}

	//Driver for the dram block interface's READY signal
	if 	(
			(currentState == IA_BUFFER_WRITE_STATE_ACCESS) 
		)
	{
		*pOutAcceptData = TRUE;
	}
}

void updateIABufferWriter (
		//Inputs
		t_input_buffer_tile_buffer_packet control,
		t_flag validControl,

		t_dram_block_ia_to_pe taggedDramBlock,
		t_flag validDramBlock,

		//Modified buffer and buffers
		unsigned short numTBPerStrip[],
		t_activation_dram_block cacheIABlocks [2][IA_CACHE_DEPTH],
		t_ia_buffer_w_state* pCurrentState,
		t_ia_buffer_write_registers* pCurrentRegisters,

		//Auxillary
		int colID
	)
{
	switch (*pCurrentState) {
		case IA_BUFFER_WRITE_STATE_DECODE: {
			if (validControl == true) {
				/**
				 * Set registers
				 */
				pCurrentRegisters->iaBlockInfo.addressBase = control.iaDramBlockAddressBase;
				pCurrentRegisters->iaBlockInfo.colStride = control.iaDramBlockColStride;
				//pCurrentRegisters->iaBlockInfo.rowStride = control.iaDramBlockRowStride;
				pCurrentRegisters->iaBlockInfo.colContribution = 0;
				//pCurrentRegisters->iaBlockInfo.rowContribution = 0;

				//pCurrentRegisters->tileInfo.iRow = 0;
				pCurrentRegisters->tileInfo.iCol = 0;
				/*
					Number of strips in WRITER should be interpreted as the total number of strips to arrive.
				*/
				pCurrentRegisters->tileInfo.numStripsCol = control.numStripsCol;
				//pCurrentRegisters->tileInfo.numStripsRow = control.numStripsRow;

				pCurrentRegisters->accessBank = control.controlBits & 0x01;
				pCurrentRegisters->iterAccess = 0;
				pCurrentRegisters->numIAAccess = 0;

				/**
				 * Update state
				 */
				*pCurrentState = IA_BUFFER_WRITE_STATE_ACCESS;
				EMULATOR_PRINT(("[kernelIABuffer Writer %d] START processing instruction. "
						"iaDramBlockAddressBase (in terms of dram block) = %#010x, "
						"iaDramBlockColStride (in terms of dram block) = %#010x, "
						"numStripsCol=%d, "
						"accessBank=%#04x\n\n",
						colID, 
						control.iaDramBlockAddressBase,
						control.iaDramBlockColStride,
						(unsigned int) control.numStripsCol,
						(unsigned int) pCurrentRegisters->accessBank));
				
			} // if validControl == TRUE
		}
		break; //IA_BUFFER_WRITE_STATE_DECODE

		case IA_BUFFER_WRITE_STATE_ACCESS: {
			/**
			 * Write incoming dram block into the cache, and update the counters
			 */
			if (validDramBlock == TRUE)
			{
				t_streamblock_address numIATransferBlocks = taggedDramBlock.numTB;
				t_activation_dram_block dramBlock = taggedDramBlock.dramBlock;
				t_flag flagIsLastInStrip = (taggedDramBlock.route >> 7) & 0x01;

				numTBPerStrip[(pCurrentRegisters->accessBank) & 0x01] = numIATransferBlocks;

				cacheIABlocks[(pCurrentRegisters->accessBank) & 0x01]
					[(pCurrentRegisters->iterAccess) 
						+ (pCurrentRegisters->iaBlockInfo.addressBase)
						+ (pCurrentRegisters->iaBlockInfo.colContribution)]
					= dramBlock;

				EMULATOR_PRINT(("[kernelIABuffer Writer %d] Writing new dram block to bank %d, "
							"iterAccess=%d, "
							"addrBase=%d, "
							"colContribution=%d\n, "
							"data[0]=%#04x, "
							"data[1]=%#04x, "
							"data[2]=%#04x, "
							"data[3]=%#04x\n",
							colID, (unsigned char) (pCurrentRegisters->accessBank),
							(unsigned int)(pCurrentRegisters->iterAccess),
							(unsigned int)(pCurrentRegisters->iaBlockInfo.addressBase),
							(unsigned int)(pCurrentRegisters->iaBlockInfo.colContribution),
							(unsigned int)dramBlock.values[0],
							(unsigned int)dramBlock.values[1],
							(unsigned int)dramBlock.values[2],
							(unsigned int)dramBlock.values[3]
							));

				pCurrentRegisters->iterAccess += 0x1;

				// EMULATOR_PRINT(("[kernelIABuffer Writer %d] Read one dram block.\n",
				// 			colID));

				if (flagIsLastInStrip == TRUE)
				{
					*pCurrentState = IA_BUFFER_WRITE_STATE_UPDATE_STRIP;
				}
			}
			
		}
		break; //IA_BUFFER_WRITE_STATE_ACCESS

		case IA_BUFFER_WRITE_STATE_UPDATE_STRIP: {
			/**
			 * Increment tile counter, cache counters
			 */
			pCurrentRegisters->iterAccess = 0;
			pCurrentRegisters->tileInfo.iCol += 0x1;
			pCurrentRegisters->iaBlockInfo.colContribution += pCurrentRegisters->iaBlockInfo.colStride;

			*pCurrentState = IA_BUFFER_WRITE_STATE_ACCESS;
			EMULATOR_PRINT(("[kernelIABuffer WRITER %d] Strip update.\n", colID));

			if (pCurrentRegisters->tileInfo.iCol == pCurrentRegisters->tileInfo.numStripsCol)
			{
				*pCurrentState = IA_BUFFER_WRITE_STATE_DECODE;
				EMULATOR_PRINT(("[kernelIABuffer WRITER %d] FINISHED processing instruction.\n\n", colID));
			}
		}
		break; //IA_BUFFER_WRITE_STATE_UPDATE_STRIP

		default: {
			// Keep the status-quo
		}
	} //end of state switch
} //updateIABufferWriter

void getIABufferReaderOutput (
		//Inputs
		t_ia_buffer_r_state currentState,
		t_ia_buffer_read_registers currentRegisters,

		//Buffers to read from
		unsigned short numTBPerStrip[],
		t_activation_dram_block cacheIABlocks [2][IA_CACHE_DEPTH],

		//Outputs
		t_flag* pOutAcceptInstruction,
		
		t_pe_a_block* pTaggedBlock,
		t_flag* pSendTransferBlock,

		//Auxillary,
		int colID
	)
{
	if (currentState == IA_BUFFER_READ_STATE_DECODE)
	{
		*pOutAcceptInstruction = TRUE;
	}

	if (currentState == IA_BUFFER_READ_STATE_ACCESS)
	{
		*pSendTransferBlock = TRUE;
		
		t_activation_dram_block dramBlock = cacheIABlocks[(currentRegisters.accessBank) & 0x01]
			[currentRegisters.iaBlockInfo.addressBase 
				+ currentRegisters.iaBlockInfo.colContribution 
				+ ((unsigned short)(currentRegisters.iterAccess >> ACTIVATION_WIDE_SIZE_OFFSET))];	

		//TODO: Change this
		// unsigned char isLastTemp =  (
		// 	((currentRegisters.iterAccess + 1) == currentRegisters.numTBPerStrip) 
		// 	&& (currentRegisters.stripUpdateMode == IA_BUFFER_READ_STRIP_UPDATE_DONE) 
		// 	&& (currentRegisters.flagIsLastRow == TRUE))
		// 	? TRUE : FALSE;

		unsigned char idxTBInDramBlock = (currentRegisters.iterAccess) & ACTIVATION_WIDE_SIZE_REMAINDER_MASK;
		#pragma unroll PE_ACTIVATION_BLOCK_SIZE_IN_WORD
		for (int i=0; i<PE_ACTIVATION_BLOCK_SIZE_IN_WORD; i++)
		{
			unsigned char idxValueInDramBlock = (idxTBInDramBlock << PE_ACTIVATION_BLOCK_SIZE_IN_WORD_OFFSET) + i;
			(*pTaggedBlock).values[i] = dramBlock.values[idxValueInDramBlock];

		}

		// setMaxTransferID(pTaggedBlock, currentRegisters.maxPeRowID);
		// setIsLast(pTaggedBlock, isLastTemp);
		(*pTaggedBlock).maxTransportID = currentRegisters.maxPeRowGroupID;
	}
}

void updateIABufferReader (
		//Inputs
		t_input_buffer_tile_buffer_packet control,
		t_flag validControl,

		t_flag writeSuccessTaggedBlock,

		//Buffers to read from
		unsigned short numTBPerStrip[],

		//Modified buffer and buffers
		t_ia_buffer_r_state* pCurrentState,
		t_ia_buffer_read_registers* pCurrentRegisters,

		//Auxillary
		int colID
	)
{
	switch (*pCurrentState) {
		case IA_BUFFER_READ_STATE_DECODE: {
			if (validControl == true) 
			{
				/**
				 * Set registers
				 */
				pCurrentRegisters->iaBlockInfo.addressBase = control.iaDramBlockAddressBase;
				pCurrentRegisters->iaBlockInfo.colStride = control.iaDramBlockColStride;
				pCurrentRegisters->iaBlockInfo.colContribution = 0;

				pCurrentRegisters->tileInfo.iCol = 0;
				pCurrentRegisters->tileInfo.numStripsCol = control.numStripsCol;

				pCurrentRegisters->accessBank = control.controlBits & 0x01;
				pCurrentRegisters->iterAccess = 0;

				pCurrentRegisters->maxPeRowGroupID = control.maxPeRowGroupID;
				pCurrentRegisters->flagIsLastRow = control.flagIsLastRow & 0x01;

				/**
				 * Update state
				 */
				{
					*pCurrentState = IA_BUFFER_READ_STATE_ACCESS;
					EMULATOR_PRINT(("[kernelIABuffer Reader %d] START processing instruction. "
						"iaDramBlockAddressBase=%#010x, "
						"iaDramBlockColStride=%#010x, "
						"numStripsCol=%d, "
						"accessBank=%#04x\n\n",
						colID, 
						control.iaDramBlockAddressBase,
						control.iaDramBlockColStride,
						(unsigned int) control.numStripsCol,
						(unsigned int) pCurrentRegisters->accessBank));

					pCurrentRegisters->numTBPerStrip = numTBPerStrip [(pCurrentRegisters->accessBank) & 0x01]; 

					/*
					 * Reset strip counters
					*/
					pCurrentRegisters->iterAccess = 0;

					/**
					 * increment the tile pointer and TB count pointers in advance,
					 * obtain the strip update mode
					 */
					// pCurrentRegisters->stripUpdateMode = IA_BUFFER_READ_STRIP_UPDATE_HORIZONTAL;
					// if ((unsigned char ) 1 == pCurrentRegisters->tileInfo.numStripsCol)
					// {
					// 	pCurrentRegisters->stripUpdateMode = IA_BUFFER_READ_STRIP_UPDATE_DONE;
					// }
				}
					
			} // if validControl == TRUE
		}
		break; //IA_BUFFER_READ_STATE_DECODE

		// case IA_BUFFER_READ_STATE_UPDATE_STRIP: {
			
		// 	*pCurrentState = IA_BUFFER_READ_STATE_ACCESS;

		// 	/*
		// 	 * Reset strip counters
		// 	*/
		// 	pCurrentRegisters->iterAccess = 0;

		// 	/**
		// 	 * increment the tile pointer and TB count pointers in advance,
		// 	 * obtain the strip update mode
		// 	 */
		// 	pCurrentRegisters->tileInfo.iCol += 0x1;
		// 	pCurrentRegisters->stripUpdateMode = IA_BUFFER_READ_STRIP_UPDATE_HORIZONTAL;

		// 	if (pCurrentRegisters->tileInfo.iCol == pCurrentRegisters->tileInfo.numStripsCol)
		// 	{
		// 		pCurrentRegisters->stripUpdateMode = IA_BUFFER_READ_STRIP_UPDATE_DONE;
		// 	}
		// }
		// break;

		case IA_BUFFER_READ_STATE_ACCESS: {
			/**
			 * Update the ia strip counters and possibly move the ia pointers if the write to the
			 * PE array is successful
			 */
			if (writeSuccessTaggedBlock == TRUE)
			{
				pCurrentRegisters->iterAccess += 1;

				/**
				 * Update IA cache pointer and reader state
				 */
				if (pCurrentRegisters->iterAccess == pCurrentRegisters->numTBPerStrip)
				{
					pCurrentRegisters->iterAccess = 0;
					pCurrentRegisters->iaBlockInfo.colContribution +=
							pCurrentRegisters->iaBlockInfo.colStride;
					pCurrentRegisters->tileInfo.iCol += 0x1;
					if (pCurrentRegisters->tileInfo.iCol == pCurrentRegisters->tileInfo.numStripsCol)
					{
						*pCurrentState = IA_BUFFER_READ_STATE_DECODE;
						EMULATOR_PRINT(("[kernelIABuffer READER %d] FINISHED processing instruction.\n\n", colID));
					}
					// if (pCurrentRegisters->stripUpdateMode == IA_BUFFER_READ_STRIP_UPDATE_HORIZONTAL)
					// {
					// 	pCurrentRegisters->iaBlockInfo.colContribution +=
					// 		pCurrentRegisters->iaBlockInfo.colStride;

					// 	*pCurrentState = IA_BUFFER_READ_STATE_UPDATE_STRIP;
					// }
					// else if (pCurrentRegisters->stripUpdateMode == IA_BUFFER_READ_STRIP_UPDATE_DONE)
					// {
					// 	*pCurrentState = IA_BUFFER_READ_STATE_DECODE;
					// 	EMULATOR_PRINT(("[kernelIABuffer READER %d] FINISHED processing instruction.\n\n", colID));
					// }
				}
			}
			
		}
		break;

		default: {
			// Keep the status-quo
		}
	} //end of state switch
} //update IA buffer reader

void getIABufferDispatcherOutput (
		//Current state
		t_ia_buffer_d_state currentState,
		bool unblockReader,
		bool unblockWriter,

		//Instruction buffer
		t_input_buffer_tile_buffer_packet controlBuffer,

		//instruction channel interface
		t_flag* pOutReadyForInstruction,

		//accessor interface
		t_flag* pOutValidReaderInstruction,
		t_flag* pOutValidWriterInstruction,
		t_input_buffer_tile_buffer_packet* pOutReaderControl,
		t_input_buffer_tile_buffer_packet* pOutWriterControl,

		//Auxillary
		int colID
	)
{
	/**
	 * Default values
	 */
	*pOutReadyForInstruction = FALSE;
	*pOutValidWriterInstruction = FALSE;
	*pOutValidReaderInstruction = FALSE;

	/**
	 * Tile control packet interface signal
	 */
	if (currentState == IA_BUFFER_INSTRUCTION_STATE_DECODE)
	{
		*pOutReadyForInstruction = TRUE;
	}

	/**
	 * Writer control interface
	 */
	if (currentState == IA_BUFFER_INSTRUCTION_STATE_SEND_TO_WRITER)
	{
		if (unblockWriter == true)
		{
			*pOutValidWriterInstruction = TRUE;
			*pOutWriterControl = controlBuffer;
		}
	}

	/**
	 * Reader control interface.
	 * Send instruction to the reader once the writer has finished loading the data
	 */
	if (currentState == IA_BUFFER_INSTRUCTION_STATE_SEND_TO_READER)
	{
		if (unblockReader == true)
		{
			*pOutValidReaderInstruction = TRUE;
			*pOutReaderControl = controlBuffer;
		}
			
	}
} //getIABufferDispatcherOutput

void updateIABufferDispatcher (
		//instruction channel interface
		t_flag inInstructionValid,
		t_input_buffer_tile_buffer_packet inControl,

		//accessor interface
		t_flag inReaderReady,
		t_flag inWriterReady,

		//context
		t_ia_buffer_d_state* pState,
		t_input_buffer_tile_buffer_packet* pControlBuffer,

		//sync
		bool unblockReader,
		bool unblockWriter,

		//Auxillary
		int colID
	)
{
	switch (*pState) {
		case IA_BUFFER_INSTRUCTION_STATE_DECODE: {
			if (inInstructionValid == TRUE) {
				*pControlBuffer = inControl;
				//Check whether the instruction is for the writer
				if (((inControl.controlBits >> 0x1) & 0x01) == 0x0)
				{
					*pState = IA_BUFFER_INSTRUCTION_STATE_SEND_TO_WRITER;
				}
				//Check whether the instruction is for the reader
				else if (((inControl.controlBits >> 0x1) & 0x01) == 0x1)
				{
					*pState = IA_BUFFER_INSTRUCTION_STATE_SEND_TO_READER;
				}
			}
		}
		break; //IA_BUFFER_INSTRUCTION_STATE_DECODE

		case IA_BUFFER_INSTRUCTION_STATE_SEND_TO_WRITER: {
			if ((inWriterReady == TRUE) && (unblockWriter == true))
			{
				*pState = IA_BUFFER_INSTRUCTION_STATE_DECODE;

				EMULATOR_PRINT(("[kernelIABuffer Dispatcher %d] Sent instruction to WRITER.\n",
							colID));
			}
		}
		break; //IA_BUFFER_INSTRUCTION_STATE_SEND_TO_WRITER

		case IA_BUFFER_INSTRUCTION_STATE_SEND_TO_READER: {
			if ((inReaderReady == TRUE) && (unblockReader == true))
			{
				*pState = IA_BUFFER_INSTRUCTION_STATE_DECODE;
				EMULATOR_PRINT(("[kernelIABuffer Dispatcher %d] Sent instruction to READER.\n",
							colID));
			}
		}
		break; //IA_BUFFER_INSTRUCTION_STATE_SEND_TO_READER

		default: {

		}
	} // switch statement
} //updateIABufferDispatcher

__attribute__((max_global_work_dim(0)))
__kernel void kernelIATileController (
	VOLATILE __global const t_ia_tile_controller_instruction* restrict pInstruction,
	unsigned int numInstructions
	)
{
	//The activities in one instruction cycle consists of
	//sending one IA write instruciton, and then sending all the IA read instructions
	//that stream the IA buffer.
	//To faciliate double buffering,
	//The first cycle consists only of sending the IA write instruction,
	//and the last cycle consists only of sending the IA read instruction 
	unsigned int numInstructionCycles = numInstructions + 1;
	unsigned int iInstruction = 0;
	uint1_t writeSideIndex = 0;


	t_ia_tile_controller_instruction regInstruction;

	for (unsigned int iInstructionCycle=0; iInstructionCycle < numInstructionCycles; iInstructionCycle++)
	{
		/*
		 * Obtain the snapshots of the instruction registers before
		 * the new instruciton cycle overwrites them
		 * 
		*/
		t_ia_tile_controller_instruction drainInstruction = regInstruction;
		/*
		1. Read the instruction of the tile from the memory reader
		*/
		if (iInstructionCycle < numInstructions)
		{
			t_ia_tile_controller_instruction instruction = pInstruction[iInstruction];

			unsigned char inputTileWidth = instruction.localTileWidth;
		    unsigned char inputTileHeight = instruction.localTileHeight;
		    unsigned char numActivePeCols = instruction.flagPadBitmaskCatNumActiveCols & 0x7F;
		    unsigned short iaCacheColStride = instruction.cacheIAStripColStride;

			/*
			2. Send load instructions to the tile buffer
			*/

			EMULATOR_PRINT(("[kernelIATileController] START sending the buffer refresh command for instruction=%d\n"
				"numStripsRow: %d, "
				"numStripsCol: %d\n"
				,iInstructionCycle
				,(unsigned int)inputTileHeight
				,(unsigned int)inputTileWidth));
			{
				t_input_buffer_tile_buffer_packet tileBufferControlPacket;
				tileBufferControlPacket.iaDramBlockAddressBase = 0;

				tileBufferControlPacket.iaDramBlockColStride = iaCacheColStride;
				
				tileBufferControlPacket.controlBits = ((numActivePeCols-1) << 0x3) 
					| (((unsigned char) writeSideIndex) & 0x01);

	            /*
	             *	For the writer, 
	             *	the number of strips should be interpreted as the total number of strips
	             */
			    tileBufferControlPacket.numStripsCol = inputTileWidth*inputTileHeight;

				write_channel_intel(channel_control_to_ia_buffer[0], tileBufferControlPacket);
			}
			EMULATOR_PRINT(("[kernelIATileController] FINISHED sending the buffer refresh instructions for iInstructionCycle=%d .\n\n", iInstructionCycle));

			regInstruction = instruction;
			iInstruction++;
		} //End of sending load instructions

		/*
		3. Send the streaming instructions to the tile buffer
		*/
		if (iInstructionCycle > 0)
		{
			unsigned short iFilterInGroup = 0;
			unsigned char iInputTileWidth = 0;
			unsigned char iInputTileHeight = 0;
			unsigned char iRowInTile = 0;

			unsigned char inputTileWidth = drainInstruction.localTileWidth;
		    unsigned char inputTileHeight = drainInstruction.localTileHeight;
		    unsigned char strideY = drainInstruction.kernelStrideY;
		    unsigned char strideX = drainInstruction.kernelStrideX;
		    unsigned char kernelSizeHeight = drainInstruction.kernelSizeHeight;
		    unsigned char kernelSizeWidth = drainInstruction.kernelSizeWidth;
	        unsigned int numOutputInstructions = drainInstruction.numOutputInstructions;
		    unsigned char numActivePeCols = drainInstruction.flagPadBitmaskCatNumActiveCols & 0x7F;
		    unsigned short numOutputChannelsInGroup = drainInstruction.numOutputChannelsInGroup;
			unsigned short iaCacheColStride = drainInstruction.cacheIAStripColStride;
			unsigned short iaCacheColStrideMultiplier = drainInstruction.cacheIAStripColStrideMultiplier;
			unsigned char numStripsToPEPerInstruction = drainInstruction.numStripsToPEPerInstruction;


	        for (unsigned int i=0; i<numOutputInstructions; i++)
			{
				unsigned char numActivePeRows = ((numOutputChannelsInGroup - iFilterInGroup) < (unsigned short) (PE_ROWS)) ?
					(unsigned char) (numOutputChannelsInGroup - iFilterInGroup) : PE_ROWS;

				unsigned char iStripInTile = (iInputTileHeight + iRowInTile) * inputTileWidth + iInputTileWidth;

				t_input_buffer_tile_buffer_packet tileBufferControlPacket;
				tileBufferControlPacket.iaDramBlockAddressBase = ((unsigned short) iStripInTile) * ((unsigned short) iaCacheColStride);
				/**
				 * e.g. numActivePeRows = 5, PE_ROWS_PER_GROUP = 4
				 * then we need ceil (5 / 4) = 2 PE row groups
				 * and the maximum PE row group ID is 1
				 */
				tileBufferControlPacket.maxPeRowGroupID = (numActivePeRows - 1) >> DIVIDE_BY_PE_ROWS_PER_GROUP_SHIFT;
				
				tileBufferControlPacket.iaDramBlockColStride = iaCacheColStride * iaCacheColStrideMultiplier;

				unsigned char sendInstructionType = 0x2; //Stream from the buffer
				tileBufferControlPacket.controlBits =
					(sendInstructionType & 0x2)
					| (((unsigned char) (~writeSideIndex)) & 0x01)
					| ((numActivePeCols-1) << 0x3);
				
				tileBufferControlPacket.numStripsCol = numStripsToPEPerInstruction;
				tileBufferControlPacket.flagIsLastRow = ((iRowInTile + 1) == kernelSizeHeight) ? TRUE : FALSE;

				//bool success = write_channel_nb_intel(channel_control_to_ia_buffer[0], tileBufferControlPacket);
				write_channel_intel(channel_control_to_ia_buffer[0], tileBufferControlPacket);	
				/*
					Parameters update
				*/
				//if (success)
				//{e
				EMULATOR_PRINT(("[kernelIATileController] Sent a buffer stream command. "
				"iInstructionCycle=%d, numActivePeRows=%d, iInputTileHeight=%d, iInputTileWidth=%d.\n\n", 
				iInstructionCycle, numActivePeRows, iInputTileHeight, iInputTileWidth));
					
				iRowInTile++;
				if (iRowInTile == kernelSizeHeight)
				{
					iRowInTile = 0;

					if ((iInputTileWidth + kernelSizeWidth) >= inputTileWidth)
					{
						iInputTileWidth = 0;

						if ((iInputTileHeight + kernelSizeHeight) >= inputTileHeight)
						{
							iInputTileHeight = 0;
							iFilterInGroup += numActivePeRows;
						}
						else
						{
							iInputTileHeight += strideY;
						}
					}
					else
					{
						iInputTileWidth += strideX;
					}
				}
			} // for
		}
		//End of sending streaming instructions

		//SWAP the read side and the write side
		writeSideIndex = (~writeSideIndex) & 0x01;
	}
} //kernelIATileController



__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelIAControlTee ()
{
	int colID = get_compute_id(0);

	while (true)
	{
		t_input_buffer_tile_buffer_packet controlPacket = read_channel_intel(channel_control_to_ia_buffer[colID]);

		unsigned char maxColID = (controlPacket.controlBits) >> 0x3;

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
	/**
	 * Receive the header of a transfer strip, then use the routing information in the strip to
	 * transfer the remaining data blocks in the strip to the respective destination
	 */
	int colID = get_compute_id(0);

	while (true)
	{

		//bool readSuccess = false;

		t_dram_block_ia_tagged taggedBlock = read_channel_intel(channel_ia_wide[colID]);
		t_activation_dram_block dramBlock = taggedBlock.dramBlock;

		// if (readSuccess == true)
		// {
			int destinationCol = (int) (taggedBlock.route & 0x3F);
			//flag2Misc is TRUE doesn't mean that this block necessarily 
			//goes to this column.
			t_flag flag2Misc = (taggedBlock.route >> 0x6) & 0x01;
			t_flag flagIsLastInStrip = (taggedBlock.route >> 0x7) & 0x01;
			signed char actualColIndex = taggedBlock.iColInSPTile;
			unsigned char colSPStride = taggedBlock.colSPStride;
			unsigned char colSPWidth = taggedBlock.colSPWidth;

			t_flag flagRoute2MiscActual = FALSE;
			t_flag flagRoute2ConvActual = FALSE;
			if ( (((signed char) colSPWidth) > actualColIndex)
							&& (actualColIndex >= 0)
						)
			{
				flagRoute2MiscActual = flag2Misc;
				flagRoute2ConvActual = (~flag2Misc) & 0x01;
			}

			taggedBlock.iColInSPTile -= (signed char) colSPStride;

			EMULATOR_PRINT(("[kernelIATee %d] Read a block. "
						"actualColIndex=%d, "
						"colSPStride=%d, "
						"colSPWidth=%d, "
						"flagRoute2Misc=%#03x. "
						"flagRoute2Conv=%#03x\n", 
						colID, 
						actualColIndex, 
						colSPStride, 
						colSPWidth, 
						((unsigned char) flagRoute2MiscActual), 
						((unsigned char) flagRoute2ConvActual)));

			//Forward to the next column
			if (colID < (PE_COLS - 1))
			{
				if (destinationCol > colID)
				{
					write_channel_intel(channel_ia_wide[colID+1], taggedBlock);
				}
			}

			if (flagRoute2ConvActual == TRUE)
			{
				t_dram_block_ia_to_pe blockToPE;
				blockToPE.dramBlock = dramBlock;
				blockToPE.numTB = taggedBlock.numTB;
				blockToPE.route = taggedBlock.route;
				write_channel_intel(channel_ia_wide_local[colID], blockToPE);
			}

			//Logic for routing dram blocks to the MISC unit
			//Only forward data blocks	
			if (colID < MISC_COLS)
			{
				if (flagRoute2MiscActual == TRUE)
				{
					t_dram_block_ia_to_misc blockToMisc;
					blockToMisc.dramBlock = taggedBlock.dramBlock;
					blockToMisc.miscLeftShiftAmount = taggedBlock.miscLeftShiftAmount;
					write_channel_intel(channel_ia_wide_misc[colID], blockToMisc);
				}
			}
		
		// } // if read is successful
	}
}
#endif //IA_MEMORY

#if defined(MISC_ENGINE)
__attribute__((max_global_work_dim(0)))
__kernel void kernelMiscControlMover (
		__global t_misc_instruction* restrict pInstruction,
		unsigned int numInstruction
	)
{
	for (int i=0; i<numInstruction; i++)
	{
		t_misc_instruction instruction = pInstruction[i];
		unsigned short numDramBlocksToReduce = instruction.numDramBlocksToReduce;
		unsigned short numOutputBlocksPerUnit = instruction.numOutputBlocksPerUnit;
		//unsigned short numOutputBlocksPerStrip = instruction.numOutputBlocksPerStrip;
		unsigned char outputModifierControl = instruction.outputModifierControl;
		//for (unsigned short iChunk=0; iChunk<numOutputBlocksPerStrip; iChunk++)
		//{
			t_misc_control_packet packet;
			packet.controlBits = instruction.controlBits;
			packet.numDramBlocksToReduce = numDramBlocksToReduce;
			packet.numOutputBlocks	=	numOutputBlocksPerUnit;
			packet.outputModifierControl = outputModifierControl;

			write_channel_intel(channel_misc_instruction[0], packet);
			// EMULATOR_PRINT(("[kernelMiscControlMover] Sent instruction (%d:%d) \n",
			// 				i, iChunk));
			EMULATOR_PRINT(("[kernelMiscControlMover] Sent instruction %d \n", i));

		//}
	}
}

#define KERNEL_MISC_TEE_FETCH 0x0
#define KERNEL_MISC_TEE_RESEND 0x1
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(MISC_COLS)))
__kernel void kernelMiscControlTee ()
{
	typedef uint1_t t_state;
	int colID = get_compute_id(0);
	t_misc_control_packet regControlPacket;
	t_flag regNeedResendToSelf = FALSE;
	t_flag regNeedResendToOther = FALSE;
	t_state regState = KERNEL_MISC_TEE_FETCH;

	#pragma ii 1
	#pragma speculated_iterations 0
	while (1)
	{
		//Local signals
		t_misc_control_packet sigControlPacket = regControlPacket;
		t_flag sigNeedResendToSelf = regNeedResendToSelf;
		t_flag sigNeedResendToOther = regNeedResendToOther;
		t_flag sigState = regState;

		//Interaction with the upstream channel
		t_flag readUpstreamSuccess = FALSE;
		if (regState == KERNEL_MISC_TEE_FETCH)
		{
			bool success = false;
			sigControlPacket = read_channel_nb_intel(channel_misc_instruction[colID], &success);
			readUpstreamSuccess = (success == true) ? TRUE : FALSE;
		}

		//Interaction with the local channel
		t_flag sendToSelfSuccess = (regNeedResendToSelf==TRUE) ? FALSE : TRUE;
		if (
			(readUpstreamSuccess == TRUE)
			|| ((regState == KERNEL_MISC_TEE_RESEND) && (regNeedResendToSelf == TRUE))
		   )
		{
			bool success = 
				write_channel_nb_intel(channel_misc_instruction_local[colID], sigControlPacket);
			sendToSelfSuccess = (success == true) ? TRUE : FALSE;
		}

		//Interaction with the downstream channel
		t_flag sendToOtherSuccess = TRUE;
		if (colID < (MISC_COLS-1))
		{
			sendToOtherSuccess = (regNeedResendToOther==TRUE) ? FALSE : TRUE;
			uint4_t numActiveCol = (sigControlPacket.controlBits & 0x0F);
			if (
				(readUpstreamSuccess == TRUE)
				|| ((regState == KERNEL_MISC_TEE_RESEND) && (regNeedResendToOther == TRUE))
			   )
			{
				if (colID < (numActiveCol - 1))
				{
					bool success = 
						write_channel_nb_intel(channel_misc_instruction[colID+1], sigControlPacket);
					sendToOtherSuccess = (success == true) ? TRUE : FALSE;
				}
				else
				{
					sendToOtherSuccess = TRUE;
				}
			}
		}

		//State update
		if (regState == KERNEL_MISC_TEE_FETCH)
		{
			if ((sendToSelfSuccess == FALSE) || (sendToOtherSuccess == FALSE))
			{
				sigState = KERNEL_MISC_TEE_RESEND;
				if (sendToSelfSuccess == FALSE)
				{
					sigNeedResendToSelf = TRUE;
				}

				if (sendToOtherSuccess == FALSE)
				{
					sigNeedResendToOther = TRUE;
				}
			}
		}
		else //regState == KERNEL_MISC_TEE_RESEND
		{
			if (sendToSelfSuccess == TRUE)
			{
				sigNeedResendToSelf = FALSE;
			}

			if (sendToOtherSuccess == TRUE)
			{
				sigNeedResendToOther = FALSE;
			}

			if ((sendToSelfSuccess == TRUE) && (sendToOtherSuccess == TRUE))
			{
				sigState = KERNEL_MISC_TEE_FETCH;
			}
		}

		//Loop-variable update
		regControlPacket = sigControlPacket;
		regNeedResendToSelf = sigNeedResendToSelf;
		regNeedResendToOther = sigNeedResendToOther;
		regState = sigState;
	}

}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(MISC_COLS)))
__kernel void kernelMisc ()
{
	int colID = get_compute_id(0);
	while (true)
	{
		bool instructionReadSuccess = false;
		t_misc_control_packet controlPacket = 
			read_channel_nb_intel(channel_misc_instruction_local[colID], 
				&instructionReadSuccess);


		if (instructionReadSuccess == true)
		{
			//Decode
			//OpCode. 00: Add; 01: Max Pooling; 10: Stream
			uint2_t opcode = (controlPacket.controlBits >> 4) & 0x03;
			unsigned short numDramBlocksToReduce = controlPacket.numDramBlocksToReduce;
			unsigned short numOutputBlocks = controlPacket.numOutputBlocks;
			unsigned char shiftDirectionCatShiftAmount = controlPacket.outputModifierControl & 0x01F;
			t_flag enableRelu = (controlPacket.outputModifierControl >> 0x05) & 0x01;

			EMULATOR_PRINT(("[kernelMisc %d] Received command "
							"opcode=%#04x, "
							"shiftDirectionCatShiftAmount=%04x, "
							"enableRelu=%03x, "
							"numOutputBlocks=%d, "
							"numDramBlocksToReduce=%d \n",
							colID, 
							((unsigned char)opcode), 
							shiftDirectionCatShiftAmount,
							(unsigned char) enableRelu,
							numOutputBlocks, 
							numDramBlocksToReduce));

			//Limit the concurrency if BURST_SIZE_BYTE > 16
			// #if (BURST_SIZE_BYTE > 16)
			// #pragma max_concurrency 2
			// #endif
			for (unsigned short iOutput=0; iOutput < numOutputBlocks; iOutput++)
			{
				#if (defined(ARRIA10) || defined(STRATIX10))
					t_accumulator reductionBlock[ACTIVATION_BURST_SIZE_BYTE] __attribute__((__register__));
				#else
					t_accumulator reductionBlock[ACTIVATION_BURST_SIZE_BYTE];
				#endif
				//Initialize the reductionBlock
				#pragma unroll
				for (int iVal=0; iVal < ACTIVATION_BURST_SIZE_BYTE; iVal++)
				{
					//If max pooling, then intialize the values to the minimum, else zero
					t_accumulator min = ACCUM_MIN;
					reductionBlock[iVal] = (opcode == ((uint2_t) 0x01)) ? 
						min : 0x0000;
				}

				//Perform reduction
				unsigned short iBlock = 0;
				#pragma ii 1
				#pragma speculated_iterations 0
				while (iBlock <numDramBlocksToReduce)
				{
					bool blockReadSuccess = false;

					t_dram_block_ia_to_misc inputDramBlockTagged = 
						read_channel_nb_intel(channel_ia_wide_misc[colID], &blockReadSuccess);

					if (blockReadSuccess == true)
					{
						unsigned char numLeftShiftAmount = inputDramBlockTagged.miscLeftShiftAmount;
						t_activation_dram_block inputDramBlock = inputDramBlockTagged.dramBlock;

						EMULATOR_PRINT(("[Kernel MISC (%d)] iInputBlock=%d, numBlocksToReduce=%d, iOutput=%d, numOutputBlocks=%d\n"
							"Inputblock[0-3]: %#04x %#04x %#04x %#04x \n",
							colID, iBlock, numDramBlocksToReduce, iOutput, numOutputBlocks,
							inputDramBlock.values[0] & 0xFF, 
							inputDramBlock.values[1] & 0xFF, 
							inputDramBlock.values[2] & 0xFF, 
							inputDramBlock.values[3] & 0xFF));

						#pragma unroll ACTIVATION_BURST_SIZE_BYTE
						#pragma ii 1
						#pragma speculated_iterations 0
						for (int iValue=0; iValue < ACTIVATION_BURST_SIZE_BYTE; iValue++)
						{
							t_accumulator rawInputValue = (t_accumulator) 
									inputDramBlock.values[iValue];

							//Left-shift input
							t_accumulator inputValue = rawInputValue << numLeftShiftAmount;

							t_accumulator currentValue = reductionBlock[iValue];

							t_accumulator newValue;
							if (opcode == ((uint2_t) 0x00))
							{
								newValue = inputValue + currentValue;
							}
							else if (opcode == ((uint2_t) 0x01))
							{
								newValue = (inputValue >= currentValue) ? inputValue : currentValue;
							}
							else
							{
								newValue = inputValue;
							}

							reductionBlock[iValue]
								= newValue;
						}

						iBlock++;
					} //if (blockReadSuccess == true)
				} //while (iBlock <numDramBlocksToReduce)

				//Modify the output
				t_output_activation_dram_block_tagged outputTagged;
				#pragma unroll
				for (int i=0; i<ACTIVATION_BURST_SIZE_BYTE; i++)
				{
					outputTagged.dramBlock.values[i] = modifyOutput(
							reductionBlock[i],
							shiftDirectionCatShiftAmount,
							enableRelu
						);
				}

				//Let the OATee set the isFromLastColumn flag
				bool writeSuccess = false;
				while (writeSuccess == false)
				{
					writeSuccess = 
						write_channel_nb_intel(
							channel_misc_to_oa_tee[colID], 
							outputTagged
							);
				}

				EMULATOR_PRINT(("[kernelMisc %d] Finished processing output block %d / %d of the command.\n", colID, iOutput, numOutputBlocks));
			}  //iOutput
			
			EMULATOR_PRINT(("[kernelMisc %d] Finished processing a command\n", colID));
		}
	}
}
#endif

#ifdef MEMORY_WRITER
__attribute__((max_global_work_dim(0)))
__kernel void kernelOAMover (
		VOLATILE __global signed char* restrict pOA,

		VOLATILE __global const t_oa_mover_instruction* restrict pInstruction,
		unsigned int numInstruction,
		//Starting offset to read the instruction from
		unsigned int offsetInstruction
	)
{
	//Use while loop instead of for-loop
	#pragma max_concurrency 1
	for (unsigned int iInst=0; iInst < numInstruction; iInst += 8)
	{
		t_oa_mover_instruction cacheInstruction[8];
		unsigned char numInstInTile = (iInst + 8) < numInstruction ? 
			8 : numInstruction - iInst;
		#pragma unroll 
		for (unsigned char i=0; i<8; i++)
		{
			cacheInstruction[i] = pInstruction[i+iInst+offsetInstruction];
		}

		unsigned char iInstTile = 0;
		while (iInstTile < numInstInTile)
		{
			/*! Read the instruction and decode the packed field*/
			t_oa_mover_instruction inst = cacheInstruction[iInstTile];
			t_flag enableSendSync = (inst.memSelectCatSparseFlagCatSyncFlagCatNumActiveCols >> 4) & 0x01;
			unsigned char numActivePeCols = inst.memSelectCatSparseFlagCatSyncFlagCatNumActiveCols & 0x0F;
			
			unsigned char numOutputTileHeightPerCol = inst.tileHeight;
			unsigned char numOutputTileWidthPerCol = inst.columnTileWidth;
			unsigned short numNominalDramBlocksPerStrip = inst.numNominalDramBlocksPerStrip;

			unsigned short numNominalDramBlocksAcrossActivePeCols = 
				((unsigned short) numActivePeCols)
				* ((unsigned short) numNominalDramBlocksPerStrip);

			unsigned char numOutputTileHeightxWidthPerCol = numOutputTileHeightPerCol * numOutputTileWidthPerCol;

			//Control variables
			unsigned char iOutputHeightInColTile = 0;
			unsigned char iOutputWidthInColTile = 0;

			//Memory pointer contribution
			signed int addrOAColContribution = 0;
			signed int addrOARowContribution = 0;
			//signed int addrOAGroupContribution = 0;

			bool instructionProceed = true;
			if (enableSendSync == TRUE)
			{
				instructionProceed = write_channel_nb_intel(channel_activation_sync, (unsigned char ) 0x01);
			}

			mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);

			if (instructionProceed == true)
			{
				EMULATOR_PRINT(("[kernelOAMover] START processing instruction. "
								"offsetInstruction=%d, "
								"iInst+iInstTile=%d, "
								"memOAStart=%#08x, "
								"memOAPEColStride=%#08x, "
								"memOAColStride=%#08x, "
								"memOARowStride=%#08x, "
								"numActivePeCols=%d, " 
		                        "numNominalDramBlocksPerStrip=%d, "
		                        "numNominalDramBlocksAcrossActivePeCols=%d, "
								"numOutputTileWidthPerCol=%#03x, "
								"numOutputTileHeightPerCol=%#03x, "
								"numActivePeCols=%d\n",
								offsetInstruction,
								iInst+iInstTile, 
								(unsigned int) inst.memOAStart,
								(unsigned int) inst.memOAPEColStride,
								(unsigned int) inst.memOAColStride,
								(unsigned int) inst.memOARowStride,
								(unsigned int) numActivePeCols,
								(unsigned int) numNominalDramBlocksPerStrip,
								(unsigned int) numNominalDramBlocksAcrossActivePeCols,
								(unsigned int) numOutputTileWidthPerCol,
								(unsigned int) numOutputTileHeightPerCol,
								(unsigned int) numActivePeCols));

				for (
						unsigned char iterOutputTileHeightxWidthPerCol=0;
						iterOutputTileHeightxWidthPerCol < numOutputTileHeightxWidthPerCol;
						iterOutputTileHeightxWidthPerCol++
					)
				{
					unsigned short iterNominalDramBlocksAcrossActivePeCols = 0;

					//iterator for the PE columns
					unsigned char iPeCol = 0;
					unsigned short iNominalDramBlockInStrip = 0;
					signed int addrOAPeColContribution = 0;
					signed int addrOABase = 
						inst.memOAStart + addrOARowContribution + addrOAColContribution;

					#pragma ii 1
					#pragma speculated_iterations 0
					while (iterNominalDramBlocksAcrossActivePeCols<numNominalDramBlocksAcrossActivePeCols)
					{
						int addrOA = 
							addrOABase 
							+ addrOAPeColContribution 
							+ (((int) iNominalDramBlockInStrip) << ACTIVATION_BURST_SIZE_BYTE_OFFSET);

						bool readSuccess = false;
						t_output_activation_dram_block_tagged receivedBlock 
							= read_channel_nb_intel(channel_output_wide[0], &readSuccess);

						if (readSuccess == true)
						{
							EMULATOR_PRINT(("[kernelOAMover] Received a dram block.\n"
									"iPeCol=%#03x, "
									"iNominalDramBlockInStrip=%d, "
									"iterNominalDramBlocksAcrossActivePeCols=%d, "
									"numNominalDramBlocksAcrossActivePeCols=%d, "
									"iOutputWidthInColTile=%#04x, "
									"iOutputHeightInColTile=%d\n",
									(unsigned int) iPeCol,
									(unsigned int) iNominalDramBlockInStrip,
									(unsigned int) iterNominalDramBlocksAcrossActivePeCols,
									(unsigned int) numNominalDramBlocksAcrossActivePeCols,
									(unsigned int) iOutputWidthInColTile,
									(unsigned int) iOutputHeightInColTile
									));

							/**
							 * Write to the exteran memory
							 * Use loop unrolling to coalescing the access
							 */
							#pragma unroll
							for (unsigned int i=0; i<ACTIVATION_BURST_SIZE_BYTE; i++)
							{
								pOA[addrOA+i] = receivedBlock.dramBlock.values[i];
							}
							

							/*State update*/
							iterNominalDramBlocksAcrossActivePeCols++;

							iPeCol++;
							addrOAPeColContribution += (signed int) inst.memOAPEColStride;

							if (iPeCol == numActivePeCols)
							{
								iPeCol = 0x0;
								addrOAPeColContribution = 0;

								//Move along the output channel direction
								iNominalDramBlockInStrip++;
							}
						} //(readSuccess == true)
					} //while iterNominalDramBlocksAcrossActivePeCols


					iOutputWidthInColTile++;
					addrOAColContribution += (signed int) inst.memOAColStride;
					
					if (iOutputWidthInColTile == numOutputTileWidthPerCol)
					{
						iOutputWidthInColTile = 0;
						addrOAColContribution = 0;
						addrOARowContribution += (signed int) inst.memOARowStride;

						iOutputHeightInColTile++;
					} //if (iOutputWidthInColTile == numOutputTileWidthPerCol)
				} //iterOutputTileHeightxWidthPerCol 

				//Increment the instruction count.
				iInstTile++;
			} // if proceed
		} //while. over instruction
	}
} //kernelOAMover
#endif //MEMORY_WRITER 

#ifdef OA_MEMORY
//#if ((defined(ARRIA10) || defined(STRATIX10)) && defined(OA_PING_PONG))
#define OA_BUFFER_WRITER_STATE_DECODE 0x0
#define OA_BUFFER_WRITER_STATE_NUM_ACCESS 0x1
#define OA_BUFFER_WRITER_STATE_UPDATE_STRIP 0x2
#define OA_BUFFER_WRITER_STATE_ACCESS 0x3

#define OA_BUFFER_READER_STATE_DECODE 0x0
#define OA_BUFFER_READER_STATE_NUM_ACCESS 0x1
#define OA_BUFFER_READER_STATE_UPDATE_STRIP 0x2
#define OA_BUFFER_READER_STATE_ACCESS 0x3

#define OA_BUFFER_DISPATCH_STATE_IDLE 0x0
#define OA_BUFFER_DISPATCH_STATE_SEND_WRITER 0x1
#define OA_BUFFER_DISPATCH_STATE_SEND_READER 0x2

typedef struct __attribute__((packed)) {
	//Starting activation address of the output strip that is to be accessed. Init = 0
	unsigned short stripStartOutputIndex;
	//Number of output channels. Init = 0;
	unsigned short numOutputsPerStrip;
	//Number of 1x1 strips to be accessed in the tile. Init = 0;
	unsigned char numStripsToAccess;
	//Stride between the start of successive strips in the cache. Counted in activations. Init 0
	unsigned short oaStridePerCol;
	//Iterator of the number of strips that have been accessed. Init 0
	unsigned char iStrip;
	//Number of access interations per strip.
	unsigned short numLoopsPerStip;
	//Iterator of access interations. Init 0
	unsigned short iLoopPerStip;
	//Index in the OA cache. Init 0
	unsigned short indexOutput; 

	//Access bank
	t_flag accessBank;
} t_oa_buffer_access_info;

typedef struct __attribute__((packed)) {
	t_oa_buffer_access_info accessInfo;

	//Init to FALSE
	// uint1_t flagSourceIsMisc;
	//Init to 0
	unsigned char accumulatorShiftDirCatShiftAmount;
	//Init to FALSE
	uint1_t enableRelu;
} t_oa_buffer_writer_info;

typedef struct __attribute__((packed)) {
	t_oa_buffer_access_info accessInfo;

	// //Init to FALSE
	// uint1_t enableSparsification;
	// //Init to 0
	// unsigned short numClustersToDrain;
	// //Init to 0
	// unsigned char iClustersInWindowFetched;
	// //Init to 0
	// unsigned short iOutputChannelFetched;
	// //Init to 0
	// unsigned short iClustersFetched;
	//Init to 0
	// unsigned char iGroupsFetched;
	//Init to 0
	// unsigned char numGroupsCurrentLayer;
	// //Init to 0
	// unsigned short numChannelsInGroupNextLayer;
} t_oa_buffer_reader_info;

/**
 * State type for oa_buffer_writer
 */
typedef uint2_t t_oa_buffer_writer_state;

/*
 * State type for oa_buffer_reader
*/
typedef uint2_t t_oa_buffer_reader_state;

/**
 * State type for the instruction generator
 */
typedef uint2_t t_oa_buffer_dispatcher_state;

typedef struct {
	char values[PE_ROWS_PER_GROUP];
} t_oa_buffer_access_block;


void updateOABufferWriter (
	//Inputs from the instruction dispatcher
	t_output_tile_buffer_packet _control,
	t_flag _validControl,

	//Signals from the interface with the PE column
	t_accumulator _wideOutputFromPEs[],
	t_flag _requestValueFromPE,
	t_flag _validValueFromPE,

	//Modified buffers
	t_oa_buffer_access_block cacheOutputActivations0[OA_CACHE_DEPTH][ACTIVATION_BURST_SIZE_IN_PE_ROW_GROUP],
	t_oa_buffer_access_block cacheOutputActivations1[OA_CACHE_DEPTH][ACTIVATION_BURST_SIZE_IN_PE_ROW_GROUP],

	//State variables
	t_oa_buffer_writer_info *pRegisters,

	//State
	t_oa_buffer_writer_state *pState,

	//Auxillary
	int colID
	);

void getOABufferWriterOutput (
	//Current state
	t_oa_buffer_writer_state _currentState,

	//Current context
	t_oa_buffer_writer_info _currentContext,

	//Interface with the dispatcher
	t_flag* pOutAcceptInstruction,

	//Interface with the PE column channel
	t_flag* pOutAcceptDataFromPE
	);

void updateOABufferReader (
	//Inputs from the instruction dispatcher
	t_output_tile_buffer_packet _control,
	t_flag _validControl,

	//Interface with the channel to OA tee
	t_flag _writeSuccessOATee,
	t_flag _requestWrite2OATee,

	//State variables,
	t_oa_buffer_reader_info *pRegisters,

	//State,
	t_oa_buffer_reader_state *pState,

	//Auxillary
	int colID
	);

void getOABufferReaderOutput (
	//Current state
	t_oa_buffer_reader_state _currentState,
	//Current context
	t_oa_buffer_reader_info _currentContext,

	//Interface with the dispatcher
	t_flag* pOutAcceptInstruction,

	//Buffer, read only
	const t_oa_buffer_access_block cacheOutputActivations0[OA_CACHE_DEPTH][ACTIVATION_BURST_SIZE_IN_PE_ROW_GROUP],
	const t_oa_buffer_access_block cacheOutputActivations1[OA_CACHE_DEPTH][ACTIVATION_BURST_SIZE_IN_PE_ROW_GROUP],

	//Interface with the channel to OA Tee
	t_flag* pToOATeeValid,
	t_output_activation_dram_block_tagged* pToOATeeData
	);

void updateOABufferDispatcher (
		//instruction channel interface
		t_flag _inInstructionValid,
		t_output_tile_buffer_packet _inControl,

		//accessor interface
		t_flag _inReaderReady,
		t_flag _inWriterReady,

		//context
		t_oa_buffer_dispatcher_state* pState,
		t_output_tile_buffer_packet* pControlBuffer,

		//sync,
		bool _unblockReader,

		//Auxillary
		int colID
	);


void getOABufferDispatcherOutput (
		//Current state
		t_oa_buffer_dispatcher_state _currentState,
		//sync,
		bool _unblockReader,

		//Instruction buffer
		t_output_tile_buffer_packet _controlBuffer,

		//instruction channel interface
		t_flag* pOutReadyForInstruction,

		//accessor interface
		t_flag* pOutValidReaderInstruction,
		t_flag* pOutValidWriterInstruction,
		t_output_tile_buffer_packet* pOutReaderControl,
		t_output_tile_buffer_packet* pOutWriterControl,

		//Auxillary
		int colID
	);

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelOABuffer ()
{
	int colID = get_compute_id(0);

	/**
	 * Activation buffer
	 * TODO: might need to change the indexing in order to get the BRAM arrangement
	 */
	t_oa_buffer_access_block cacheOutputActivations0 [OA_CACHE_DEPTH][ACTIVATION_BURST_SIZE_IN_PE_ROW_GROUP] __attribute__((
                   numbanks(ACTIVATION_BURST_SIZE_IN_PE_ROW_GROUP), bankwidth(PE_ROWS_PER_GROUP), singlepump));
	t_oa_buffer_access_block cacheOutputActivations1 [OA_CACHE_DEPTH][ACTIVATION_BURST_SIZE_IN_PE_ROW_GROUP] __attribute__((
                   numbanks(ACTIVATION_BURST_SIZE_IN_PE_ROW_GROUP), bankwidth(PE_ROWS_PER_GROUP), singlepump));

	/**
	 * Writer state and registers
	 */
	t_oa_buffer_writer_info regWriterContext = {
		.accessInfo = {
			.stripStartOutputIndex = 0,
			.numOutputsPerStrip = 0,
			.numStripsToAccess = 0,
			.oaStridePerCol = 0,
			.iStrip = 0,
			.numLoopsPerStip = 0,
			.iLoopPerStip = 0,
			.indexOutput = 0,
			.accessBank = 0x0
		},
		// .flagSourceIsMisc = FALSE,
		.accumulatorShiftDirCatShiftAmount = 0x0,
		.enableRelu = FALSE
	};
	t_oa_buffer_writer_state regWriterState = OA_BUFFER_WRITER_STATE_DECODE;

	/**
	 * Reader state and registers 
	 * TODO: elminate some variables
	 */
	t_oa_buffer_reader_info regReaderContext = {
		.accessInfo = {
			.stripStartOutputIndex = 0,
			.numOutputsPerStrip = 0,
			.numStripsToAccess = 0,
			.oaStridePerCol = 0,
			.iStrip = 0,
			.numLoopsPerStip = 0,
			.iLoopPerStip = 0,
			.indexOutput = 0,
			.accessBank = 0x0
		},
		// .enableSparsification = FALSE,
		// .numClustersToDrain = 0,
		// .iClustersInWindowFetched = 0,
		// .iOutputChannelFetched = 0,
		// .iClustersFetched = 0,
		// .iGroupsFetched = 0,
		// .numGroupsCurrentLayer = 0,
		// .numChannelsInGroupNextLayer = 0
	};

	t_oa_buffer_reader_state regReaderState = OA_BUFFER_READER_STATE_DECODE;

	/**
	 * Dispatcher state and registers
	 */
	t_output_tile_buffer_packet regDispatcherInstructionBuffer;
	t_oa_buffer_dispatcher_state regDispatcherState = OA_BUFFER_DISPATCH_STATE_IDLE;


	//Runtime logic
	//#pragma ivdep array(cacheOutputActivations0)
	//#pragma ivdep array(cacheOutputActivations1)
	#pragma ii 1
	#pragma speculated_iterations 0
	while (true) {
		/**
		 * oa buffer instruction channel <===> dispatcher
		 */
		t_flag dispatcherReadyForInstruction = FALSE;
		t_flag dispatcherInstructionValid = FALSE;
		t_output_tile_buffer_packet dispatcherNewInstruction;

		/**
		 * dispatcher <==> writer
		 */
		t_flag writerInstructionValid = FALSE;
		t_flag writerInstructionRequest = FALSE;
		t_output_tile_buffer_packet writerNewInstruction;

		/**
		 * dispatcher <==> reader
		 */
		t_flag readerInstructionValid = FALSE;
		t_flag readerInstructionRequest = FALSE;
		t_output_tile_buffer_packet readerNewInstruction;

		/**
		 * writer <===> channel from Pe
		 */
		t_flag writerBlockFromPEValid = FALSE;
		t_flag writerBlockFromPERequest = FALSE;
		t_accumulator writerBlockFromPEData[PE_ROWS_PER_GROUP];

		/**
		 * reader <===> channel to OA tee
		 */
		t_flag readerBlockToOATeeValid = FALSE;
		t_flag readerBlockToOATeeSent = FALSE;
		t_output_activation_dram_block_tagged readerBlockToOATeeData;

		/**
		 * Reader-writer synchornization
		 */
		bool unblockReader = 
			(regWriterState == OA_BUFFER_WRITER_STATE_DECODE)
			|| (regWriterContext.accessInfo.accessBank != ((regDispatcherInstructionBuffer.controlBits >> 0x9) & 0x01));


		/**
		 * Derive current interface outputs from the modules
		 */
		getOABufferWriterOutput (
			//Current state
			regWriterState,

			//t_oa_buffer_writer_info _currentContext,
			regWriterContext,

			//t_flag* pOutAcceptInstruction,
			&writerInstructionRequest,

			//t_flag* pOutAcceptDataFromPE,
			&writerBlockFromPERequest
			);

		getOABufferReaderOutput (
			//t_oa_buffer_reader_state _currentState,
			regReaderState,
			
			//t_oa_buffer_reader_info _currentContext,
			regReaderContext,

			//t_flag* pOutAcceptInstruction,
			&readerInstructionRequest,

			//const char cacheOutputActivations[2][OA_CACHE_DEPTH][CLUSTER_SIZE],
			cacheOutputActivations0,
			cacheOutputActivations1,

			//t_flag* pToOATeeValid,
			&readerBlockToOATeeValid,
			//t_output_cluster_tagged* pToOATeeData,
			&readerBlockToOATeeData
			);

		getOABufferDispatcherOutput (
			//t_oa_buffer_dispatcher_state _currentState,
			regDispatcherState,
			//bool _unblockReader,
			unblockReader,

			//t_output_tile_buffer_packet _controlBuffer,
			regDispatcherInstructionBuffer,

			//t_flag* pOutReadyForInstruction,
			&dispatcherReadyForInstruction,

			//t_flag* pOutValidReaderInstruction,
			&readerInstructionValid,
			//t_flag* pOutValidWriterInstruction,
			&writerInstructionValid,
			//t_output_tile_buffer_packet* pOutReaderControl,
			&readerNewInstruction,
			//t_output_tile_buffer_packet* pOutWriterControl,
			&writerNewInstruction,

			//Auxillary
			colID
		);

		/**
		 * Interface with the channels
		 */
		//Dispatcher <==> instruction channel
		if (dispatcherReadyForInstruction == TRUE)
		{
			bool success = false;
			dispatcherNewInstruction =
				read_channel_nb_intel(channel_control_to_oa_buffer_local[colID], &success);
			if (success == true)
			{
				dispatcherInstructionValid = TRUE;
			}
		}

		//Writer <===> data channel from PE
		if (writerBlockFromPERequest == TRUE)
		{
			bool success = false;
			t_conv_drain_multiple_tagged wideOutputTagged = 
				read_channel_nb_intel(channel_drain_conv[PE_ROW_GROUPS-1][colID], &success);
			if (success == true)
			{
				#pragma unroll
				for (int i=0; i<PE_ROWS_PER_GROUP; i++)
				{
					writerBlockFromPEData[i] = wideOutputTagged.values[i];
				}
				writerBlockFromPEValid = TRUE;
			}
		}


		//Reader <===> OA tee
		if (readerBlockToOATeeValid == TRUE)
		{
			bool success = 
				write_channel_nb_intel(channel_oa_buffer_to_oa_tee[colID], readerBlockToOATeeData);
			if (success == true)
			{
				readerBlockToOATeeSent = TRUE;
			}
		}

		/**
		 * State update
		 */
		//Update the writer
		updateOABufferWriter (
			//t_output_tile_buffer_packet _control,
			writerNewInstruction,
			//t_flag _validControl,
			writerInstructionValid,

			//t_accumulator _wideOutputFromPEs[],
			writerBlockFromPEData,
			//t_flag _requestValueFromPE,
			writerBlockFromPERequest,
			//t_flag _validValueFromPE,
			writerBlockFromPEValid,

			//char cacheOutputActivations[OA_CACHE_DEPTH][CLUSTER_SIZE][2],
			cacheOutputActivations0,
			cacheOutputActivations1,

			//t_oa_buffer_writer_info *pRegisters,
			&regWriterContext,

			//t_oa_buffer_writer_state *pState,
			&regWriterState,

			//int colID
			colID
			);


		//Update the reader
		updateOABufferReader (
			//t_output_tile_buffer_packet _control,
			readerNewInstruction,
			//t_flag _validControl,
			readerInstructionValid,

			//t_flag _writeSuccessOATee,
			readerBlockToOATeeSent,
			//t_flag _requestWrite2OATee,
			readerBlockToOATeeValid,

			//t_oa_buffer_reader_info *pRegisters,
			&regReaderContext,

			//t_oa_buffer_reader_state *pState,
			&regReaderState,

			//int colID
			colID
			);

		//Update the dispatcher
		updateOABufferDispatcher (
			//t_flag _inInstructionValid,
			dispatcherInstructionValid,
			//t_output_tile_buffer_packet _inControl,
			dispatcherNewInstruction,

			//t_flag _inReaderReady,
			readerInstructionRequest,
			//t_flag _inWriterReady,
			writerInstructionRequest,

			//t_oa_buffer_dispatcher_state* pState,
			&regDispatcherState,
			//t_output_tile_buffer_packet* pControlBuffer,
			&regDispatcherInstructionBuffer,

			//bool _unblockReader,
			unblockReader,

			//int colID
			colID
		);
	} // while-loop
}

void updateOABufferWriter (
	//Inputs from the instruction dispatcher
	t_output_tile_buffer_packet _control,
	t_flag _validControl,

	//Signals from the interface with the PE column
	t_accumulator _wideOutputFromPEs[],
	t_flag _requestValueFromPE,
	t_flag _validValueFromPE,

	//Modified buffers
	t_oa_buffer_access_block cacheOutputActivations0 [OA_CACHE_DEPTH][ACTIVATION_BURST_SIZE_IN_PE_ROW_GROUP],
	t_oa_buffer_access_block cacheOutputActivations1 [OA_CACHE_DEPTH][ACTIVATION_BURST_SIZE_IN_PE_ROW_GROUP],

	//State variables
	t_oa_buffer_writer_info *pRegisters,

	//State
	t_oa_buffer_writer_state *pState,

	//Auxillary
	int colID
	)
{
	switch (*pState) {
		case (OA_BUFFER_WRITER_STATE_DECODE) : {
			//Obtain new instruction and update the context
			if (_validControl == TRUE)
			{
				//Accessor update
				(*pRegisters).accessInfo.stripStartOutputIndex = _control.startOutputIndex;
				(*pRegisters).accessInfo.numOutputsPerStrip = _control.numOutputsPerStrip;
				(*pRegisters).accessInfo.numStripsToAccess = _control.numStripsToAccess;
				(*pRegisters).accessInfo.oaStridePerCol = _control.iaStridePerCol;
				(*pRegisters).accessInfo.iStrip = 0x0;
				(*pRegisters).accessInfo.accessBank = (_control.controlBits >> 9) & 0x01;

				//Update registers specific to the writer
				(*pRegisters).accumulatorShiftDirCatShiftAmount = _control.controlBits & 0x1F;
				(*pRegisters).enableRelu = (_control.controlBits >> 7) & 0x1;


				//State update
				*pState = OA_BUFFER_WRITER_STATE_NUM_ACCESS;

				EMULATOR_PRINT(("[kernelOABuffer WRITER %d] START processing instruction. "
						"stripStartOutputIndex=%d, "
						"numOutputsPerStrip=%d, "
						"numStripsToAccess=%d, "
						"oaStridePerCol=%d, "
						"enableRelu=%#03x, "
						"accessBank=%#03x, "
						"accumulatorShiftDirCatShiftAmount=%#07x \n\n", 
						colID, 
						_control.startOutputIndex, 
						_control.numOutputsPerStrip,
						_control.numStripsToAccess,
						_control.iaStridePerCol,
						(unsigned char) (*pRegisters).enableRelu,
						(unsigned char) (*pRegisters).accessInfo.accessBank,
						(*pRegisters).accumulatorShiftDirCatShiftAmount));
			} //if, _validControl is TRUE
		}
		break;
		case (OA_BUFFER_WRITER_STATE_NUM_ACCESS) : {
			(*pRegisters).accessInfo.numLoopsPerStip = 
				1 
				+ (((*pRegisters).accessInfo.numOutputsPerStrip - 1) >> DIVIDE_BY_PE_ROWS_PER_GROUP_SHIFT);
			(*pRegisters).accessInfo.iLoopPerStip = 0x0;
			(*pRegisters).accessInfo.indexOutput = (*pRegisters).accessInfo.stripStartOutputIndex;
			*pState =  OA_BUFFER_WRITER_STATE_ACCESS;

		}
		break;
		case (OA_BUFFER_WRITER_STATE_UPDATE_STRIP) : {
			//Default state transition
			*pState = OA_BUFFER_WRITER_STATE_NUM_ACCESS;
			(*pRegisters).accessInfo.iStrip += 0x01;
			(*pRegisters).accessInfo.stripStartOutputIndex += (*pRegisters).accessInfo.oaStridePerCol;
			if ((*pRegisters).accessInfo.iStrip == (*pRegisters).accessInfo.numStripsToAccess)
			{
				*pState = OA_BUFFER_WRITER_STATE_DECODE;
				EMULATOR_PRINT(("[kernelOABuffer WRITER %d] Finished processing processing instruction.\n", 
					colID));
			}
		}
		break;
		case (OA_BUFFER_WRITER_STATE_ACCESS) : {

			int addressBase = (((*pRegisters).accessInfo.accessBank & 0x01) == 0x00) ?
					0x0 : OA_CACHE_DEPTH;

			if (_validValueFromPE == TRUE)
			{
				unsigned int idxDramBlock = 
					((*pRegisters).accessInfo.indexOutput >> ACTIVATION_BURST_SIZE_BYTE_OFFSET);
				unsigned int idxOABlockInDramBlock = 
					((*pRegisters).accessInfo.indexOutput >> DIVIDE_BY_PE_ROWS_PER_GROUP_SHIFT) 
						& ACTIVATION_WIDE_SIZE_IN_PE_ROW_GROUP_MASK;
				
				//Assumption:
				//All the output values from a PE row-group land in the same output activation dram block
				//Specifically, indexOutput aligns with PE_ROW_GROUP blocks and ACTIVATION_BURST blocks
				t_oa_buffer_access_block oaBlock;
				#pragma unroll
				for (unsigned int i=0; i<PE_ROWS_PER_GROUP; i++)
				{
					t_operand shortOutput = modifyOutput(
						_wideOutputFromPEs[i], 
						(*pRegisters).accumulatorShiftDirCatShiftAmount, 
						(*pRegisters).enableRelu
					);

					oaBlock.values[i] = shortOutput;

					EMULATOR_PRINT(("[kernelOABuffer %d] Read and processed values from PEs. "
					 "Value: %#04x, %d out of %d values of the strip are read.\n\n", 
					 colID, 
					 shortOutput, 
					 (*pRegisters).accessInfo.iLoopPerStip + i, 
					 (*pRegisters).accessInfo.numOutputsPerStrip));
				}
				//cacheOutputActivations[indexOutput] = shortOutput;
				if (((*pRegisters).accessInfo.accessBank & 0x01) == 0x00) 
				{
					// cacheOutputActivations0
					// 	[idxDramBlock]
					// 	[(idxInDramBlockBase+i) & ACTIVATION_WIDE_SIZE_BYTE_MASK]
					// 	= shortOutput;
					cacheOutputActivations0
						[idxDramBlock]
						[idxOABlockInDramBlock]
						= oaBlock;
				}
				else
				{
					// cacheOutputActivations1
					// 	[idxDramBlock]
					// 	[(idxInDramBlockBase+i) & ACTIVATION_WIDE_SIZE_BYTE_MASK]
					// 	= shortOutput;
					cacheOutputActivations1
						[idxDramBlock]
						[idxOABlockInDramBlock]
						= oaBlock;
				}


				//Loop variable updates
				(*pRegisters).accessInfo.indexOutput += PE_ROWS_PER_GROUP;
				(*pRegisters).accessInfo.iLoopPerStip += 0x1;

				if ((*pRegisters).accessInfo.iLoopPerStip == (*pRegisters).accessInfo.numLoopsPerStip)
	            {
	                *pState = OA_BUFFER_WRITER_STATE_UPDATE_STRIP;
	            }
			}	
		}	
		break;
		default:
		break;
	} // switch
} //updateOABufferWriter

void getOABufferWriterOutput (
	//Current state
	t_oa_buffer_writer_state _currentState,

	//Current context
	t_oa_buffer_writer_info _currentContext,

	//Interface with the dispatcher
	t_flag* pOutAcceptInstruction,

	//Interface with the PE column channel
	t_flag* pOutAcceptDataFromPE
	)
{
	//Defaults
	*pOutAcceptInstruction = FALSE;
	*pOutAcceptDataFromPE = FALSE;

	switch (_currentState) {
		case (OA_BUFFER_WRITER_STATE_DECODE): {
			*pOutAcceptInstruction = TRUE;
		}
		break;
		case (OA_BUFFER_WRITER_STATE_ACCESS): {
			*pOutAcceptDataFromPE = TRUE;
		}
		default:
		break;
	} // switch _currentState
} //getOABufferWriterOutput

void updateOABufferReader (
	//Inputs from the instruction dispatcher
	t_output_tile_buffer_packet _control,
	t_flag _validControl,

	//Interface with the channel to OA tee
	t_flag _writeSuccessOATee,
	t_flag _requestWrite2OATee,

	//State variables,
	t_oa_buffer_reader_info *pRegisters,

	//State,
	t_oa_buffer_reader_state *pState,

	//Auxillary
	int colID
	)
{
	switch (*pState) {
		case (OA_BUFFER_READER_STATE_DECODE): {
			//Obtain new instruction and update the context
			if (_validControl == TRUE)
			{
				//Accessor update
				(*pRegisters).accessInfo.indexOutput = 0x0;
				(*pRegisters).accessInfo.numOutputsPerStrip = _control.numOutputsPerStrip;
				(*pRegisters).accessInfo.numStripsToAccess = _control.numStripsToAccess;
				(*pRegisters).accessInfo.oaStridePerCol = _control.iaStridePerCol;
				(*pRegisters).accessInfo.iStrip = 0x0;
				(*pRegisters).accessInfo.accessBank = (_control.controlBits >> 9) & 0x01;
				(*pRegisters).accessInfo.numLoopsPerStip = 
					1 + ((_control.numOutputsPerStrip - 1) >> ACTIVATION_BURST_SIZE_BYTE_OFFSET);
				(*pRegisters).accessInfo.iLoopPerStip = 0x0;
				

				//State update
				*pState = OA_BUFFER_READER_STATE_ACCESS;

				EMULATOR_PRINT(("[kernelOABuffer READER %d] START processing instruction. "
						"stripStartOutputIndex =%d, "
						"numOutputsPerStrip=%d, "
						"numStripsToAccess=%d, "
						"oaStridePerCol=%d, "
						"accessBank=%#03x \n\n ",
						colID, 
						_control.startOutputIndex, 
						_control.numOutputsPerStrip,
						_control.numStripsToAccess,
						_control.iaStridePerCol,
						(unsigned char) (*pRegisters).accessInfo.accessBank));
			} //if, _validControl is TRUE
			// else
			// {
			// 	EMULATOR_PRINT(("[kernelOABuffer READER %d] WAITING for instruction. \n\n", 
			// 					colID));
			// }
		} //OA_BUFFER_ACCESS_STATE_DECODE
		break;
		case (OA_BUFFER_READER_STATE_UPDATE_STRIP): {
			//Default state transition
			*pState = OA_BUFFER_READER_STATE_ACCESS;

			(*pRegisters).accessInfo.iLoopPerStip = 0x0;
			(*pRegisters).accessInfo.indexOutput = (*pRegisters).accessInfo.iStrip * (*pRegisters).accessInfo.oaStridePerCol;
			if ((*pRegisters).accessInfo.iStrip == (*pRegisters).accessInfo.numStripsToAccess)
			{
				*pState = OA_BUFFER_READER_STATE_DECODE;
				EMULATOR_PRINT(("[kernelOABuffer READER %d] Finished processing processing instruction.\n", 
					colID));
			}
		} //OA_BUFFER_ACCESS_STATE_UPDATE_STRIP
		break;
		case (OA_BUFFER_READER_STATE_ACCESS): {
			if (_writeSuccessOATee == TRUE) {
				(*pRegisters).accessInfo.indexOutput += ACTIVATION_BURST_SIZE_BYTE;
				(*pRegisters).accessInfo.iLoopPerStip += 0x01;
			}
			// else
			// 	{
			// 		EMULATOR_PRINT(("[kernelOABuffer READER %d] WAITING to send data to the OA Tee channel.\n\n",
			// 						colID));
			// 	}
			if ((*pRegisters).accessInfo.iLoopPerStip == (*pRegisters).accessInfo.numLoopsPerStip)
            {
            	(*pRegisters).accessInfo.iStrip += 0x01;
                *pState = OA_BUFFER_READER_STATE_UPDATE_STRIP;
            }
		} //OA_BUFFER_READER_STATE_ACCESS
		break;
		default:
		break;
	} //switch, *pState
} //updateOABufferReader

void getOABufferReaderOutput (
	//Current state
	t_oa_buffer_reader_state _currentState,
	//Current context
	t_oa_buffer_reader_info _currentContext,

	//Interface with the dispatcher
	t_flag* pOutAcceptInstruction,

	//Buffer, read only
	const t_oa_buffer_access_block cacheOutputActivations0 [OA_CACHE_DEPTH][ACTIVATION_BURST_SIZE_IN_PE_ROW_GROUP],
	const t_oa_buffer_access_block cacheOutputActivations1 [OA_CACHE_DEPTH][ACTIVATION_BURST_SIZE_IN_PE_ROW_GROUP],

	//Interface with the channel to OA Tee
	t_flag* pToOATeeValid,
	t_output_activation_dram_block_tagged* pToOATeeData
	)
{
	//Defaults:
	*pOutAcceptInstruction = FALSE;
	*pToOATeeValid = FALSE;


	switch (_currentState) {
		case (OA_BUFFER_READER_STATE_DECODE): {
			*pOutAcceptInstruction = TRUE;
		} //OA_BUFFER_READER_STATE_DECODE
		break;
		case (OA_BUFFER_READER_STATE_ACCESS): {
			
			t_output_activation_dram_block_tagged taggedOutput;

			int addressBase = ((_currentContext.accessInfo.accessBank & 0x01) == 0x00) ?
					0x0 : OA_CACHE_DEPTH;
			int idxDramBlock = (_currentContext.accessInfo.indexOutput >> ACTIVATION_BURST_SIZE_BYTE_OFFSET);
			//fetch the cluster
			#pragma unroll
			for (unsigned char i=0; i<ACTIVATION_BURST_SIZE_IN_PE_ROW_GROUP; i++)
			{
				t_oa_buffer_access_block block = ((_currentContext.accessInfo.accessBank & 0x01) == 0x00) ?
					cacheOutputActivations0
							[idxDramBlock]
							[i & ACTIVATION_WIDE_SIZE_BYTE_MASK]
					: cacheOutputActivations1
							[idxDramBlock]
							[i & ACTIVATION_WIDE_SIZE_BYTE_MASK];
				#pragma unroll
				for (unsigned char j=0; j<PE_ROWS_PER_GROUP; j++)
				{
					taggedOutput.dramBlock.values[i*PE_ROWS_PER_GROUP + j]
						= block.values[j];
				}
			}
			
			//Only set the isLastInStrip flag.
			//The OA tee is responsible for setting the other flag
			//isFromLastColumn

			*pToOATeeValid = TRUE;
			*pToOATeeData = taggedOutput;
		} //OA_BUFFER_READER_STATE_ACCESS
		break;
		default:
		break;
	} //switch, _currentState
}

void updateOABufferDispatcher (
		//instruction channel interface
		t_flag _inInstructionValid,
		t_output_tile_buffer_packet _inControl,

		//accessor interface
		t_flag _inReaderReady,
		t_flag _inWriterReady,

		//context
		t_oa_buffer_dispatcher_state* pState,
		t_output_tile_buffer_packet* pControlBuffer,

		//sync,
		bool _unblockReader,

		//Auxillary
		int colID
	)
{
	switch (*pState) {
		case (OA_BUFFER_DISPATCH_STATE_IDLE): {
			if (_inInstructionValid == TRUE) {
				*pControlBuffer = _inControl;

				//Depends on the drain/fill bit of the control packet
				//we might transition to OA_BUFFER_DISPATCH_STATE_SEND_WRITER
				//or OA_BUFFER_DISPATCH_STATE_SEND_READER
				if (((_inControl.controlBits >> 0x8) & 0x01) == 0x01)
				{
					//Drain/read from the buffer
					*pState = OA_BUFFER_DISPATCH_STATE_SEND_READER;

				}
				else
				{
					*pState  = OA_BUFFER_DISPATCH_STATE_SEND_WRITER;
				}
			}
		} //OA_BUFFER_DISPATCH_STATE_IDLE
		break;
		case (OA_BUFFER_DISPATCH_STATE_SEND_WRITER): {
			if (_inWriterReady == TRUE)
			{
				*pState = OA_BUFFER_DISPATCH_STATE_IDLE;
				EMULATOR_PRINT(("[kernelOABuffer Dispatcher %d] Sent instruction to WRITER.\n",
							colID));
			}

		} //OA_BUFFER_DISPATCH_STATE_SEND_WRITER
		break;
		case (OA_BUFFER_DISPATCH_STATE_SEND_READER): {
			if ((_inReaderReady == TRUE) && (_unblockReader == true))
			{
				//State transition out of  OA_BUFFER_DISPATCH_STATE_SEND_READER 
				//needs to take consideration of the synchornization flag.
				*pState = OA_BUFFER_DISPATCH_STATE_IDLE;
				EMULATOR_PRINT(("[kernelOABuffer Dispatcher %d] Sent instruction to READER.\n",
							colID));
			}

		} //OA_BUFFER_DISPATCH_STATE_SEND_WRITER
		break;
		default:
		break;
	}
} //updateOABufferDispatcher

void getOABufferDispatcherOutput (
		//Current state
		t_oa_buffer_dispatcher_state _currentState,
		//sync,
		bool _unblockReader,

		//Instruction buffer
		t_output_tile_buffer_packet _controlBuffer,

		//instruction channel interface
		t_flag* pOutReadyForInstruction,

		//accessor interface
		t_flag* pOutValidReaderInstruction,
		t_flag* pOutValidWriterInstruction,
		t_output_tile_buffer_packet* pOutReaderControl,
		t_output_tile_buffer_packet* pOutWriterControl,

		//Auxillary
		int colID
	)
{
	/**
	 * Default values
	 */
	*pOutReadyForInstruction = FALSE;
	*pOutValidWriterInstruction = FALSE;
	*pOutValidReaderInstruction = FALSE;

	/**
	 * Tile control packet interface signal
	 */
	if (_currentState == OA_BUFFER_DISPATCH_STATE_IDLE)
	{
		*pOutReadyForInstruction = TRUE;
	}

	/**
	 * Writer control interface
	 */
	if (_currentState == OA_BUFFER_DISPATCH_STATE_SEND_WRITER)
	{
		*pOutValidWriterInstruction = TRUE;
		*pOutWriterControl = _controlBuffer;
	}

	/**
	 * Reader control interface.
	 * Send instruction to the reader once the writer has finished loading the data
	 */
	if (_currentState == OA_BUFFER_DISPATCH_STATE_SEND_READER)
	{
		if (_unblockReader == true)
		{
			*pOutValidReaderInstruction = TRUE;
			*pOutReaderControl = _controlBuffer;
		}
			
	}
} //getOABufferDispatcherOutput


__attribute__((max_global_work_dim(0)))
__kernel void kernelOATileController (
	VOLATILE  __global const t_oa_tile_controller_instruction* restrict pInst,
	unsigned int numInstructions
	)
{
	/*!
		The activities in one instruction cycle consists of
		sending one OA write instruciton, and then sending all the IA read instructions
		that stream from the OA buffer.
		To faciliate double buffering,
		The first cycle consists only of sending the OA write instruction,
		and the last cycle consists only of sending the OA read instruction
	*/
	uint1_t writeSideIndex = 0;

	for (unsigned int iInstruction=0; iInstruction<numInstructions; iInstruction++)
	{
		/**
		 * 1. Send the streaming (cache--> dram) command, if possible
		 */
		t_oa_tile_controller_instruction inst = pInst[iInstruction];
		unsigned char numOutputTileHeightxWidth = inst.numLocalTilePerColHxW;
		unsigned short numFoldsInGroupCurrentLayer = inst.numFoldsInGroupCurrentLayer;
	    unsigned short numFullFoldsInGroupCurrentLayer = inst.numFullFoldsInCurrentLayer;
	    unsigned short numActiveElementsInFullFold = inst.numActiveElementsInFullFold;
	    unsigned short numActiveRowsInPartialFolds = inst.numActiveElementsInPartialFold;

	    unsigned short numBurstAlignedChannelsPerCurrentGroup = inst.numBurstAlignedChannelsPerCurrentGroup;

	    unsigned char outputModifierBits = inst.flagSparseCatFlagReluCatFlagSourceCatShift;
	    unsigned char numActivePeCols = inst.numActiveCols;
	    unsigned char flagDrainFromMisc = (outputModifierBits >> 0x06) & 0x01;
	    if (flagDrainFromMisc == FALSE)
		{		

		    /*
		    2. Send instruction to drain from the PE array
		    */
		    unsigned short iChannelInGroup = 0;
		    unsigned short iFoldInGroup = 0;
		    //unsigned short iOutputTileHxWDrain = 0;

		   	EMULATOR_PRINT(("[kernelOATileController] START sending the drain-from-array instruction for instruction %d\n\n", 
					iInstruction));

		    for  (unsigned short i=0; i < inst.numDrainInstructions; i++)
		    {
		    	unsigned short numActivePeRows = (iFoldInGroup < numFullFoldsInGroupCurrentLayer) ?
		    		numActiveElementsInFullFold : numActiveRowsInPartialFolds;
		    	unsigned short startOutputIndex = iChannelInGroup;

		    	t_output_tile_buffer_packet_tagged bufferPacketTagged;
		    	bufferPacketTagged.bufferPacket.startOutputIndex = startOutputIndex;
		    	bufferPacketTagged.bufferPacket.numOutputsPerStrip = numActivePeRows;
		    	bufferPacketTagged.bufferPacket.numStripsToAccess = numOutputTileHeightxWidth;
		    	bufferPacketTagged.bufferPacket.iaStridePerCol = numBurstAlignedChannelsPerCurrentGroup;
		    	bufferPacketTagged.bufferPacket.controlBits = 
		    		( (((unsigned short) writeSideIndex) & 0x01) << 0x9)
		    		| ((unsigned short) outputModifierBits);
		    	bufferPacketTagged.maxColID = (numActivePeCols - 1);

		    	write_channel_intel(channel_oa_noc_control[0], bufferPacketTagged);

		    	/*
		    	Parameter updates
		    	*/

	    		iFoldInGroup++;
	    		iChannelInGroup += numActivePeRows;
		    } //while-loop.  Send instruction to drain from the PE array
		    EMULATOR_PRINT(("[kernelOATileController] FINISHED sending the drain-from-array instruction for instruction %d\n\n", 
					iInstruction));
		} // Send the drain (compute --> cache) command

		{
			EMULATOR_PRINT(("[kernelOATileController] START sending the write-to-memory instruction for instruction %d\n\n", 
				iInstruction));

		    t_output_tile_buffer_packet_tagged bufferPacketTagged;
	    	bufferPacketTagged.bufferPacket.startOutputIndex = 0x0;
			bufferPacketTagged.bufferPacket.numOutputsPerStrip = numBurstAlignedChannelsPerCurrentGroup;
	    	bufferPacketTagged.bufferPacket.numStripsToAccess = numOutputTileHeightxWidth;
	    	bufferPacketTagged.bufferPacket.iaStridePerCol = numBurstAlignedChannelsPerCurrentGroup;
	    	bufferPacketTagged.bufferPacket.controlBits = 
	    		( (((unsigned short) (writeSideIndex)) & 0x01) << 0x9)
	    		| (((unsigned short) 0x1) << 0x8 ) 
	    		| ((unsigned short) outputModifierBits);

	    	bufferPacketTagged.maxColID = (numActivePeCols - 1);

	    	/**
	    	 * Information for OA Tee only
	    	 */
	    	bufferPacketTagged.bufferPacket.numNominalDramBlocksPerOATee = 
	    		inst.numNominalDramBlocksPerStrip * inst.numLocalTilePerColHxW;

	    	write_channel_intel(channel_oa_noc_control[0], bufferPacketTagged);
	    	EMULATOR_PRINT(("[kernelOATileController] FINISHED sending the write-to-memory instruction for instruction cycle %d\n\n", 
				iInstruction));
		}

	    //SWAP the read side and the write side
		writeSideIndex = (~writeSideIndex) & 0x01;
	} // iterate over instructions cycles
}

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
		t_flag isFromMisc = (controlPacketTagged.bufferPacket.controlBits & 0x040) >> 6;

		//Only send command to the OA buffer is the results are drained from the convolution PE array
		if (isFromMisc == FALSE)
		{
			write_channel_intel(channel_control_to_oa_buffer_local[colID], controlPacketTagged.bufferPacket);
		}

		//Send instruction to the OA tee, if this is a drain command
		t_output_tile_tee_packet teePacket;
		teePacket.numNominalDramBlocksPerOATee = controlPacketTagged.bufferPacket.numNominalDramBlocksPerOATee;
		teePacket.flagSourceCatFlagSparseFlagMaxColID = (
				((unsigned char) (maxColID & 0x0F))
				| (controlPacketTagged.bufferPacket.controlBits & 0x060));
		uint1_t drainBuffer = (controlPacketTagged.bufferPacket.controlBits & 0x100) >> 8;
		if (drainBuffer == TRUE)
		{
			//Write to the OA Tee
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


#define OA_TEE_STATE_DECODE 0X0
#define OA_TEE_STATE_DRAIN_SELF 0X1
#define OA_TEE_STATE_RESEND_SELF 0x2
#define OA_TEE_STATE_DRAIN_OTHER 0X3
#define OA_TEE_STATE_RESEND_OTHER 0X4
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelOATee ()
{
	typedef uint3_t t_state;
	int colID = get_compute_id(0);

	/**
	 * Loop-carried registers -- commmon to both sparse and dense
	 */
	t_state regState = OA_TEE_STATE_DECODE;
	//Number of nominal blocks that this unit has created. [COUNTER]
	unsigned short regCountSelfNominalBlocks = 0;
	//Total number of nominal blocks that this unit should create. [THRESHOLD]
	unsigned short regTotalSelfNominalBlocks = 0;
	//Flag that indicates the source
	t_flag regDrainFromMisc = FALSE;

	//Flag. Whether this unit is the last column
	t_flag regIsLastCol = (colID == (PE_COLS-1)) ? TRUE : FALSE;

	//Register bank of the clusters
	t_output_activation_dram_block_tagged regDramBlockTagged;

	#pragma ii 1
	#pragma speculated_iterations 0
	while (true)
	{
		// EMULATOR_PRINT(("[kernelOATee %d] regState=%#04x\n",
		// 					(unsigned int) colID, 
		// 					(unsigned int) regState
		// 					));
		/**
		 * Local signals -- seen by both type of architectures
		 */
		t_state sigState = regState;
		unsigned short sigCountSelfNominalBlocks = regCountSelfNominalBlocks;
		unsigned short sigTotalSelfNominalBlocks = regTotalSelfNominalBlocks;
		t_flag sigIsLastCol = (colID == (PE_COLS-1)) ? TRUE : regIsLastCol;
		t_flag sigDrainFromMisc = regDrainFromMisc;
		t_output_activation_dram_block_tagged sigDramBlockTagged = regDramBlockTagged;

		/**
		 * Interface signals of the self-draining channel
		 */
		t_flag readSuccessInstruction = FALSE;

		t_flag readSuccessDrainSelf = FALSE;

		/**
		 * Interface signals of the other-draining channel
		 */
		t_flag readRequestDrainOther = FALSE;
		t_flag readSuccessDrainOther = FALSE;

		/**
		 * Interface signals of the sending channel
		 */
		t_flag writeRequestSendBlock = FALSE;
		t_flag writeSuccessSendBlock = FALSE;

		/**
		 * Interaction with the instruction channel
		 * Modify: sigTeeControl
		 */
		t_output_tile_tee_packet sigTeeControl;
		if (regState == OA_TEE_STATE_DECODE)
		{
			bool readSuccess = false;
			sigTeeControl = read_channel_nb_intel(channel_oa_tee_local[colID], &readSuccess);
			if (readSuccess == true)
			{
				readSuccessInstruction = TRUE;
			}
		}

		/**
		 * Interaction with the local cluster input channel
		 */
		t_output_activation_dram_block_tagged sigOutputFromSelfCompute;
		if (regState == OA_TEE_STATE_DRAIN_SELF)
		{
			bool readSuccess = false;
			
			if (sigDrainFromMisc == FALSE)
			{
				sigOutputFromSelfCompute = 
					read_channel_nb_intel(channel_oa_buffer_to_oa_tee[colID], &readSuccess);
			}
			else
			{
				if (colID < MISC_COLS)
				{
					sigOutputFromSelfCompute = 
						read_channel_nb_intel(channel_misc_to_oa_tee[colID], &readSuccess);
				}
			}
			if (readSuccess == true)
			{
				readSuccessDrainSelf = TRUE;
			}
		}

		/**
		 * Interaction with the drain-other channel
		 */
		if (colID < (PE_COLS-1))
		{
			if (regState == OA_TEE_STATE_DRAIN_OTHER)
			{
				readRequestDrainOther = TRUE;
			}

			if (readRequestDrainOther == TRUE)
			{
				bool readSuccess = false;
				sigDramBlockTagged = 
					read_channel_nb_intel(channel_output_wide[colID+1], &readSuccess);
				if (readSuccess == true)
				{
					readSuccessDrainOther = TRUE;
				}
			}
		}

		/**
		 * Interaction with the send channel
		 */
		if (
				(readSuccessDrainOther == TRUE)
				|| (readSuccessDrainSelf == TRUE)
				|| (regState == OA_TEE_STATE_RESEND_SELF)
				|| (regState == OA_TEE_STATE_RESEND_OTHER)	
		   )
		{
			writeRequestSendBlock = TRUE;

			//If the block will take on this unit's data,
			//then we need to set the flags and payload accordingly.
			if (readSuccessDrainSelf == TRUE)
			{
				//Set the flags: is valid cat whether this is the last col
				sigDramBlockTagged = sigOutputFromSelfCompute;
				sigDramBlockTagged.isFromLastColumn = sigIsLastCol;
			}
		}

		if (writeRequestSendBlock == TRUE)
		{
			bool writeSuccess = write_channel_nb_intel(channel_output_wide[colID], sigDramBlockTagged);
			writeSuccessSendBlock = (writeSuccess == true) ? TRUE : FALSE;

			#if defined(EMULATOR)
				if (writeSuccess == true)
				{
					EMULATOR_PRINT(("[kernelOATee %d] Sent a dram block.\n"
							"regCountSelfNominalBlocks=%d, "
							"regState=%#04x, "
							"regIsLastCol=%d\n",
							(unsigned int) colID, 
							(unsigned int) regCountSelfNominalBlocks,
							(unsigned int) regState,
							(unsigned int) regIsLastCol
							));
				}
			#endif
		}

		/**
		 * State transition and variable update
		 */
		switch (regState) {
			case (OA_TEE_STATE_DECODE): {
				if (readSuccessInstruction == TRUE)
				{
					sigCountSelfNominalBlocks = 0;
					sigTotalSelfNominalBlocks = sigTeeControl.numNominalDramBlocksPerOATee;

					unsigned char maxColID = 
						sigTeeControl.flagSourceCatFlagSparseFlagMaxColID & 0x0F;
					sigIsLastCol = (maxColID > colID) ? FALSE : TRUE;
					sigDrainFromMisc = 
						(sigTeeControl.flagSourceCatFlagSparseFlagMaxColID & 0x040) >> 0x6;

					//State transition
					sigState = OA_TEE_STATE_DRAIN_SELF;

					EMULATOR_PRINT(("[kernelOATee %d] Received instruction \n"
							"numNominalDramBlocksPerOATee=%d, "
							"Source is from MISC=%d, "
							"maxColID=%d\n",
							(unsigned int) colID, 
							(unsigned int) sigTotalSelfNominalBlocks,
							(unsigned int) sigDrainFromMisc,
							(unsigned int) maxColID
							));
				}
			} //case (OA_TEE_STATE_DECODE)
			break;
			case (OA_TEE_STATE_DRAIN_SELF): {
				if (readSuccessDrainSelf == TRUE)
				{
					// sigDramBlockTagged.block.clusters[sigIdxClusterInDramBlock] 
					// 	= sigClusterTagged.cluster;

					sigState = OA_TEE_STATE_RESEND_SELF;
					if (writeSuccessSendBlock == TRUE)
					{
						sigCountSelfNominalBlocks++;

						sigState = OA_TEE_STATE_DRAIN_OTHER;
						if (sigIsLastCol == TRUE)
						{
							if (sigCountSelfNominalBlocks == regTotalSelfNominalBlocks)
							{
								sigState = OA_TEE_STATE_DECODE;
							}
							else // sigCountSelfNominalBlocks < regTotalSelfNominalBlocks
							{
								sigState = OA_TEE_STATE_DRAIN_SELF;
							} //sigCountSelfNominalBlocks < regTotalSelfNominalBlocks
						} //sigIsLastCol == TRUE
					} //writeSuccessSendBlock == TRUE
				}//readSuccessDrainSelf == TRUE
			} //case (OA_TEE_STATE_DRAIN_CONV)
			break;
			case (OA_TEE_STATE_RESEND_SELF): {
				if (writeSuccessSendBlock == TRUE)
				{
					sigCountSelfNominalBlocks++;

					sigState = OA_TEE_STATE_DRAIN_OTHER;
					if (sigIsLastCol == TRUE)
					{
						if (sigCountSelfNominalBlocks == regTotalSelfNominalBlocks)
						{
							sigState = OA_TEE_STATE_DECODE;
						}
						else // sigCountSelfNominalBlocks < regTotalSelfNominalBlocks
						{
							sigState = OA_TEE_STATE_DRAIN_SELF;
						} //sigCountSelfNominalBlocks < regTotalSelfNominalBlocks
					} //sigIsLastCol == TRUE
				} //writeSuccessSendBlock == TRUE
			} //(OA_TEE_STATE_RESEND_SELF_DATA)
			break;
			case (OA_TEE_STATE_DRAIN_OTHER): {
				if (readSuccessDrainOther == TRUE)
				{
					sigState = OA_TEE_STATE_RESEND_OTHER;

					if (writeSuccessSendBlock == TRUE)
					{
						sigState = OA_TEE_STATE_DRAIN_OTHER;
						t_flag sigIsFromLastCol = sigDramBlockTagged.isFromLastColumn;
						if (sigIsFromLastCol == TRUE)
						{
							if (sigCountSelfNominalBlocks == regTotalSelfNominalBlocks)
							{
								sigState = OA_TEE_STATE_DECODE;
							}
							else // sigCountSelfNominalBlocks < regTotalSelfNominalBlocks
							{
								sigState = OA_TEE_STATE_DRAIN_SELF;
							} //sigCountSelfNominalBlocks < regTotalSelfNominalBlocks
						} //(sigIsFromLastCol == TRUE)
					} //(writeSuccessSendBlock == TRUE)
				} //readSuccessDrainOther == TRUE
			} //case (OA_TEE_STATE_DRAIN_OTHER_DATA)
			break;
			case (OA_TEE_STATE_RESEND_OTHER): {
				if (writeSuccessSendBlock == TRUE)
				{
					sigState = OA_TEE_STATE_DRAIN_OTHER;
					t_flag sigIsFromLastCol = sigDramBlockTagged.isFromLastColumn;
					if (sigIsFromLastCol == TRUE)
					{
						if (sigCountSelfNominalBlocks == regTotalSelfNominalBlocks)
						{
							sigState = OA_TEE_STATE_DECODE;
						}
						else // sigCountSelfNominalBlocks < regTotalSelfNominalBlocks
						{
							sigState = OA_TEE_STATE_DRAIN_SELF;
						} //sigCountSelfNominalBlocks < regTotalSelfNominalBlocks
					} //(sigIsFromLastCol == TRUE)
				} //(writeSuccessSendBlock == TRUE)
			} //case (OA_TEE_STATE_RESEND_OTHER_DATA)
			break;
			default: break;
		} //switch (regState)

		/**
		 * Register update assignment
		 */
		regState = sigState;
		regCountSelfNominalBlocks = sigCountSelfNominalBlocks;
		regTotalSelfNominalBlocks = sigTotalSelfNominalBlocks;
		regIsLastCol = sigIsLastCol;
		regDramBlockTagged = sigDramBlockTagged;
		regDrainFromMisc = sigDrainFromMisc;

	}//while
} //kernelOATee

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

#define STATE_FILTER_STREAMER_READ_CACHE_WAIT 0X0
#define STATE_FILTER_STREAMER_READ_CACHE_READ 0X1
//#define STATE_FILTER_STREAMER_READ_CACHE_WAIT 0X4

/*! kernelFilterStreamer
	\brief Stream filter values to the PE array
*/
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_ROWS)))
__kernel void kernelFilterBuffer ()
{
	int rowID = get_compute_id(0);

	typedef uint2_t t_filter_streamer_write_state;
	typedef uint1_t t_filter_streamer_read_state;

	typedef struct {
    	t_char values[WEIGHT_BURST_SIZE_VALUE_BYTE];
	} t_weight_dram_block_values;
	//important to size the bankwidth, otherwise the default 32 bit will be used, resulting in complex store logic
	t_weight_dram_block_values cacheNzBlocks [2][KERNEL_CACHE_DEPTH] __attribute__((bankwidth(WEIGHT_BURST_SIZE_VALUE_BYTE))); 
	#if defined(SPW_SYSTEM)
	typedef struct {
    	t_uchar indices[WEIGHT_WIDE_SIZE*INDEX_CHAR_ARRAY_SIZE];
	} t_weight_dram_block_indices;
	t_weight_dram_block_indices cacheIndices [2][KERNEL_CACHE_DEPTH] __attribute__((bankwidth(WEIGHT_WIDE_SIZE*INDEX_CHAR_ARRAY_SIZE))); 
	#endif
	uint1_t regWriteSide = 0x0;
	unsigned int regOutputxTransferBlocks[2];
	//unsigned char maxOutputHeightTileSize[2]; //maxTP
	//unsigned char maxOutputWidthTileSize[2]; //maxTQ 
	unsigned short maxTransferBlockInFilter[2]; //maxCg

	unsigned char maxPeCols[2];
	t_bias cacheBias[2];
	t_flag regIsRealFilter[2];
	#if defined(SPW_SYSTEM)
	unsigned char regNumNZClustersInPruneRange[2];
	#endif

	//=================Write into cache variables=================
	t_filter_streamer_write_state stateWriteCache = STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL;
	unsigned short iTransferBlockInFilterWrite; //iCg

	//=================Read from cache variables=================
	t_filter_streamer_read_state stateReadCache = STATE_FILTER_STREAMER_READ_CACHE_WAIT;
	unsigned short iTransferBlockInFilterRead = 0; //iCg
	//unsigned char iWidthInOutputTileRead; //pq*A
	//unsigned char iHeightInOutputTileRead; //p
	unsigned int iOutputXTransferBlocksRead = 0;
	#if defined(SPW_SYSTEM)
	unsigned char iNZClusterInPruneRange = 0;
	#endif



	//#pragma ivdep array(cacheNzBlocks)
	#pragma ivdep
	#pragma ii 1
	#pragma speculated_iterations 0
	while (true)
	{
		//===============Write side====================
		t_filter_streamer_write_state nextStateWriteCache = stateWriteCache;
		{
			bool success = false;
			t_weight_dram_block writeBlock;
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
					regOutputxTransferBlocks[regWriteSide] = control.numOutputsXNumTransferBlocks;
					maxPeCols[regWriteSide] = control.maxPeCols;
					maxTransferBlockInFilter[regWriteSide] = control.numTransferBlocks;
					cacheBias[regWriteSide] = control.bias;
					t_flag flagIsReal = (t_flag) control.flagIsReal;
					regIsRealFilter[regWriteSide] = flagIsReal;
					#if defined(SPW_SYSTEM)
					regNumNZClustersInPruneRange[regWriteSide] = control.numNZClustersPerPruneRange;
					#endif
					
					iTransferBlockInFilterWrite = 0;


					EMULATOR_PRINT(("[kernelFilterBuffer %d] Received setup packet for a new filter. Number of transfer blocks to follow: %d\n\n", rowID, control.numTransferBlocks));

					//If this filter feeder is to provide padding,
					//then the filter mover won't provide anymore blocks
					//so the writer should transition to the wait state
					nextStateWriteCache = (flagIsReal == TRUE) ?
						STATE_FILTER_STREAMER_WRITE_CACHE_WRITE:
						STATE_FILTER_STREAMER_WRITE_CACHE_WAIT;
				}
			} // STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL
			else if (stateWriteCache == STATE_FILTER_STREAMER_WRITE_CACHE_WRITE)
			{
				if (success)
				{
					unsigned short dramBlockIndex = (iTransferBlockInFilterWrite >> WEIGHT_WIDE_SIZE_OFFSET);
					
					t_weight_dram_block_values values;

					#pragma unroll
					for (unsigned char i=0; i<WEIGHT_BURST_SIZE_VALUE_BYTE; i++)
					{
						values.values[i] = writeBlock.values[i];
					}


					cacheNzBlocks[regWriteSide][dramBlockIndex] = values;

					#if defined(SPW_SYSTEM)
						t_weight_dram_block_indices indices;
						#pragma unroll
						for (unsigned char i=0; i<WEIGHT_BURST_SIZE_INDEX_BYTE; i++)
						{
							indices.indices[i] = writeBlock.indices[i];
						}
						cacheIndices[regWriteSide][dramBlockIndex] = indices;
					#endif

					iTransferBlockInFilterWrite += WEIGHT_WIDE_SIZE;
					if (iTransferBlockInFilterWrite >= maxTransferBlockInFilter[regWriteSide])
					{
						nextStateWriteCache = STATE_FILTER_STREAMER_WRITE_CACHE_WAIT;
					}
				}
			} // STATE_FILTER_STREAMER_WRITE_CACHE_WRITE
		} // WRITE

		t_filter_streamer_read_state nextStateReadCache = stateReadCache;
		//t_transferblock_tagged weightBlockTagged;
		
		// if (stateReadCache == STATE_FILTER_STREAMER_READ_CACHE_SETUP)
		// {
		// 	iTransferBlockInFilterRead = 0;
		// 	if (iOutputRead == maxOutputCount[(~regWriteSide) & 0x1])
		// 	{
		// 		nextStateReadCache = STATE_FILTER_STREAMER_READ_CACHE_WAIT;
		// 		iOutputRead = 0;
		// 		EMULATOR_PRINT(("[kernelFilterBuffer %d] FINISHED stream all the weights in the buffer for the tile.\n\n", rowID));
		// 	}
		// 	else
		// 	{
		// 		nextStateReadCache = STATE_FILTER_STREAMER_READ_CACHE_READ;
		// 		//iOutputRead++;
		// 	}
		// } // STATE_FILTER_STREAMER_READ_CACHE_SETUP
		// Send bias, then followed by the clusters
		if ( stateReadCache == STATE_FILTER_STREAMER_READ_CACHE_READ)
		{
			t_pe_w_block peWeightBlock;

			t_flag tempIsLastBlockInFilter = FALSE;
			#if defined(SPW_SYSTEM)
				t_flag tempIsLastBlockInPruneRange = FALSE; 
			#endif

			// if (iTransferBlockInFilterRead > 0)
			// {
				// unsigned short dramIndex = (iTransferBlockInFilterRead - 1) >> WEIGHT_WIDE_SIZE_OFFSET;
				// unsigned short indexInDramBlock = (iTransferBlockInFilterRead - 1) & WEIGHT_WIDE_SIZE_REMAINDER_MASK;
				unsigned short dramIndex = iTransferBlockInFilterRead >> WEIGHT_WIDE_SIZE_OFFSET;
				unsigned short indexInDramBlock = iTransferBlockInFilterRead & WEIGHT_WIDE_SIZE_REMAINDER_MASK;

				t_weight_dram_block_values valueBlock = cacheNzBlocks[(~regWriteSide) & 0x1][dramIndex];
				//Bridge the weight values
				#pragma unroll 
				for (int v=0; v<PE_SIMD_SIZE*CLUSTER_SIZE; v++)
				{
					//Handle zero-padded filter
					if (regIsRealFilter[(~regWriteSide) & 0x1] == TRUE)
					{
						peWeightBlock.values[v] = valueBlock.values[
							(indexInDramBlock << (PE_SIMD_SIZE_CLUSTER_OFFSET + VALUE_TO_CLUSTER_SHIFT)) + v
							];
					}
					else
					{
						peWeightBlock.values[v] = 0x0;
					}
				}

				#if defined(SPW_SYSTEM)
					t_weight_dram_block_indices indicesBlock = cacheIndices[(~regWriteSide) & 0x1][dramIndex];
					#pragma unroll
					for (unsigned char iChar=0; iChar<INDEX_CHAR_ARRAY_SIZE; iChar++)
					{
						unsigned char index0 = iChar << 1; //*2
						unsigned char index1 = (iChar << 1) + 1; //*2, +1
						unsigned char val = indicesBlock.indices[
									(indexInDramBlock << INDEX_CHAR_ARRAY_SIZE_OFFSET)
									+ iChar
								];

						t_spw_index val0 = val & CHAR_TO_SPW_INDEX_MASK;
						t_spw_index val1 = (val >> 0x04) & CHAR_TO_SPW_INDEX_MASK;
						if (index0 < PE_SIMD_SIZE)
						{
							peWeightBlock.indices[index0] = val0;
						}
						if (index1 < PE_SIMD_SIZE)
						{
							peWeightBlock.indices[index1] = val1;
						}
					}
				#endif

				//The comparsion is greater or equal to 
				//since iTransferBlockInFilterRead is also incremented to transmitting the bias
				tempIsLastBlockInFilter = 
					((iTransferBlockInFilterRead + (unsigned short) 1) >= maxTransferBlockInFilter[(~regWriteSide) & 0x1]) ?
					TRUE : FALSE;
				#if defined(SPW_SYSTEM)
					tempIsLastBlockInPruneRange = 
						((iNZClusterInPruneRange + (unsigned short) 1) == regNumNZClustersInPruneRange[(~regWriteSide) & 0x1]) ?
						TRUE : FALSE;
				#endif
			// }
			// else //Bias
			// {
			// 	t_bias bias = (regIsRealFilter[(~regWriteSide) & 0x1] == TRUE) ?
			// 		cacheBias[(~regWriteSide) & 0x1] : 0x0;
			// 	peWeightBlock.values[0] = bias & 0x0FF;
			// 	peWeightBlock.values[1] = (bias >> 0x08) & 0x0FF;
			// 	//weightBlockTagged.isLast = false;
			// }
			peWeightBlock.bias = (regIsRealFilter[(~regWriteSide) & 0x1] == TRUE) ?
					cacheBias[(~regWriteSide) & 0x1] : 0x0;
			//Set the control signals for the block
			peWeightBlock.isLastInFilter = tempIsLastBlockInFilter;
			peWeightBlock.maxTransportID = maxPeCols[(~regWriteSide) & 0x1];
			#if defined(SPW_SYSTEM)
				peWeightBlock.isLastInPruneRange = tempIsLastBlockInPruneRange;
			#endif
			
			// EMULATOR_PRINT(("[kernelFilterBuffer %d] Attempt to send transfer block %d / %d, in the %d / %d time.\n\n", 
			// 		rowID, iTransferBlockInFilterRead, maxTransferBlockInFilter[(~regWriteSide) & 0x1], iOutputRead, maxOutputCount[(~regWriteSide) & 0x1]));
			bool success = false;
			success = write_channel_nb_intel(channel_weight[rowID][0], peWeightBlock);
			if (success)
			{
				#if !defined(SPW_SYSTEM)
					EMULATOR_PRINT((
						"[kernelFilterBuffer %d]"
						"Sent transfer block %d / %d. Total transfer: %d / %d.\n"
						"TB[0-3]: %#04x %#04x %#04x %#04x. \n"
						"isLastInFilter: %#03x, maxTransportID: %#04x.\n",
						rowID, 
						iTransferBlockInFilterRead, 
						maxTransferBlockInFilter[(~regWriteSide) & 0x1], 
						iOutputXTransferBlocksRead, 
						regOutputxTransferBlocks[(~regWriteSide) & 0x1],
						peWeightBlock.values[0],
	                    peWeightBlock.values[1],
	                    peWeightBlock.values[2],
	                    peWeightBlock.values[3],
	                    (unsigned int) peWeightBlock.isLastInFilter,
	                    (unsigned int) peWeightBlock.maxTransportID
						));
				#else
					EMULATOR_PRINT((
						"[kernelFilterBuffer %d]"
						"Sent transfer block %d / %d. Total transfer: %d / %d.\n"
						"TB[0-3]: %#04x %#04x %#04x %#04x. \n"
						"isLastInFilter: %#03x, isLastInPruneRange: %#03x, maxTransportID: %#04x.\n",
						rowID, 
						iTransferBlockInFilterRead, 
						maxTransferBlockInFilter[(~regWriteSide) & 0x1], 
						iOutputXTransferBlocksRead, 
						regOutputxTransferBlocks[(~regWriteSide) & 0x1],
						peWeightBlock.values[0],
	                    peWeightBlock.values[1],
	                    peWeightBlock.values[2],
	                    peWeightBlock.values[3],
	                    (unsigned int) peWeightBlock.isLastInFilter,
	                    (unsigned int) peWeightBlock.isLastInPruneRange,
	                    (unsigned int) peWeightBlock.maxTransportID
						));
				#endif

				//Update the counters
				#if defined(SPW_SYSTEM)
					//Only update iNZClusterInPruneRange when sending weights, 
					//NOT when sending biases
					// if (iTransferBlockInFilterRead > 0)
					// {
						iNZClusterInPruneRange++;
						if (iNZClusterInPruneRange == regNumNZClustersInPruneRange[(~regWriteSide) & 0x1])
						{
							iNZClusterInPruneRange = 0x0;
						}
					// }
				#endif
				// //Omit plus 1 to send the bias
				// if ((iTransferBlockInFilterRead + 1) >= maxTransferBlockInFilter[(~regWriteSide) & 0x1])
				if (tempIsLastBlockInFilter == TRUE)
				{
					iTransferBlockInFilterRead = 0;
				}
				else
				{
					iTransferBlockInFilterRead++;
				}

				iOutputXTransferBlocksRead++;
				if (iOutputXTransferBlocksRead == regOutputxTransferBlocks[(~regWriteSide) & 0x1]) {
					nextStateReadCache = STATE_FILTER_STREAMER_READ_CACHE_WAIT;
					iOutputXTransferBlocksRead = 0;
					EMULATOR_PRINT(("[kernelFilterBuffer %d] FINISHED stream all the weights in the buffer for the tile.\n\n", rowID));
					
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

#define STATE_DRAIN_TRANSPORT_DRAIN_SELF 0X0
#define STATE_DRAIN_TRANSPORT_DRAIN_OTHERS 0x1

typedef uint1_t t_drain_state;

__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__((num_compute_units(PE_ROW_GROUPS, PE_COLS)))
__attribute__((autorun))
__kernel void kernelDrainTransport ()
{
	//Obtain kernel location
	#ifdef FULL_SYSTEM
		int idx = get_compute_id(1);
		int idy = get_compute_id(0);
	#else
		int idx = 0;
		int idy = 0;
	#endif	

	#if defined(EMULATOR) && defined(EMUPRINT)
		unsigned short countPrint = 0x0;
	#endif

	/**
	 * Registers
	 */
	t_drain_state regDrainState = (idy==0) ? 
		STATE_DRAIN_TRANSPORT_DRAIN_SELF : STATE_DRAIN_TRANSPORT_DRAIN_OTHERS;
	t_conv_drain_tagged regDrainPacket;


	#pragma ii 1
	#pragma speculated_iterations 0
	while (1)
	{
		/**
		 * 	Local signals
		 */
		t_drain_state nextDrainState = regDrainState;

		t_conv_drain_multiple_tagged sigDrainPacket;

		/**
		 * Data path
		 */
		if (regDrainState == STATE_DRAIN_TRANSPORT_DRAIN_SELF)
		{
			sigDrainPacket = read_channel_intel(
					channel_drain_conv_local[idy][idx]
				);
		} //if (drainState == STATE_DRAIN_TRANSPORT_DRAIN_SELF)
		else //(regDrainState == STATE_DRAIN_TRANSPORT_DRAIN_OTHERS)
		{
			if (idy > 0)
			{
				sigDrainPacket = read_channel_intel(
						channel_drain_conv[idy-1][idx]
					);
			}
		} //else if (drainState == STATE_DRAIN_TRANSPORT_DRAIN_OTHERS)


		write_channel_intel(
				channel_drain_conv[idy][idx],
				sigDrainPacket
			);

		/**
		 * State update
		 *  #define STATE_DRAIN_TRANSPORT_DRAIN_SELF 0X0
			#define STATE_DRAIN_TRANSPORT_SEND_SELF_RETRY 0X1
			#define STATE_DRAIN_TRANSPORT_DRAIN_OTHERS 0x2
			#define STATE_DRAIN_TRANSPORT_SEND_OTHERS_RETRY 0x3
		 */
		switch (regDrainState) {
			case STATE_DRAIN_TRANSPORT_DRAIN_SELF: {
				if (idy == 0)
				{
					nextDrainState = STATE_DRAIN_TRANSPORT_DRAIN_SELF;
				}
				else
				{
					nextDrainState = STATE_DRAIN_TRANSPORT_DRAIN_OTHERS;
				}
			}	
			break; //STATE_DRAIN_TRANSPORT_DRAIN_SELF
			case STATE_DRAIN_TRANSPORT_DRAIN_OTHERS: {
				nextDrainState = STATE_DRAIN_TRANSPORT_DRAIN_OTHERS;
				t_flag isLast = sigDrainPacket.flagIsLast;
				uint5_t sourceRowID = sigDrainPacket.sourceRowGroupID;
				if ((isLast == FALSE) && (sourceRowID == ((uint5_t) (idy-1))))
				{
					nextDrainState = STATE_DRAIN_TRANSPORT_DRAIN_SELF;
				}
			}	
			break; //STATE_DRAIN_TRANSPORT_DRAIN_SELF
			default:
			break;
		} //switch (regDrainState)


		/**
		 * Register updates
		 */
		regDrainState = nextDrainState;
	}
} //kernelDrainTransport



