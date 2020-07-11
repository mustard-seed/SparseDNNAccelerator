#include "params.hpp"
#include "device_structures.hpp"
#include "channels.hpp"
#include "device_utils.hpp"
#include "ihc_apint.h"
#include "rtl_lib.hpp"

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
		VOLATILE __global const t_dram_block* restrict pIA,

		#if defined(SPARSE_SYSTEM)
			// Memory port for TB count per strip
			VOLATILE __global const t_streamblock_address* restrict pTBCount,
		#endif

		//Memory port for transfer instructions
		VOLATILE __global const t_ia_mover_instruction* restrict pInstruction,
		//Number of transfer instructions
		unsigned int numInstruction
	)
{
	for (unsigned int iInst=0; iInst<numInstruction; iInst++)
	{
		//Read the instruction
		t_ia_mover_instruction inst = pInstruction[iInst];

		/*! Unpackethe concatenated fields of the instruction */
		//Number of compute columns that are active in this transfer
		unsigned char numActiveCols = inst.memRegionCatSparseFlagCatDestinationCatSyncCatNumActiveCols & 0x0F;
		//Flag for waiting for synchornization flag from the output writer AFTER performing transfers
		t_flag syncWithOA = (inst.memRegionCatSparseFlagCatDestinationCatSyncCatNumActiveCols >> 0x04) & 0x01;
		//Flag for the transfer destignatiion. 1 for MISC channel, 0 for CONV PE array.
		t_flag destinationMisc = (inst.memRegionCatSparseFlagCatDestinationCatSyncCatNumActiveCols >> 0x05) & 0x01;
		//Flag for whether the input is sparse
		t_flag sparseInput = (inst.memRegionCatSparseFlagCatDestinationCatSyncCatNumActiveCols >> 0x06) & 0x01;
		//Flag for which memory region to fetch the input and TB from. TODO: remove this if not necessary.
		t_flag memRegion = (inst.memRegionCatSparseFlagCatDestinationCatSyncCatNumActiveCols >> 0x07) & 0x01;
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
		uint4_t iColSPUnitIndex = hInitSPIndex;
		uint4_t iRowSPUnitIndex = vInitSPIndex;

		//Iterators of the column and row positions of the strip inside the tile
		signed char iColInSPTile = 0;
		signed char iRowInSPTile = 0;

		//Address offset contributions from row and column movements in the tile
		signed int offsetIADramBlockRow = 0;
		signed int offsetIADramBlockCol = 0;
		#if defined(SPARSE_SYSTEM)
			signed int offsetTBCountRow = 0;
			signed int offsetTBCountCol = 0;
		#endif

		//iterate over all IA strips in tile
		for (unsigned short iter=0; iter < inst.tileSPWidthxTileSPHeight; iter++)
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
						// if (memRegion == 0x0)
						// {
						// 	numTBInStrip = pTBCount1[addressTBCountDDR];
						// }
						// else
						// {
						// 	numTBInStrip = pTBCount2[addressTBCountDDR];
						// }
						numTBInStrip = pTBCount[addressTBCountDDR];
					}
					else
					{
						numTBInStrip = (t_ushort) inst.numCWOrTBInGroup;
					}
				}
				else
				{
					numTBInStrip = (t_ushort) inst.numCWOrTBInGroup;
				}
			#else
				unsigned short numTBInStrip = (t_ushort) inst.numTBPerStrip;
			#endif

			//dramBlockCount = ceil(numTBInStrip / WIDE_SIZE)
			//Compute the number of dram block transfers needed for the strip
			unsigned short dramBlockCount = 1 + ( (numTBInStrip-1) >> WIDE_SIZE_OFFSET );
			//The actual number of transfer is one more than the number of DRAM block.
			//The extra block is in the beginning, and it contains routing information
			//as well as the number of TB count, which is required by the convolution PE array
			unsigned short numTransferActions = dramBlockCount + 1;

			EMULATOR_PRINT(("[kernelIAMover] START strip transfer. "
						"iInst=%d, "
						"iStrip=%d, "
						"tileSPWidthxTileSPHeight=%d, "
						"iRowInSPTile=%d, " 
                        "iColInSPTile=%d, "
						"rowIsDense=%#03x, "
						"colIsDense=%#03x, "
						"numTBInStrip=%d, "
						"numActiveCols=%d, "
						"addressIADramBlockDDR=%#010x, "
						"destinationMisc=%#03x\n",
						iInst,
						iter,
						(unsigned int) inst.tileSPWidthxTileSPHeight, 
						iRowInSPTile,
						iColInSPTile,
						rowIsDense,
						colIsDense,
						numTBInStrip,
						numActiveCols,
						(unsigned int) addressIADramBlockDDR,
						(unsigned int) destinationMisc));

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
						// iaBlock.dramBlock = (memRegion == 0x0) ? 
						// 	pIA1[addressIADramBlockDDR] : pIA2[addressIADramBlockDDR];
						iaBlock.dramBlock = pIA[addressIADramBlockDDR];
						addressIADramBlockDDR++;
					}
					else  //Strip is padding
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

			EMULATOR_PRINT(("[kernelIAMover] FINISHED strip transfer.\n\n"));

			/*! Loop carried variable updates*/
			//TODO: Double check the iColSPUnitIndex and iRowSPUnitIndex update conditions
			iColInSPTile++;
			if ( (iColInSPTile > ((signed char) tileLeftPadding)) 
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
				#if defined(SPARSE_SYSTEM)
					offsetTBCountCol = 0;
				#endif

				iRowInSPTile++;
				if ( (iRowInSPTile > ((signed char) tileTopPadding)) 
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

		if (syncWithOA == TRUE)
		{
			DEBUG_PRINT(("[kernelIAMover] SYNC: Waiting for transfer token from the OA mover. \n"));	
		}

		//Gotcha: Need mem_fence here
		mem_fence(CLK_CHANNEL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		
		//Wait for the wait for the synchronous signal from the OA buffer
		t_flag wait = syncWithOA;

		//Gotcha: use non-blocking read
		while (wait == TRUE)
		{
			bool readSuccess = false;
			unsigned char token = read_channel_nb_intel(channel_activation_sync, &readSuccess);
			if (readSuccess == true)
			{
				wait = FALSE;
				DEBUG_PRINT(("[kernelIAMover] SYNC: Received transfer token from the OA mover. \n"));
			}
		}


	} // for. iInst
}

__attribute__((max_global_work_dim(0)))
__kernel void kernelWMover (
		//Memory port for instructions
		VOLATILE __global const t_weight_mover_instruction* restrict pInst,
		VOLATILE __global const t_dram_block* restrict pW,
		VOLATILE __global const t_accumulator* restrict pBias,
		#if defined(SPARSE_SYSTEM)
		 //Pointer to filter transfer block count
		 VOLATILE __global const t_streamblock_address* restrict pFilterTBCount,
		#endif //SPARSE_SYSTEM
		unsigned int numInstruction
	)
{
	for (unsigned int iInst=0; iInst<numInstruction; iInst++)
	{
		t_weight_mover_instruction inst = pInst[iInst];

		//Iterator of filter fold count
		unsigned short iFilterFold = 0;
		//Iterator of the number of filters that have been transferred within this fold
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
				unsigned short numTransferBlockInFilter = pFilterTBCount[addrWeightTB];
			#else
				unsigned short numTransferBlockInFilter = inst.numTBPerFilter;
			#endif
			
			t_accumulator bias = pBias[addrBias];

			unsigned short numDramBlockInFilter = ((numTransferBlockInFilter-1) >> WIDE_SIZE_OFFSET) + 1;
			
			t_filter_streamer_control control;
			control.numOutputs = inst.filterReuse;
			control.bias = bias;
			control.numTransferBlocks = numTransferBlockInFilter;
			control.maxPeCols = (inst.numActivePeCols - 1);

			t_dram_block dramControl = filterStreamerControl2dramBlock(control);

			int iDramBlock = addrWeightFilterBase;

			EMULATOR_PRINT(("[kernelWMover] START filter transfer. "
						"iInst=%d, "
						"iFilterInGroup=%d, " 
                        "iFilterInFold=%d, "
						"iFilterFold=%d, "
						"num. active PE cols=%d, "
						"num. filter reuse=%d, "
						"bias=%#04x, "
						"numTransferBlocks=%d\n\n",
						iInst, 
						iFilterInGroup,
						iFilterInFold,
						iFilterFold,
						inst.numActivePeCols,
						inst.filterReuse,
						bias,
						numTransferBlockInFilter));

			//one extra for filter stream control
			for (unsigned short iTransmitCount=0; iTransmitCount<=numDramBlockInFilter; iTransmitCount++)
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

			EMULATOR_PRINT(("[kernelWMover] FINISHED filter transfer.\n"));

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
		} //for loop over the filters in one group
	}  //for loop over instructions
}

#endif //MEMORY_READER

#ifdef IA_MEMORY
#define IA_BUFFER_WRITE_STATE_DECODE 0X0
#define IA_BUFFER_WRITE_STATE_ACCESS 0X1
#define IA_BUFFER_WRITE_STATE_UPDATE_STRIP 0X2

#define IA_BUFFER_READ_STATE_DECODE 0X0
#define IA_BUFFER_READ_STATE_COMP_NUM_ACCESS 0X1
#define IA_BUFFER_READ_STATE_ACCESS 0X2
#define IA_BUFFER_READ_STATE_UPDATE_STRIP 0X3

#define IA_BUFFER_INSTRUCTION_STATE_DECODE 0X0
#define IA_BUFFER_INSTRUCTION_STATE_SEND_TO_READER 0X1
#define IA_BUFFER_INSTRUCTION_STATE_SEND_TO_WRITER 0x2

#define IA_BUFFER_READ_STRIP_UPDATE_HORIZONTAL 0X0
#define IA_BUFFER_READ_STRIP_UPDATE_VERTICAL 0X1
#define IA_BUFFER_READ_STRIP_UPDATE_DONE 0x2

typedef uint3_t t_ia_buffer_w_state;
typedef uint3_t t_ia_buffer_r_state;
typedef uint2_t t_ia_buffer_d_state;

/**
 * Helper data bundle for accessing the dram_block cache in IA buffers
 */
typedef struct __attribute__((packed)) 
{
	unsigned short addressBase;
	unsigned short colStride;
	unsigned short rowStride;
	unsigned short colContribution;
	unsigned short rowContribution;
} t_ia_data_buffer_access_info;


#if defined(SPARSE_SYSTEM)
/**
 * Helper data struct for accessing the tb counts in each IA strip. 
 * Only used in the sparse architecture
 */
typedef struct __attribute__((packed)) 
{
	//Iterators that are used to access the TB count cache
	//tbAddress = tbAddressBase + tbAddressRowContribution + tbAddressColContribution
	unsigned char addressBase;
	//Don't uncomment. unsigned char colStride = 1;
	unsigned char rowStride;
	unsigned char colContribution;
	unsigned char rowContribution;
} t_ia_tb_buffer_access_info;

#endif

/**
 * Helper data struct for accessing the position in tile
 */
typedef struct __attribute__((packed)) 
{
	unsigned char iRow;
	unsigned char iCol;
	unsigned char numStripsRow;
	unsigned char numStripsCol;
} t_ia_tile_access_info;

/**
 * IA Buffer writer state and variables encapsulation
 */
typedef struct __attribute__((packed)) {

	t_ia_data_buffer_access_info iaBlockInfo;
	t_ia_tile_access_info tileInfo;

	#if defined(SPARSE_SYSTEM)
		t_ia_tb_buffer_access_info tbCountInfo;
	#endif

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
 * @param      cacheIAStreamBlockAddress  Cache of TB counts. (Available for sparse architecture only)
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

		t_dram_block dramBlock,
		t_flag validDramBlock,

		//Modified buffer and buffers
		#if defined(SPARSE_SYSTEM)
			t_streamblock_address cacheIAStreamBlockAddress [2][256],
		#else
			unsigned short numTBPerStrip[2],
		#endif
		t_dram_block cacheIABlocks [2][IA_CACHE_DEPTH],
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

	#if defined(SPARSE_SYSTEM)
		t_ia_tb_buffer_access_info tbCountInfo;

		//Flag that indicates whether the IA strip is in fact dense, 
		//hence the buffer needs to insert sparse bitmask when streaming IA strips to the PE array
		uint1_t flagPadBitmask;
		//Number of transfer blocks in an uncompressed compression window
		unsigned char iTBInCW; //Only useful for dense input
		unsigned char partialBitmask[COMPRESSION_WINDOW_SIZE / 8];
	#endif

	//Number of dram blocks in each strip during buffer loading, 
	unsigned short numTBPerStrip;

	//Iterator for the buffer access count (in each strip?)
	unsigned short iterAccess;

	//Maximum convolution PE row that will be affected by the buffer read operation
	unsigned char maxPeRowID;

	//Instruction on how the dram cache pointer should be updated in update_strip state
	//Also used to indicate whether the current strip is the last one to be streamed
	uint2_t stripUpdateMode;
	

	//Which cache bank to read from
	t_flag accessBank;
} t_ia_buffer_read_registers;

/**
 * @brief      Gets the ia buffer reader interface outputs.
 *
 * @param[in]  currentState               The current state
 * @param[in]  currentRegisters           Current indices, etc
 * @param      cacheIAStreamBlockAddress  The cache ia stream block address
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
		#if defined(SPARSE_SYSTEM)
			t_streamblock_address cacheIAStreamBlockAddress [2][256],
		#else
			unsigned short numTBPerStrip[2],
		#endif
		t_dram_block cacheIABlocks [2][IA_CACHE_DEPTH],

		//Outputs
		t_flag* pOutAcceptInstruction,
		
		t_transferblock_tagged* pTaggedBlock,
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
 * @param      cacheIAStreamBlockAddress  The cache ia stream block address
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
		#if defined(SPARSE_SYSTEM)
			t_streamblock_address cacheIAStreamBlockAddress [2][256],
		#else
			unsigned short numTBPerStrip[2],
		#endif

		//Modified buffer and buffers
		t_ia_buffer_r_state* pCurrentState,
		t_ia_buffer_read_registers* pCurrentRegisters,

		//Auxillary
		int colID
	);

void getIABufferDispatcherOutput (
		//Current state
		t_ia_buffer_d_state currentState,

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

		//Auxillary
		int colID
	);

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelIABuffer ()
{
	int colID = get_compute_id(0);

	t_dram_block cacheIABlocks [2][IA_CACHE_DEPTH] __attribute__((bankwidth(BURST_SIZE_BYTE)));

	//TODO: Determine whether the attribute for the TB count cache is correct
	#if defined(SPARSE_SYSTEM)
		t_streamblock_address cacheIAStreamBlockAddress [2][256] __attribute__((numbanks(1)));
	#else
		t_streamblock_address numTBPerStrip [2];
	#endif

	/**
	 * Writer state and registers
	 */
	t_ia_buffer_writer_registers regWriterContext;
	t_ia_buffer_w_state regWriterState = IA_BUFFER_WRITE_STATE_DECODE;

	/**
	 * Reader state and registers
	 */
	t_ia_buffer_read_registers regReaderContext;
	t_ia_buffer_r_state regReaderState = IA_BUFFER_READ_STATE_DECODE;


	/**
	 * Dispatcher state and registers
	 */
	t_input_buffer_tile_buffer_packet regDispatcherInstructionBuffer;
	t_ia_buffer_d_state regDispatcherState = IA_BUFFER_INSTRUCTION_STATE_DECODE;

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
		t_dram_block writerNewBlock;

		/**
		 * PE transfer block channel <===> reader interface
		 */
		t_flag readerBlockValid = FALSE;
		t_flag readerBlockSent = FALSE;
		t_transferblock_tagged readerTB;

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

			#if defined(SPARSE_SYSTEM)
				cacheIAStreamBlockAddress,
			#else
				numTBPerStrip,
			#endif
			cacheIABlocks,

			&readerReadyForInstruction,

			&readerTB,
			&readerBlockValid,

			colID
			);

		getIABufferDispatcherOutput (
			regDispatcherState,

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
			}
		}

		if (readerBlockValid == TRUE)
		{
			bool success = false;
			success = write_channel_nb_intel(channel_activation[0][colID], readerTB);
			if (success == true)
			{
				readerBlockSent = TRUE;

				EMULATOR_PRINT(("[kernelIABuffer %d] Sent TB %d / %d. TB[0-3]: %#04x %#04x %#04x %#04x \n\n",
					colID, regReaderContext.iterAccess, regReaderContext.numIAAccess
					,readerTB.values.values[0]
					,readerTB.values.values[1]
					,readerTB.values.values[2]
					,readerTB.values.values[3]
					));
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

			#if defined(SPARSE_SYSTEM)
				cacheIAStreamBlockAddress,
			#else
				numTBPerStrip,
			#endif	
			cacheIABlocks,
			regWriterState,
			regWriterContext
			);

		updateIABufferReader (
			readerNewInstruction,
			readerInstructionValid,

			readerBlockSent,

			#if defined(SPARSE_SYSTEM)
				cacheIAStreamBlockAddress,
			#else
				numTBPerStrip,
			#endif	

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
			(currentState == IA_BUFFER_WRITE_STATE_DECODE) 
			|| (currentState == IA_BUFFER_WRITE_STATE_COMP_NUM_ACCESS)
		)
	{
		*pOutAcceptData = TRUE;
	}
}

void updateIABufferWriter (
		//Inputs
		t_input_buffer_tile_buffer_packet control,
		t_flag validControl,

		t_dram_block dramBlock,
		t_flag validDramBlock,

		//Modified buffer and buffers
		#if defined(SPARSE_SYSTEM)
			t_streamblock_address cacheIAStreamBlockAddress [2][256],
		#else
			unsigned short numTBPerStrip[2],
		#endif
		t_dram_block cacheIABlocks [2][IA_CACHE_DEPTH],
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
				pCurrentRegisters->iaBlockInfo.rowStride = control.iaDramBlockRowStride;
				pCurrentRegisters->iaBlockInfo.colContribution = 0;
				pCurrentRegisters->iaBlockInfo.rowContribution = 0;

				pCurrentRegisters->tileInfo.iRow = 0;
				pCurrentRegisters->tileInfo.iCol = 0;
				pCurrentRegisters->tileInfo.numStripsCol = control.numStripsCol;
				pCurrentRegisters->tileInfo.numStripsRow = control.numStripsRow;

				pCurrentRegisters->accessBank = control.controlBits & 0x01;
				pCurrentRegisters->iterAccess = 0;
				pCurrentRegisters->numIAAccess = 0;

				#if defined(SPARSE_SYSTEM)
					pCurrentRegisters->tbCountInfo.addressBase = control.tbAddressBase;
					pCurrentRegisters->tbCountInfo.rowStride = control.tbAddressRowStride;
					pCurrentRegisters->colContribution = 0;
					pCurrentRegisters->rowContribution = 0;
				#endif

				/**
				 * Update state, if this is not a ghost command
				 */
				if (control.numStripsRow != 0)
				{
					*pCurrentState = IA_BUFFER_WRITE_STATE_COMP_NUM_ACCESS;
					#if defined(SPARSE_SYSTEM)
					EMULATOR_PRINT(("[kernelIABuffer Writer %d] START processing instruction. "
						"iaDramBlockAddressBase=%#010x, "
						"iaDramBlockColStride=%#010x, "
						"iaDramBlockRowStride=%#010x, "
						"tbCountAddressBase=%#010x, "
						"tbCountRowStride=%#010x, "
						"numStripsRow=%d, "
						"numStripsCol=%d\n\n"
						colID, 
						control.iaDramBlockAddressBase,
						control.iaDramBlockColStride,
						control.iaDramBlockRowStride,
						control.tbAddressBase,
						control.tbAddressRowStride,
						(unsigned int) control.numStripsRow,
						(unsigned int) control.numStripsCol));
					#else
						EMULATOR_PRINT(("[kernelIABuffer Writer %d] START processing instruction. "
							"iaDramBlockAddressBase=%#010x, "
							"iaDramBlockColStride=%#010x, "
							"iaDramBlockRowStride=%#010x, "
							"numStripsRow=%d, "
							"numStripsCol=%d\n\n"
							colID, 
							control.iaDramBlockAddressBase,
							control.iaDramBlockColStride,
							control.iaDramBlockRowStride,
							(unsigned int) control.numStripsRow,
							(unsigned int) control.numStripsCol));
					#endif
				}
				
			} // if validControl == TRUE
		}
		break; //IA_BUFFER_WRITE_STATE_DECODE

		case IA_BUFFER_WRITE_STATE_COMP_NUM_ACCESS: {
			if (validDramBlock == TRUE)
			{
				t_streamblock_address numIATransferBlocks = getTBCount(dramBlock);

				pCurrentRegisters->numIAAccess = 
					1 + ((numIATransferBlocks-1) >> WIDE_SIZE_OFFSET);
				pCurrentRegisters->iterAccess = 0;

				#if defined(SPARSE_SYSTEM)
					cacheIAStreamBlockAddress
						[(pCurrentRegisters->accessBank) & 0x01]
						[pCurrentRegisters->tbCountInfo.addressBase 
							+ pCurrentRegisters->tbCountInfo.colContribution 
							+ pCurrentRegisters->tbCountInfo.rowContribution] 
						= numIATransferBlocks;
				#else
					numTBPerStrip[(pCurrentRegisters->accessBank) & 0x01] = numIATransferBlocks;
				#endif

				*pCurrentState = IA_BUFFER_WRITE_STATE_ACCESS;
			}
		}
		break;

		case IA_BUFFER_WRITE_STATE_ACCESS: {
			/**
			 * Write incoming dram block into the cache, and update the counters
			 */
			if (validDramBlock == TRUE)
			{
				cacheIABlocks[(pCurrentRegisters->accessBank) & 0x01]
					[(pCurrentRegisters->iterAccess) 
						+ (pCurrentRegisters->iaBlockInfo.addressBase)
						+ (pCurrentRegisters->iaBlockInfo.colContribution)
						+ (pCurrentRegisters->iaBlockInfo.rowContribution)]
					= dramBlock;

				pCurrentRegisters->iterAccess += 0x1;

				if ((pCurrentRegisters->iterAccess) == (pCurrentRegisters->numIAAccess))
				{
					*pCurrentState = IA_BUFFER_WRITE_STATE_UPDATE_STRIP;
				}
			}
			
		}
		break;

		case IA_BUFFER_WRITE_STATE_UPDATE_STRIP: {
			/**
			 * Increment tile counter, cache counters
			 */
			pCurrentRegisters->iterAccess = 0;
			pCurrentRegisters->tileInfo.iCol += 0x1;
			pCurrentRegisters->iaBlockInfo.colContribution += pCurrentRegisters->iaBlockInfo.colStride;
			#if defined(SPARSE_SYSTEM)
				pCurrentRegisters->tbCountInfo.colContribution += 0x1;
			#endif

			*pCurrentState = IA_BUFFER_WRITE_STATE_COMP_NUM_ACCESS;

			if (pCurrentRegisters->tileInfo.iCol == pCurrentRegisters->tileInfo.numStripsCol)
			{
				pCurrentRegisters->tileInfo.iCol = 0x0;
				pCurrentRegisters->tileInfo.iRow += 0x1;
				pCurrentRegisters->iaBlockInfo.colContribution = 0;
				pCurrentRegisters->iaBlockInfo.rowContribution += pCurrentRegisters->iaBlockInfo.rowStride;
				#if defined(SPARSE_SYSTEM)
					pCurrentRegisters->tbCountInfo.colContribution 0x0;
					pCurrentRegisters->tbCountInfo.rowContribution += pCurrentRegisters->tbCountInfo.rowStride;
				#endif

				if (pCurrentRegisters->tileInfo.iRow == pCurrentRegisters->tileInfo.numStripsRow)
				{
					*pCurrentState = IA_BUFFER_WRITE_STATE_DECODE;
					EMULATOR_PRINT(("[kernelIABuffer WRITER %d] FINISHED processing instruction.\n\n", colID));
				}
			}
		}
		break;

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
		#if defined(SPARSE_SYSTEM)
			t_streamblock_address cacheIAStreamBlockAddress [2][256],
		#else
			unsigned short numTBPerStrip[2],
		#endif
		t_dram_block cacheIABlocks [2][IA_CACHE_DEPTH],

		//Outputs
		t_flag* pOutAcceptInstruction,
		
		t_transferblock_tagged* pTaggedBlock,
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
		
		t_dram_block dramBlock = cacheIABlocks[currentRegisters.accessBank & 0x01]
			[currentRegisters.iaBlockInfo.addressBase 
				+ currentRegisters.iaBlockInfo.colContribution 
				+ currentRegisters.iaBlockInfo.rowContribution 
				+ ((unsigned short)(currentRegisters.iterAccess >> WIDE_SIZE_OFFSET))];	

		//TODO: Change this
		unsigned char isLastTemp =  (((currentRegisters.iterAccess + 1) == currentRegisters.numTBPerStrip) 
			&& (currentRegisters.stripUpdateMode == IA_BUFFER_READ_STRIP_UPDATE_DONE) )
			TRUE : FALSE;

		#if defined(SPARSE_SYSTEM)
			//Insert the bitmask
			if ((currentRegisters.iTBInCW == 0) && (currentRegisters.flagPadBitmask == TRUE))
			{
				#pragma unroll
				for (int i=0; i<TRANSFER_SIZE*CLUSTER_SIZE; i++)
				{
					if (i < (COMPRESSION_WINDOW_SIZE / 8))
					{
						(*pTaggedBlock).values.values[i] = ((currentRegisters.numTBPerStrip - currentRegisters.iterAccess) < (COMPRESSION_WINDOW_SIZE / TRANSFER_SIZE)) ?
							currentRegisters.partialBitmask[i] : 0xFF;
					}
					else
					{
						(*pTaggedBlock).values.values[i] = 0x00;
					}
				}
				isLastTemp = FALSE;
			}
			else
			{
				(*pTaggedBlock).values = dramBlock.transferBlocks[(currentRegisters.iterAccess) & WIDE_SIZE_REMAINDER_MASK];
			}
		#else
			(*pTaggedBlock).values = dramBlock.transferBlocks[(currentRegisters.iterAccess) & WIDE_SIZE_REMAINDER_MASK];
		#endif

		setMaxTransferID(pTaggedBlock, currentRegisters.maxPeRowID);
		setIsLast(pTaggedBlock, isLastTemp);
	}
}

void updateIABufferReader (
		//Inputs
		t_input_buffer_tile_buffer_packet control,
		t_flag validControl,

		t_flag writeSuccessTaggedBlock,

		//Buffers to read from
		#if defined(SPARSE_SYSTEM)
			t_streamblock_address cacheIAStreamBlockAddress [2][256],
		#else
			unsigned short numTBPerStrip[2],
		#endif

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
				pCurrentRegisters->iaBlockInfo.rowStride = control.iaDramBlockRowStride;
				pCurrentRegisters->iaBlockInfo.colContribution = 0;
				pCurrentRegisters->iaBlockInfo.rowContribution = 0;

				pCurrentRegisters->tileInfo.iRow = 0;
				pCurrentRegisters->tileInfo.iCol = 0;
				pCurrentRegisters->tileInfo.numStripsCol = control.numStripsCol;
				pCurrentRegisters->tileInfo.numStripsRow = control.numStripsRow;

				pCurrentRegisters->accessBank = control.controlBits & 0x01;
				pCurrentRegisters->iterAccess = 0;
				//Number of IA dram block cache per-strip
				pCurrentRegisters->numTBPerStrip = 0;

				pCurrentRegisters->maxPeRowID = control.maxPeRowID;

				#if defined(SPARSE_SYSTEM)
					//tbCountInfo
					pCurrentRegisters->tbCountInfo.addressBase = control.tbAddressBase;
					pCurrentRegisters->tbCountInfo.rowStride = control.tbAddressRowStride;
					pCurrentRegisters->colContribution = 0;
					pCurrentRegisters->rowContribution = 0;

					pCurrentRegisters->flagPadBitmask = (control.controlBits >> 2) & 0x01;
					pCurrentRegisters->iTBInCW = 0;
					#pragma unroll
					for (int i=0; i<(COMPRESSION_WINDOW_SIZE / 8); i++)
					{
						pCurrentRegisters->partialBitmask[i] = control.partialBitmask[i];
					}
				#endif

				/**
				 * Update state
				 */
				{
					*pCurrentState = IA_BUFFER_READ_STATE_UPDATE_STRIP;
					#if defined(SPARSE_SYSTEM)
					EMULATOR_PRINT(("[kernelIABuffer Reader %d] START processing instruction. "
						"iaDramBlockAddressBase=%#010x, "
						"iaDramBlockColStride=%#010x, "
						"iaDramBlockRowStride=%#010x, "
						"tbCountAddressBase=%#010x, "
						"tbCountRowStride=%#010x, "
						"numStripsRow=%d, "
						"numStripsCol=%d\n\n"
						colID, 
						control.iaDramBlockAddressBase,
						control.iaDramBlockColStride,
						control.iaDramBlockRowStride,
						control.tbAddressBase,
						control.tbAddressRowStride,
						(unsigned int) control.numStripsRow,
						(unsigned int) control.numStripsCol));
					#else
						EMULATOR_PRINT(("[kernelIABuffer Reader %d] START processing instruction. "
							"iaDramBlockAddressBase=%#010x, "
							"iaDramBlockColStride=%#010x, "
							"iaDramBlockRowStride=%#010x, "
							"numStripsRow=%d, "
							"numStripsCol=%d\n\n"
							colID, 
							control.iaDramBlockAddressBase,
							control.iaDramBlockColStride,
							control.iaDramBlockRowStride,
							(unsigned int) control.numStripsRow,
							(unsigned int) control.numStripsCol));
					#endif
				}
				
			} // if validControl == TRUE
		}
		break; //IA_BUFFER_WRITE_STATE_DECODE

		case IA_BUFFER_WRITE_STATE_UPDATE_STRIP: {
			
			*pCurrentState = IA_BUFFER_READ_STATE_ACCESS;

			/**
			 * Access the number of IA cache number of access in the upcoming strip
			 */
			#if defined(SPARSE_SYSTEM)
				pCurrentRegisters->numTBPerStrip = 
					cacheIAStreamBlockAddress
						[pCurrentRegisters->accessBank & 0x01] 
						[pCurrentRegisters->tbCountInfo.addressBase 
							+ pCurrentRegisters->tbCountInfo.colContribution 
							+ pCurrentRegisters->tbCountInfo.rowContribution];
				pCurrentRegisters->iTBInCW = 0;
			#else
				pCurrentRegisters->numTBPerStrip = numTBPerStrip [pCurrentRegisters->accessBank & 0x01]; 
			#endif

			/*
			 * Reset strip counters
			*/
			pCurrentRegisters->iterAccess = 0;

			/**
			 * increment the tile pointer and TB count pointers in advance,
			 * obtain the strip update mode
			 */
			pCurrentRegisters->tileInfo.iCol += 0x1;
			pCurrentRegisters->stripUpdateMode = IA_BUFFER_READ_STRIP_UPDATE_HORIZONTAL;
			#if defined(SPARSE_SYSTEM)
				pCurrentRegisters->tbCountInfo.colContribution += 0x1;
			#endif

			if (pCurrentRegisters->tileInfo.iCol == pCurrentRegisters->tileInfo.numStripsCol)
			{
				pCurrentRegisters->tileInfo.iCol = 0x0;
				pCurrentRegisters->tileInfo.iRow += 0x1;

				pCurrentRegisters->stripUpdateMode = IA_BUFFER_READ_STRIP_UPDATE_VERTICAL;

				#if defined(SPARSE_SYSTEM)
					pCurrentRegisters->tbCountInfo.colContribution = 0x0;
					pCurrentRegisters->tbCountInfo.rowContribution += pCurrentRegisters->tbCountInfo.rowStride;
				#endif

				if (pCurrentRegisters->tileInfo.iRow == pCurrentRegisters->tileInfo.numStripsRow)
				{
					pCurrentRegisters->stripUpdateMode = IA_BUFFER_READ_STRIP_UPDATE_DONE;
				}
			}
		}
		break;

		case IA_BUFFER_WRITE_STATE_ACCESS: {
			/**
			 * Update the ia strip counters and possibly move the ia pointers if the write to the
			 * PE array is successful
			 */
			if (writeSuccessTaggedBlock == TRUE)
			{
				#if defined(SPARSE_SYSTEM)
					if ((pCurrentRegisters->iTBInCW > 0) 
						|| (pCurrentRegisters->flagPadBitmask == FALSE))
					{
						pCurrentRegisters->iterAccess += 1;
					}

					pCurrentRegisters->iTBInCW += 1;
					if ( (pCurrentRegisters->iTBInCW) == (COMPRESSION_WINDOW_SIZE / TRANSFER_SIZE) )
					{
						pCurrentRegisters->iTBInCW = 0;
					}
				#else
					pCurrentRegisters->iterAccess += 1;
				#endif

				/**
				 * Update IA cache pointer and reader state
				 */
				if (pCurrentRegisters->iterAccess == pCurrentRegisters->numTBPerStrip)
				{
					if (pCurrentRegisters->stripUpdateMode == IA_BUFFER_READ_STRIP_UPDATE_HORIZONTAL)
					{
						pCurrentRegisters->iaBlockInfo.colContribution +=
							pCurrentRegisters->iaBlockInfo.colStride;

						*pCurrentState = IA_BUFFER_READ_STATE_UPDATE_STRIP;
					}
					else if (pCurrentRegisters->stripUpdateMode == IA_BUFFER_READ_STRIP_UPDATE_VERTICAL)
					{
						pCurrentRegisters->iaBlockInfo.colContribution = 0;

						pCurrentRegisters->iaBlockInfo.rowContribution +=
							pCurrentRegisters->iaBlockInfo.rowStride;

						*pCurrentState = IA_BUFFER_READ_STATE_UPDATE_STRIP;
					}
					else if (pCurrentRegisters->stripUpdateMode == IA_BUFFER_READ_STRIP_UPDATE_DONE)
					{
						*pCurrentState = IA_BUFFER_READ_STATE_DECODE;
					}
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
		*pOutValidWriterInstruction = TRUE;
		*pOutWriterControl = controlBuffer;
	}

	/**
	 * Reader control interface
	 */
	if (currentState == IA_BUFFER_INSTRUCTION_STATE_SEND_TO_READER)
	{
		*pOutValidReaderInstruction = TRUE;
		*pOutReaderControl = controlBuffer;
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
			if (inWriterReady == TRUE)
			{
				*pState = IA_BUFFER_INSTRUCTION_STATE_DECODE;

				EMULATOR_PRINT(("[kernelIABuffer Dispatcher %d] Sent instruction to WRITER.\n",
							colID));
			}
		}
		break; //IA_BUFFER_INSTRUCTION_STATE_SEND_TO_WRITER

		case IA_BUFFER_INSTRUCTION_STATE_SEND_TO_READER: {
			if (inReaderReady == TRUE)
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
		{
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
				tileBufferControlPacket.iaDramBlockRowStride = iaCacheRowStride;
				
				tileBufferControlPacket.controlBits = ((numActivePeCols-1) << 0x3) 
					| ((unsigned char) writeSideIndex) & 0x01

				#if defined(SPARSE_SYSTEM)
	                tileBufferControlPacket.tbAddressBase = 0;
	                tileBufferControlPacket.tbAddressRowStride = inputTileWidth;
			    #endif

			    tileBufferControlPacket.numStripsCol = inputTileWidth;
			    if (iInstructionCycle < numInstructions)
				{
			    	tileBufferControlPacket.numStripsRow = inputTileHeight;
			    }
			    else //HACK. Set the number of rows to zero represent a ghost command.
			    {
			    	tileBufferControlPacket.numStripsRow = 0;
			    }

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

			unsigned char inputTileWidth = drainInstruction.localTileWidth;
		    unsigned char inputTileHeight = drainInstruction.localTileHeight;
		    unsigned char stride = drainInstruction.kernelStride;
		    unsigned char kernelSize = drainInstruction.kernelSize;
	        unsigned int numOutputInstructions = drainInstruction.numOutputInstructions;
		    unsigned char numActivePeCols = drainInstruction.flagPadBitmaskCatNumActiveCols & 0x7F;
		    unsigned short numOutputChannelsInGroup = drainInstruction.numOutputChannelsInGroup;
		    unsigned short iaCacheColStride = drainInstruction.cacheIAStripColStride;
			
			//while (iFilterInGroup < numOutputChannelsInGroup)
	        for (unsigned int i=0; i<numOutputInstructions; i++)
			{
				unsigned char numActivePeRows = ((numOutputChannelsInGroup - iFilterInGroup) < (unsigned short) (PE_ROWS)) ?
					(unsigned char) (numOutputChannelsInGroup - iFilterInGroup) : PE_ROWS;

				unsigned char iStripInTile = iInputTileHeight * inputTileWidth + iInputTileWidth;

				t_input_buffer_tile_buffer_packet tileBufferControlPacket;
				tileBufferControlPacket.iaDramBlockAddressBase = ((unsigned short) iStripInTile) * ((unsigned short) iaCacheColStride);
				tileBufferControlPacket.maxPeRowID = (numActivePeRows - 1);
				
				tileBufferControlPacket.iaDramBlockColStride = iaCacheColStride;
				tileBufferControlPacket.iaDramBlockRowStride = iaCacheRowStride;

				#if defined(SPARSE_SYSTEM)
			    	tileBufferControlPacket.tbAddressBase = iStripInTile;
			    	tileBufferControlPacket.tbAddressRowStride = inputTileWidth;
			    #endif
				unsigned char sendInstructionType = 0x2; //Stream from the buffer
				tileBufferControlPacket.controlBits =
					(sendInstructionType & 0x2)
					| (((unsigned char) (~writeSideIndex)) & 0x01)
					| ((numActivePeCols-1) << 0x3);
				#if defined(SPARSE_SYSTEM)
					//The sparse system needs to know whether there is the need to insert operand bitmask to the ia stream
					tileBufferControlPacket.controlBits |= ((unsigned char) flagPadBitmask) << 2;
					#pragma unroll
					for (int i=0; i<COMPRESSION_WINDOW_SIZE / 8; i++)
					{
						tileBufferControlPacket.partialBitmask[i] = instruction.partialBitmask[i];
					}
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
				#if defined(SPARSE_SYSTEM)
					EMULATOR_PRINT(("[kernelIATileController] Sent a buffer stream command. "
					"iInstructionCycle=%d, numActivePeRows=%d, iInputTileHeight=%d, iInputTileWidth=%d. flagPadBitmask=%#03x\n\n", 
					iInstructionCycle, numActivePeRows, iInputTileHeight, iInputTileWidth, ((unsigned char) flagPadBitmask)));
				#else
					EMULATOR_PRINT(("[kernelIATileController] Sent a buffer stream command. "
					"iInstructionCycle=%d, numActivePeRows=%d, iInputTileHeight=%d, iInputTileWidth=%d.\n\n", 
					iInstructionCycle, numActivePeRows, iInputTileHeight, iInputTileWidth));
				#endif
					

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
			} // for
		}
		//End of sending streaming instructions

		//SWAP the read side and the write side
		writeSideIndex ~= writeSideIndex;
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

#define IA_TEE_COMMAND_READ_STRIP_HEADER 0X0
#define IA_TEE_COMMAND_TRANSFER 0x1
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

					if ( (((signed char) colSPWidth) > actualColIndex)
							&& (actualColIndex >= 0)
						)
					{
						nextFlagRoute2Misc = flag2Misc;
						nextFlagRoute2Conv = (~flag2Misc) & 0x01;
					}
					else
					{
						nextFlagRoute2Misc = FALSE;
						nextFlagRoute2Conv = FALSE;
					}

					//Adjust the col index seen by the next compute column
					#if (WIDE_SIZE > 1)
						taggedBlock.dramBlock.transferBlocks[1].values[0] = actualColIndex - ((signed char) colSPStride);
					#else
						taggedBlock.dramBlock.transferBlocks[0].values[4] = actualColIndex - ((signed char) colSPStride);
					#endif

					EMULATOR_PRINT(("[kernelIATee %d] Detected a strip head. "
						"actualColIndex=%d, colSPStride=%d, colSPWidth=%d, nextFlagRoute2Misc=%#03x. nextFlagRoute2Conv=%#03x\n", 
						colID, actualColIndex, colSPStride, colSPWidth, ((unsigned char) nextFlagRoute2Misc), ((unsigned char) nextFlagRoute2Conv)));

					nextState = IA_TEE_COMMAND_TRANSFER;
				} //IA_TEE_COMMAND_READ_STRIP_HEADER
				break;
				case (IA_TEE_COMMAND_TRANSFER) :
				{
					if (flagIsLastInStrip == TRUE)
					{
						nextState = IA_TEE_COMMAND_READ_STRIP_HEADER;

						EMULATOR_PRINT(("[kernelIATee %d] Finished processing a strip\n", colID));
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

			//Logic for routing the dram block to the conv engine.
			//Forward the header block, as well as the subsequent data block
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

			//Logic for routing dram blocks to the MISC unit
			//Only forward data blocks	
			bool write2Misc = false;
			if ( (regState == IA_TEE_COMMAND_TRANSFER) && (regFlagRoute2Misc == TRUE))
			{
				write2Misc = true;
			}

			if (write2Misc == true)
			{
				write_channel_intel(channel_ia_wide_misc[colID], dramBlock);
			}
		
		} // if read is successful

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
		unsigned int numInstruction
	)
{
	for (int i=0; i<numInstruction; i++)
	{
		t_misc_instruction instruction = pInstruction[i];
		t_misc_control_packet packet;
		packet.controlBits = instruction.controlBits;
		packet.numDramBlocksToReduce = instruction.numDramBlocksToReduce;
		packet.numOutputBlocks	=	instruction.numOutputBlocks;
		packet.numEffectiveValuesInLastStrip = instruction.numEffectiveValuesInLastStrip;
		write_channel_intel(channel_misc_instruction[0], packet);
		EMULATOR_PRINT(("[kernelMiscControlMover] Sent instruction %d \n",
						i));
	}
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelMisc ()
{
	int colID = get_compute_id(0);
	while (true)
	{
		signed char reductionBlock[BURST_SIZE_BYTE];

		t_misc_control_packet controlPacket = read_channel_intel(channel_misc_instruction[colID]);

		//Handle the passing over the control Packet
		if (colID < (PE_COLS - 1))
		{
			uint4_t numActiveCol = (controlPacket.controlBits & 0x0F);
			if (colID < (numActiveCol - 1))
			{
				write_channel_intel(channel_misc_instruction[colID+1], controlPacket);
			}
		}

		//Decode
		//OpCode. 00: Add; 01: Max Pooling; 10: Stream
		uint2_t opcode = (controlPacket.controlBits >> 4) & 0x03;
		unsigned char numDramBlocksToReduce = controlPacket.numDramBlocksToReduce;
		unsigned char numOutputBlocks = controlPacket.numOutputBlocks;
		unsigned char numEffectiveValuesInLastStrip = controlPacket.numEffectiveValuesInLastStrip;

		EMULATOR_PRINT(("[kernelMisc %d] Received command "
						"opcode=%#04x, numOutputBlocks=%d, numDramBlocksToReduce=%d, numEffectiveValues=%d \n",
						colID, ((unsigned char)opcode), numOutputBlocks, numDramBlocksToReduce, numEffectiveValuesInLastStrip));

		for (unsigned char iOutput=0; iOutput < numOutputBlocks; iOutput++)
		{
			unsigned char numEffectiveValues = (iOutput < (numOutputBlocks-1)) ? BURST_SIZE_BYTE : numEffectiveValuesInLastStrip;
			//Initialize the reductionBlock
			#pragma unroll
			for (int iCluster=0; iCluster<NUM_CLUSTER_IN_DRAM_SIZE; iCluster++)
			{
				#pragma unroll
				for (int iVal=0; iVal < BURST_SIZE_BYTE; iVal++)
				{
					//If max pooling, then intialize the values to the minimum, else zero
					reductionBlock[iVal] = (opcode == 0x01) ? 
						0x80 : 0x00;
				}
			}

			//Perform reduction
			for (unsigned char iBlock=0; iBlock<numDramBlocksToReduce; iBlock++)
			{
				t_dram_block inputDramBlock = read_channel_intel(channel_ia_wide_misc[colID]);
				#pragma unroll
				for (int iValue=0; iValue < BURST_SIZE_BYTE; iValue++)
				{
					signed char inputValue = inputDramBlock
						.transferBlocks[iValue >> (VALUE_TO_CLUSTER_SHIFT + CLUSTER_TO_TRANSFER_SIZE_SHIFT)]
						.values[iValue & VALUE_DIVIDED_BY_SIMD_SIZE_REMAINDER_MASK];

					signed char currentValue = reductionBlock[iValue];

					signed char newValue;
					if (opcode == 0x00)
					{
						newValue = inputValue + currentValue;
					}
					else if (opcode == 0x01)
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
			}

			//Drain the output
			for (unsigned char iVal=0; iVal < BURST_SIZE_BYTE; iVal++)
			{
				if (iVal<numEffectiveValues)
				{
					write_channel_intel(channel_drain_misc[colID], reductionBlock[iVal]);
				}
			}

			EMULATOR_PRINT(("[kernelMisc %d] Finished processing output block %d / %d of the command.\n", colID, iOutput, numOutputBlocks));
		}  //iOutput
		

		EMULATOR_PRINT(("[kernelMisc %d] Finished processing a command\n", colID));
	}
}
#endif

#ifdef MEMORY_WRITER
__attribute__((max_global_work_dim(0)))
__kernel void kernelOAMover (
		VOLATILE __global t_output_dram_block* restrict pOA,

		#if defined(SPARSE_SYSTEM)
			VOLATILE __global t_streamblock_address* restrict pTBCount,
		#endif

		VOLATILE __global const t_oa_mover_instruction* restrict pInstruction,
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
		// __global t_output_dram_block* pOA;
		// pOA = (outputMemSelect == 0x01) ? pOA1 : pOA0;
		// #if defined(SPARSE_SYSTEM)
		// 	__global t_streamblock_address* pTB;
		// 	pTB = (outputMemSelect == 0x01) ? pTBCount1 : pTBCount0;
		// #endif

		//Control variables
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
			EMULATOR_PRINT(("[kernelOAMover] START strip transfer. "
						"iInst=%d, "
						"iCol=%d, " 
                        "iOutputWidthInColTile=%d, "
						"iOutputHeightInColTile=%d, "
						"enableSparsification=%#03x, "
						"addrOA=%#08x, "
						"numActivePeCols=%d\n",
						iInst, 
						iCol,
						iOutputWidthInColTile,
						iOutputHeightInColTile,
						(unsigned char) enableSparsification,
						addrOA,
						numActivePeCols));
			#if defined(SPARSE_SYSTEM)
				//int addrTB = inst.memTBStart + addrTBGroupContribution + addrTBRowContribution + addrTBColContribution + addrTBPeColContribution;
				int addrTB = inst.memTBStart + addrTBRowContribution + addrTBColContribution + addrTBPeColContribution;
				bool proceed = true;
				unsigned short clusterCount = 0;
				while (proceed)
				{
					bool readSuccess = false;
					t_output_dram_block_tagged receivedBlock = read_channel_nb_intel(channel_output_wide[0], &readSuccess);
					if (readSuccess == true)
					{
						if ((receivedBlock.isLastFlag & 0x1) == TRUE)
						{
							proceed = false;
						}
							clusterCount = outputDramBlock2ClusterCount(receivedBlock.block);
						
						//When output sparsification is disabled, no TB count is transferred
						if (((receivedBlock.isLastFlag & 0x1) == FALSE) || (enableSparsification == FALSE))
						{
							//Store the dram block
							pOA[addrOA++] = receivedBlock.block;
						}
					}
				} //while

				t_streamblock_address tbBlockCount = ((clusterCount-1) >> CLUSTER_TO_TRANSFER_BLOCK_SHIFT) + 1;
				
				if (enableSparsification == TRUE)
				{
					//Store the cluster count
					pTBCount[addrTB++] = tbBlockCount;
				}
			#else
				for (unsigned int i=0; i<inst.numDramBlockPerStrip;)
				{
					bool readSuccess = false;
					t_output_dram_block_tagged receivedBlock = read_channel_nb_intel(channel_output_wide[0], &readSuccess);
					if (readSuccess == true)
					{
						pOA[addrOA++] = receivedBlock.block;
						i++;
					}
				}
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
					addrTBColContribution += (signed int) inst.memTBColStride;
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
				}
			}
			EMULATOR_PRINT(("[kernelOAMover] FINISHED a strip transfer.\n"));
		} //for. strip inside the OA tile

		mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
		//Send the activation sync if needed

		t_flag keepSending = enableSendSync;
		while (keepSending == TRUE)
		{
			bool sendSuccess = false;
			sendSuccess = write_channel_nb_intel(channel_activation_sync, 0x0);
			if (sendSuccess == true)
			{
				keepSending = FALSE;
				DEBUG_PRINT(("[kernelOAMover] SYNC: Sent the transfer token. \n"));
			}
		}
		// if (enableSendSync == TRUE)
		// {
		// 	write_channel_intel(channel_activation_sync, 0x0);
		// 	DEBUG_PRINT(("[kernelOAMover] SYNC: Sent the transfer token. \n"));
		// }
	} //for. over instruction
} //kernelOAMover
#endif //MEMORY_WRITER 

#ifdef OA_MEMORY
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
	//TODO: weak review
	typedef uint3_t t_state;

	int colID = get_compute_id(0);
	char cacheOutputActivations[OA_CACHE_DEPTH][CLUSTER_SIZE] __attribute__((
                   numbanks(CLUSTER_SIZE), singlepump));

	/*
	 *Loop carried variables
	*/
	t_state currentState = OA_BUFFER_STATE_DECODE;

	//Starting address of the output strip that is to be accessed
	unsigned short stripStartOutputIndex = 0X0;
	//Number of output per strip
	unsigned short numOutputsPerStrip = 0X0;
	//Number of strips to access in the access tile
	unsigned short numStripsToAccess = 0x0;
	//Stride between the start of successive strips in the cache
	unsigned short oaStridePerCol = 0x0;
	
	uint1_t isDrainBuffer = FALSE;
	uint1_t flagSourceIsMisc = FALSE; 

	//Information relevant for loading the cache only
	unsigned char accumulatorShiftDirCatShiftAmount = 0x0;
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
				stripStartOutputIndex = controlPacket.startOutputIndex;
				numOutputsPerStrip = controlPacket.numOutputsPerStrip;
				numStripsToAccess = controlPacket.numStripsToAccess;
				oaStridePerCol = controlPacket.iaStridePerCol;
                isDrainBuffer = (controlPacket.controlBits >> 7) & 0x1;

				//Information relevant for loading the cache only
                accumulatorShiftDirCatShiftAmount = controlPacket.controlBits & 0xF;
                enableRelu = (controlPacket.controlBits >> 6) & 0x1;
                enableSparsification = ( controlPacket.controlBits >> 4) & 0x1;
                flagSourceIsMisc = (controlPacket.controlBits >> 5) & 0x1;

                iStrip = 0;
				
				nextState = OA_BUFFER_STATE_NUM_ACCESS;

				EMULATOR_PRINT(("[kernelOABuffer %d] START processing instruction. "
						"isDrainBuffer=%#03x, stripStartOutputIndex=%d, numOutputsPerStrip=%d, "
						"numStripsToAccess=%d, enableRelu=%#03x, enableSparsification=%#03x, flagSourceIsMisc=%#03x \n\n", 
						colID, (unsigned char) isDrainBuffer, stripStartOutputIndex, numOutputsPerStrip,
						numStripsToAccess, ((unsigned char) enableRelu), ((unsigned char)enableSparsification), ((unsigned char) flagSourceIsMisc)));
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
			indexOutput = stripStartOutputIndex;

			nextState = OA_BUFFER_STATE_ACCESS;
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
				t_accumulator wideOutput;

				if (flagSourceIsMisc == FALSE)
				{
					t_conv_drain_tagged wideOutputTagged;
					wideOutputTagged = read_channel_nb_intel(channel_drain_conv[0][colID], &readSuccess);
					wideOutput = wideOutputTagged.value;
				}
				else
				{
					signed char miscOutput = read_channel_nb_intel(channel_drain_misc[colID], &readSuccess);
					//OpenCL should handle sign extension
					wideOutput = (t_accumulator) miscOutput;
				}
				
				
				if (readSuccess == true) {
					t_operand shortOutput = modifyOutput(wideOutput, accumulatorShiftDirCatShiftAmount, enableRelu);
					//cacheOutputActivations[indexOutput] = shortOutput;
					cacheOutputActivations[indexOutput >> VALUE_TO_CLUSTER_SHIFT][indexOutput & VALUE_DIVIDED_BY_CLUSTER_SIZE_REMAINDER_MASK] = shortOutput;

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

						//Fetch all the values in the cluster
						#pragma unroll
						for (unsigned char i=0; i<CLUSTER_SIZE; i++)
						{
							unsigned short index = indexOutput + i;
							unsigned short tempOC = iOutputChannelFetched + i;
							char tempValue = (tempOC >= numOutputsPerStrip) ?
								0x0 : cacheOutputActivations[index >> VALUE_TO_CLUSTER_SHIFT][i];
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
				#else //DENSE_SYSTEM
					t_output_cluster_tagged taggedCluster;
					//fetch the cluster
					#pragma unroll
					for (unsigned char i=0; i<CLUSTER_SIZE; i++)
					{
						unsigned short index = indexOutput + i;
						unsigned short tempOC = iOutputChannelFetched + i;
						char tempValue = (tempOC >= numOutputsPerStrip) ?
								0x0 : cacheOutputActivations[index >> VALUE_TO_CLUSTER_SHIFT][i];
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
			stripStartOutputIndex += oaStridePerCol;
			if (iStrip == numStripsToAccess)
			{
				delayCount = 0;
				nextState = OA_BUFFER_STATE_PADD;
				EMULATOR_PRINT(("[kernelOABuffer %d] Finished processing processing instruction.\n", colID));
			}
		}

		currentState = nextState;
	} //end while
} //kernelOABuffer

__attribute__((max_global_work_dim(0)))
__kernel void kernelOATileController (
	VOLATILE  __global const t_oa_tile_controller_instruction* restrict pInst,
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
	    unsigned char numActiveElementsInFullFold = inst.numActiveElementsInFullFold;
	    unsigned char numActiveRowsInPartialFolds = inst.numActiveElementsInPartialFold;

	    unsigned short numChannelsInGroupCurrentLayer = inst.numLocalChannelsPerCurrentGroup;
	    unsigned short numChannelsInGroupNextLayer = inst.numLocalChannelsPerNextGroup;
	    unsigned char outputModifierBits = inst.flagSparseCatFlagReluCatFlagSourceCatShift;
	    unsigned char numActivePeCols = inst.numActiveCols;

	    unsigned short numOutputChannels = inst.numLocalChannels;

	    /*
	    2. Send instruction to drain from the PE array
	    */
	    unsigned short iChannelCurrentLayer = 0;
	    unsigned short iChannelInGroup = 0;
	    unsigned short iFoldInGroup = 0;
	    //unsigned short iOutputTileHxWDrain = 0;

	   	EMULATOR_PRINT(("[kernelOATileController] START sending the drain-from-array instruction for instruction %d\n\n", 
				iInst));

	    for  (unsigned short i=0; i < inst.numDrainInstructions; i++)
	    {
	    	unsigned char numActivePeRows = (iFoldInGroup < numFullFoldsInGroupCurrentLayer) ?
	    		numActiveElementsInFullFold : numActiveRowsInPartialFolds;
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

	} // iterate over instructions
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


#define OA_TEE_INSTRUCTION_DRAIN_CONV 0X0
#define OA_TEE_INSTRUCTION_DRAIN_PADD 0X1
#define OA_TEE_INSTRUCTION_SEND_SELF 0x2
#define OA_TEE_INSTRUCTION_DRAIN_OTHERS 0X3
#define OA_TEE_INSTRUCTION_DECODE_COMMAND 0X4
#define OA_TEE_INSTRUCTION_SEND_COUNT 0x5
#define OA_TEE_INSTRUCTION_LOOP_UPDATE 0x6
//#define OA_TEE_INSTRUCTION_DRAIN_MISC 0x7
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_COLS)))
__kernel void kernelOATee ()
{
	//TODO: Weak review
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
		bool sendDramBlockEnable = false;

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
					sendDramBlockEnable = true;
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
			sendDramBlockEnable = true;
		}
		// else if (regInstruction == OA_TEE_INSTRUCTION_DRAIN_MISC)
		// {
		// 	t_output_dram_block miscOutput = read_channel_nb_intel(channel_drain_misc[colID], &readSuccess);
		// 	if (readSuccess)
		// 	{
		// 		tempDramBlockTagged.block = miscOutput;
		// 		tempDramBlockTagged.isLastFlag = ((regIsLastTee & 0x1) << 0x1) | 0x1;
		// 		sendDramBlockEnable = true;
		// 	}
		// }
		#if defined(SPARSE_SYSTEM)
			else if (regInstruction == OA_TEE_INSTRUCTION_SEND_COUNT)
			{
				tempDramBlockTagged.isLastFlag = ((regIsLastTee & 0x1) << 0x1) | 0x01;
				t_output_dram_block countDramBlock = clusterCount2OutputDramBlock(iClustersInStrip);
				tempDramBlockTagged.block = countDramBlock;
				sendDramBlockEnable = true;
            }
		#endif


		//Write channels: output dram block passing
		if ( sendDramBlockEnable == true)
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
			break;
			// case (OA_TEE_INSTRUCTION_DRAIN_MISC) :
			// {
			// 	if (readSuccess == true)
			// 	{
			// 		tempInstruction = OA_TEE_INSTRUCTION_DRAIN_OTHERS;
			// 	}
			// } //OA_TEE_INSTRUCTION_DRAIN_MISC
			// break;
			case (OA_TEE_INSTRUCTION_DECODE_COMMAND) :
			{
				if (readSuccess == true)
				{
					regIsLastTee = ((tempTeeControl.flagSourceCatFlagSparseFlagMaxColID & 0x0F) > colID) ? FALSE: TRUE;
					regStripsInTile = tempTeeControl.numLocalTileHeightxLocalTileWidth;
					iStripsInTile = 0;
					regFlagSparse = tempTeeControl.flagSourceCatFlagSparseFlagMaxColID >> 4;
					//uint1_t tempFlagDrainMisc = tempTeeControl.flagSourceCatFlagSparseFlagMaxColID >> 5;

					// if (tempFlagDrainMisc == TRUE)
					// {
					// 	tempInstruction = OA_TEE_INSTRUCTION_DRAIN_MISC;
					// }
					// else
					// {
					// 	tempInstruction = OA_TEE_INSTRUCTION_DRAIN_CONV;
					// }
					tempInstruction = OA_TEE_INSTRUCTION_DRAIN_CONV;
					//regFlagDrainMisc = tempFlagDrainMisc;
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

					// if (regFlagDrainMisc == TRUE)
					// {
					// 	tempInstruction = OA_TEE_INSTRUCTION_DRAIN_MISC;
					// }
					// else
					// {
					// 	tempInstruction = OA_TEE_INSTRUCTION_DRAIN_CONV;
					// }
					tempInstruction = OA_TEE_INSTRUCTION_DRAIN_CONV;
				}
				
			} //OA_TEE_INSTRUCTION_LOOP_UPDATE
			break;
			#if defined(SPARSE_SYSTEM)
				case (OA_TEE_INSTRUCTION_SEND_COUNT) :
				{
					tempInstruction = OA_TEE_INSTRUCTION_DRAIN_OTHERS;
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
				regConvDramBlock.clusters[i] = regConvDramBlock.clusters[i+1];
			}
		}
		if (shiftInNewCluster == true)
		{
			regConvDramBlock.clusters[NUM_CLUSTER_IN_DRAM_SIZE-1] = tempClusterTagged.cluster;
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
	
		#if defined(ARRIA10) || defined(STRATIX10)
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
			//TODO: Try registering the output of pePopCounter
			unsigned char tempNumClusterLeft = pePopCounter(&nextWeightBitmaskBytes);
			#if defined(ARRIA10) || defined(STRATIX10)
				nextNumWeightClusterLeft = __fpga_reg(tempNumClusterLeft);
			#else
				nextNumWeightClusterLeft = tempNumClusterLeft;
			#endif
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
			//TODO: Try registering the output of pePopCounter
			unsigned char tempNumClusterLeft = pePopCounter(&nextActivationBitmaskBytes);
			#if defined(ARRIA10) || defined(STRATIX10)
				nextNumActivationClusterLeft = __fpga_reg(tempNumClusterLeft);
			#else
				nextNumActivationClusterLeft = tempNumClusterLeft;
			#endif

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
				EMULATOR_PRINT(("[Op Filter (%d, %d)] Commit. pSum value: %#04x \n", idy, idx, (signed int) drainTransportBlock.value));
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
				EMULATOR_PRINT(("[DENSE PE (%d, %d)] Commit. pSum value: %#04x \n", idy, idx, drainTransportBlock.value));
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

