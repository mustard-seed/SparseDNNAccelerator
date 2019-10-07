#include "params.hpp"
#include "device_structures.hpp"
#include "channels.cl"
#include "device_utils.hpp"

__attribute__((max_global_work_dim(0)))
__kernel void nop () {}


#define BURST_SIZE_TRANSFER_BLOCK  BURST_SIZE_BYTE/CLUSTER_SIZE/TRANSFER_SIZE

#define MAX_CACHE_SIZE 1024
#define WRITE_TO_CAHCE_SIDE 0x0
#define READ_FROM_CACHE_SIDE 0x1

/*
#define WEIGHT_WRITE_CACHE_SETUP_PE_ROW 0x0
#define WEIGHT_WRITE_CACHE_SETUP_FOLD_IN_GROUP 0x1
#define WEIGHT_WRITE_CACHE_SETUP_GROUP 0x2
#define WEIGHT_WRITE_CACHE_SETUP_TILES 0x3
#define WEIGHT_WRITE_CACHE_SETUP_AUX_UPDATE 0x4
#define WEIGHT_WRITE_CACHE_STREAM 0x5
#define WEIGHT_WRITE_CACHE_WAIT 0x6
#define WEIGHT_WRITE_CACHE_FINISH 0X7

#define WEIGHT_READ_CACHE_SETUP_DIM_IN_TILE 0x0
#define WEIGHT_READ_CACHE_SETUP_FOLD_GROUP 0x1
#define WEIGHT_READ_CACHE_SETUP_TILES 0X2
#define WEIGHT_READ_CACHE_SETUP_AUX_UPDATE 0x3
#define WEIGHT_READ_CACHE_LOAD_ADDRESS 0x4
#define WEIGHT_READ_CACHE_STREAM  0x5
#define WEIGHT_READ_CACHE_WAIT  0x6
#define WEIGHT_READ_CACHE_FINISH 0x7
*/

/*! kernelSimpleWeightStreamer
Important assumption: The address cache only stores the BRAM address of the first streaming block in each strip.
*/
__attribute__((max_global_work_dim(0)))
__kernel void kernelFilterWriter (
	 volatile __global t_dram_block* restrict pDramWeights, //Pointer to the filter weights in external memory
	 volatile __global t_streamblock_address* restrict pStreamBlockAddress,

	unsigned int strideExternalMemory, //Distance between the start of successive filters in DRAM in terms of transfer blocks

	unsigned short outputWidth, //Q
	unsigned char sizeOutputWidthTile, //TQ
	unsigned char numOutputWidthTile, // ceil (Q / TQ)
	unsigned char sizeOutputWidthTile0, //Special case: TQ for the tiles on the left boundary
	unsigned char maxPeCols0, //maximum number of PE cols in use for tile 0 in width.

	//TODO: Make the number of PE columns for the first tile variable

	unsigned short outputHeight, //P
	unsigned char sizeOutputHeightTile, //TP
	unsigned char numOutputHeightTile, // ceil (P / TP)
	unsigned char sizeOutputHeightTile0, //Special case: for the left corner

	unsigned int numFiltersInKernel, //L

	unsigned short numGroups, // L / G
	unsigned short numFiltersInGroup, // G
	unsigned short numFoldInGroup // ceil (G / F)

	) {
	//typedef uint3_t t_state;
	//Cache: NzBlocks blocks
	//t_dram_block cacheNzBlocks [2][KERNEL_CACHE_DEPTH][PE_ROWS] __attribute__ ((numbanks(PE_ROWS), bankwidth(CLUSTER_SIZE*TRANSFER_SIZE*WIDE_SIZE)));
	//t_transfer_block cacheNzBlocks [2][PE_ROWS][KERNEL_CACHE_DEPTH] __attribute__ ((numbanks(BURST_SIZE_TRANSFER_BLOCK * PE_ROWS), bankwidth(CLUSTER_SIZE*TRANSFER_SIZE)));

	//Cache: Cache address of the streaming blocks
	t_spOffset cacheStreamingBlockAddress [4096] __attribute__((numbanks(1)));

	//=============================================================
	//Read all the streaming block address in to BRAM as soon as possible
	//==============================================================
	EMULATOR_PRINT(("[kernelFilterWriter] Reading the stream block addresses\n"));
	{
		unsigned int countFilters = 0;
		while (countFilters < numFiltersInKernel) {
			//Unroll the loop to increase the transfer width and make better use of the BRAM bandwidth
			//#pragma unroll 2
			//for (unsigned char i=0; i<2; i++)
			//{
				//unsigned short filterIndex = countFilters + i;
				//if (filterIndex < numFiltersInKernel)
				//{
					cacheStreamingBlockAddress[countFilters] = pStreamBlockAddress[countFilters];
				//}
			//}
			countFilters += 1;
		}
	}

	//bool proceed = true;

	//======================Other shared variables============================
	//uint1_t regWriteSide = 0x0;

	//====================Write into cache variables=======================
	//t_state stateWriteCache = WEIGHT_WRITE_CACHE_STREAM;

	//unsigned short iOutputHeightTile = 0; //tp
	//unsigned short iOutputWidthTile = 0; //tq
	//unsigned short iGroup = 0; // gl
	//unsigned short iFoldInGroup = 0; //gf
	//unsigned char iPeRow = 0; //f
	//unsigned short iTransferBlockInFilter = 0; //iCg

	//unsigned short iFilterGlobal = 0; //iL
	//unsigned short iFilterInGroup = 0; //gf * F
	//unsigned int iTransferBlockFilterBaseDDR = 0;
	//unsigned int iTransferBlockDDR = iTransferBlockFilterBaseDDR;

	unsigned short iHeightGlobal = 0; // countPCoveredByTP
	//unsigned short iWidthGlobal = 0; //countQCovertedByTQ
	
	//unsigned char maxRowUsed = PE_ROWS < (numFiltersInGroup - iFilterInGroup) ?
	//			PE_ROWS : (numFiltersInGroup - iFilterInGroup); //maxF
	//unsigned short maxTransferBlockInFilter = cacheStreamingBlockAddress[iFilterGlobal];
	

	//Loops
	for (unsigned char iOutputHeightTile = 0; iOutputHeightTile < numOutputHeightTile; iOutputHeightTile++) //tp
	{
		unsigned char maxTP_ = (iOutputHeightTile == 0) ? sizeOutputHeightTile0 : sizeOutputHeightTile;
		unsigned char maxOutputHeightTileSize = (((unsigned short) maxTP_) < ((unsigned short) (outputHeight - iHeightGlobal)) ) ?
			maxTP_ :  (outputHeight - iHeightGlobal);

		unsigned short iWidthGlobal = 0; //countQCovertedByTQ

		for (unsigned char iOutputWidthTile=0; iOutputWidthTile<numOutputWidthTile; iOutputWidthTile++) //tq
		{
			unsigned char maxTQ_ = (iOutputWidthTile == 0) ? sizeOutputWidthTile0 : sizeOutputWidthTile;
			unsigned char maxPeCols = (iOutputWidthTile == 0) ? maxPeCols0 : PE_COLS;
			unsigned char  maxOutputWidthTileSize = ( ((unsigned short) maxTQ_) < ((unsigned short) (outputWidth - iWidthGlobal)) ) ?
				maxTQ_ :  (outputWidth - iWidthGlobal);

			unsigned short iFilterGlobal = 0; //iL
			unsigned int iTransferBlockFilterBaseDDR = 0;

			for (unsigned short iGroup=0; iGroup<numGroups; iGroup++) //gl
			{
				unsigned short iFilterInGroup = 0; //gf * F
				for (unsigned short iFoldInGroup=0; iFoldInGroup<numFoldInGroup; iFoldInGroup++) //gf
				{
					unsigned char maxRowUsed = PE_ROWS < (numFiltersInGroup - iFilterInGroup) ?
						PE_ROWS : (numFiltersInGroup - iFilterInGroup); //maxF

					for (unsigned char iPeRow=0; iPeRow<maxRowUsed; iPeRow++)
					{
						unsigned short maxTransferBlockInFilter = cacheStreamingBlockAddress[iFilterGlobal];
						unsigned short maxDramBlockInFilter = ((maxTransferBlockInFilter & WIDE_SIZE_REMAINDER_MASK) > 0x0) ?
							(maxTransferBlockInFilter >> WIDE_SIZE_OFFSET) + 1 : maxTransferBlockInFilter >> WIDE_SIZE_OFFSET;
						unsigned short maxTransmitCount = maxDramBlockInFilter+2; //one extra for filter stream control, another for max pe cols
						
						t_filter_streamer_control control;
						control.maxOutputHeightTileSize = maxOutputHeightTileSize;
						control.maxOutputWidthTileSize = maxOutputWidthTileSize;
						//control.destinationRow = iPeRow;
						control.numTransferBlocks = maxTransferBlockInFilter;
						control.maxPeCols = maxPeCols;
						t_dram_block dramControl = filterStreamerControl2dramBlock(control);

						t_dram_block dramMaxPeCol = filterStreamerMaxPeCol2DramBlock(maxPeCols);

						unsigned int iTransferBlockDDR = iTransferBlockFilterBaseDDR;

						EMULATOR_PRINT(("[kernelFilterWriter] Sending filter %d to row %d. (iHeightGlobal, iWidthGlobal): (%d, %d). Number of transfer blocks: %d\n",
							iFilterGlobal, iPeRow, iHeightGlobal, iWidthGlobal, maxTransferBlockInFilter));
						for (unsigned short iTransmitCount=0; iTransmitCount<maxTransmitCount; iTransmitCount++)
						{
							t_dram_block block;
							if (iTransmitCount == 0) 
							{
								block = dramControl;
							}
							else if (iTransmitCount == 1)
							{
								block = dramMaxPeCol;
							}
							else
							{
								block = pDramWeights[iTransferBlockDDR >> WIDE_SIZE_OFFSET];
							}

							t_dram_block_tagged taggedBlock;
							taggedBlock.dramBlock = block;
							taggedBlock.destinationRow = iPeRow;
							write_channel_intel(channel_filter_transport[0], taggedBlock);
							if (iTransmitCount>1)
							{
								iTransferBlockDDR += WIDE_SIZE;
							}
						} // iTransmitCount

						iTransferBlockFilterBaseDDR += strideExternalMemory;
						iFilterGlobal++;

					} // iPeRow

					iFilterInGroup += maxRowUsed;	
				} // iFoldInGroup
			} //iGroup
			iWidthGlobal += maxOutputWidthTileSize;
		} // iOutputWidthTile
		iHeightGlobal += maxOutputHeightTileSize;
	}	// iOutputHeightTile
}

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
		t_dram_block_tagged taggedBlock = read_channel_intel(channel_filter_transport[rowID]);

		int destinationRow = (int) taggedBlock.destinationRow;

		if (destinationRow == rowID)
		{
			write_channel_intel(channel_filter_local[rowID], taggedBlock.dramBlock);
		}
		else
		{
			if (rowID < (PE_ROWS - 1) )
			{
				write_channel_intel(channel_filter_transport[rowID+1], taggedBlock);
			}
		}
	}

		//uint1_t nextState = state;
		/*
		if (state == STATE_FILTER_TEE_HEADER)
		{
			t_filter_streamer_control control = dramBlock2FilterStreamerControl(block);
			maxTransferBlockSize = control.numTransferBlocks;
			transferBlockCount = 0;
			int destinationRow = (int) control.destinationRow;
			if (destinationRow > rowID)
			{
				direction = SWITCH_FILTER_TEE_PASS;
			}
			else
			{
				direction = SWITCH_FILTER_TEE_STREAMER;
			}
			nextState = STATE_FILTER_TEE_PAYLOAD;
		}
		else if (state == STATE_FILTER_TEE_PAYLOAD)
		{
			transferBlockCount += WIDE_SIZE;
			if (transferBlockCount >= maxTransferBlockSize)
			{
				nextState = STATE_FILTER_TEE_HEADER;
			}
		}

		//================transmission=======================

		if (direction == SWITCH_FILTER_TEE_STREAMER)
		{
			write_channel_intel(channel_filter_local[rowID], block);
		}
		else 
		{
			if (rowID < (PE_ROWS - 1) )
			{
				write_channel_intel(channel_filter_transport[rowID+1], block);
			} 
		}

		state = nextState;
		*/
	//}
}

#define STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL 0X0
#define STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_MAX_PE_COL 0x1
#define STATE_FILTER_STREAMER_WRITE_CACHE_WRITE 0X2
#define STATE_FILTER_STREAMER_WRITE_CACHE_WAIT 0X3

#define STATE_FILTER_STREAMER_READ_CACHE_SETUP0 0X0
#define STATE_FILTER_STREAMER_READ_CACHE_SETUP1 0x1
#define STATE_FILTER_STREAMER_READ_CACHE_MAX_COL_SETUP 0x2
#define STATE_FILTER_STREAMER_READ_CACHE_READ 0X3
#define STATE_FILTER_STREAMER_READ_CACHE_WAIT 0X4

/*! kernelFilterStreamer
	\brief Stream filter values to the PE array
*/
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_ROWS)))
__kernel void kernelFilterStreamer ()
{
	int rowID = get_compute_id(0);

	typedef uint3_t t_state;
	//important to size the bankwidth, otherwise the default 32 bit will be used, resulting in complex store logic
	t_dram_block cacheNzBlocks [2][KERNEL_CACHE_DEPTH] __attribute__((bankwidth(BURST_SIZE_BYTE))); 
	uint1_t regWriteSide = 0x0;
	unsigned char maxOutputHeightTileSize[2]; //maxTP
	unsigned char maxOutputWidthTileSize[2]; //maxTQ 
	unsigned short maxTransferBlockInFilter[2]; //maxCg
	unsigned char maxPeCols[2];

	//=================Write into cache variables=================
	t_state stateWriteCache = STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL;
	unsigned short iTransferBlockInFilterWrite; //iCg

	//=================Read from cache variables=================
	t_state stateReadCache = STATE_FILTER_STREAMER_READ_CACHE_WAIT;
	unsigned short iTransferBlockInFilterRead; //iCg
	unsigned char iWidthInOutputTileRead; //pq*A
	unsigned char iHeightInOutputTileRead; //p
	unsigned char maxColUsedRead; //maxA

	#pragma ivdep array(cacheNzBlocks)
	while (true)
	{
		//===============Write side====================
		t_state nextStateWriteCache = stateWriteCache;
		{
			bool success = false;
			t_dram_block writeBlock;
			if ( (stateWriteCache == STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL)
				|| (stateWriteCache == STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_MAX_PE_COL)
				|| (stateWriteCache == STATE_FILTER_STREAMER_WRITE_CACHE_WRITE) )
			{
				writeBlock = read_channel_nb_intel(channel_filter_local[rowID], &success);
			}
			
			if (stateWriteCache == STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL)
			{
				if (success)
				{
					t_filter_streamer_control control = 
						dramBlock2FilterStreamerControl(writeBlock);
					maxOutputHeightTileSize[regWriteSide] = control.maxOutputHeightTileSize;
					maxOutputWidthTileSize[regWriteSide] = control.maxOutputWidthTileSize;
					maxTransferBlockInFilter[regWriteSide] = control.numTransferBlocks;
					iTransferBlockInFilterWrite = 0;

					EMULATOR_PRINT(("[kernelFilterStreamer %d] Received setup packet for a new filter. Number of transfer blocks to follow: %d\n", rowID, control.numTransferBlocks));

					nextStateWriteCache = STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_MAX_PE_COL;
				}
			} // STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL
			else if (stateWriteCache == STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_MAX_PE_COL)
			{
				if (success)
				{
					unsigned char maxPeColsLocal = dramBlock2FilterStreamerMaxPeCol(writeBlock);
					EMULATOR_PRINT(("[kernelFilterStreamer %d] Received setup packet for the maximum number of PE cols to activate: %d\n", rowID, maxPeColsLocal));
					maxPeCols[regWriteSide] = maxPeColsLocal;

					nextStateWriteCache = STATE_FILTER_STREAMER_WRITE_CACHE_WRITE;
				}
			} //STATE_FILTER_STREAMER_READ_CACHE_MAX_COL_SETUP
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
		
		if (stateReadCache == STATE_FILTER_STREAMER_READ_CACHE_SETUP0)
		{
			iWidthInOutputTileRead = 0;
			iHeightInOutputTileRead = 0;
			//maxColUsedRead = (maxOutputWidthTileSize[(~regWriteSide) & 0x1] - iWidthInOutputTileRead) < PE_COLS ?
			//	(maxOutputWidthTileSize[(~regWriteSide) & 0x1] - iWidthInOutputTileRead) : PE_COLS;
			iTransferBlockInFilterRead = 0;
			nextStateReadCache = STATE_FILTER_STREAMER_READ_CACHE_MAX_COL_SETUP;
		} // STATE_FILTER_STREAMER_READ_CACHE_SETUP0
		else if (stateReadCache == STATE_FILTER_STREAMER_READ_CACHE_SETUP1)
		{
			nextStateReadCache = STATE_FILTER_STREAMER_READ_CACHE_MAX_COL_SETUP;

			iTransferBlockInFilterRead = 0;
			iWidthInOutputTileRead += maxColUsedRead;
			
			if (iWidthInOutputTileRead >= maxOutputWidthTileSize[(~regWriteSide) & 0x1])
			{
				iWidthInOutputTileRead = 0;
				iHeightInOutputTileRead++;

				if (iHeightInOutputTileRead >= maxOutputHeightTileSize[(~regWriteSide) & 0x1])
				{
					nextStateReadCache = STATE_FILTER_STREAMER_READ_CACHE_WAIT;
				}
			}

			//maxColUsedRead = (maxOutputWidthTileSize[(~regWriteSide) & 0x1] - iWidthInOutputTileRead) < PE_COLS ?
			//	(maxOutputWidthTileSize[(~regWriteSide) & 0x1] - iWidthInOutputTileRead) : PE_COLS;

		} // STATE_FILTER_STREAMER_READ_CACHE_SETUP1
		else if (stateReadCache == STATE_FILTER_STREAMER_READ_CACHE_MAX_COL_SETUP)
		{
			maxColUsedRead = (maxOutputWidthTileSize[(~regWriteSide) & 0x1] - iWidthInOutputTileRead) < maxPeCols[(~regWriteSide) & 0x1] ?
				(maxOutputWidthTileSize[(~regWriteSide) & 0x1] - iWidthInOutputTileRead) : maxPeCols[(~regWriteSide) & 0x1];

			nextStateReadCache = STATE_FILTER_STREAMER_READ_CACHE_READ;
		} //STATE_FILTER_STREAMER_READ_CACHE_MAX_COL_SETUP
		else if (stateReadCache == STATE_FILTER_STREAMER_READ_CACHE_READ)
		{
			unsigned short dramIndex = iTransferBlockInFilterRead >> WIDE_SIZE_OFFSET;
			unsigned short indexInDramBlock = iTransferBlockInFilterRead & WIDE_SIZE_REMAINDER_MASK;
			t_dram_block dramBlock = cacheNzBlocks[(~regWriteSide) & 0x1][dramIndex];
			t_transfer_block tblock = dramBlock.transferBlocks[indexInDramBlock];
			t_transferblock_tagged taggedBlock;
			taggedBlock.values = tblock;
			taggedBlock.maxTransportID = (maxColUsedRead - 1);
			taggedBlock.isLast = ((iTransferBlockInFilterRead + 1) >= maxTransferBlockInFilter[(~regWriteSide) & 0x1]) ?
				TRUE : FALSE;
			bool success = false;
			success = write_channel_nb_intel(channel_weightLanes[rowID][0], taggedBlock);
			if (success)
			{
				/*
				EMULATOR_PRINT(("[kernelFilterStreamer %d] Sent tb %d: %d % d %d %d\n", 
					rowID, 
					iTransferBlockInFilterRead,
					tblock.values[0].cluster_values[0],
					tblock.values[0].cluster_values[1],
					tblock.values[1].cluster_values[0],
					tblock.values[1].cluster_values[1]));
				*/
				if ((iTransferBlockInFilterRead + 1) >= maxTransferBlockInFilter[(~regWriteSide) & 0x1])
				{
					nextStateReadCache = STATE_FILTER_STREAMER_READ_CACHE_SETUP1;
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
			nextStateReadCache = STATE_FILTER_STREAMER_READ_CACHE_SETUP0;
			nextStateWriteCache = STATE_FILTER_STREAMER_WRITE_CACHE_SETUP_CONTROL;
			regWriteSide = (~regWriteSide) & 0x1; 
			EMULATOR_PRINT(("[kernelFilterStreamer %d] Swap\n", rowID));

		}

		stateReadCache = nextStateReadCache;
		stateWriteCache = nextStateWriteCache;

	} // while


}


/*! kernelTensorChecker
	\brief Examine the transfer blocks that belong to the same filter received on a given channel to the host 
	\param pHost. Pointer to the memory region for storing the transfer blocks.
	\param channelID. Index of the channel to listen.
	\param sequenceID. Index of the sequence (marked by isLast) that the checker should listen.
*/
__attribute__((max_global_work_dim(0)))
__kernel void kernelTensorChecker (
	__global t_transfer_block* restrict pHost,
	unsigned char channelID,
	unsigned int sequenceID) 
{
	bool proceed = true;
	unsigned int countSequenceID = 0;
	unsigned int dramLocation = 0;

	while (countSequenceID <= sequenceID)
	{
		t_transferblock_tagged blocks [PE_ROWS];
		bool readStatus[PE_ROWS];
		#pragma unroll 
		for (unsigned char j=0; j<PE_ROWS; j++) {
			blocks[j] = read_channel_nb_intel(channel_weightLanes[j][0], &(readStatus[j]) );
		}

		if (readStatus[channelID])
		{

			t_transferblock_tagged targetBlock = blocks[channelID];
			if (countSequenceID == sequenceID)
			{
				pHost[dramLocation++] = targetBlock.values;
			}

			if (targetBlock.isLast == TRUE)
			{
				EMULATOR_PRINT(("[kernelTensorChecker] Finished reading one sequence on target row %d\n", channelID));
				EMULATOR_PRINT(("[kernelTensorChecker] Sequence ID: %d\n", countSequenceID));
				countSequenceID++;		
			}
		}
	}
}