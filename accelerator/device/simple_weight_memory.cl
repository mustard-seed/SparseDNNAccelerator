#include "params.hpp"
#include "device_structures.hpp"
#include "channels.cl"
#include "device_utils.hpp"

__attribute__((max_global_work_dim(0)))
__kernel void nop () {}


#define BURST_SIZE_BYTE 32
#define BURST_SIZE_TRANSFER_BLOCK  BURST_SIZE_BYTE/CLUSTER_SIZE/TRANSFER_SIZE

#define MAX_CACHE_SIZE 1024
#define WRITE_TO_CAHCE_SIDE 0x0
#define READ_FROM_CACHE_SIDE 0x1

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

#define TRUE 0X1
#define FALSE 0X0

/*! kernelSimpleWeightStreamer
Important assumption: The address cache only stores the BRAM address of the first streaming block in each strip.
*/
__attribute__((max_global_work_dim(0)))
__kernel void kernelSimpleWeightStreamer (
	 volatile __global t_dram_block* restrict pDramWeights, //Pointer to the filter weights in external memory
	 volatile __global t_spOffset* restrict pStreamBlockAddress,

	unsigned int strideExternalMemory, //Distance between the start of successive filters in DRAM in terms of transfer blocks

	unsigned short outputWidth, //Q
	unsigned char sizeOutputWidthTile, //TQ
	unsigned short numOutputWidthTile, // ceil (Q / TQ)
	unsigned char sizeOutputWidthTile0, //Special case: TQ for the tiles on the left boundary

	unsigned short outputHeight, //P
	unsigned char numOutputHeightTile, //TP
	unsigned char sizeOutputHeightTile, // ceil (P / TP)
	unsigned char sizeOutputHeightTile0, //Special case: for the left corner

	unsigned int numFiltersInKernel, //L

	unsigned short numGroups, // L / G
	unsigned short numFiltersInGroup, // G
	unsigned short numFoldInGroup // ceil (G / F)

	) {
	typedef uint3_t t_state;
	//Cache: NzBlocks blocks
	t_dram_block cacheNzBlocks [2][KERNEL_CACHE_DEPTH][PE_ROWS] __attribute__ ((numbanks(PE_ROWS), bankwidth(CLUSTER_SIZE*TRANSFER_SIZE*WIDE_SIZE)));
	//t_transfer_block cacheNzBlocks [2][PE_ROWS][KERNEL_CACHE_DEPTH] __attribute__ ((numbanks(BURST_SIZE_TRANSFER_BLOCK * PE_ROWS), bankwidth(CLUSTER_SIZE*TRANSFER_SIZE)));

	//Cache: Cache address of the streaming blocks
	t_spOffset cacheStreamingBlockAddress [4096] __attribute__((numbanks(1)));

	//=============================================================
	//Read all the streaming block address in to BRAM as soon as possible
	//==============================================================
	{
		unsigned int countFilters = 0;
		while (countFilters < numFiltersInKernel) {
			//Unroll the loop to increase the transfer width and make better use of the BRAM bandwidth
			#pragma unroll 2
			for (unsigned char i=0; i<2; i++)
			{
				unsigned short filterIndex = countFilters + i;
				if (filterIndex < numFiltersInKernel)
				{
					cacheStreamingBlockAddress[filterIndex] = pStreamBlockAddress[filterIndex];
				}
			}
			countFilters += 2;
		}
	}

	bool proceed = true;

	//======================Other shared variables============================
	uint1_t regWriteSide = 0x0;

	//====================Write into cache variables=======================
	t_state stateWriteCache = WEIGHT_WRITE_CACHE_STREAM;

	unsigned short iOutputHeightTileWrite = 0; //tp
	unsigned short iOutputWidthTileWrite = 0; //tq
	unsigned short iGroupWrite = 0; // gl
	unsigned short iFoldInGroupWrite = 0; //gf
	unsigned char iPeRowWrite = 0; //f
	unsigned short iTransferBlockInFilterWrite = 0; //iCg

	unsigned short iFilterGlobalWrite = 0; //iL
	unsigned short iFilterInGroupWrite = 0; //gf * F
	unsigned int iTransferBlockFilterBaseDDR = 0;
	unsigned int iTransferBlockDDR = iTransferBlockFilterBaseDDR;
	
	unsigned char maxRowUsedWrite = PE_ROWS < (numFiltersInGroup - iFilterInGroupWrite) ?
				PE_ROWS : (numFiltersInGroup - iFilterInGroupWrite); //maxF
	unsigned short maxTransferBlockInFilterWrite = cacheStreamingBlockAddress[iFilterGlobalWrite];



	//===================Read from cache variables=========================
	t_state stateReadCache = WEIGHT_READ_CACHE_WAIT;

	unsigned short iOutputHeightTileRead = 0; //tp
	unsigned short iOutputWidthTileRead = 0; //tq
	unsigned short iGroupRead = 0; // gl
	unsigned short iFoldInGroupRead = 0; //gf
	unsigned short iFoldInOutputHeightTileRead = 0; // p
	unsigned short iFoldInOutputWidthTileRead = 0; // pq
	unsigned short iPeRowRead = 0; //f
	unsigned short iTransferBlockInFilterRead[PE_ROWS]; //iCg
	#pragma unroll
	for (int i=0; i<PE_ROWS; i++)
	{
		iTransferBlockInFilterRead[i] = 0;
	}

	unsigned short iHeightGlobalRead = 0; // countPCoveredByTP
	unsigned short iWidthGlobalRead = 0; //countQCovertedByTQ
	unsigned short iFilterGlobalRead = 0; //iL Need this to access the streaming block sizes
	unsigned short iFilterInGroupRead = 0; // gf * F
	unsigned short iWidthInTileRead = 0; // pq * A
	unsigned short iHeightInTileRead = 0;

	
	unsigned char maxOutputHeightTileSize = sizeOutputHeightTile0; //maxTP
	unsigned char maxOutputWidthTileSize = sizeOutputWidthTile0; //maxTQ
	unsigned char maxRowUsedRead = PE_ROWS < (numFiltersInGroup - iFilterInGroupRead) ?
				PE_ROWS : (numFiltersInGroup - iFilterInGroupRead); //maxF
	unsigned char maxColUsedRead = PE_COLS < (maxOutputWidthTileSize - iWidthInTileRead) ?
				PE_COLS : (maxOutputWidthTileSize - iWidthInTileRead); //maxA
	unsigned short maxTransferBlockInFilterRead [PE_ROWS];
	unsigned char iMaxRowUsedRead;
	#pragma unroll 1
	for (int i=0; i<PE_ROWS; i++)
	{
		if (i<maxRowUsedRead)
		{
			maxTransferBlockInFilterRead[i] = cacheStreamingBlockAddress[iFilterGlobalRead+i];
		}
	}


	//===============State actions=======================================
	#pragma ivdep array(cacheNzBlocks)
	while (proceed) 
	{
		t_state nextStateWriteCache = stateWriteCache;
		if (stateWriteCache == WEIGHT_WRITE_CACHE_SETUP_PE_ROW)
		{
			nextStateWriteCache = WEIGHT_WRITE_CACHE_SETUP_FOLD_IN_GROUP;

			iFilterGlobalWrite++;
			iPeRowWrite++;
		}
		else if (stateWriteCache == WEIGHT_WRITE_CACHE_SETUP_FOLD_IN_GROUP)
		{
			nextStateWriteCache = WEIGHT_WRITE_CACHE_SETUP_GROUP;
			if (iPeRowWrite >= maxRowUsedRead)
			{
				iPeRowWrite = 0;
				iFilterInGroupWrite += maxRowUsedWrite;
				iFoldInGroupWrite++;
			}
		}
		else if (stateWriteCache == WEIGHT_WRITE_CACHE_SETUP_GROUP)
		{
			nextStateWriteCache = WEIGHT_WRITE_CACHE_SETUP_TILES;

			if (iFoldInGroupWrite>=numFoldInGroup)
			{
				iFilterInGroupWrite = 0;
				iFoldInGroupWrite = 0;
				iGroupWrite++;
			}
		}
		else if (stateWriteCache == WEIGHT_WRITE_CACHE_SETUP_TILES)
		{
			nextStateWriteCache = WEIGHT_WRITE_CACHE_SETUP_AUX_UPDATE;

			if (iGroupWrite>=numGroups)
			{
				iGroupWrite = 0;

				iOutputWidthTileWrite++;
				if (iOutputWidthTileWrite>=numOutputWidthTile) {
					iOutputWidthTileWrite = 0;

					iOutputHeightTileWrite++;
				}

				iTransferBlockFilterBaseDDR = 0;
				iTransferBlockDDR = iTransferBlockFilterBaseDDR;
				iFilterGlobalWrite = 0;
			}
		}
		else if (stateWriteCache == WEIGHT_WRITE_CACHE_SETUP_AUX_UPDATE)
		{
			nextStateWriteCache = WEIGHT_WRITE_CACHE_STREAM;
			//maxF update
			if (iPeRowWrite == 0) //a fold is about to be loaded
			{
				maxRowUsedWrite = PE_ROWS <  (numFiltersInGroup - iFilterInGroupWrite) ?
					PE_ROWS : numFiltersInGroup - iFilterInGroupWrite;
				nextStateWriteCache = WEIGHT_WRITE_CACHE_WAIT;
			}

			if (iOutputHeightTileWrite>=numOutputHeightTile)
			{
				nextStateWriteCache = WEIGHT_WRITE_CACHE_FINISH;
			}
		}
		else if (stateWriteCache == WEIGHT_WRITE_CACHE_STREAM)
		{
			//Load weights in burst. Make the most use of the DRAM bandwidth
			//#pragma unroll
			//for (unsigned int i=0; i<BURST_SIZE_TRANSFER_BLOCK; i++)
			//{
			//	cacheNzBlocks[regWriteSide][iTransferBlockInFilterWrite++][iPeRowWrite]
			//		= pWeights[iTransferBlockDDR++];
			//}
			cacheNzBlocks[regWriteSide][iTransferBlockInFilterWrite >> WIDE_SIZE_OFFSET][iPeRowWrite]
				= pDramWeights[iTransferBlockDDR >> WIDE_SIZE_OFFSET];

			iTransferBlockInFilterWrite += WIDE_SIZE;
			iTransferBlockDDR += WIDE_SIZE;

			//iTransferBlockInFilterWrite += BURST_SIZE_TRANSFER_BLOCK;
			// Update the parameters once the nz values in one filter has been loaded from DRAM
			if (iTransferBlockInFilterWrite >= maxTransferBlockInFilterWrite)
			{
				iTransferBlockInFilterWrite = 0;
				nextStateWriteCache = WEIGHT_WRITE_CACHE_SETUP_PE_ROW;

				//Tricky: the following variables are updated in WEIGHT_WRITE_CACHE_SETUP too.
				iTransferBlockFilterBaseDDR += strideExternalMemory;
				iTransferBlockDDR = iTransferBlockFilterBaseDDR;			
			}
			
		} // WEIGHT_WRITE_CACHE_SETUP

		t_state nextStateReadCache = stateReadCache;
		if (stateReadCache == WEIGHT_READ_CACHE_SETUP_DIM_IN_TILE)
		{
			nextStateReadCache = WEIGHT_READ_CACHE_SETUP_FOLD_GROUP;
			//iTransferBlockInFilterRead = 0;

			#pragma unroll
			for (unsigned char i=0; i<PE_ROWS; i++)
			{
				iTransferBlockInFilterRead[i] = 0x0;
			}

			iWidthInTileRead += maxColUsedRead;
			if (iWidthInTileRead >= maxOutputWidthTileSize)
			{
				iWidthInTileRead = 0;

				iHeightInTileRead++;

			} //iWidthInTileRead
		} // WEIGHT_READ_CACHE_SETUP
		else if (stateReadCache == WEIGHT_READ_CACHE_SETUP_FOLD_GROUP)
		{
			nextStateReadCache = WEIGHT_READ_CACHE_SETUP_TILES;

			if (iHeightInTileRead >= maxOutputHeightTileSize)
			{
				iHeightInTileRead = 0;

				iFilterGlobalRead += maxRowUsedRead;
				iFilterInGroupRead += maxRowUsedRead;

				iFoldInGroupRead++;

				if (iFoldInGroupRead >= numFoldInGroup)
				{
					iFoldInGroupRead = 0;
					iFilterInGroupRead = 0;

					iGroupRead++;

				} //iFoldInGroupRead

			} // iHeightInTileRead

		} //WEIGHT_READ_CACHE_SETUP1
		else if (stateReadCache == WEIGHT_READ_CACHE_SETUP_TILES)
		{
			nextStateReadCache = WEIGHT_READ_CACHE_SETUP_AUX_UPDATE;

			if (iGroupRead >= numGroups)
			{
				iGroupRead = 0;

				iFilterGlobalRead = 0;

				iWidthGlobalRead += maxOutputWidthTileSize;
				

				iOutputWidthTileRead++;
				if (iOutputWidthTileRead >= numOutputWidthTile)
				{
					iOutputWidthTileRead = 0;
					iWidthGlobalRead = 0;

					iHeightGlobalRead += maxOutputHeightTileSize;

					iOutputHeightTileRead++;
					if (iOutputHeightTileRead >= numOutputHeightTile)
					{
						nextStateReadCache = WEIGHT_READ_CACHE_FINISH;
					}
				} // iOutputwidthTileRead
				
			} //iGroupRead

		} //WEIGHT_READ_CACHE_SETUP_AUX_UPDATE
		else if (stateReadCache == WEIGHT_READ_CACHE_SETUP_AUX_UPDATE)
		{
			nextStateReadCache = WEIGHT_READ_CACHE_STREAM;
			if ( (iWidthInTileRead == 0) 
					&& (iHeightInTileRead == 0)
					&& (iFoldInGroupRead == 0)
					&& (iGroupRead == 0) )
			{
				maxOutputWidthTileSize = sizeOutputWidthTile < (outputWidth - iWidthGlobalRead) ?
					sizeOutputWidthTile : (outputWidth - iWidthGlobalRead);

				if (iOutputWidthTileRead == 0)
				{
					maxOutputWidthTileSize = sizeOutputWidthTile0;
					maxOutputHeightTileSize = sizeOutputHeightTile < (outputHeight - iHeightGlobalRead) ?
					sizeOutputHeightTile : (outputHeight - iHeightGlobalRead);
				}

			}

			if ( (iHeightInTileRead == 0) && (iWidthInTileRead == 0) )
			{
				maxRowUsedRead = PE_ROWS < (numFiltersInGroup - iFilterInGroupRead) ?
							PE_ROWS : (numFiltersInGroup - iFilterInGroupRead);
				iMaxRowUsedRead = 0;
				nextStateReadCache = WEIGHT_READ_CACHE_LOAD_ADDRESS;
			}

			maxColUsedRead = PE_COLS < (maxOutputWidthTileSize - iWidthInTileRead) ?
				PE_COLS : maxOutputWidthTileSize - iWidthInTileRead;

		}
		else if (stateReadCache == WEIGHT_READ_CACHE_LOAD_ADDRESS)
		{
			maxTransferBlockInFilterRead[iMaxRowUsedRead] = cacheStreamingBlockAddress[iFilterGlobalRead+iMaxRowUsedRead];
			iMaxRowUsedRead++;
			if (iMaxRowUsedRead >= maxRowUsedRead)
			{
				nextStateReadCache = WEIGHT_READ_CACHE_WAIT;
			}

		} // WEIGHT_READ_CACHE_LOAD_ADDRESS

		else if (stateReadCache == WEIGHT_READ_CACHE_STREAM)
		{
			//Keep streaming until we have drained all weights once
			uint1_t finished[PE_ROWS];

			#pragma unroll
			for (unsigned char i=0; i<PE_ROWS; i++)
			{
				uint1_t finishedLocal = TRUE;
				if (i<maxRowUsedRead)
				{
					unsigned short index = iTransferBlockInFilterRead[i];
					unsigned short numTransferBlock = maxTransferBlockInFilterRead[i];
					if (index < numTransferBlock)
					{
						finishedLocal = FALSE;
						t_dram_block wideValues = cacheNzBlocks[~regWriteSide][index >> WIDE_SIZE_OFFSET][i];
						//t_dram_block wideValues = cacheNzBlocks[~regWriteSide][index][i];
						t_transfer_block values = wideValues.transferBlocks[index & WIDE_SIZE_REMAINDER_MASK];
						//t_transfer_block values = wideValues.transferBlocks[0];
						t_transferblock_tagged taggedBlock;
						taggedBlock.values = values;
						taggedBlock.maxTransportID = (maxColUsedRead - 1);
						taggedBlock.isLast = (index == (numTransferBlock - 1)) ?
							0x1 : 0x0;
						//bool writeSuccess;
						bool writeSuccess = write_channel_nb_intel(channel_weightLanes[i][0], taggedBlock);
						//write_channel_nb_intel(channel_weightLanes[i][0], taggedBlock);
						if (writeSuccess)
						{
							index++;
						}
						iTransferBlockInFilterRead[i] = index;
					}
				}
				finished[i] = finishedLocal;
			} // for
			//iTransferBlockInFilterRead++;
			uint1_t finishedGlobal = TRUE;
			#pragma unroll
			for (unsigned char i=0; i<PE_ROWS; i++)
			{	
				finishedGlobal &= finished[i];
			}
			if (finishedGlobal == TRUE)
			{
				nextStateReadCache = WEIGHT_READ_CACHE_SETUP_DIM_IN_TILE;
			}
		} // WEIGHT_READ_CACHE_STREAM
		else if ((stateReadCache == WEIGHT_READ_CACHE_WAIT) && (stateWriteCache == WEIGHT_WRITE_CACHE_WAIT))
		{
			regWriteSide = (~regWriteSide) & 0x1;
			nextStateWriteCache = WEIGHT_WRITE_CACHE_STREAM;
			nextStateReadCache = WEIGHT_READ_CACHE_STREAM;
		}
		else if ((stateReadCache == WEIGHT_READ_CACHE_WAIT) && (stateWriteCache == WEIGHT_WRITE_CACHE_FINISH))
		{
			regWriteSide = (~regWriteSide) & 0x1;
			nextStateReadCache = WEIGHT_READ_CACHE_STREAM;
		}
		else if ((stateReadCache == WEIGHT_READ_CACHE_FINISH) && (stateWriteCache == WEIGHT_WRITE_CACHE_WAIT))
		{
			regWriteSide = (~regWriteSide) & 0x1;
			nextStateWriteCache = WEIGHT_WRITE_CACHE_STREAM;
		}
		else if ((stateReadCache == WEIGHT_READ_CACHE_FINISH) && (stateWriteCache == WEIGHT_WRITE_CACHE_FINISH))
		{
			proceed = false;
		}

		//==================Final state update===================================
		stateWriteCache = nextStateWriteCache;
		stateReadCache = nextStateReadCache;
	}// while 
	
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

	bool readStatus[PE_ROWS];
	while (countSequenceID <= sequenceID)
	{
		t_transferblock_tagged blocks [PE_ROWS];
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

			if (targetBlock.isLast == 0x1)
			{
				countSequenceID++;		
			}
		}
	}
}