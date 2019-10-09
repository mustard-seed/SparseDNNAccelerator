#include "params.hpp"
#include "device_structures.hpp"
#include "channels.cl"
#include "device_utils.hpp"

__attribute__((max_global_work_dim(0)))
__kernel void nop () {}


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
			blocks[j] = read_channel_nb_intel(channel_weight[j][0], &(readStatus[j]) );
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