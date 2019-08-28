#include "params.hpp"
#include "device_structures.hpp"
#include "channels.cl"
#include "device_utils.hpp"

__attribute__((max_global_work_dim(0)))
__kernel void nop () {}


#define BURST_SIZE 8 //Unit: transfer blocks. 32 bytes / (4 bytes per cycle) = 8 / cycle

#define MAX_CACHE_SIZE 1024
#define WRITE_TO_CAHCE_SIDE 0x0
#define READ_FROM_CACHE_SIDE 0x1

#define WEIGHT_WRITE_CACHE_SETUP 0x0
#define WEIGHT_WRITE_CACHE_STREAM 0x1
#define WEIGHT_WRITE_CACHE_WAIT 0x2
#define WEIGHT_WRITE_CACHE_FINISH 0X3

#define WEIGHT_READ_CACHE_SETUP 0x0
#define WEIGHT_READ_CACHE_STREAM  0x1
#define WEIGHT_READ_CACHE_WAIT  0x2
#define WEIGHT_READ_CACHE_FINISH 0x4

/*! kernelSimpleWeightStreamer
Important assumption: The address cache only stores the BRAM address of the first streaming block in each strip.
*/
__attribute__((max_global_work_dim(0)))
__kernel void kernelSimpleWeightStreamer (
	volatile __global t_transfer_block* restrict pWeights, //Pointer to the filter weights in external memory
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
	unsigned short numFoldPerGroup // ceil (G / F)

	) {
	typedef uint3_t t_state;
	//Cache: NzBlocks blocks
	t_transfer_block cacheNzBlocks [2][KERNEL_CACHE_DEPTH][PE_ROWS] __attribute__ ((numbanks(BURST_SIZE*PE_ROWS), bankwidth(CLUSTER_SIZE*TRANSFER_SIZE)));

	//Cache: Cache address of the streaming blocks
	t_spOffset cacheStreamingBlockAddress [4096] __attribute__((numbanks(PE_ROWS)));

	//=============================================================
	//Read all the streaming block address in to BRAM as soon as possible
	//==============================================================
	{
		unsigned int countFilters = 0;
		while (countFilters < numFilters) {
			//Unroll the loop to increase the transfer width and make better use of the BRAM bandwidth
			#pragma unroll
			for (unsigned char i=0; i<PE_ROWS; i++)
			{
				unsigned short filterIndex = counterFilters + i;
				if (filterIndex < numFilters)
				{
					cacheStreamingBlockAddress[filterIndex] = pStreamBlockAddress[filterIndex];
				}
			}
			countFilters += PE_ROWS;
		}
	}

	bool proceed = true;

	//======================Other shared variables============================
	uint1_t regWriteSide = 0x0;




	//====================Write into cache variables=======================
	t_state stateWriteCache = WEIGHT_WRITE_CACHE_STREAM;

	unsigned short iOutputHeightTileWrite = 0; //tp
	unsigned short iOutputWidthTileWrite = 0; //tq
	unsigned char iPeRowWrite;
	unsigned char maxSizeOutputHeightTileWrite; //maxTP
	unsigned char maxSizeOutputWidthTileWrite; //maxTQ
	unsigned char maxRowsUsedWrite; //maxF

	unsigned short iFilterGlobalWrite = 0; //iL
	unsigned char iFilterLocalWrite = 0; //0 <= x <= maxRowUsedWrite
	unsigned short iScalarInFilterWrite = 0; //iCg
	unsigned short iFitlerInGroupWrite = 0; //gf * F

	//===================Read from cache variables=========================

	while (proceed) 
	{
		t_state nextStateWriteCache;
		if (stateWriteCache == WEIGHT_WRITE_CACHE_STREAM)
		{


		} // WEIGHT_WRITE_CACHE_STREAM	
		else if (stateWriteCache == WEIGHT_WRITE_CACHE_SETUP)
		{
			
		} // WEIGHT_WRITE_CACHE_SETUP
	}



	
}

/*! kernelTensorChecker
	\brief Transfer a select range of transfer blocks received on a given channel to the host 
	\param pHost. Pointer to the memory region for storing the transfer blocks.
	\param channelID. Index of the channel to listen.
	\param startSequenceID. Index of the first sequence (marked by isLast) that the checker should listen.
	\param endSequenceID. Index of the last sequence (marked by isLast) that the checker should listen
*/
__attribute__((max_global_work_dim(0)))
__kernel void kernelTensorChecker (
	__global t_cluster_block* restrict pHost,
	unsigned char channelID,
	unsigned short startSequenceID,
	unsigned short endSequenceID) {

	for (int i=0; i<numRead; i++) {
		t_simdblock_di_tagged blocks[PE_ROWS];
		#pragma unroll 
		for (unsigned char j=0; j<PE_ROWS; j++) {
			blocks[j] = read_channel_intel(channel_weightLanes[j][0]);
		}
		t_simdblock_host hostBlock;
		#pragma unroll
		for (unsigned char j=0; j<SIMD_SIZE; j++) {
			hostBlock.values[j] = blocks[0].values[j];
		}
		hostBlock.runLength = blocks[0].streamingBlockIndex;
		pSimdHost[i] = hostBlock;
	}
}