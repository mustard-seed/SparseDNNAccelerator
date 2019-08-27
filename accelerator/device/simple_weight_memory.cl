#include "params.hpp"
#include "device_structures.hpp"
#include "channels.cl"
#include "device_utils.hpp"

__attribute__((max_global_work_dim(0)))
__kernel void nop () {}


#define BURST_SIZE 8 //Unite: transfer blocks

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

	unsigned int strideBetweenFilterInExternalMemory,

	unsigned short outputWidth,
	unsigned char numOutputWidthTile,
	unsigned char sizeOutputWidthTile,
	unsigned char sizeOutputWidthTile0, //Special case: for the left corner

	unsigned short outputHeight,
	unsigned char numOutputHeightTile,
	unsigned char sizeOutputHeightTile,
	unsigned char sizeOutputHeightTile0, //Special case: for the left corner

	unsigned short numFiltersInKernel, //L
	) {
	//Cache: SIMD blocks
	t_transfer_block cacheSimdBlock [2][KERNEL_CACHE_DEPTH][PE_ROWS] __attribute__ ((numbanks(BURST_SIZE*PE_ROWS), bankwidth(CLUSTER_SIZE*TRANSFER_SIZE)));

	//Cache: Cache address of the streaming blocks
	t_spOffset cacheStreamingBlockAddress [4096];

	//=============================================================
	//Read all the streaming block address as soon as possible
	//==============================================================

	
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