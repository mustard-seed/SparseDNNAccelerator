#include "params.hpp"
#include "device_structures.hpp"
#include "channels.cl"
#include "device_utils.hpp"

__attribute__((max_global_work_dim(0)))
__kernel void nop () {}


#define MAX_CACHE_SIZE 1024
#define WRITE_TO_CAHCE_SIDE 0x0
#define READ_FROM_CACHE_SIDE 0x1

#define SIMPLE_WEIGHT_WRITE_CACHE_SETUP 0x0
#define SIMPLE_WEIGHT_WRITE_CACHE_STREAM 0x1
#define SIMPLE_WEIGHT_READ_CACHE_SETUP 0x0
#define SIMPLE_WEIGHT_READ_CACHE_STREAM  0x1

/*! kernelSimpleWeightStreamer
Important assumption: The address cache only stores the BRAM address of the first streaming block in each strip.
*/
__attribute__((max_global_work_dim(0)))
__kernel void kernelSimpleWeightStreamer (
	volatile __global t_simdblock_value* restrict pSimdBlock, //External memory pointer to the simd block values
	volatile __global t_simdblock_channel_offset* restrict pChannelOffset, //External memory pointer to the simd block channel offsets
	volatile __global t_spOffset* restrict pStreamBlockAddress,

	unsigned short outputWidth,
	unsigned char numOutputWidthTile,
	unsigned char sizeOutputWidthTile,

	unsigned char numStreamingBlocksPerFilter,

	unsigned short numFiltersInKernel, //L
	unsigned short numFiltersPerGroup, // G
	unsigned char numGroups, // L / G
	unsigned char numFoldsPerGroup, // ceil ( G / F )
	unsigned int numTotalFolds //=numGroups * numFoldsPerGroup * numOutputWidthTile * numOutputHeight
	) {
	//Cache: SIMD blocks
	t_simdblock_value cacheSimdBlock [KERNEL_CACHE_DEPTH][PE_ROWS][2] __attribute__ ((numbanks(2*PE_ROWS), bankwidth(SIMD_SIZE)));
	//Cache; Channel offsets of SIMD blocks
	t_simdblock_channel_offset cacheChannelOffset [KERNEL_CACHE_DEPTH][PE_ROWS][2] __attribute__ ((numbanks(2*PE_ROWS), bankwidth(1)));
	//Cache: Cache address of the streaming blocks
	t_spOffset cacheStreamingBlockAddress [4096];
	//t_spOffset cacheStreamingBlockAddress [PE_ROWS*2];
	//float cache[CACHE_SIZE];

	//Stream the number of compressed simd blocks in each filter into the
	//stream block address cache
	for (unsigned short i=0; i<numFiltersInKernel; i++) {
		cacheStreamingBlockAddress[i] = pStreamBlockAddress[i];
	}

	//iterators related to the output width
	unsigned short iterOutputWidthReadFromCache = 0;
	
	//Number of filters that have been iterated through in a group
	unsigned short iterFiltersInGroup[2] = {0, 0};


	//DRAM address used to accesss channel offset and simd blocks
	int dramOffset = 0;

	//Number of filters that have been written
	unsigned short iterFilterWriteToCache = 0;
	unsigned short iterFilterReadFromCache = 0;

	//Important to go one extra, hence less equal
	for (unsigned int iterFold = 0; iterFold <= numTotalFolds; iterFold++) {

		bool writeIntoCache = (iterFold < numTotalFolds) ? true : false;
		bool readFromCache = (iterFold > 0) ? true: false;
		
		//Bank number used to write into cache
		char bankWriteIntoCache = writeIntoCache ? 0x01 : 0x0;
		char bankReadFromCache = writeIntoCache ? 0x0 : 0x01;

		//Number of filters in a group
		unsigned short numFiltersLeftInTheGroup[2];
		#pragma unroll
		for (int i=0; i<2; i++) {
			numFiltersLeftInTheGroup[i] = numFiltersPerGroup - iterFiltersInGroup[i]; 
		}

		//Number of effective filter lanes
		unsigned char naf[2];
		#pragma unroll
		for (unsigned char i=0; i<2; i++) {
			naf[i] = (numFiltersLeftInTheGroup[i] < PE_ROWS) ?
				((unsigned char) numFiltersLeftInTheGroup[i]) : PE_ROWS;
		}

		unsigned char iterFilterLane = 0;

		//Number of effective output width
		unsigned short numOutputLeftInWidthReadFromCache = outputWidth - iterOutputWidthReadFromCache;
		unsigned char naqReadFromCache = (numOutputLeftInWidthReadFromCache < sizeOutputWidthTile) ? 
			((unsigned char) numOutputLeftInWidthReadFromCache) : sizeOutputWidthTile;

		//States for writing into or reading from the cache
		unsigned char stateWriteToCache = SIMPLE_WEIGHT_WRITE_CACHE_STREAM;

		//Number of streaming blocks in a filter that has been read
		unsigned char iterSimdBlocksInFilterWriteToCache = 0;

		//Value of the streaming block address to be written
		unsigned short streamingBlockAddressWrite = 0;

		//Number of simd blocks inside each filter to be loaded into the cache
		unsigned short numSimdBlocksInFilterWriteToCache[PE_ROWS];

		//Load the number simd blocks to be loaded into the cache from it filter
		//for (unsigned char i=0; i<PE_ROWS; i++) {
		//	numSimdBlocksInFilterWriteToCache[i] = 
		//		cacheStreamingBlockAddress[ (unsigned short) (iterFilterWriteToCache+i) ];
		//}


		//===================================================
		unsigned char stateReadFromCache = SIMPLE_WEIGHT_READ_CACHE_STREAM;

		//Index of the output in the width dimension of an output tile
		unsigned char iterOutputInTileRead = 0;

		unsigned char parallelSizeOutput = ((naqReadFromCache - iterOutputInTileRead) >= PE_COLS) ? PE_COLS: (naqReadFromCache - iterOutputInTileRead);

		unsigned short iterSimdBlockReadInFilterFromCache = 0;

		unsigned short numSimdBlocksInFilterReadFromCache[PE_ROWS];

		//#pragma unroll
		for (unsigned char i=0; i<PE_ROWS; i++) {
			numSimdBlocksInFilterWriteToCache[i] = 
				cacheStreamingBlockAddress[ (unsigned short) (iterFilterWriteToCache+i) ];
			numSimdBlocksInFilterReadFromCache[i] = 
				cacheStreamingBlockAddress[ (unsigned short) (iterFilterReadFromCache+i)];
		}

		//These flags will be modified by the loop
		bool proceedWriteIntoCache = writeIntoCache;
		bool proceedReadFromCache = readFromCache;
		//Flag for initiating the ping-pong motion
		bool proceed = true;
		
		#pragma ivdep array(cacheSimdBlock)
		#pragma ivdep array(cacheChannelOffset)
		#pragma ivdep array(cacheStreamingBlockAddress)
		while (proceed) {

			//Reading simd blocks, channel offsets from DRAM into the cache,
			//and generate cache addresses to the start of each streaming block

			if (proceedWriteIntoCache){

				if (stateWriteToCache == SIMPLE_WEIGHT_WRITE_CACHE_STREAM) {
					t_simdblock_channel_offset channelOffsetBlob = pChannelOffset[dramOffset];

					//Decode the channel offset
	                char channelOffset = channelOffsetBlob;

	                //Read the simd block from the cache
					t_simdblock_value simdValue = pSimdBlock[dramOffset];

					dramOffset++;

					cacheSimdBlock[streamingBlockAddressWrite][iterFilterLane][bankWriteIntoCache]
						= simdValue;
					cacheChannelOffset[streamingBlockAddressWrite][iterFilterLane][bankWriteIntoCache]
						= channelOffset;
					
					iterSimdBlocksInFilterWriteToCache++;
					streamingBlockAddressWrite++;

					if (iterSimdBlocksInFilterWriteToCache == numSimdBlocksInFilterWriteToCache[iterFilterLane]) {
							stateWriteToCache = SIMPLE_WEIGHT_WRITE_CACHE_SETUP;
					}
				} // case SIMPLE_WEIGHT_WRITE_CACHE_STREAM
				//break;

				else if (stateWriteToCache == SIMPLE_WEIGHT_WRITE_CACHE_SETUP) {
					//If we have read all the streaming blocks in one filter
						//Write the tail address of the filter;

						iterSimdBlocksInFilterWriteToCache = 0;
						streamingBlockAddressWrite = 0;

						unsigned char tempIterFilterLane = iterFilterLane+1;

						if (tempIterFilterLane == naf[WRITE_TO_CAHCE_SIDE]) {
							proceedWriteIntoCache = false;
							iterFilterLane = 0;

						}
						else {
							iterFilterLane = tempIterFilterLane;
						}
						stateWriteToCache = SIMPLE_WEIGHT_WRITE_CACHE_STREAM;

				} // SIMPLE_WEIGHT_WRITE_CACHE_SETUP
			} //if write into caches

			
			//Reading simd blocks and channel offsets from the cache into the FIFOs
			//Need to take care of padding
			if (proceedReadFromCache) {

				//switch (stateReadFromCache) {
					if (stateReadFromCache == SIMPLE_WEIGHT_READ_CACHE_STREAM) {
						bool launchNewCacheRead = true;
						#pragma unroll
						for (unsigned char i=0; i<PE_ROWS; i++) {
							
							bool laneReadInProgress = (iterSimdBlockReadInFilterFromCache < numSimdBlocksInFilterReadFromCache[i]);
							bool streamLaneI = laneReadInProgress && (i < naf[READ_FROM_CACHE_SIDE]);
							if (streamLaneI) {
								t_simdblock_value simdValue = cacheSimdBlock[iterSimdBlockReadInFilterFromCache][i][bankReadFromCache];
								t_simdblock_channel_offset offset = cacheChannelOffset[iterSimdBlockReadInFilterFromCache][i][bankReadFromCache];
								bool isLast = false;
								unsigned short numSimdBlockToRead = numSimdBlocksInFilterReadFromCache[i];
								unsigned short tempIterSimdBlockReadI = (unsigned short) (iterSimdBlockReadInFilterFromCache + 1);
								if ( tempIterSimdBlockReadI == numSimdBlockToRead) {
									isLast = true;
								}

								t_simdblock_di_tagged simdBlockSend;
								#pragma unroll
								for (unsigned char j=0; j<SIMD_SIZE; j++) {
									simdBlockSend.values[j] = simdValue.values[j];
								}
								simdBlockSend.streamingBlockIndex = offset;
								simdBlockSend.isLast = isLast;
								simdBlockSend.maxTransportID = parallelSizeOutput;

								//bool writeSuccess = write_channel_nb_intel(channel_weightLanes[i][0], simdBlockSend);
								write_channel_intel(channel_weightLanes[i][0], simdBlockSend);
								//if (writeSuccess) {
								if (tempIterSimdBlockReadI < numSimdBlockToRead) {
									launchNewCacheRead = false;
								}
								//}
								//else {
									//launchNewCacheRead = false;
								//}

							}
						} // parallel for
						iterSimdBlockReadInFilterFromCache++;
						if (launchNewCacheRead) {
							stateReadFromCache = SIMPLE_WEIGHT_READ_CACHE_SETUP;
						}
					} // READ_CACHE_STREAM
					else if (stateReadFromCache == SIMPLE_WEIGHT_READ_CACHE_SETUP) {
						iterOutputInTileRead += parallelSizeOutput;
						unsigned short parallelSizeOutputTemp = naqReadFromCache - iterOutputInTileRead;
						parallelSizeOutput = (parallelSizeOutputTemp >= PE_COLS) ? 
							PE_COLS : parallelSizeOutputTemp;
					
						iterSimdBlockReadInFilterFromCache = 0;
						if (parallelSizeOutputTemp == 0x0) {
							proceedReadFromCache = false;
						}
						stateReadFromCache = SIMPLE_WEIGHT_READ_CACHE_STREAM;
					} // SIMPLE_WEIGHT_READ_CACHE_SETUP

				//} // switch readFromCache
				
			} // if read from caches


			//Final proceed condition
			proceed = proceedWriteIntoCache || proceedReadFromCache;

		}

		//Update loop carried variables
		if (writeIntoCache) {
			unsigned short newIterFiltersInGroup =  iterFiltersInGroup[WRITE_TO_CAHCE_SIDE] + (unsigned short) naf[WRITE_TO_CAHCE_SIDE];
			iterFiltersInGroup[WRITE_TO_CAHCE_SIDE] = newIterFiltersInGroup >= numFiltersPerGroup ? 0 : newIterFiltersInGroup;
			unsigned short newIterFilterWriteToCache = iterFilterWriteToCache + (unsigned short) naf[WRITE_TO_CAHCE_SIDE];
			if (newIterFilterWriteToCache >= numFiltersInKernel) {
				dramOffset = 0;
				iterFilterWriteToCache = 0;
			}
		} // if writeIntCache

		if (readFromCache) {
			unsigned short newIterFiltersInGroup =  iterFiltersInGroup[READ_FROM_CACHE_SIDE] + (unsigned short) naf[READ_FROM_CACHE_SIDE];
			iterFiltersInGroup[READ_FROM_CACHE_SIDE] = newIterFiltersInGroup >= numFiltersPerGroup ? 0 : newIterFiltersInGroup;
		
			unsigned short newIterFilterReadFromCache = iterFilterReadFromCache + (unsigned short) naf[READ_FROM_CACHE_SIDE];
			bool finishedOnePatchReadFromCache = newIterFilterReadFromCache >= numFiltersInKernel;
			iterFilterReadFromCache = finishedOnePatchReadFromCache ? 0 : newIterFilterReadFromCache;

			//Update the iterators related to outputwidth
			if (finishedOnePatchReadFromCache) {
				unsigned short newIterOutputWidth = iterOutputWidthReadFromCache + (unsigned short) naqReadFromCache;
				iterOutputWidthReadFromCache = (newIterOutputWidth >= outputWidth) ? 0 : newIterOutputWidth;
			}
		} // if readFromCache
	} // for fold
	
}

__attribute__((max_global_work_dim(0)))
__kernel void kernelSimpleWeightStreamerChecker (
	__global t_simdblock_host* restrict pSimdHost,
	unsigned int numRead) {

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