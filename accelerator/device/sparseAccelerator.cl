#include "ihc_apint.h"
#include "params.hpp"

//Must include the following line in order to use channel
#pragma OPENCL EXTENSION cl_intel_channels : enable

typedef short t_spWeight;
typedef short t_spOffset;

/*! t_tokenFillWeightCache
	Token used to command filling of the sparse weight cache
*/
typedef struct {
	int ddrIndexStartOffset; //Word offset of the indices of the kernel relative to the start of the kernel indices region
	int ddrWeightStartOffset; //Word offset of the weights of the kernel relative to the start of the kernel region
	int filterStart; //Index of the first filter to be streamed into the cache
	int filterEnd; //Index of the last filter to be streamed into the cache

	int numEncodingBlocksInFilter;
	int numWeightsInFilter;

	uint1_t fillBankNumber; //Which bank to fill. Either 0 or 1;
} t_tokenFillWeightCache;


/*! t_tokenDrainWeightCache
	Token used to command draining of the sparse weight cache
*/
typedef struct {
	int laneStart; //First lane to be streamed
	int laneEnd; //Last lane to be streamed
	int cbStart; //Index of the first encoder block inside each channel to be streamed
	int cbEnd; //Index of the last encoder block insider each channel to be streamed

	int R; //Height of the kernel
	int S; //width of the kernel
	int CB; //Number of encoding blocks along the channel per filter

	uint1_t drainBankNumber; //Which bank to drain. Either 0 or 1;
} t_tokenDrainWeightCache;


channel t_tokenFillWeightCache channel_tokenFillWeightCacheControl __attribute__((depth(0)));
channel bool channel_tokenFillWeightCacheFinish __attribute__((depth(0)));

channel t_tokenDrainWeightCache channel_tokenDrainWeightCacheControl __attribute__((depth(0)));
channel bool channel_tokenDrainWeightCacheFinish __attribute__((depth(0)));



/*! Kernel. kernelSparseWeightCache
	\brief The multibank sparse weight and indices cache 
*/
__kernel void kernelSparseWeightCache(
	__global t_spWeight * restrict pSpWeight,
	__global t_spOffset * restrict pSpOffset 
	) 
{
	//Declare the buffers
	__local t_spWeight bufferWeightValues0 [KERNEL_CACHE_DEPTH][KERNEL_CACHE_LANES];
	__local t_spOffset bufferWeightIndex0 [KERNEL_INDEX_CACHE_DEPTH][KERNEL_INDEX_CACHE_LANES];

	__local t_spWeight bufferWeightValues1 [KERNEL_CACHE_DEPTH][KERNEL_CACHE_LANES];
	__local t_spOffset bufferWeightIndex1 [KERNEL_INDEX_CACHE_DEPTH][KERNEL_INDEX_CACHE_LANES];

	while (true) {
		//Checks whether there is any read or drain request
		bool requestFill, requestDrain;
		t_tokenFillWeightCache tokenFill = read_channel_nb_intel(
				channel_tokenFillWeightCacheControl
				, &requestFill
			);
		t_tokenDrainWeightCache tokenDrain = read_channel_nb_intel(
				channel_tokenDrainWeightCacheControl
				, &requestDrain
			);

		if (requestFill) {
			//Unpack the control information
			int ddrIndexStartOffset = tokenFill.ddrIndexStartOffset;
			int ddrWeightStartOffset = tokenFill.ddrWeightStartOffset;
			int fillfilterStart = tokenFill.filterStart;
			int fillFilterEnd = tokenFill.filterEnd;
			int fillKRange = tokenFill.filterStart - tokenFill.filterEnd + 1;
			int fillNumEncodingBlocksInFilter = tokenFill.numEncodingBlocksInFilter;
			int fillNumWeightsInFilter = tokenFill.numWeightsInFilter;
			int ddrIndexOffset = ddrIndexStartOffset + (fillNumEncodingBlocksInFilter+1)*fillfilterStart;
			int ddrWeightOffset = ddrWeightStartOffset + fillNumWeightsInFilter*fillfilterStart;
			uint1_t fillNumber = tokenFill.fillBankNumber;

			//Move the pointers and the values from the DDR to each lane.
			for (int iterFilter=fillfilterStart, iterLane=0;
				 iterFilter<fillFilterEnd; 
				 iterFilter++, iterLane++, ddrIndexOffset+=(fillNumEncodingBlocksInFilter+1),
				 ddrWeightOffset+=fillNumWeightsInFilter) {
				//Read the indices
				int iterEncodingBlocks;
				int offsetFilterBegin, offsetFilterEnd;
				offsetFilterBegin = pSpOffset[ddrIndexOffset];
				for (iterEncodingBlocks=0;
				 	iterEncodingBlocks<fillNumEncodingBlocksInFilter; 
				 	iterEncodingBlocks++){
					//Read from the DDR
					t_spOffset index = pSpOffset[ddrIndexOffset + iterEncodingBlocks]; 

					//Steer to the correct buffer
					if (fillNumber) {
						bufferWeightIndex1
						[iterEncodingBlocks & KERNEL_INDEX_CACHE_DEPTH_MASK][iterLane & KERNEL_CACHE_LANE_MASK] 
						= index;
					}
					else {
						bufferWeightIndex0
						[iterEncodingBlocks & KERNEL_INDEX_CACHE_DEPTH_MASK][iterLane & KERNEL_CACHE_LANE_MASK]
						 = index;
					}
				}
				offsetFilterEnd = pSpOffset[ddrIndexOffset + iterEncodingBlocks];

				//Read the weights themselves
				int numNZWeightsInFilter = offsetFilterEnd - offsetFilterBegin;
				for (int iterWeight=0;
				     iterWeight<numNZWeightsInFilter; 
				     iterWeight++){
					 t_spWeight weight = pSpWeight[ddrWeightOffset + iterWeight];

					//Steer to the correct buffer
					if (fillNumber){
						bufferWeightValues1
						[iterWeight & KERNEL_CACHE_DEPTH_MASK][iterLane & KERNEL_CACHE_LANE_MASK] 
						= weight;
					}
					else {
						bufferWeightValues0
						[iterWeight & KERNEL_CACHE_DEPTH_MASK][iterLane & KERNEL_CACHE_LANE_MASK] 
						= weight;
					}
				}
			}
			write_channel_nb_intel(channel_tokenFillWeightCacheFinish, true);
		}
	}
}