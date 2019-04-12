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
	__global volatile short * restrict pMem; //Pointer to the start of the global memory
	int ddrKernelIndexStartOffset; //Word offset of the indices of the kernel relative to the start of the global memory
	int ddrKernelWeightStartOffset; //Word offset of the weights of the kernel relative to the start of the global memory
	int filterStart; //Index of the first filter to be streamed into the cache
	int numFilterToStream; //Number of filters to be streamed

	int cbStart; //The first encoded block to be streamed. Index 0 corresponds to the beginning of the row
	int cbEnd; //The last encoded block to be streamed. Index 0 corresponds to the beginning of the row

	int numEncodingBlocksInFilter; //Number of encoding blocks in a filter. R*S*CB
	int numWeightsInFilter; //Number of weights in a filter, if no compression is applied. R*S*C

	//uint1_t fillSetNumber; //Which bank to fill. Either 0 or 1;
} t_tokenFillWeightCache;


typedef union {
	t_spWeight weightAndOffset;
	t_spOffset offset;
} u_index_data;

/*! t_packetDMAToWeightFeeder
	Structure encapsulating the data from the Sparse Weight DMA to the Sparse Weight feeders
*/
typedef struct {
	u_index_data packet;
	short laneNumber;
	short depth;
	uint1_t isIndex;
} t_packetDMAToWeightFeeder;


/*! t_tokenDrainWeightCache
	Token used to command draining of the sparse weight cache
*/
typedef struct {
	char laneStart; //First lane to be streamed
	char laneEnd; //Last lane to be streamed
	/*
	Index of the first encoder block inside each lane's index cache line to be streamed. 
	The block at the start of the cache line has index 0
	*/
	int cbStart; 
	/*
	Index of the last encoder block plus one inside the each lane's index cache line to be streamed
	*/
	int cbEnd; //Index of the last encoder block insider each filter to be streamed 

	//uint1_t drainSetNumber; //Which bank to drain. Either 0 or 1;
} t_tokenDrainWeightCache;


channel t_tokenFillWeightCache channel_spWeightDMA __attribute__((depth(0)));

channel t_packetDMAToWeightFeeder channel_packetDMAToWeightFeeder [KERNEL_CACHE_LANES] __attribute__((depth(0)));
channel uint1_t channel_packetDMAToWeightFeederLoopBack __attribute__((depth(0)));

channel t_tokenDrainWeightCache channel_tokenDrainWeightCacheControl[KERNEL_CACHE_LANES] __attribute__((depth(0)));
channel bool channel_tokenDrainWeightCacheFinish[KERNEL_CACHE_LANES] __attribute__((depth(0)));

channel uint1_t channel_spWeightFeederDrainSelect [KERNEL_CACHE_LANES] __attribute__((depth(0)));
channel uint1_t channel_spWeightFeederDrainSelectLoopBack __attribute__((depth(0)));

channel t_spWeight channel_sparseWeights[PE_ROWS][PE_COLS] __attribute__((depth(0)));



/*! Kernel. DMA for the kernelSparseWeightCache
*/
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void kernelSparseWeightDMA()
{
	bool request;
	t_tokenFillWeightCache token = read_channel_nb_intel(channel_spWeightDMA, &request);
	if (request){
		__global volatile short * restrict pMem = token.pMem;
		int ddrKernelIndexStartOffset = token.ddrKernelIndexStartOffset;
		int ddrKernelWeightStartOffset = token.ddrKernelWeightStartOffset;
		int filterStart = token.filterStart;
		int numFilterToStream = token.numFilterToStream;
		int cbStart = token.cbStart;
		int cbEnd = token.cbEnd;
		int numEncodingBlocksInFilter = token.numEncodingBlocksInFilter;
		int numWeightsInFilter = token.numWeightsInFilter;

		for (int laneID=0; laneID < numFilterToStream; laneID++){
			/*
				First obtain the indicies, then stream it
				Then obtain the weights
			*/
			for (int iterCb=cbStart, ddrAddress=ddrKernelIndexStartOffset+filterStart*numEncodingBlocksInFilter;
				iterCb < cbEnd + 1;
				iterCb++)

		}
	} 

}


/*! Kernel. kernelSparseWeightCache
	\brief The multibank sparse weight and indices cache 
*/
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(KERNEL_CACHE_LANES)))
__kernel void kernelSparseWeightFeeder() 
{
	int laneID = get_compute_id(0);

	//Declare the buffers
	__local t_spWeight  __attribute__ ((numbanks(2), bankwidth(2))) bufferWeightValues [KERNEL_CACHE_DEPTH][2];
	__local t_spOffset __attribute__ ((numbanks(2), bankwidth(2))) bufferWeightIndex [KERNEL_INDEX_CACHE_DEPTH][2];

	//Register controlling the drain select
	uint1_t drainSelectReg;

	//Checks whether there is any read or drain request
	bool isDataInput, requestDrain, isDrainSelect;

	#pragma ivdep array(bufferWeightValues)
	#pragma ivdep array(bufferWeightIndex)
	while (true){

		t_packetDMAToWeightFeeder inputPacket = read_channel_nb_intel(
				channel_packetDMAToWeightFeeder [laneID]
				, &isDataInput
			);

		t_tokenDrainWeightCache tokenDrain = read_channel_nb_intel(
				channel_tokenDrainWeightCacheControl[laneID]
				, &requestDrain
			);

		uint1_t requestDrainSelect = read_channel_nb_intel(
				channel_spWeightFeederDrainSelect[laneID]
				, &isDrainSelect
			);

		//Swap the drain/fill side if requested
		if (isDrainSelect){
			drainSelectReg = requestDrainSelect;
			if (laneID < KERNEL_CACHE_LANES-1) {
				write_channel_intel(channel_spWeightFeederDrainSelect[laneID+1], requestDrainSelect);
			}
			else {
				write_channel_intel(channel_spWeightFeederDrainSelectLoopBack, 1);
			}
		}

		if (isDataInput) {

			//Unpack the packet
			u_index_data packet = inputPacket.packet;
			short dstLaneNumber = inputPacket.laneNumber;
			short dstDepth = inputPacket.depth;
			uint1_t isIndex = inputPacket.isIndex;

			if (laneID == dstLaneNumber){
				switch(isIndex){
					case 0:
						bufferWeightIndex[dstDepth][(~drainSelectReg) & 0x1] = packet.offset;
						break;
					default:
						bufferWeightValues[dstDepth][(~drainSelectReg) & 0x1] = packet.weightAndOffset;
				}
			}

			//mem_fence(CLK_CHANNEL_MEM_FENCE);

			//pass the token to the next lane
			if (laneID < KERNEL_CACHE_LANES-1){
				write_channel_intel(channel_packetDMAToWeightFeeder [laneID+1 & KERNEL_CACHE_LANE_MASK], inputPacket);
			}
			else {
				write_channel_intel(channel_packetDMAToWeightFeederLoopBack, 1);
			}
		}

		if (requestDrain){
			//pass the token to the next lane
			if (laneID < KERNEL_CACHE_LANES-1){
				write_channel_intel(channel_tokenDrainWeightCacheControl[laneID+1 & KERNEL_CACHE_LANE_MASK], tokenDrain);
			}
			//Unpack the control information required for draining
			char drainLaneStart = tokenDrain.laneStart;
			char drainLaneEnd = tokenDrain.laneEnd;
			char drainCbStart = tokenDrain.cbStart;
			char drainCbEnd = tokenDrain.cbEnd;

			if (laneID>=drainLaneStart && laneID<drainLaneEnd){
				t_spOffset laneHeadOffset = bufferWeightIndex[0][drainSelectReg];

				t_spOffset drainStartIndex = bufferWeightIndex
				[drainCbStart & KERNEL_INDEX_CACHE_DEPTH_MASK][drainSelectReg & 0x1] - laneHeadOffset;

				t_spOffset drainEndIndex = bufferWeightIndex
				[(drainCbEnd + 1) & KERNEL_INDEX_CACHE_DEPTH_MASK][drainSelectReg & 0x1] - laneHeadOffset;

				#pragma ivdep array(bufferWeightValues)
				for (int iterIndex=drainStartIndex;
						iterIndex<drainEndIndex;
						iterIndex++
					) {
					t_spWeight weight = bufferWeightValues
					[iterIndex & KERNEL_INDEX_CACHE_DEPTH_MASK][drainSelectReg & 0x1];
					write_channel_intel(channel_sparseWeights[laneID & KERNEL_CACHE_LANE_MASK][0]
							, weight);

				}
			}
			//mem_fence(CLK_CHANNEL_MEM_FENCE);
			if (laneID > 0){
				read_channel_intel(channel_tokenDrainWeightCacheFinish[laneID-1 & KERNEL_CACHE_LANE_MASK]);
			}
			write_channel_intel(channel_tokenDrainWeightCacheFinish[laneID & KERNEL_CACHE_LANE_MASK], true & 0x1);
		}
	}
}