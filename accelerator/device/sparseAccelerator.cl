#include "ihc_apint.h"
#include "params.hpp"

//Must include the following line in order to use channel
#pragma OPENCL EXTENSION cl_intel_channels : enable

typedef short t_spWeight;
typedef ushort t_spOffset;

/*! t_tokenFillWeightCache
	Token used to command filling of the sparse weight cache
*/
typedef struct {
	__global volatile short * restrict pMem; //Pointer to the start of the global memory
    unsigned int ddrKernelIndexStartOffset; //Word offset of the indices of the kernel relative to the start of the global memory
    unsigned int ddrKernelWeightStartOffset; //Word offset of the weights of the kernel relative to the start of the global memory
    unsigned int filterStart; //Index of the first filter to be streamed into the cache
    short int numFiltersToStream;
    unsigned int cbStart; //The first encoded block to be streamed. Index 0 corresponds to the beginning of the row
    unsigned int cbEnd; //The last encoded block to be streamed. Index 0 corresponds to the beginning of the row

    unsigned int numEncodingBlocksInFilter; //Number of encoding blocks in a filter. R*S*CB
    unsigned int numWeightsInFilter; //Number of weights in a filter, if no compression is applied. R*S*C

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
        unsigned int ddrKernelIndexStartOffset = token.ddrKernelIndexStartOffset;
        unsigned int ddrKernelWeightStartOffset = token.ddrKernelWeightStartOffset;
        short numFiltersToStream = token.numFiltersToStream;
        unsigned int filterStart = token.filterStart;
        unsigned int cbStart = token.cbStart;
        unsigned int cbEnd = token.cbEnd;
        unsigned short numCbToStream = (unsigned short)(cbEnd- cbStart + 1);
        unsigned int numEncodingBlocksInFilter = token.numEncodingBlocksInFilter;
        unsigned int numWeightsInFilter = token.numWeightsInFilter;

        unsigned int tokenCount=0;
        unsigned int numTokenToCollect=0;
        for (unsigned int laneID=0, iterFilter=filterStart;
            laneID < numFiltersToStream;
			laneID++, iterFilter++){

            t_spOffset offsetHead;
            t_spOffset offsetEnd;
            t_spOffset numWeightsToStream;

            short depth;

			//	First obtain the indicies, then stream it
			//	Then obtain the weights
            depth=0;
            for (unsigned int ddrAddress=ddrKernelIndexStartOffset+iterFilter*numEncodingBlocksInFilter+cbStart;
                depth < (short) numCbToStream;
                depth++, ddrAddress++)
			{
                bool loopBackTokenRead;
                t_spOffset index = (t_spOffset) pMem[ddrAddress];
				u_index_data data;
				data.offset = index;
				t_packetDMAToWeightFeeder packet;
                packet.depth = depth;
                packet.isIndex = 1;
                packet.laneNumber = (short) laneID;
                packet.packet = data;
                write_channel_intel(channel_packetDMAToWeightFeeder[0], packet);
                if (depth==0){
                    offsetHead = index;
                }
                if (depth == (short) (numCbToStream - 1)) {
                    offsetEnd = index;
                }
                read_channel_nb_intel(channel_packetDMAToWeightFeederLoopBack, &loopBackTokenRead);
                if (loopBackTokenRead){
                    tokenCount++;
                }
			}
            numTokenToCollect += (unsigned int) numCbToStream;
            numWeightsToStream = offsetEnd - offsetHead;

            depth=0;
            for (unsigned int ddrAddress=ddrKernelWeightStartOffset+iterFilter*numWeightsInFilter+ (unsigned int) offsetHead;
                 depth < (short) numWeightsToStream;
                 depth++, ddrAddress++) {
                bool loopBackTokenRead;
                t_spWeight weight = (t_spWeight) pMem[ddrAddress];
                u_index_data data;
                data.weightAndOffset = weight;
                t_packetDMAToWeightFeeder packet;
                packet.depth = depth;
                packet.isIndex = 0;
                packet.laneNumber = (short) laneID;
                packet.packet = data;
                write_channel_intel(channel_packetDMAToWeightFeeder[0], packet);
                read_channel_nb_intel(channel_packetDMAToWeightFeederLoopBack, &loopBackTokenRead);
                if (loopBackTokenRead){
                    tokenCount++;
                }
            }

            numTokenToCollect += (unsigned int) numWeightsToStream;
		}
        //Wait for all the tokens to arrive
        while (tokenCount < numTokenToCollect){
            bool loopBackTokenRead;
            read_channel_nb_intel(channel_packetDMAToWeightFeederLoopBack, &loopBackTokenRead);
            if (loopBackTokenRead){
                tokenCount++;
            }
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
	enum e_states {IDLE, STREAM_HEAD_SETUP, STREAM_START_SETUP, STREAM_END_SETUP, STREAM, STREAM_COMMIT_WAIT, STREAM_COMMIT_WRITE};
	enum e_states state = IDLE, next_state;
	int laneID = get_compute_id(0);

	//Declare the buffers
	__local t_spWeight  __attribute__ ((numbanks(2), bankwidth(2))) bufferWeightValues [KERNEL_CACHE_DEPTH][2];
	__local t_spOffset __attribute__ ((numbanks(2), bankwidth(2))) bufferWeightIndex [KERNEL_INDEX_CACHE_DEPTH][2];

	//Register controlling the drain select
	uint1_t drainSelectReg = 0x1;

	//Internal registers
	//Unpack the control information required for draining
	short drainCbStartReg;
	short drainCbEndReg;
	int drainWeightIndexStartReg;
	int drainWeightIndexEndReg;

	t_tokenDrainWeightCache tokenDrain;
	uint1_t requestDrainSelect;
	t_packetDMAToWeightFeeder inputPacket;

	t_spOffset laneHeadOffset;

	t_spOffset drainIterIndexReg;

	t_spOffset drainEndIndexReg;

	#pragma ivdep array(bufferWeightValues)
	#pragma ivdep array(bufferWeightIndex)
	while (true){
		bool isDataInput, isDrainSelect, requestDrain;

		t_packetDMAToWeightFeeder inputPacket = read_channel_nb_intel(
				channel_packetDMAToWeightFeeder [laneID]
				, &isDataInput
			);
		//mem_fence(CLK_CHANNEL_MEM_FENCE);
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

			//pass the token to the next lane
			if (laneID < KERNEL_CACHE_LANES-1){
				write_channel_intel(channel_packetDMAToWeightFeeder [laneID+1 & KERNEL_CACHE_LANE_MASK], inputPacket);
			}
			else {
				write_channel_intel(channel_packetDMAToWeightFeederLoopBack, 1);
			}
		}

		switch (state) {
			case (IDLE):
				tokenDrain = read_channel_nb_intel(
					channel_tokenDrainWeightCacheControl[laneID]
					, &requestDrain
				);

				requestDrainSelect = read_channel_nb_intel(
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

				if (requestDrain){
					if (laneID < KERNEL_CACHE_LANES-1){
						write_channel_intel(channel_tokenDrainWeightCacheControl[laneID+1 & KERNEL_CACHE_LANE_MASK], tokenDrain);
					}

					if (laneID>=tokenDrain.laneStart && laneID<tokenDrain.laneEnd){
						drainCbStartReg = tokenDrain.cbStart;
						drainCbEndReg = tokenDrain.cbEnd;
						state = STREAM_HEAD_SETUP; //State update
					}
				}
				break;
			case (STREAM_HEAD_SETUP):
				laneHeadOffset = bufferWeightIndex[0][drainSelectReg];
				state = STREAM_START_SETUP;
				break;
			case (STREAM_START_SETUP):
				drainIterIndexReg = bufferWeightIndex
				[drainCbStartReg & KERNEL_INDEX_CACHE_DEPTH_MASK][drainSelectReg & 0x1] - laneHeadOffset;
				state = STREAM_END_SETUP;
				break;
			case (STREAM_END_SETUP):
				drainEndIndexReg = bufferWeightIndex
				[(drainCbEndReg + 1) & KERNEL_INDEX_CACHE_DEPTH_MASK][drainSelectReg & 0x1] - laneHeadOffset;
				state = STREAM;
				break;
			case (STREAM):

				write_channel_intel(channel_sparseWeights[laneID & KERNEL_CACHE_LANE_MASK][0]
						, bufferWeightValues [drainIterIndexReg & KERNEL_INDEX_CACHE_DEPTH_MASK][drainSelectReg & 0x1]);

				if (drainIterIndexReg+1 == drainEndIndexReg) {
					state = STREAM_COMMIT_WAIT;
				}
				drainIterIndexReg++;
				break;
			case (STREAM_COMMIT_WAIT):
				if (laneID > 0){
					read_channel_intel(channel_tokenDrainWeightCacheFinish[laneID-1 & KERNEL_CACHE_LANE_MASK]);
				}
				state = STREAM_COMMIT_WRITE;
				break;
			case (STREAM_COMMIT_WRITE):
				write_channel_intel(channel_tokenDrainWeightCacheFinish[laneID & KERNEL_CACHE_LANE_MASK], true & 0x1);
				state = IDLE;
				break;
			}


		}
}
