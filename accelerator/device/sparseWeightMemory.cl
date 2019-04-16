#ifndef WEIGHT_MEMORY_KERNEL_DEF
#define WEIGHT_MEMORY_KERNEL_DEF
#include "params.hpp"
#include "device_structures.hpp"
#include "channels.cl"




/*! Kernel. DMA for the kernelSparseWeightCache
*/

__attribute__((max_global_work_dim(0)))

__kernel void kernelSparseWeightDMA(
        __global volatile short * restrict pMem)
{
	bool request;
	t_tokenFillWeightCache token = read_channel_nb_intel(channel_spWeightDMA, &request);
	if (request){
        unsigned int ddrKernelIndexStartOffset = token.ddrKernelIndexStartOffset;
        unsigned int ddrKernelWeightStartOffset = token.ddrKernelWeightStartOffset;
        unsigned char numFiltersToStream = token.numFiltersToStream;
        unsigned short filterStart = token.filterStart;
        unsigned short cbStart = token.cbStart;
        unsigned short cbEnd = token.cbEnd + 1;
        unsigned short numCbToStream = (unsigned short)(cbEnd- cbStart + 1);
        unsigned short numEncodingBlocksInFilter = token.numEncodingBlocksInFilter;
        unsigned int numWeightsInFilter = token.numWeightsInFilter;

        unsigned int tokenCount=0;
        unsigned int numTokenToCollect=0;
        unsigned char laneID = 0;
        unsigned short iterFilter;
        for (laneID=0, iterFilter=filterStart;
            laneID < numFiltersToStream;
			laneID++, iterFilter++){

            t_spOffset offsetHead;
            t_spOffset offsetEnd;
            t_spOffset numWeightsToStream;

            unsigned short depth;

			//	First obtain the indicies, then stream it
			//	Then obtain the weights
            depth=0;
            for (unsigned int ddrAddress=ddrKernelIndexStartOffset+iterFilter*numEncodingBlocksInFilter+cbStart;
                depth < (unsigned short) numCbToStream;
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
    enum e_states state = IDLE;
	int laneID = get_compute_id(0);

	//Declare the buffers
	__local t_spWeight  __attribute__ ((numbanks(2), bankwidth(2))) bufferWeightValues [KERNEL_CACHE_DEPTH][2];
	__local t_spOffset __attribute__ ((numbanks(2), bankwidth(2))) bufferWeightIndex [KERNEL_INDEX_CACHE_DEPTH][2];

	//Register controlling the drain select
	uint1_t drainSelectReg = 0x1;

	//Internal registers
	//Unpack the control information required for draining
    unsigned short drainCbStartReg;
    unsigned short drainCbEndReg;

	t_tokenDrainWeightCache tokenDrain;
	uint1_t requestDrainSelect;

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
                        drainCbStartReg = (unsigned short) tokenDrain.cbStart;
                        drainCbEndReg = (unsigned short) tokenDrain.cbEnd;
						state = STREAM_HEAD_SETUP; //State update
					}
                    else {
                        state = STREAM_COMMIT_WAIT;
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
#endif
