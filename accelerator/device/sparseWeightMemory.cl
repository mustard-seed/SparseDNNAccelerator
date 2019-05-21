#ifndef WEIGHT_MEMORY_KERNEL_DEF
#define WEIGHT_MEMORY_KERNEL_DEF
#include "params.hpp"
#include "device_structures.hpp"
#include "channels.cl"

#ifdef EMULATOR
	#define EMULATOR_PRINT(format) printf format
#else
	#define EMULATOR_PRINT(format)
#endif


#pragma OPENCL EXTENSION cl_intel_channels : enable

//channel t_spWeightAndOffset channel_sparseWeights[PE_ROWS] __attribute__((depth(0)));
/*! Kernel. DMA for the kernelSparseWeightCache
*/

__attribute__((max_global_work_dim(0)))

__kernel void kernelSparseWeightDMA(
        __global volatile short * restrict pWeightMem
        ,__global volatile short * restrict pIndexMem)
{
	EMULATOR_PRINT ( ("[Kernel SpW DMA]: Launched\n") );

	bool keepGoing = true;

	#pragma unroll 1
	while (keepGoing){
		bool stopSignal = false;
		bool stop = read_channel_nb_intel(channel_spWDMAStop, &stopSignal);

		if (stopSignal) {
			keepGoing = false;
		}

		bool request = false;
		t_tokenFillWeightCache token = read_channel_nb_intel(channel_spWeightDMA, &request);
		//mem_fence(CLK_CHANNEL_MEM_FENCE);
		if (request){
			EMULATOR_PRINT ( ("[Kernel SpW DMA]: Request received!\n") );
	        unsigned int ddrKernelIndexStartOffset = token.ddrKernelIndexStartOffset;
	        unsigned int ddrKernelWeightStartOffset = token.ddrKernelWeightStartOffset;
	        unsigned char numFiltersToStream = token.numFiltersToStream;
	        unsigned short filterStart = token.filterStart;
	        unsigned short cbStart = token.cbStart;
	        unsigned short cbEnd = token.cbEnd;
            unsigned short numCbToStream = (unsigned short)(cbEnd- cbStart + 1);
	        unsigned short numEncodingBlocksInFilter = token.numEncodingBlocksInFilter;
	        unsigned int numWeightsInFilter = token.numWeightsInFilter;

	        unsigned int tokenCount=0;
	        unsigned int numTokenToCollect=0;
	        unsigned char laneID = 0;
	        unsigned short iterFilter;
	        #pragma unroll 1
	        for (laneID=0, iterFilter=filterStart;
	            laneID < numFiltersToStream;
				laneID++, iterFilter++){

	            t_spOffset offsetHead;
	            t_spOffset offsetEnd;
	            t_spOffset numWeightsToStream;

	            unsigned short depth;

				//	First obtain the indicies, then stream it
				//	Then obtain the weights
				//  CONFUSION; numEncodingBlocksInFilter or numEncodingBlocksInFilterSrip?
	            depth=0;
	            #pragma unroll 1
                for (unsigned int ddrAddress=ddrKernelIndexStartOffset+iterFilter*(numEncodingBlocksInFilter + 1)+cbStart;
                    depth <= (unsigned short) numCbToStream; //CAUTION: Need to be <=, because the extra cb contains the pointer!
	                depth++, ddrAddress++)
				{
	                bool loopBackTokenRead = false;

	                t_spOffset index = (t_spOffset) pIndexMem[ddrAddress];
					
					u_index_data data;
					data.offset = index;
					t_packetDMAToWeightFeeder packet;
	                packet.depth = depth;
	                packet.isIndex = 1;
	                packet.laneNumber = (unsigned short) laneID;
	                packet.packet = data;

	               	bool packetSent = false;
	               	while (packetSent == false) {
	               		packetSent = write_channel_nb_intel(channel_packetDMAToWeightFeeder[0], packet);

	               		read_channel_nb_intel(channel_packetDMAToWeightFeederLoopBack, &loopBackTokenRead);

		                if (loopBackTokenRead){
		                    tokenCount++;
		                }
	               	}
	                //write_channel_intel(channel_packetDMAToWeightFeeder[0], packet);

	                if (depth==0){
	                    offsetHead = index;
	                }
                    if (depth == (unsigned short) (numCbToStream)) {//Pointer of the next cb, which marks the end of this stream run!
	                    offsetEnd = index;
	                }

				}
	            numTokenToCollect += ((unsigned int)  numCbToStream + 1); //CAUTION: Need to be +1 , because the extra cb contains the pointer!
	            numWeightsToStream = offsetEnd - offsetHead;

	            depth=0;
	            #pragma unroll 1
	            for (unsigned int ddrAddress=ddrKernelWeightStartOffset+iterFilter*numWeightsInFilter+ (unsigned int) offsetHead;
	                 depth < (unsigned short) numWeightsToStream;
	                 depth++, ddrAddress++) {
	                bool loopBackTokenRead;

	                t_spWeightAndOffset weight = (t_spWeightAndOffset) pWeightMem[ddrAddress];
	                
	                u_index_data data;
	                data.weightAndOffset = weight;
	                t_packetDMAToWeightFeeder packet;
	                packet.depth = depth;
	                packet.isIndex = 0;
	                packet.laneNumber = (unsigned short) laneID;
	                packet.packet = data;

	                bool packetSent = false;
	               	while (packetSent == false) {
	               		packetSent = write_channel_nb_intel(channel_packetDMAToWeightFeeder[0], packet);
	               		
	               		read_channel_nb_intel(channel_packetDMAToWeightFeederLoopBack, &loopBackTokenRead);

		                if (loopBackTokenRead){
		                    tokenCount++;
		                }
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
	        EMULATOR_PRINT ( ("[Kernel SpW DMA]: Commiting the reqeust!\n") );

	        write_channel_intel(channel_spWeightDMACommit, 0x1);
	        
	        EMULATOR_PRINT ( ("[Kernel SpW DMA]: Request serviced!\n") );
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
	EMULATOR_PRINT( ("[SpW Feeder %i]: Launching\n", laneID) );
	//Declare the buffers
	__private t_spWeightAndOffset  __attribute__ ((numbanks(2), bankwidth(2))) bufferWeightValues [KERNEL_CACHE_DEPTH][2];
	__private t_spOffset __attribute__ ((numbanks(2), bankwidth(2))) bufferWeightIndex [KERNEL_INDEX_CACHE_DEPTH][2];

	//Register controlling the drain select
	uint1_t drainSelectReg = 0x1;

	//Internal registers
	//Unpack the control information required for draining
    unsigned short drainCbStartReg = 0;
    unsigned short drainCbEndReg = 0;

	t_tokenDrainWeightCache tokenDrain;
	//uint1_t requestDrainSelect;

	t_spOffset laneHeadOffset = 0;
	t_spOffset drainIterIndexReg = 0;
	t_spOffset drainEndIndexReg = 0;

	#pragma ivdep array(bufferWeightValues)
	#pragma ivdep array(bufferWeightIndex)
	#pragma unroll 1
	while (true){
		bool isDataInput=0, isDrainSelect=0, requestDrain=0;

		t_packetDMAToWeightFeeder inputPacket = read_channel_nb_intel(
				channel_packetDMAToWeightFeeder [laneID]
				, &isDataInput
			);

		//mem_fence(CLK_CHANNEL_MEM_FENCE);
		
		if (isDataInput) {
			//Unpack the packet
			u_index_data packet = inputPacket.packet;
			unsigned short dstLaneNumber = inputPacket.laneNumber;
			unsigned short dstDepth = inputPacket.depth;
			uint1_t isIndex = inputPacket.isIndex;

			if (laneID == (int) dstLaneNumber){
				switch(isIndex){
					case 1:

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
				EMULATOR_PRINT( ("[SpW Feeder %i]: Waiting to acknowledge the write reqeust from the DMA!\n", laneID) );

				write_channel_intel(channel_packetDMAToWeightFeederLoopBack, 1);

				EMULATOR_PRINT( ("[SpW Feeder %i]: Acknowledge to write reqeust from the DMA!\n", laneID) );
			}
		}

		switch (state) {
			case (IDLE):

				tokenDrain = read_channel_nb_intel(
					channel_tokenDrainWeightCacheControl[laneID]
					, &requestDrain
				);

				//EMULATOR_PRINT ( ("[SpWFeeder %i]: State IDLE\n", laneID));

				read_channel_nb_intel(
					channel_spWeightFeederDrainSelect[laneID]
					, &isDrainSelect
				);

						//Swap the drain/fill side if requested
				if (isDrainSelect){
					EMULATOR_PRINT( ("[SpW Feeder %i]: Swap request received!\n", laneID) );
					drainSelectReg = ~drainSelectReg;
					//EMULATOR_PRINT ( ("[SpWFeeder %i]: State IDLE. Servicing swap.\n", laneID));
					if (laneID < KERNEL_CACHE_LANES-1) {

						write_channel_intel(channel_spWeightFeederDrainSelect[laneID+1], 0x1);

						EMULATOR_PRINT( ("[SpW Feeder %i]: Sending the swap request down the daisy chain, to %u!\n", laneID, laneID+1) );
					}
					else {

						write_channel_intel(channel_spWeightFeederDrainSelectCommit, 1);

						EMULATOR_PRINT( ("[SpW Feeder %i]: Last SpW feeder. Commited Swap Request!\n", laneID) );
					}
				}

				if (requestDrain){
					EMULATOR_PRINT ( ("[SpW Feeder %i]: State IDLE. Received request to drain.\n", laneID));
					if (laneID < KERNEL_CACHE_LANES-1){

						write_channel_intel(channel_tokenDrainWeightCacheControl[laneID+1 & KERNEL_CACHE_LANE_MASK], tokenDrain);

					}

                    if ( ((unsigned char) laneID) >=tokenDrain.laneStart && ((unsigned char) laneID) <tokenDrain.laneEnd){
                        drainCbStartReg = (unsigned short) tokenDrain.cbStart;
                        drainCbEndReg = (unsigned short) tokenDrain.cbEnd;
						state = STREAM_HEAD_SETUP; //State update
					}
                    else {
                        state = STREAM_COMMIT_WAIT;
                    }
				}
				//EMULATOR_PRINT ( ("[SpWFeeder %i]: State will remain IDLE\n", laneID));
				break;
			case (STREAM_HEAD_SETUP):
				EMULATOR_PRINT( ("[SpW Feeder %i]: State is STREAM_HEAD_SETUP\n", laneID) );

				laneHeadOffset = bufferWeightIndex[0][drainSelectReg];
				
				state = STREAM_START_SETUP;
				break;
			case (STREAM_START_SETUP):
				EMULATOR_PRINT( ("[SpW Feeder %i]: State is STREAM_START_SETUP\n", laneID) );

				drainIterIndexReg = bufferWeightIndex
				[drainCbStartReg & KERNEL_INDEX_CACHE_DEPTH_MASK][drainSelectReg & 0x1] - laneHeadOffset;
				
				state = STREAM_END_SETUP;
				break;
			case (STREAM_END_SETUP):
				EMULATOR_PRINT( ("[SpW Feeder %i]: State is STREAM_END_SETUP\n", laneID) );

				drainEndIndexReg = bufferWeightIndex
				[(drainCbEndReg + 1) & KERNEL_INDEX_CACHE_DEPTH_MASK][drainSelectReg & 0x1] - laneHeadOffset;
				
				state = STREAM;
				break;
			case (STREAM):
				EMULATOR_PRINT( ("[SpW Feeder %i]: State is STREAM\n", laneID) );
#ifndef INCLUDE_COMPUTE_CORE

				write_channel_intel(channel_sparseWeights[laneID & KERNEL_CACHE_LANE_MASK]
						, bufferWeightValues [drainIterIndexReg & KERNEL_INDEX_CACHE_DEPTH_MASK][drainSelectReg & 0x1]);

#else

				write_channel_intel(channel_sparseWeights[laneID & KERNEL_CACHE_LANE_MASK][0]
						, bufferWeightValues [drainIterIndexReg & KERNEL_INDEX_CACHE_DEPTH_MASK][drainSelectReg & 0x1]);

#endif
				if (drainIterIndexReg+1 == drainEndIndexReg) {
					state = STREAM_COMMIT_WAIT;
				}
				drainIterIndexReg++;
				break;
			case (STREAM_COMMIT_WAIT):
				EMULATOR_PRINT( ("[SpW Feeder %i]: State is STREAM_COMMIT_WAIT\n", laneID) );
				if (laneID > 0){

					read_channel_intel(channel_drainWeightCacheInternalCommit[laneID-1 & KERNEL_CACHE_LANE_MASK]);

				}
				state = STREAM_COMMIT_WRITE;
				break;
			case (STREAM_COMMIT_WRITE):
				EMULATOR_PRINT( ("[SpW Feeder %i]: State is STREAM_COMMIT_WRITE\n", laneID) );
				if (laneID < KERNEL_CACHE_LANES - 1){

					write_channel_intel(channel_drainWeightCacheInternalCommit[laneID & KERNEL_CACHE_LANE_MASK], true & 0x1);

				}
				else{

					write_channel_intel(channel_drainWeightCacheCommit, true & 0x1);

				}
				state = IDLE;
				break;

			default:
				EMULATOR_PRINT( ("[SpW Feeder %i]: State is UNDEFINED!\n", laneID) );
				state = IDLE;
			}


		}
}
#endif
