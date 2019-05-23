#ifndef WEIGHT_MEMORY_KERNEL_DEF
#define WEIGHT_MEMORY_KERNEL_DEF
#include "params.hpp"
#include "device_structures.hpp"
#include "channels.cl"

/*
printf enabled during SW emulation
*/
#ifdef EMULATOR
	#define EMULATOR_PRINT(format) printf format
#else
	#define EMULATOR_PRINT(format)
#endif

/*
printf enabled on HW if -HW_DEBUG flag is set
*/
#ifdef HW_DEBUG
	#define DEBUG_PRINT(format) printf format
#else
	#define DEBUG_PRINT(format)
#endif


#pragma OPENCL EXTENSION cl_intel_channels : enable


/*! \brief Kernel. DMA for the sparse weight and pointer cache. 
*/

__attribute__((max_global_work_dim(0)))

__kernel void kernelSparseWeightDMA(
        __global volatile short * restrict pWeightMem
        ,__global volatile short * restrict pIndexMem)
{
	EMULATOR_PRINT ( ("[Kernel SpW DMA]: Launched\n") );

    //! Flag enabling the while loop to run
	/*!
        If the sequencer stop issues a stop signal, this flag will be set to false.
        Then the DMA will stop until it is relaunched by the host.
        This variable should require register flip-flops.
	*/
	bool keepGoing = true;

    #pragma max_concurrency 1
	while (keepGoing){

        /*!
         * \brief stopSignal
         * \details Flag. Indicates whether stop signal has been received from the sequencer
         */
		bool stopSignal = false;

        /*!
         * \brief stop
         * \details Unused. The stop signal from the sequencer.
         */
		bool stop = read_channel_nb_intel(channel_spWDMAStop, &stopSignal);

        /*!
         * \brief timeOutCount
         * \details Keeps track whether overtime has occured when servicing one DMA request.
         */
		uint24_t timeOutCount = 0;

        /*
         * Stops the while loop upon detecting a stop signal from the sequencer.
        */
		if (stopSignal) {
			keepGoing = false;
		}

        /*!
         * \brief request
         * \details Channel read flag.
         *          Indicates whether a request to fill the SpW cache has been received by the DMA
         */
		bool request = false;

        //Obtain the token required for transferring SpW and the compression block pointers
		t_tokenFillWeightCache token = read_channel_nb_intel(channel_spWeightDMA, &request);
		if (request){
			EMULATOR_PRINT ( ("[Kernel SpW DMA]: Request received!\n") );

            // Unpack the token.

            /*!
             * \brief ddrKernelIndexStartOffset
             * \details DDR address offset (measured relative to the start of the pointer region)
             *  of the start of the SpW pointers of this kernel
             * A kernel refers to all the weights that are on the edges going to all the neurons of a layer
             */
	        unsigned int ddrKernelIndexStartOffset = token.ddrKernelIndexStartOffset;

            /*!
             * \brief ddrKernelWeightStartOffset
             * \details DDR address offset (measured relative to the start of the SpW region)
             *  of the start of the SpW offset+value of this kernel
             */
            unsigned int ddrKernelWeightStartOffset = token.ddrKernelWeightStartOffset;
	        unsigned char numFiltersToStream = token.numFiltersToStream;
	        unsigned short filterStart = token.filterStart;
            /*!
             * \brief cbStart.
             * First compression block. This is the first compression block within each filter that will be streamed
           */
            unsigned short cbStart = token.cbStart;

            /*!
             * \brief cbEnd.
             * Final compression block. This is the last compression block within each filter that will be streamed
           */
            unsigned short cbEnd = token.cbEnd;
            unsigned short numCbToStream = (unsigned short)(cbEnd- cbStart + 1);

            /*!
             * \brief numEncodingBlocksInFilter
             * Total number of encoding block within each filter
             */
	        unsigned short numEncodingBlocksInFilter = token.numEncodingBlocksInFilter;

	        unsigned int numWeightsInFilter = token.numWeightsInFilter;

            // Tracker of the number of commit tokens that the SpW cache should send to this DMA.
            // Should be synthesized as a register FF.
	        unsigned int tokenCount=0;

            // Tracker of the number of commit tokens that this DMA has actually collected from the SpW cache.
            // Should be synthesized as a register FF.
            unsigned int numTokenToCollect=0;

            // Counter of the SpW cache lane to stream to.
            // Should be synthesized as a register FF.
            unsigned char laneID = 0;

            // Pointer to the current filter being streamed.
            // Should be synthesized as  register FF.
            unsigned short iterFilter;
	        #pragma max_concurrency 1
	        for (laneID=0, iterFilter=filterStart;
	            laneID < numFiltersToStream;
				laneID++, iterFilter++){

                //Tracker of the index of the first weight in the SpW cache lane to be streamed.
                //FF
	            t_spOffset offsetHead;

                //Tracker of the index of the last weight in the SpW cache lane to be streamed.
                //FF
                t_spOffset offsetEnd;

                //FF
	            t_spOffset numWeightsToStream;

                //	First load the compress block pointes from DDR, then stream them to the SpW cache lane.
                //	Then obtain the weights in the compression blocks and stream them to the SpW cache lane.

                // Tracker of the destination depth of the SpW lane that the DMA should write to.
                // FF
                unsigned short depth = 0;
	            #pragma unroll 1
                for (unsigned int ddrAddress=ddrKernelIndexStartOffset+iterFilter*(numEncodingBlocksInFilter + 1)+cbStart;
                    //CAUTION: Need to be <= in order to account for the extra CB pointer at the end!
                    depth <= (unsigned short) numCbToStream;
	                depth++, ddrAddress++)
				{
                    // Flag. Indicate whether a commit token from the SpW cache has been received.
	                bool loopBackTokenRead = false;

                    // Read the pointer to the compression block
	                t_spOffset index = (t_spOffset) pIndexMem[ddrAddress];
					
                    // Package the packet.
					u_index_data data;
					data.offset = index;
					t_packetDMAToWeightFeeder packet;
	                packet.depth = depth;
	                packet.isIndex = 1;
	                packet.laneNumber = (unsigned short) laneID;
	                packet.packet = data;

	                //DEBUG_PRINT ( ("[Kernel SpW DMA]: Read index from address %u\n", ddrAddress) );
	                //DEBUG_PRINT ( ("[Kernel SpW DMA]: Sending index %u to lane %u\n", index, laneID) );

                    // Keep trying to send the packet until success or timeout.
	               	bool packetSent = false;
	               	while (packetSent == false && timeOutCount < TIMEOUT) {
	               		packetSent = write_channel_nb_intel(channel_packetDMAToWeightFeeder[0], packet);

	               		read_channel_nb_intel(channel_packetDMAToWeightFeederLoopBack, &loopBackTokenRead);

		              if (loopBackTokenRead){
		                    tokenCount++;
		                }

		              timeOutCount++;
	               	} // while
	                
//	                bool packetSent = false;
//	                unsigned char attempCount = 0;
//	                do { 
//	                	packetSent = write_channel_nb_intel(channel_packetDMAToWeightFeeder[0], packet);
//	                	attempCount++;
//	                }
//	                while (!packetSent && attempCount < 0xFF);
//
//	                if (attempCount == 0xFF) {
//	                	DEBUG_PRINT ( ("[KERNEL SpW DMA]: Write to channel_packetDMAToWeightFeeder[0] timeout\n") );
//	                }

//	                read_channel_nb_intel(channel_packetDMAToWeightFeederLoopBack, &loopBackTokenRead);

					//if (loopBackTokenRead){
		            //        tokenCount++;
		            //    }

                    // If the destination depth is zero, then the pointer points to the first weight in the SpW cache lane.
                    // Store it.
                    if (depth==0){
	                    offsetHead = index;
	                }

                    // If the destination depth equals to the number of CB we expect to stream to the SpW cache lane.
                    // then the pointer points the next null position, NOT the last weight!!!!,  in the SpW cache lane
                    // Store it.
                    if (depth == (unsigned short) (numCbToStream)) {//Pointer of the next cb, which marks the end of this stream run!
	                    offsetEnd = index;
	                }

				} // For. Pointers to the compression blocks.

                //CAUTION: Need to be +1 , because the extra cb contains the pointer!
                numTokenToCollect += ((unsigned int)  numCbToStream + 1);

                //CAUTION: Do not +1. offsetEnd does not point to any weight that need to be streamed.!
	            numWeightsToStream = offsetEnd - offsetHead;

                // Now we will stream to the weight/zCount lane. Reset the destination depth tracker.
	            depth=0;
	            #pragma unroll 1
	            for (unsigned int ddrAddress=ddrKernelWeightStartOffset+iterFilter*numWeightsInFilter+ (unsigned int) offsetHead;
	                 depth < (unsigned short) numWeightsToStream;
	                 depth++, ddrAddress++) {
                    // Flag. Indicate whether a commit token from the SpW cache has been received.
	                bool loopBackTokenRead;

                    // Read the weight/Zcount from the DDR.
	                t_spWeightAndOffset weight = (t_spWeightAndOffset) pWeightMem[ddrAddress];
	                
                    // Prepare the packet.
	                u_index_data data;
	                data.weightAndOffset = weight;
	                t_packetDMAToWeightFeeder packet;
	                packet.depth = depth;
	                packet.isIndex = 0;
	                packet.laneNumber = (unsigned short) laneID;
	                packet.packet = data;

	                //DEBUG_PRINT ( ("[Kernel SpW DMA]: Read weight and offset from address %u\n", ddrAddress) );

	                //DEBUG_PRINT ( ("[Kernel SpW DMA]: Sending weight & offset %u to lane %u\n", weight, laneID) );

                    // Keep sending the packet until success or timeout.
	                bool packetSent = false;
	               	while (packetSent == false && timeOutCount < 0xFFFFFF) {
	               		packetSent = write_channel_nb_intel(channel_packetDMAToWeightFeeder[0], packet);
	               	
	               		read_channel_nb_intel(channel_packetDMAToWeightFeederLoopBack, &loopBackTokenRead);

		                if (loopBackTokenRead){
		                    tokenCount++;
		                }

		                timeOutCount++;
	               	} // while

	               	//bool packetSent = false;
	                //unsigned char attempCount = 0;
	                //do { 
	                //	packetSent = write_channel_nb_intel(channel_packetDMAToWeightFeeder[0], packet);
	                //	attempCount++;
	                //}
	                //while (!packetSent && attempCount < 0xFF);

	                //if (attempCount == 0xFF) {
	                //	DEBUG_PRINT ( ("[KERNEL SpW DMA]: Write to channel_packetDMAToWeightFeeder[0] timeout\n") );
	                //}

	                //read_channel_nb_intel(channel_packetDMAToWeightFeederLoopBack, &loopBackTokenRead);

					//if (loopBackTokenRead){
		            //       tokenCount++;
		            //}
	            } // For. Weight/zCount stream
 
	            numTokenToCollect += (unsigned int) numWeightsToStream;
			} //For. Filters

            //Wait for all the commit tokens to arrive
	        while (tokenCount < numTokenToCollect && timeOutCount < 0xFFFFFF){
	            bool loopBackTokenRead;

	            read_channel_nb_intel(channel_packetDMAToWeightFeederLoopBack, &loopBackTokenRead);

	            if (loopBackTokenRead){
	                tokenCount++;
	            }

	            timeOutCount++;
            } // while

	        if (timeOutCount == TIMEOUT) {
	        	EMULATOR_PRINT ( ("[Kernel SpW DMA]: Timeout!\n") );
            }

            // Commit
	        EMULATOR_PRINT ( ("[Kernel SpW DMA]: Commiting the reqeust!\n") );

	        write_channel_intel(channel_spWeightDMACommit, 0x1);
	        
	        EMULATOR_PRINT ( ("[Kernel SpW DMA]: Request serviced!\n") );

        } // IF
    } // while
}



/*! Kernel. kernelSparseWeightCache
    \brief The multibank sparse weight and pointer cache
*/
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(KERNEL_CACHE_LANES)))
__kernel void kernelSparseWeightFeeder() 
{
    //Internal state definitions. Used to implement the drain mode.
	enum e_states {IDLE, STREAM_HEAD_SETUP, STREAM_START_SETUP, STREAM_END_SETUP, STREAM, STREAM_COMMIT_WAIT, STREAM_COMMIT_WRITE};

    // Storage variable of the state.
    // FF
    enum e_states state = IDLE;

    int laneID = get_compute_id(0);

	EMULATOR_PRINT( ("[SpW Feeder %i]: Launching\n", laneID) );

	//Declare the buffers
    //By default, the lower dimension is used for banking!
    //M10/20K
	__private t_spWeightAndOffset  __attribute__ ((numbanks(2), bankwidth(2))) bufferWeightValues [KERNEL_CACHE_DEPTH][2];
	__private t_spOffset __attribute__ ((numbanks(2), bankwidth(2))) bufferWeightIndex [KERNEL_INDEX_CACHE_DEPTH][2];

	//Register controlling the drain select
    //FF
	uint1_t drainSelectReg = 0x1;

    //Registers that keep track of the first and last compression block to be drained.
    //FFs
    unsigned short drainCbStartReg = 0;
    unsigned short drainCbEndReg = 0;

    //Variable for holding the drain token.
	t_tokenDrainWeightCache tokenDrain;

    //Registers that keep track the position of the first effectual W+NZ count,
    //the position of the first value to be streamed,
    //and the position of the last value to be streamed.
    //FFs
	t_spOffset laneHeadOffset = 0;
	t_spOffset drainIterIndexReg = 0;
	t_spOffset drainEndIndexReg = 0;

    //Need these pragmas to enable double buffering.
	#pragma ivdep array(bufferWeightValues)
	#pragma ivdep array(bufferWeightIndex)
	#pragma unroll 1
	while (true){
        //Flags that record whether the requests have arrived from various channels.
		bool isDataInput=false, isDrainSelect=false, requestDrain=false;

        //Global timeout tracker.
        //FFs
        uint24_t timeOutCount = 0x0;

		t_packetDMAToWeightFeeder inputPacket = read_channel_nb_intel(
				channel_packetDMAToWeightFeeder [laneID]
				, &isDataInput
			);
		
        //If a data packet from the SpW DMA has arrived ...
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

            //pass the packet to the next lane
			if (laneID < KERNEL_CACHE_LANES-1){

				bool writeSuccess = false;

				while (!writeSuccess && timeOutCount < TIMEOUT) {
					writeSuccess = write_channel_nb_intel
					(channel_packetDMAToWeightFeeder [laneID+1 & KERNEL_CACHE_LANE_MASK], inputPacket);

					timeOutCount++;
				}

			}
            //The last SpW cache lane is responsible for issuing the commit token to the SpW DMA
			else {
				//EMULATOR_PRINT( ("[SpW Feeder %i]: Waiting to acknowledge the write reqeust from the DMA!\n", laneID) );
                //unsigned int timeOutCount = 0X0;
				bool writeSuccess = false;

                //Keep trying until timeout.
				while (!writeSuccess && timeOutCount < TIMEOUT) {
					writeSuccess = write_channel_nb_intel
					(channel_packetDMAToWeightFeederLoopBack, 1);

					timeOutCount++;
				}

				//EMULATOR_PRINT( ("[SpW Feeder %i]: Acknowledge to write reqeust from the DMA!\n", laneID) );
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

                //Handle drain/fill requests
				if (isDrainSelect){
					EMULATOR_PRINT( ("[SpW Feeder %i]: Swap request received!\n", laneID) );

                    //Swap the bank flag.
					drainSelectReg = ~drainSelectReg;

					//EMULATOR_PRINT ( ("[SpWFeeder %i]: State IDLE. Servicing swap.\n", laneID));
					if (laneID < KERNEL_CACHE_LANES-1) {

                        //unsigned int timeOutCount = 0X0;
                        //Pass the request to the next SpW cache lane.
						bool writeSuccess = false;

						while (!writeSuccess && timeOutCount < TIMEOUT) {
							writeSuccess = write_channel_nb_intel
							(channel_spWeightFeederDrainSelect[laneID+1], 0x1);

							timeOutCount++;
						}

						EMULATOR_PRINT( ("[SpW Feeder %i]: Sending the swap request down the daisy chain, to %u!\n", laneID, laneID+1) );
					}
					else {
                        //The last SpW cache lane should commit the swap request.
						bool writeSuccess = false;

						while (!writeSuccess && timeOutCount < TIMEOUT) {
							writeSuccess = write_channel_nb_intel
							(channel_spWeightFeederDrainSelectCommit, 1);

							timeOutCount++;
						}

						EMULATOR_PRINT( ("[SpW Feeder %i]: Last SpW feeder. Commited Swap Request!\n", laneID) );
					}
				}

                // Handle drain requests
				if (requestDrain){
					EMULATOR_PRINT ( ("[SpW Feeder %i]: State IDLE. Received request to drain.\n", laneID));
					if (laneID < KERNEL_CACHE_LANES-1){

                        //unsigned int timeOutCount = 0X0;

                         //Pass
						bool writeSuccess = false;

						while (!writeSuccess && timeOutCount < TIMEOUT) {
							writeSuccess = write_channel_nb_intel
							(channel_tokenDrainWeightCacheControl[laneID+1 & KERNEL_CACHE_LANE_MASK], tokenDrain);

							timeOutCount++;
						}
					}

					timeOutCount = 0x0;

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
				//EMULATOR_PRINT( ("[SpW Feeder %i]: State is STREAM_HEAD_SETUP\n", laneID) );

				laneHeadOffset = bufferWeightIndex[0][drainSelectReg];
				
				state = STREAM_START_SETUP;
				break;
			case (STREAM_START_SETUP):
				EMULATOR_PRINT( ("[SpW Feeder %i]: State is STREAM_START_SETUP\n", laneID) );

                //CAUTION: The mask should be the one used for the pointer cache!
				drainIterIndexReg = bufferWeightIndex
                [drainCbStartReg & KERNEL_INDEX_CACHE_DEPTH_MASK][drainSelectReg] - laneHeadOffset;

				EMULATOR_PRINT( ("[SpW Feeder %i]: drainIterIndexReg is %u\n", laneID, drainIterIndexReg) );
				
				state = STREAM_END_SETUP;
				break;
			case (STREAM_END_SETUP):
				EMULATOR_PRINT( ("[SpW Feeder %i]: State is STREAM_END_SETUP\n", laneID) );

                //CAUTION: The mask should be the one used for the pointer cache!
				drainEndIndexReg = bufferWeightIndex
                [(drainCbEndReg + 1) & KERNEL_INDEX_CACHE_DEPTH_MASK][drainSelectReg] - laneHeadOffset;

				EMULATOR_PRINT( ("[SpW Feeder %i]: drainEndIndexReg is %u\n", laneID, drainEndIndexReg) );
				
				state = STREAM;
				break;
			case (STREAM):
				//EMULATOR_PRINT( ("[SpW Feeder %i]: State is STREAM\n", laneID) );
					{
						bool spwWriteSuccess = false;
#ifndef INCLUDE_COMPUTE_CORE
                        //CAUTION: The mask should be the one used for the weight/zCount cache!
						spwWriteSuccess = write_channel_nb_intel(channel_sparseWeights[laneID & KERNEL_CACHE_LANE_MASK]
								, bufferWeightValues [drainIterIndexReg & KERNEL_CACHE_DEPTH_MASK][drainSelectReg & 0x1]);
#else
                         //CAUTION: The mask should be the one used for the weight/zCount cache!
						spwWriteSuccess = write_channel_nb_intel(channel_sparseWeights[laneID & KERNEL_CACHE_LANE_MASK][0]
								, bufferWeightValues [drainIterIndexReg & KERNEL_CACHE_DEPTH_MASK][drainSelectReg & 0x1]);

#endif
						if (spwWriteSuccess){
							drainIterIndexReg++;
						}

						timeOutCount++;

                        if (drainIterIndexReg == drainEndIndexReg || timeOutCount == TIMEOUT) {
								state = STREAM_COMMIT_WAIT;
								//timeOutCount = 0x0;
						}
					}
				break;
			case (STREAM_COMMIT_WAIT):
				//EMULATOR_PRINT( ("[SpW Feeder %i]: State is STREAM_COMMIT_WAIT\n", laneID) );
				{
					if (laneID > 0 && timeOutCount < TIMEOUT){
						bool commitReadSuccess = false;

						read_channel_nb_intel(channel_drainWeightCacheInternalCommit[laneID-1 & KERNEL_CACHE_LANE_MASK], 
							&commitReadSuccess);

						timeOutCount++;

						if (commitReadSuccess) {state = STREAM_COMMIT_WRITE;}

					}
					else {
						state = STREAM_COMMIT_WRITE;
					}
				}
				break;
			case (STREAM_COMMIT_WRITE):
				{
					bool commitSuccess = false;
					//EMULATOR_PRINT( ("[SpW Feeder %i]: State is STREAM_COMMIT_WRITE\n", laneID) );
					if (laneID < KERNEL_CACHE_LANES - 1){

						commitSuccess = write_channel_nb_intel
							(channel_drainWeightCacheInternalCommit[laneID & KERNEL_CACHE_LANE_MASK], true & 0x1);

					}
					else{

						commitSuccess = write_channel_nb_intel
							(channel_drainWeightCacheCommit, true & 0x1);

					}

					if (commitSuccess) {
						if (timeOutCount == TIMEOUT) {
							EMULATOR_PRINT ( ("[SpW Feeder %i]: Timeout\n", laneID) );
						}
						state = IDLE;
					}
				}
				break;

			default:
				EMULATOR_PRINT( ("[SpW Feeder %i]: State is UNDEFINED!\n", laneID) );
				state = IDLE;
			}


		}
}
#endif
