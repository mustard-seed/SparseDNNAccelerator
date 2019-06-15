#include "params.hpp"
#include "device_structures.hpp"
#include "channels.cl"
#include "ihc_apint.h"
#include "device_utils.hpp"

//Must include the following line in order to use channel
#pragma OPENCL EXTENSION cl_intel_channels : enable

#if KERNEL_CACHE_LANES > 8
#error "Too many cache lanes for the test case! Maximal is 8"
#endif
#if MAX_INSTRUCTION_TYPE_BITS == 16
#elif MAX_INSTRUCTION_TYPE_BITS == 8
#else
#error "Invalid number of inflight instructions. Only supports 8 or 16"    
#endif 

void checkCommits (uint4_t* pInstrInFlightCount) {
  bool dmaCommitRead=false, drainCacheCommitRead=false, CacheSelectCommitRead=false, collectCommitRead=false;

  read_channel_nb_intel(channel_spWeightDMACommit, &dmaCommitRead);
  if (dmaCommitRead) {
    EMULATOR_PRINT ( ("[Kernel Sequencer]: COMMIT. Header is %u.\n", OPCODE_FILL_WEIGHT_BUFFER));
    pInstrInFlightCount[OPCODE_FILL_WEIGHT_BUFFER & 0x1F]--; 
  }

  read_channel_nb_intel(channel_drainWeightCacheCommit, &drainCacheCommitRead);
  if (drainCacheCommitRead) {
    EMULATOR_PRINT ( ("[Kernel Sequencer]: COMMIT. Header is %u.\n", OPCODE_DRAIN_WEIGHT_BUFFER));
    pInstrInFlightCount[OPCODE_DRAIN_WEIGHT_BUFFER & 0x1F]--; 
  }

  read_channel_nb_intel(channel_spWeightFeederDrainSelectCommit, &CacheSelectCommitRead);
  if (CacheSelectCommitRead) {
    EMULATOR_PRINT ( ("[Kernel Sequencer]: COMMIT. Header is %u.\n", OPCODE_SWAP_WEIGHT_BUFFER));
    pInstrInFlightCount[OPCODE_SWAP_WEIGHT_BUFFER & 0x1F]--; 
  }

  read_channel_nb_intel(channel_weightCollectControlCommit, &collectCommitRead);
  if (collectCommitRead) {
    EMULATOR_PRINT ( ("[Kernel Sequencer]: COMMIT. Header is %u.\n", OPCODE_COLLECT_WEIGHT));
    //DEBUG_PRINT ( ("[Kernel Sequencer]: COMMIT. Header is %u.\n", OPCODE_COLLECT_WEIGHT) );
    pInstrInFlightCount[OPCODE_COLLECT_WEIGHT & 0x1F]--; 
  }
}

__attribute__((task))
__attribute__((max_global_work_dim(0)))
__kernel void kernelSequencer(
        __global volatile t_instruction* restrict pInstruction,
        unsigned short numInstructions
        )
{

      EMULATOR_PRINT ( ("[Kernel Sequencer]: Launched\n") );
      uint4_t instructionInFlightCount[MAX_INSTRUCTION_TYPE_BITS];

      //Initialize the inflight instruction count of each instruction type to
      //0
      #pragma unroll 1
      for (uint5_t iter=0;
          iter < MAX_INSTRUCTION_TYPE_BITS;
          iter++) {
        instructionInFlightCount[iter] = 0x0;
      }

      //Read the instructions sequentially
      #pragma max_concurrency 1
      for (unsigned short iterInstruction=0;
           iterInstruction<numInstructions;
           iterInstruction++) {

          t_instruction instruction = pInstruction[iterInstruction];

          //Scan the dependency and the header
          //Then wait for the dependency to be met before issuing the instruction
          unsigned short dependency =
          ((unsigned short) instruction.dependencyList[1]) << 8 |   
          ((unsigned short) instruction.dependencyList[0]);

          unsigned char header = instruction.header;
          unsigned char instructionSizeBytes = instruction.instructionSizeBytes;

          uint1_t wait = 0x1;

           EMULATOR_PRINT ( ("[Kernel Sequencer]: Wait to send instruction %u. Header is %u.\n", iterInstruction, header));
          #pragma unroll 1
          while (wait){
            uint1_t wait_next = 0x0;

            checkCommits(instructionInFlightCount);

            //Check the dependency
            #pragma unroll
            for (uint5_t iter=0;
                iter < MAX_INSTRUCTION_TYPE_BITS;
                iter++){
              wait_next |= 
                ((dependency & (0x1 << iter)) >> iter) & 
                (instructionInFlightCount[iter & 0x1F] > 0);   
            }

            //Stalls the instruciton issue if 
            // 1) the dependency isn't met, or
            // 2) The number of instruciton of the same type in flight is too high
            wait = wait_next || 
            (instructionInFlightCount[header] == MAX_INSTRUCTION_IN_FLIGHT_COUNT_PER_TYPE);
          }


          //Issue the instructions
          instructionInFlightCount[header]++;

          //Need +1 cycle to send the header
          for (unsigned char iter=instructionSizeBytes+1;
            iter > 0;
            iter--
            ){

            bool instructionSent = false;

            unsigned short payload = (iter == instructionSizeBytes+1) ? 
                ((unsigned short) header) << 8 : ((unsigned short) header << 8 )| (instruction.words[(iter-1) & 0x01F]);

            do {
              //Send the instructions
              //The header occupies the upper byte,
              //The receiver occupies the lower byte
              instructionSent = write_channel_nb_intel(channel_instructions[0], 
                      payload) ;

              checkCommits(instructionInFlightCount);

             }
            while (!instructionSent);

            DEBUG_PRINT( ("[Kernel Sequencer]: Sent instruction %u byte %u. Header is %u. Packet is %u.\n",
                iterInstruction, iter, header, (instruction.words[(iter-1) & 0x01F]) );
            ); //do-while
          } // for
          EMULATOR_PRINT ( ("[Kernel Sequencer]: Sent instruction %u. Header is %u No. Bytes is %u.\n"
            , iterInstruction, header, instructionSizeBytes));
      }

      //Wait for outstanding commands to finish
      uint1_t wait = 0x1;
      #pragma max_concurrency 1
      while (wait){
        uint1_t wait_next = 0;
        checkCommits(instructionInFlightCount);

        //Check the dependency
        #pragma unroll
        for (uint5_t iter=0;
            iter < MAX_INSTRUCTION_TYPE_BITS;
            iter++){
          wait_next |= 
            (instructionInFlightCount[iter] > 0);   
        }
        wait = wait_next;
      }
       
      //Send the stop signals
      write_channel_intel(channel_weightCollectorStop[0], 0x1);

      write_channel_intel(channel_spWDMAStop, 0x1);

}

/*! Kernel. weightCollector
    \brief Collects the sparse weights and store them to cache
*/
 void weightCollector(
      __global volatile short * pMem,
      unsigned char laneID
    )
{
    EMULATOR_PRINT (("[Weight Collector %u]: Launched\n", laneID));
    //DEBUG_PRINT ( ("[Weight Collector %u]: Launched\n", laneID) );
    //mem_fence(CLK_GLOBAL_MEM_FENCE);


    enum e_states {IDLE, STREAM, COMMIT_WAIT, COMMIT};
    enum e_states state=IDLE;

    uint24_t weightIndexLast = 0x0;
    uint24_t weightIndexTracker = 0x0;
    unsigned int weightAddressOffset = 0x0;
    bool collectWeightRequest = false;

    bool keepGoing = true;
    bool stopSignal = false;

    t_spValueAndZCount weightAndOffset = 0x0;
    t_zCount zCount = 0x0;
    short weight = 0x0;

    //Used in unit teset. Delete this later!
    unsigned int timeOutCount = 0;

    #pragma unroll 1
    while (keepGoing) {

      collectWeightRequest = false;

      t_weightCollectToken controlToken
           = read_channel_nb_intel(channel_weightCollectControl[laneID], &collectWeightRequest);

      
      if (collectWeightRequest && (laneID < KERNEL_CACHE_LANES - 1)) {

          write_channel_intel(channel_weightCollectControl[laneID+1], controlToken);

      }

      bool stop = read_channel_nb_intel(channel_weightCollectorStop[laneID], &stopSignal);
      if (stopSignal) {
          if (laneID < KERNEL_CACHE_LANES - 1) {
            write_channel_intel (channel_weightCollectorStop[laneID+1], 0x1);
          }
         // DEBUG_PRINT( ("[Weight Collector %u] Shutting down\n", laneID) );
          keepGoing = false;

      }

      switch (state) {
        case (IDLE):
          if (collectWeightRequest) {
              if ( laneID < controlToken.numFiltersToCollect) { //For each collector in the activate range

                weightAddressOffset = 
                  controlToken.ddrKernelWeightStartOffset
                  + (unsigned int) controlToken.numWeightsInFilter 
                  * ( (unsigned int) controlToken.filterStart + (unsigned int) laneID );

                weightIndexTracker=0;
                weightIndexLast = controlToken.numWeightsInFilter;

                state = STREAM;
                EMULATOR_PRINT (("[Weight Collector %u]: Starting to collect weights! Number of weights in uncompressed row: %u\n", laneID, weightIndexLast));
                DEBUG_PRINT ( ("[Weight Collector %u]: Starting to collect weights! Number of weights in uncompressed row: %u\n", laneID, weightIndexLast) );
              }
              else { //Inactivate collectors should start waiting for commits
                  //DEBUG_PRINT( ("[Weight Collector %u]: Collector is skipped.\n", laneID) );
                  //printf("Number of filters to collect is: %u\n", controlToken.numFiltersToCollect);
                  //printf("Number of weights in the filter is %u\n", controlToken.numWeightsInFilter);
                  //printf("Filter start is %u\n", controlToken.filterStart);
                  //printf("DDR start is %u\n", controlToken.ddrKernelWeightStartOffset);

                  state = COMMIT_WAIT;
              }
              timeOutCount = 0;
          } 
          break;
        case (STREAM):
              {
                  bool weightReadSuccess = false;

                  //Wait for weight-offset tuple to arrive
                  weightAndOffset = 
                    read_channel_nb_intel(channel_sparseWeights[laneID], &weightReadSuccess);


                  if (weightReadSuccess){
                    //Parse the offset and the weight
                    zCount = (weightAndOffset & WEIGHT_ZCOUNT_MASK)
                                        >> WEIGHT_ZCOUNT_BITOFFSET;
                    weight = (weightAndOffset & WEIGHT_MASK)
                                        >> WEIGHT_BITOFFSET;

                    
                    //Store the weight
                    pMem[weightAddressOffset] = weight;

                    //DEBUG_PRINT( ("[Weight Collector %u] Writing %u to %u \n", laneID, weight, weightAddressOffset) );

                    //Update the index tracker
                    //weightIndexTracker += ( (uint24_t) 0X01
                    //              + ((uint24_t) zCount) & 0x0F );
                    //CAUTION: BIT-wise AND has LOWER precedence than + !!!!
                    weightIndexTracker += ( (uint24_t) 0X01
                                  + ( (uint24_t) zCount & 0x0F) );

                    EMULATOR_PRINT( ("[Weight Collector %u] zCount %u weightIndexTracker %u, increment %u\n", 
                      laneID, ((uint24_t) zCount) & 0x0F, weightIndexTracker, (  1 
                                  + ( ((uint24_t) zCount) & 0x0F ) ) ) );

                    weightAddressOffset++;
                 }

                 if (weightIndexTracker == weightIndexLast || timeOutCount == TIMEOUT){
                      state = COMMIT_WAIT;
                      if (timeOutCount == TIMEOUT) {
                        DEBUG_PRINT ( ("[Weight Collector %u] Read weight timeout. weightIndexTracker is %u\n", laneID, weightIndexTracker) );
                        EMULATOR_PRINT ( ("[Weight Collector %u] Read weight timeout. weightIndexTracker is %u\n", laneID, weightIndexTracker) );
                      }
                    }
              }

          break;
        case (COMMIT_WAIT):
            {
                bool previousCollectCommit = false;

                //EMULATOR_PRINT ( ("[Weight Collector %u] State is COMMIT_WAIT\n", laneID) );
                if (laneID > 0){

                  read_channel_nb_intel (
                    channel_weightCollectControlCommitInternal[laneID-1],
                    &previousCollectCommit);

                  if (previousCollectCommit) {
                    state = COMMIT;
                  }
                }
                else {
                  state = COMMIT;
                }
            }
          break;
        case (COMMIT):
          {
              bool commitSuccess = false;
              EMULATOR_PRINT ( ("[Weight Collector %u] State is COMMIT\n", laneID) );
              if (laneID < KERNEL_CACHE_LANES - 1) {
//                  commitSuccess = write_channel_nb_intel(
//                    channel_weightCollectControlCommitInternal[laneID],
//                    0x1);
                  commitSuccess = write_channel_nb_intel(
                    channel_weightCollectControlCommitInternal[laneID],
                    0x1);

                }
              else {
                  commitSuccess = write_channel_nb_intel(
                    channel_weightCollectControlCommit,
                    0x1);

//                  write_channel_intel(
//                   channel_weightCollectControlCommit,
//                    0x1);
              }
              if (commitSuccess){
                  EMULATOR_PRINT (("[Weight Collector %u]: Committed! Number of weights in uncompressed row: %u\n", laneID, weightIndexLast));
                  state = IDLE;
              }
//              EMULATOR_PRINT (("[Weight Collector %u]: Committed! Number of weights in uncompressed row: %u\n", laneID, weightIndexLast));
//              state = IDLE;
          }
              break;
      }
      //Timeout counter
      timeOutCount++;  
    }
}

#define COLLECTOR_GEN(copy) \
\
__attribute__((task)) \
__attribute__((max_global_work_dim(0))) \
__kernel void kernelWeightCollector ## copy  (\
    __global volatile short * pMem \
    ) { \
    weightCollector(pMem, copy); \
  }

COLLECTOR_GEN(0)

#if KERNEL_CACHE_LANES > 1
  COLLECTOR_GEN(1)
#endif
#if KERNEL_CACHE_LANES > 2
  COLLECTOR_GEN(2)
#endif
#if KERNEL_CACHE_LANES > 3
  COLLECTOR_GEN(3)
#endif
#if KERNEL_CACHE_LANES > 4
  COLLECTOR_GEN(4)
#endif
#if KERNEL_CACHE_LANES > 5
  COLLECTOR_GEN(5)
#endif
#if KERNEL_CACHE_LANES > 6
  COLLECTOR_GEN(6)
#endif
#if KERNEL_CACHE_LANES > 7
  COLLECTOR_GEN(7)
#endif

/*! transport_fill_weight_buffer
    \brief The instruction transport module of fill_weight_buffer command
    \details Assumption of the order in which instruction arrive
    Each instruction contains 25 bytes
    t0  Insruction[24], Header (1)
    t1 - t4: Instruction[23] - Instruction[20], numWeightsInFilter (4)
    t5 - t8: Instruction[19]-[18], numEncodingBlocks (2)
    t9 - t10: Instruction[17]-[16], cbEnd (2)
    t11 - t12: Instruction[15]-[14], cbStart (2)
    t13: Instruction[13], numFilterToStream (1)
    t14: Instruction[12]-[11], filterStart (2)
    t15: Instruction[10]-[7], ddrKernelWeightStartOffset (4)
    t16: Instruction[6]-[3], ddrKernelIndexStartOffset (4)

*/
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void transport_fill_weight_buffer ()
{
  EMULATOR_PRINT (("[transport_fill_weight_buffer]: Launched\n"));
  uint5_t wordCount=0;
  //unsigned char header = 0x0;
  t_tokenFillWeightCache token = {};

  #pragma unroll 1
   while (1) {
      bool readNewPacket = false;
      unsigned short word = read_channel_nb_intel(channel_instructions[TRANSPORT_ID_FILL_WEIGHT_BUFFER], &readNewPacket);
      

      if (readNewPacket) {
        //EMULATOR_PRINT (("[transport_fill_weight_buffer]: New word!\n"));
        //Pass the packet to the next one, if this transport isn't the last
        if (TRANSPORT_ID_FILL_WEIGHT_BUFFER + 0x1 < NUM_TRANSPORTS) {
            write_channel_intel(channel_instructions[TRANSPORT_ID_FILL_WEIGHT_BUFFER + 0x1], word);
        }

        unsigned char header = (unsigned char) (word >> 8);
        unsigned char instruction = (unsigned char) (0x0FF & word);

        if (header == OPCODE_FILL_WEIGHT_BUFFER) {
          switch (wordCount) {
            case 0:
              
                EMULATOR_PRINT (("[transport_fill_weight_buffer]: Detected message \n"));
                wordCount++;
              break;

            case 1:
              token.numWeightsInFilter = ((unsigned int) instruction) << 24;
              wordCount++;
              break;

            case 2:
              token.numWeightsInFilter = token.numWeightsInFilter | ( ((unsigned int) instruction) << 16 );
              wordCount++;
              break;

            case 3:
              token.numWeightsInFilter = token.numWeightsInFilter | ( ((unsigned int) instruction) << 8 );
              wordCount++;
              break;

            case 4:
              token.numWeightsInFilter = token.numWeightsInFilter | ( ((unsigned int) instruction ) );
              wordCount++;
              break;

            case 5:
              token.numEncodingBlocksInFilter = ((unsigned short) instruction) << 8;
              wordCount++;
              break;

            case 6:
              token.numEncodingBlocksInFilter = token.numEncodingBlocksInFilter | ( ((unsigned short) instruction) );
              wordCount++;
              break;

            case 7:
              token.cbEnd = ((unsigned short) instruction) << 8;
              wordCount++;
              break;

            case 8:
              token.cbEnd = token.cbEnd | ( ((unsigned short) instruction) );
              wordCount++;
              break;

            case 9:
              token.cbStart = ((unsigned short) instruction) << 8;
              wordCount++;
              break;

            case 10:
              token.cbStart = token.cbStart | ( ((unsigned short) instruction) );
              wordCount++;
              break;

            case 11:
              token.numFiltersToStream =  (unsigned char) instruction;
              wordCount++;
              break;

            case 12:
              token.filterStart = ((unsigned short) instruction) << 8;
              wordCount++;
              break;

            case 13:
              token.filterStart = token.filterStart | ( ((unsigned short) instruction) );
              wordCount++;
              break;

            case 14:
              token.ddrKernelWeightStartOffset = ((unsigned int) instruction) << 24;
              wordCount++;
              break;

            case 15:
              token.ddrKernelWeightStartOffset = token.ddrKernelWeightStartOffset | ( ((unsigned int) instruction) << 16 );
              wordCount++;
              break;

            case 16:
              token.ddrKernelWeightStartOffset = token.ddrKernelWeightStartOffset | ( ((unsigned int) instruction) << 8 );
              wordCount++;
              break;

            case 17:
              token.ddrKernelWeightStartOffset = token.ddrKernelWeightStartOffset | ( ((unsigned int) instruction) );
              wordCount++;
              break;

            case 18:
              token.ddrKernelIndexStartOffset = ((unsigned int) instruction) << 24;
              wordCount++;
              break;

            case 19:
              token.ddrKernelIndexStartOffset = token.ddrKernelIndexStartOffset | ( ((unsigned int) instruction) << 16 );
              wordCount++;
              break;

            case 20:
              token.ddrKernelIndexStartOffset = token.ddrKernelIndexStartOffset | ( ((unsigned int) instruction) << 8 );
              wordCount++;
              break;

            case 21:
              token.ddrKernelIndexStartOffset = token.ddrKernelIndexStartOffset | ( ((unsigned int) instruction) );

              EMULATOR_PRINT ( ("[SpW Fill Transport]: Sending instruction to SpW DMA.....\n") );
              write_channel_intel(channel_spWeightDMA, token);
              EMULATOR_PRINT ( ("[SpW Fill Transport]: Sent instruction to SpW DMA!\n") );
              wordCount = 0x0;
              break;

            default:
              wordCount++; 

          }
        }
       //EMULATOR_PRINT (("[transport_fill_weight_buffer]: WordCount is %u \n", wordCount));
      }
   }
}

/*! transport_drain_weight_buffer
    \brief The instruction transport module of drain_weight_buffer command

*/
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void transport_drain_weight_buffer ()
{
  EMULATOR_PRINT (("[transport_drain_weight_buffer]: Launched\n"));
  uint5_t wordCount=0;
  //unsigned char header = 0x0;
  t_tokenDrainWeightCache token = {};

  #pragma unroll 1
   while (1) {
      bool readNewPacket = false;
      unsigned short word = read_channel_nb_intel(channel_instructions[TRANSPORT_ID_DRAIN_WEIGHT_BUFFER], &readNewPacket);
      

      if (readNewPacket) {
        if (TRANSPORT_ID_DRAIN_WEIGHT_BUFFER + 0x1 < NUM_TRANSPORTS) {
            write_channel_intel(channel_instructions[TRANSPORT_ID_DRAIN_WEIGHT_BUFFER + 0x1], word);
        }

        unsigned char header = (unsigned char) (word >> 8);
        unsigned char instruction = (unsigned char) (0x0FF & word);

        if (header == OPCODE_DRAIN_WEIGHT_BUFFER) {
          switch (wordCount) {
            case 0:
                EMULATOR_PRINT ( ("[SpW Drain Transport]: Message detected!\n") );
                wordCount++;
              break;

            case 1:
              token.cbEnd = ((unsigned short) instruction) << 8;
              wordCount++;
              break;

            case 2:
              token.cbEnd = token.cbEnd | ((unsigned short) instruction);
              wordCount++;
              break;

            case 3:
              token.cbStart = ((unsigned short) instruction) << 8;
              wordCount++;
              break;

            case 4:
              token.cbStart = token.cbStart | ( ((unsigned short) instruction) );
              wordCount++;
              break;

            case 5:
              token.laneEnd = ((unsigned char) instruction);
              wordCount++;
              break;

            case 6:
              token.laneStart = ((unsigned char) instruction);
              wordCount = 0x0;

              write_channel_intel(channel_tokenDrainWeightCacheControl[0], token);
              EMULATOR_PRINT ( ("[SpW Drain Transport]: Message sent to the SpW Feeders!\n") );
              break;

            default:
              wordCount++; 

          }
        }
      }
   }
}

/*! transport_collect_weights
    \brief The instruction transport module of collect_weights command

*/
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void transport_collect_weights ()
{
  EMULATOR_PRINT (("[Transport_collect_weights]: Launched\n"));

  //uint5_t wordCount=0;
  uint5_t wordCount = 0;

  t_weightCollectToken token = {};

  #pragma unroll 1
   while (1) {
      bool readNewPacket = false;
      unsigned short word = read_channel_nb_intel(channel_instructions[TRANSPORT_ID_COLLECT_WEIGHT], &readNewPacket);
      
      //mem_fence(CLK_CHANNEL_MEM_FENCE);

      if (readNewPacket) {
        //EMULATOR_PRINT (("[Transport_collect_weights]: New word!\n"));
        if (TRANSPORT_ID_COLLECT_WEIGHT + 0x1 < NUM_TRANSPORTS) {
            write_channel_intel(channel_instructions[TRANSPORT_ID_COLLECT_WEIGHT + 0x1], word);
        }

        unsigned char header = (unsigned char) (word >> 8);
        unsigned char instruction = (unsigned char) (0x0FF & word);

         if (header == OPCODE_COLLECT_WEIGHT) {
          switch (wordCount) {
            case 0:
                EMULATOR_PRINT (("[Transport_collect_weights]: Message detected!\n"));
                wordCount++;
              break;

            case 1:
              token.numWeightsInFilter = ((uint24_t) instruction) << 16;
              wordCount++;
              break;

            case 2:
              token.numWeightsInFilter = token.numWeightsInFilter | 
                              (((uint24_t) instruction) << 8);
              wordCount++;
              break;

            case 3:
              token.numWeightsInFilter = token.numWeightsInFilter | 
                              ((uint24_t) instruction);
              wordCount++;
              break;

            case 4:
              token.numFiltersToCollect = ((unsigned short) instruction) << 0x8;
              wordCount++;
              break;

            case 5:
              token.numFiltersToCollect = token.numFiltersToCollect 
                                          | ((unsigned short) instruction);
              wordCount++;
              break;

            case 6:
              token.filterStart = ((unsigned short) instruction) << 0x8;
              wordCount++;
              break;

            case 7:
              token.filterStart = token.filterStart 
                                          | ((unsigned short) instruction);
              wordCount++;
              break;

            case 8:
              token.ddrKernelWeightStartOffset = ((unsigned int) instruction) << 24;
              wordCount++;
              break;

            case 9:
              token.ddrKernelWeightStartOffset 
                      = token.ddrKernelWeightStartOffset | (((unsigned int) instruction) << 16);
              wordCount++;
              break;

            case 10:
              token.ddrKernelWeightStartOffset 
                      = token.ddrKernelWeightStartOffset | (((unsigned int) instruction) << 8);
              wordCount++;
              break;

            case 11:
              token.ddrKernelWeightStartOffset 
                      = token.ddrKernelWeightStartOffset | ((unsigned int) instruction);
              wordCount = 0x0;
              write_channel_intel(channel_weightCollectControl[0], token);
              EMULATOR_PRINT (("[Transport_collect_weights]: Message sent to the collectors!\n"));
              break;

            default:
              wordCount++; 

          }
        //EMULATOR_PRINT (("[Transport_collect_weights]: WordCount is %u \n", wordCount));
        }
      }
   }
}

/*! transport_drain_weight_select
    \brief The instruction transport module of drain_weight_swap command

*/
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void transport_drain_weights_select ()
{
  EMULATOR_PRINT (("[transport_drain_weights_select]: Launched\n"));
  //uint5_t wordCount=0;
  uint1_t wordCount=0;

  #pragma unroll 1
   while (1) {
      bool readNewPacket = false;
      unsigned short word = read_channel_nb_intel(channel_instructions[TRANSPORT_ID_SWAP_WEIGHT_BUFFER], &readNewPacket);

      if (readNewPacket) {
        if (TRANSPORT_ID_SWAP_WEIGHT_BUFFER + 0x1 < NUM_TRANSPORTS) {
            write_channel_intel(channel_instructions[TRANSPORT_ID_SWAP_WEIGHT_BUFFER + 0x1], word);
        }

        unsigned char header = (unsigned char) (word >> 8);
        unsigned char instruction = (unsigned char) (0x0FF & word);

        if (header == OPCODE_SWAP_WEIGHT_BUFFER) {

          switch (wordCount) {
            case 0:
                EMULATOR_PRINT (("[transport_drain_weights_select]: Message detected!\n"));
                wordCount++;
                break;
            case 1:
                write_channel_intel(channel_spWeightFeederDrainSelect[0], 0x1);
                EMULATOR_PRINT (("[transport_drain_weights_select]: Swap sent!\n"));
                wordCount=0;
                break;
            default:
                wordCount++;
          }

        }
         

      }
    }
}
