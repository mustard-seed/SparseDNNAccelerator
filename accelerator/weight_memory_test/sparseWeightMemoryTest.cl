#include "params.hpp"
#include "device_structures.hpp"
#include "channels.cl"
#include "ihc_apint.h"
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


#ifdef EMULATOR
  #define EMULATOR_PRINT(format) printf format
#else
  #define EMULATOR_PRINT(format)
#endif


void checkCommits (uint4_t* pInstrInFlightCount) {
  bool read;
  read_channel_nb_intel(channel_spWeightDMACommit, &read);
  if (read) {
    EMULATOR_PRINT ( ("[Kernel Sequencer]: COMMIT. Header is %u.\n", OPCODE_FILL_WEIGHT_BUFFER));
    pInstrInFlightCount[OPCODE_FILL_WEIGHT_BUFFER & 0x1F]--; 
  }

  read_channel_nb_intel(channel_drainWeightCacheCommit, &read);
  if (read) {
    EMULATOR_PRINT ( ("[Kernel Sequencer]: COMMIT. Header is %u.\n", OPCODE_DRAIN_WEIGHT_BUFFER));
    pInstrInFlightCount[OPCODE_DRAIN_WEIGHT_BUFFER & 0x1F]--; 
  }

  read_channel_nb_intel(channel_spWeightFeederDrainSelectCommit, &read);
  if (read) {
    EMULATOR_PRINT ( ("[Kernel Sequencer]: COMMIT. Header is %u.\n", OPCODE_SWAP_WEIGHT_BUFFER));
    pInstrInFlightCount[OPCODE_SWAP_WEIGHT_BUFFER & 0x1F]--; 
  }

  read_channel_nb_intel(channel_weightCollectControlCommit, &read);
  if (read) {
    EMULATOR_PRINT ( ("[Kernel Sequencer]: COMMIT. Header is %u.\n", OPCODE_COLLECT_WEIGHT));
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
      for (uint5_t iter=0;
          iter < MAX_INSTRUCTION_TYPE_BITS;
          iter++) {
        instructionInFlightCount[iter] = 0x0;
      }

      //Read the instructions sequentially
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

          mem_fence(CLK_CHANNEL_MEM_FENCE);

          //CAUTION: Sending the header just by itself is strictly not required, unless the transports waste the first payload
          write_channel_intel(channel_instructions[0], header << 8);
          //Issue the instructions minus the dependency
          instructionInFlightCount[header]++;

          for (unsigned char iter=instructionSizeBytes;
            iter > 0;
            iter-- ){

            //Still need to perform the commit check in every cycle
            checkCommits(instructionInFlightCount);

            //Send the instructions
            //The header occupies the upper byte,
            //The receiver occupies the lower byte
            write_channel_intel(channel_instructions[0], 
              (header << 8) | (instruction.words[(iter-1) & 0x1F]) );

            mem_fence(CLK_CHANNEL_MEM_FENCE);
          }
          EMULATOR_PRINT ( ("[Kernel Sequencer]: Sent instruction %u. Header is %u No. Bytes is %u.\n"
            , iterInstruction, header, instructionSizeBytes));
      }

      //Wait for outstanding commands to finish
      uint1_t wait = 0x1;
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
}

/*! Kernel. weightCollector
    \brief Collects the sparse weights and store them to cache
*/
 void weightCollector(
      __global short * pMem,
      unsigned char laneID
    )
{
    EMULATOR_PRINT (("[Weight Collector %u]: Launched\n", laneID));
    enum e_states {IDLE, STREAM, COMMIT_WAIT, COMMIT};
    enum e_states state=IDLE;

    uint24_t weightIndexLast;
    uint24_t weightIndexTracker;
    unsigned int weightAddressOffset;
    bool collectWeightRequest;
    bool previousCollectCommit;
    bool weightReadSuccess;
    bool commitSuccess;

    t_spWeightAndOffset weightAndOffset;
    t_zCount zCount;
    short weight;

    while (1) {
      t_weightCollectToken controlToken
           = read_channel_nb_intel(channel_weightCollectControl[laneID], &collectWeightRequest);
      
      if (collectWeightRequest && (laneID < KERNEL_CACHE_LANES - 1)) {
          write_channel_intel(channel_weightCollectControl[laneID+1], controlToken);
      }

      switch (state) {
        case (IDLE):
          if (collectWeightRequest) {
              if (laneID < controlToken.numFiltersToCollect) { //For each collector in the activate range

                weightAddressOffset = 
                  controlToken.ddrKernelWeightStartOffset
                  + (unsigned int) controlToken.numWeightsInFilter 
                  * ( (unsigned int) controlToken.filterStart + (unsigned int) laneID );

                weightIndexTracker=0;
                weightIndexLast = controlToken.numWeightsInFilter;

                state = STREAM;
                EMULATOR_PRINT (("[Weight Collector %u]: Starting to collect weights! Number of weights in uncompressed row: %u\n", laneID, weightIndexLast));
              }
              else { //Inactivate collectors should start waiting for commits
                  state = COMMIT_WAIT;
              }
          } 
          break;
        case (STREAM):
              //Wait for weight-offset tuple to arrive
              weightAndOffset = 
                read_channel_nb_intel(channel_sparseWeights[laneID], &weightReadSuccess);

              if (weightReadSuccess){
                //Parse the offset and the weight
                zCount = (weightAndOffset & WEIGHT_ZCOUNT_MASK)
                                    >> WEIGHT_ZCOUNT_BITOFFSET;
                weight = (weightAndOffset & WEIGHT_MASK)
                                    >> WEIGHT_BITOFFSET;

                
                //Move the weight 
                pMem[weightAddressOffset] = weight;

                //Update the index tracker
                weightIndexTracker += ((uint24_t) 0X01
                              + (uint24_t) zCount);

                weightAddressOffset++;

                if (weightIndexTracker == weightIndexLast){
                  state = COMMIT_WAIT;
                }
              }
          break;
        case (COMMIT_WAIT):
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
          break;
        case (COMMIT):
              if (laneID < KERNEL_CACHE_LANES - 1) {
//                  commitSuccess = write_channel_nb_intel(
//                    channel_weightCollectControlCommitInternal[laneID],
//                    0x1);
                  write_channel_intel(
                    channel_weightCollectControlCommitInternal[laneID],
                    0x1);
                }
              else {
//                  commitSuccess = write_channel_nb_intel(
//                    channel_weightCollectControlCommit,
//                    0x1);
                  write_channel_intel(
                    channel_weightCollectControlCommit,
                    0x1);
              }
//              if (commitSuccess){
//                  EMULATOR_PRINT (("[Weight Collector %u]: Committed! Number of weights in uncompressed row: %u\n", laneID, weightIndexLast));
//                  state = IDLE;
//              }
              EMULATOR_PRINT (("[Weight Collector %u]: Committed! Number of weights in uncompressed row: %u\n", laneID, weightIndexLast));
              state = IDLE;
          break;
      }
    }
}

#define COLLECTOR_GEN(copy) \
\
__kernel void kernelWeightCollector ## copy  (\
    __global short * restrict pMem \
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
  unsigned char header = 0x0;
  bool readNewPacket;

   while (1) {
      unsigned short word = read_channel_nb_intel(channel_instructions[TRANSPORT_ID_FILL_WEIGHT_BUFFER], &readNewPacket);
      
      t_tokenFillWeightCache token;

      if (readNewPacket) {
        //EMULATOR_PRINT (("[transport_fill_weight_buffer]: New word!\n"));
        //Pass the packet to the next one, if this transport isn't the last
        if (TRANSPORT_ID_FILL_WEIGHT_BUFFER + 0x1 < NUM_TRANSPORTS) {
            write_channel_intel(channel_instructions[TRANSPORT_ID_FILL_WEIGHT_BUFFER + 0x1], word);
        }

        unsigned char header = (unsigned char) (word >> 8);
        unsigned char instruction = (unsigned char) (0x0FF & word);

        switch (wordCount) {
          case 0:
            if (header == OPCODE_FILL_WEIGHT_BUFFER) {
              EMULATOR_PRINT (("[transport_fill_weight_buffer]: Detected message \n"));
              wordCount++;
            }
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
            token.ddrKernelWeightStartOffset = ((unsigned int) instruction) << 24;
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
  unsigned char header = 0x0;
  bool readNewPacket;

   while (1) {
      unsigned short word = read_channel_nb_intel(channel_instructions[TRANSPORT_ID_DRAIN_WEIGHT_BUFFER], &readNewPacket);
      
      t_tokenDrainWeightCache token;

      if (readNewPacket) {
        if (TRANSPORT_ID_DRAIN_WEIGHT_BUFFER + 0x1 < NUM_TRANSPORTS) {
            write_channel_intel(channel_instructions[TRANSPORT_ID_DRAIN_WEIGHT_BUFFER + 0x1], word);
        }

        unsigned char header = (unsigned char) (word >> 8);
        unsigned char instruction = (unsigned char) (0x0FF & word);

        switch (wordCount) {
          case 0:
            if (header == OPCODE_DRAIN_WEIGHT_BUFFER) {
              EMULATOR_PRINT ( ("[SpW Drain Transport]: Message detected!\n") );
              wordCount++;
            }
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

/*! transport_collect_weights
    \brief The instruction transport module of collect_weights command

*/
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void transport_collect_weights ()
{
  EMULATOR_PRINT (("[Transport_collect_weights]: Launched\n"));
  uint5_t wordCount=0;
  bool readNewPacket;

   while (1) {
      unsigned short word = read_channel_nb_intel(channel_instructions[TRANSPORT_ID_COLLECT_WEIGHT], &readNewPacket);
      
      t_weightCollectToken token;

      if (readNewPacket) {
        //EMULATOR_PRINT (("[Transport_collect_weights]: New word!\n"));
        if (TRANSPORT_ID_COLLECT_WEIGHT + 0x1 < NUM_TRANSPORTS) {
            write_channel_intel(channel_instructions[TRANSPORT_ID_COLLECT_WEIGHT + 0x1], word);
        }

        unsigned char header = (unsigned char) (word >> 8);
        unsigned char instruction = (unsigned char) (0x0FF & word);

        switch (wordCount) {
          case 0:
            if (header == OPCODE_COLLECT_WEIGHT) {
              EMULATOR_PRINT (("[Transport_collect_weights]: Message detected!\n"));
              wordCount++;
            }
            break;

          case 1:
            token.numWeightsInFilter = (uint24_t) (((unsigned int) instruction) << 16);
            wordCount++;
            break;

          case 2:
            token.numWeightsInFilter = token.numWeightsInFilter | 
                            (uint24_t) (((unsigned int) instruction) << 8);
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

/*! transport_drain_weight_select
    \brief The instruction transport module of drain_weight_swap command

*/
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void transport_drain_weights_select ()
{
  EMULATOR_PRINT (("[transport_drain_weights_select]: Launched\n"));
  //uint5_t wordCount=0;
  bool readNewPacket;
  uint1_t wordCount=0;

   while (1) {
      unsigned short word = read_channel_nb_intel(channel_instructions[TRANSPORT_ID_SWAP_WEIGHT_BUFFER], &readNewPacket);

      if (readNewPacket) {
        if (TRANSPORT_ID_SWAP_WEIGHT_BUFFER + 0x1 < NUM_TRANSPORTS) {
            write_channel_intel(channel_instructions[TRANSPORT_ID_SWAP_WEIGHT_BUFFER + 0x1], word);
        }

        unsigned char header = (unsigned char) (word >> 8);
        unsigned char instruction = (unsigned char) (0x0FF & word);

        switch (wordCount) {
          case 0:
              if (header == OPCODE_SWAP_WEIGHT_BUFFER) {
                EMULATOR_PRINT (("[transport_drain_weights_select]: Message detected!\n"));
                wordCount++;
              }
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
