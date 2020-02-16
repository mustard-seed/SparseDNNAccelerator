#include "params.hpp"
#include "device_structures.hpp"
#include "device_utils.hpp"
#include "prototypePE_structs.hpp"
#include "channels.hpp"


#define PE_NUM_X PE_COLS
#define PE_NUM_Y PE_ROWS

typedef struct __attribute__((packed)) {
	unsigned char numPSumToProcess;
	unsigned char numBitsToRightShift;
	uint1_t enableRelu;
} t_outputInstruction;

channel t_operand channel_processedDrain __attribute__((depth(0)));


channel t_outputInstruction channel_outputInstruction __attribute__((depth(0)));

__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__ ((autorun))
__kernel void kernelOutputModifier (
	) 
{

	while (true)
	{
		//Decode the instruction
		t_outputInstruction instruction = read_channel_intel(channel_outputInstruction);
		unsigned char numPSum = instruction.numPSumToProcess;
		//unsigned char rndRightShift = instruction.numBitsToRightShift - 1;
		uint1_t enableRelu = instruction.enableRelu;

		unsigned char countPSum = 0;
		while (countPSum < numPSum)
		{
			t_conv_drain_tagged drain = read_channel_intel(channel_drain[0][0]);

			EMULATOR_PRINT ( ("[kernelOutputModifier]: Collected %d out of %d raw output. Value = %#04x \n", countPSum, numPSum, drain.value) );

			t_accumulator accumulator = drain.value;
			
			t_operand result = modifyOutput(accumulator, instruction.numBitsToRightShift, enableRelu);

			EMULATOR_PRINT ( ("[kernelOutputModifier]: Processed Value = %#04x\n", result) );
			
			write_channel_intel(channel_processedDrain, result);
			
			countPSum++;		
		}
	}
}


__attribute__((task))
__attribute__((max_global_work_dim(0)))
__kernel void kernelTestInterface (
		volatile __global t_transfer_block* restrict pActivationInput,
		volatile __global t_transfer_block* restrict pActivationOutput,
		volatile __global t_transfer_block* restrict pWeightInput,
		volatile __global t_transfer_block* restrict pWeightOutput,
		volatile __global char * restrict pDrainOut,
		short bias,
		volatile __global t_output_instruction_host* restrict pOutputInstruction,
		unsigned short numInputActivationBlocks,
		unsigned short numOutputActivationBlocks,
		unsigned short numInputWeightBlocks,
		unsigned short numOutputWeightBlocks,
		unsigned short numOutputDrain
	)
{
	unsigned short countInputActivationBlocks = 0,
				   countOutputActivationBlocks = 0,
				   countInputWeightBlocks = 0,
				   countOutputWeightBlocks = 0,
				   countOutputDrain = 0,
				   countOutputInstruction = 0;

	while (
			(countOutputActivationBlocks < numOutputActivationBlocks)
			||
			(countOutputWeightBlocks < numOutputWeightBlocks)
			||
			(countOutputDrain < numOutputDrain)
		) {

		if (countInputActivationBlocks < numInputActivationBlocks) {
			bool valid;
			t_transfer_block block;
			t_transferblock_tagged taggedBlock;
			
			block = pActivationInput[countInputActivationBlocks];

			#pragma unroll
			for (unsigned char i=0; i<TRANSFER_SIZE; i++) {
				taggedBlock.values.values[i] = block.values[i];
			}

			//uint6_t indexInStreamingBlock = indexActivationTracker + (uint6_t) block.runLength; 
			uint1_t isLast = (countInputActivationBlocks == (numInputActivationBlocks - 1)) ?
				TRUE : FALSE;
            setIsLast(&taggedBlock, isLast);
            setMaxTransferID(&taggedBlock, 1);

			valid = write_channel_nb_intel (channel_activation[0][0], taggedBlock);
			if (valid) {
				countInputActivationBlocks++;
				//numActivationTracker = unpackedValue.indices[COMPRESSION_VEC_SIZE-1];
				//indexActivationTracker = (indexInStreamingBlock == (SYNC_SIZE - 1)) ? 0x0 : indexInStreamingBlock + 0x1;
			}
		}

		if (countOutputActivationBlocks < numOutputActivationBlocks) {
			bool valid;
			t_transfer_block block;
			t_transferblock_tagged taggedBlock;
			taggedBlock = read_channel_nb_intel(channel_activation[1][0], &valid);
			if (valid) {
				#pragma unroll
				for (unsigned char i=0; i<TRANSFER_SIZE; i++) {
					block.values[i] = taggedBlock.values.values[i];
				}
				//hostBlock.runLength = block.streamingBlockIndex;
				//pActivationOutput[countOutputActivationBlocks++] = block;
				countOutputActivationBlocks++;
				EMULATOR_PRINT ( ("[kernelTestInferace]: Collected %d out of %d activation blocks\n", countOutputActivationBlocks, numOutputActivationBlocks) );
				//DEBUG_PRINT ( ("[kernelTestInferace]: Collected %d out of %d activation blocks\n", countOutputActivationBlocks, numOutputActivationBlocks) );
			}
		}

		//Need an extra one for bias
		if (countInputWeightBlocks < (numInputWeightBlocks+1)) {
			bool valid;

			t_transfer_block block;
			t_transferblock_tagged taggedBlock;

			block = (countInputWeightBlocks == ((unsigned short) 0)) ?
				bias2TransferBlock( (t_accumulator) bias)
				: pWeightInput[countInputWeightBlocks-1];
			//block = bias2TransferBlock( (t_accumulator) bias);

			#pragma unroll
			for (unsigned char i=0; i<TRANSFER_SIZE; i++) {
				taggedBlock.values.values[i] = block.values[i];
			}

			//Need an extra one for bias
			uint1_t isLast = (countInputWeightBlocks == (numInputWeightBlocks)) ? 
				TRUE : FALSE;

           setIsLast(&taggedBlock, isLast);
           setMaxTransferID(&taggedBlock, 1);

			valid = write_channel_nb_intel (channel_weight[0][0], taggedBlock);
			if (valid) {
				countInputWeightBlocks++;
				//numActivationTracker = unpackedValue.indices[COMPRESSION_VEC_SIZE-1];
				//indexWeightTracker = (indexInStreamingBlock == (SYNC_SIZE - 1)) ? 0x0 : indexInStreamingBlock + 0x1;
			}
		}

		//Need an extra one for bias
		if (countOutputWeightBlocks < numOutputWeightBlocks+1) {
			bool valid;
			t_transfer_block block;
			t_transferblock_tagged taggedBlock;

			taggedBlock = read_channel_nb_intel(channel_weight[0][1], &valid);

			if (valid) {
				
				#pragma unroll
				for (unsigned char i=0; i<TRANSFER_SIZE; i++) {
					block.values[i] = taggedBlock.values.values[i];
				}
				//hostBlock.runLength = block.streamingBlockIndex;
				//pWeightOutput[countOutputWeightBlocks++] = block;
				countOutputWeightBlocks++;
				EMULATOR_PRINT ( ("[kernelTestInferace]: Collected %d out of %d weight blocks\n", countOutputWeightBlocks, numOutputWeightBlocks) );
				//DEBUG_PRINT ( ("[kernelTestInferace]: Collected %d out of %d weight blocks\n", countOutputWeightBlocks, numOutputWeightBlocks) );
			}
		}

		if (countOutputDrain < numOutputDrain) {
			bool valid;
			char value = read_channel_nb_intel(channel_processedDrain, &valid);
			if (valid) {
				pDrainOut [countOutputDrain] = (char) value;
				EMULATOR_PRINT ( ("[kernelTestInferace]: Collected %d out of %d output. Value = %d\n", countOutputDrain, numOutputDrain, value) );
				countOutputDrain++;
			}
		}

		if (countOutputInstruction < 1)
		{
			t_outputInstruction instruction;
			t_output_instruction_host hostOutputInstruction = pOutputInstruction[0];
			instruction.numPSumToProcess = hostOutputInstruction.numPSumToProcess;
			instruction.numBitsToRightShift = hostOutputInstruction.numBitsToRightShift;
			instruction.enableRelu = hostOutputInstruction.enableRelu ? TRUE : FALSE;
			bool valid = write_channel_nb_intel(channel_outputInstruction, instruction);
			if (valid)
			{
				countOutputInstruction++;
			}
		}
		
	}
}