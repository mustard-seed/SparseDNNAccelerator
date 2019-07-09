#include "params.hpp"
#include "device_structures.hpp"
#include "ihc_apint.h"
#include "device_utils.hpp"
#include "prototypePE_structs.hpp"
#include "peComponents.hpp"
#include "rtl_lib.hpp"


#define PE_NUM_X PE_COLS
#define PE_NUM_Y PE_ROWS


/*
Used to control whether the weight/activation transport should forward the block received
*/
typedef struct __attribute__((packed)) {

  	//Whether to forward teh activation block to the compute unit / pSum manager
  	uint2_t forwardEnable;
} t_convBlockTransportInstruction;

typedef struct __attribute__((packed)) {
	uint2_t biasForwardEnable;
	unsigned char numPSumToSend;
	unsigned char accumulatorFracWidth;
	unsigned char biasFracWidth;
	unsigned char outputFracWidth;
} t_drainTransportInstruction;

/*
typedef struct __attribute__((packed)) {
	unsigned char mode;
	// Number of bits assigned to the fraction width
  	char fracW;
  	char fracDin;
  	char fracDout;
  	uint1_t enable;
} t_pSumManagerInstruction;
*/

typedef struct __attribute__((packed)) {
	t_operand nzWeight;
	t_operand nzActivation;
	uint1_t isLast;
} t_macOperands;

//With the max transport length and last bit annotation
typedef struct __attribute__((packed)){
	char values [SIMD_SIZE];
	uint6_t streamingBlockIndex;
	uint1_t isLast;
	char maxTransportID;

} t_simdblock_di_tagged;

//With the final array
typedef struct __attribute__((packed)){
	char values [SIMD_SIZE];
	uint6_t streamingBlockIndex;
	uint1_t isLast;
} t_simdblock_di;

//MAC Operands
typedef struct __attribute__((packed)) {
	char values [SIMD_SIZE];
} t_simdblock_mac;

t_accumulator mac (t_simdblock_mac activations, t_simdblock_mac weights) {
	t_accumulator output = 0x0;

	#pragma unroll
	for(int i=0; i<SIMD_SIZE/4; i++){
		//output += input.data[i]*weights.data[i];
		// use packed DSP blocks to improve efficiency
		#if defined (ARRIA10)
			output += a10_mac_8bitx4(
				activations.values[i*4],
				weights.values[i*4],
				activations.values[i*4+1],
				weights.values[i*4+1],
				activations.values[i*4+2],
				weights.values[i*4+2],
				activations.values[i*4+3],
				weights.values[i*4+3]
				);
		#elif defined (C5SOC)
			output += c5_mac_8bitx4(
					activations.values[i*4],
					weights.values[i*4],
					activations.values[i*4+1],
					weights.values[i*4+1],
					activations.values[i*4+2],
					weights.values[i*4+2],
					activations.values[i*4+3],
					weights.values[i*4+3]
					);
		#else
		#error Unsupported FPGA type!
		#endif
	}
	return output;
}



channel t_simdblock_di_tagged channel_activationInput __attribute__((depth(1)));
channel t_simdblock_di_tagged channel_activationOutput __attribute__((depth(1)));
channel t_simdblock_di_tagged channel_weightInput __attribute__((depth(1)));
channel t_simdblock_di_tagged channel_weightOutput __attribute__((depth(1)));
channel t_operand channel_biasInput __attribute__((depth(1)));
channel t_operand channel_biasOutput __attribute__((depth(1)));
channel t_accumulator channel_drainInput __attribute__((depth(1)));
channel t_operand channel_drainOutput __attribute__((depth(1)));
channel t_pe_prototype_instruction channel_instructionInput __attribute__((depth(1)));
channel t_pe_prototype_instruction channel_instructionOutputVertical __attribute__((depth(1)));
channel t_pe_prototype_instruction channel_instructionOutputHorizontal __attribute__((depth(1)));

channel t_operand channel_peBiasInput __attribute__((depth(0)));
//channel t_pSumManagerInstruction channel_pSumManagerInstructionInput __attribute__((depth(0)));
//channel t_convBlockTransportInstruction channel_activationTransportInstructionInput __attribute__((depth(0)));
//channel t_convBlockTransportInstruction channel_weightTransportInstructionInput __attribute__((depth(0)));
channel t_drainTransportInstruction channel_drainTransportInstruction __attribute__((depth(0)));

//channel t_spValueAndZCountUnpacked channel_peWeightInput __attribute__((depth(0)));
//channel t_spValueAndZCountUnpacked channel_peActivationInput __attribute__((depth(0)));

channel t_simdblock_di channel_dpWeightInput __attribute__((depth(PE_VEC_FIFO_SIZE)));
channel t_simdblock_di channel_dpActivationInput __attribute__((depth(PE_VEC_FIFO_SIZE)));
//channel t_operand channel_pSumManagerActivationInput __attribute__((depth(0)));
channel t_macOperands channel_macOperandsInput __attribute__((depth(0)));
//channel t_accumulator channel_pSumManagerMacInput __attribute__((depth(0)));

channel t_accumulator channel_peDrainOutput __attribute__((depth(0)));





__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__ ((autorun))
__kernel void kernelInstructionTransport (
		//int idx,
		//int idy
	)
{
	int idx = IDX;
	int idy = IDY;
		t_pe_prototype_instruction instruction = read_channel_intel(channel_instructionInput);

		//Forward the instruction to other the activaion transport
		//t_convBlockTransportInstruction activationInstruction, weightInstruction;
		//activationInstruction.forwardEnable = (idy < instruction.maxIDY) ? 
		//	0x1 : 0x0;
		//weightInstruction.forwardEnable = (idx < instruction.maxIDX) ? 
		//	0x1 : 0x0;

		t_drainTransportInstruction drainInstruction;
		drainInstruction.biasForwardEnable = (idx < instruction.maxIDX) ? 
			0x1 : 0x0;
		drainInstruction.numPSumToSend = (unsigned char) (instruction.maxIDY - idy + 1);
		drainInstruction.accumulatorFracWidth = instruction.fracW + instruction.fracDin;
		drainInstruction.biasFracWidth = instruction.fracW;
		drainInstruction.outputFracWidth = instruction.fracDout;

		//write_channel_intel(channel_activationTransportInstructionInput, activationInstruction);
		//write_channel_intel(channel_weightTransportInstructionInput, weightInstruction);
		write_channel_intel(channel_drainTransportInstruction, drainInstruction);

		if ( ( (0x1FF & idy) == 0 ) && ( (0x1FF & idx) < (instruction.maxIDX) ) ) {
			//pass right
			write_channel_intel(channel_instructionOutputHorizontal, instruction);
		}
		if ( (0x1FF & idy) < (instruction.maxIDY)) {
			//pass down
			write_channel_intel(channel_instructionOutputVertical, instruction);
		}	
}

__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__ ((autorun))
__kernel void kernelDrainTransport (
		//int idx,
		//int idy
	)
{
	int idx = IDX;
	int idy = IDY;

	uint1_t biasState = 0x0,
		    drainState = 0x0;

	t_accumulator bias;
	unsigned char drainCount = 0;
	unsigned char biasLeftShift;
	unsigned char accumulatorRightShift;
	unsigned char numPSumToSend;
	uint2_t biasForwardEnable;

	while (true) {
		if ( (drainState == 0x0) 
			&& (biasState == 0x0)) {
			bool success;
			t_drainTransportInstruction instruction = read_channel_nb_intel(channel_drainTransportInstruction, &success);
			if (success){
				biasState = 0x1;
				drainState = 0x1;
				accumulatorRightShift= (unsigned char) (instruction.accumulatorFracWidth - instruction.outputFracWidth - 1);
				biasLeftShift = instruction.accumulatorFracWidth - instruction.biasFracWidth;
				biasForwardEnable = instruction.biasForwardEnable;
				numPSumToSend = instruction.numPSumToSend;
			}
		}
		else {
			if (biasState == 0x1) {
				bool readSuccess;
				t_operand tempBias = read_channel_nb_intel(channel_biasInput, &readSuccess);
				if (readSuccess) {
					bias = ((t_accumulator) tempBias) 
					<< biasLeftShift;
					biasState = 0x0;
				}
				if (idx < (PE_COLS - 1)) {
					if (readSuccess & (biasForwardEnable == 0x1)) {
						write_channel_intel(channel_biasOutput, tempBias);
					}
				}
			} // biasState

			if (drainCount < numPSumToSend) {
				t_accumulator accumulator;
				bool success;
				if (drainCount == 0) {
					accumulator =
						read_channel_nb_intel(channel_peDrainOutput, &success);
				}
				else {
					accumulator =  read_channel_nb_intel(channel_drainInput, &success);
				}
					//Add bias, shift, and saturate
				if (success) {
					if (drainCount == 0) {
						accumulator += bias;
					}

					t_operand result;
					
					t_accumulator signExtensionMask;
					if(accumulator>=0)
						signExtensionMask = 0x00;
					else
						signExtensionMask = ~(0xFFFFFFFF>> accumulatorRightShift); // ">>" is logic shift, then perform sign extension manually

					t_accumulator accumulatorWithRndBit = 
						(signExtensionMask 
						| (accumulator >> 
							( accumulatorRightShift )));


					t_accumulator accumulatorBiased;
					if(accumulatorWithRndBit >= ((t_accumulator) 256))
						accumulatorBiased = 0x0FF; //=255
					else if(accumulatorWithRndBit <((t_accumulator) -256))
						accumulatorBiased = 0x0100; //=-256
					else
						accumulatorBiased = (t_accumulator) ((0x1FF & accumulatorWithRndBit)+ (t_accumulator) 0x01);

					// final truncation
					result = 0xFF & (accumulatorBiased>>0x01);  // remove the last rounding bit
					
					EMULATOR_PRINT ( ("[kernelDrainTransport]: Waiting to write pSum %d \n", drainCount) );
					//write_channel_intel(channel_drainOutput, result);
					write_channel_intel(channel_drainOutput, result);
					drainCount++;
				}

				if (drainCount == numPSumToSend) {
					EMULATOR_PRINT ( ("[kernelDrainTransport]: Committed the pSum\n") );
					drainCount = 0;
					drainState = 0x0;
				}

			} // drainState
		}// else
	}	
}


__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__ ((autorun))
__kernel void kernelWeightTransport (
	)
{
	

	int idx = IDX;
	int idy = IDY;

	t_simdblock_di_tagged block = read_channel_intel(channel_weightInput);
	t_simdblock_di peBlock;
	#pragma unroll
	for (unsigned char i=0; i<SIMD_SIZE; i++) {
		peBlock.values[i] = block.values[i];
	}
	peBlock.streamingBlockIndex = block.streamingBlockIndex;
	peBlock.isLast = block.isLast;

	if (idy < (PE_ROWS - 1)){
		if ( idy < block.maxTransportID ) {
			EMULATOR_PRINT ( ("[kernelWeightTransport]: Waiting to pass a weight block to the output\n") );
			write_channel_intel(channel_weightOutput, block);
		}
	}

	write_channel_intel(channel_dpWeightInput, peBlock); 
}

__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__ ((autorun))
__kernel void kernelActivationTransport (
	)
{
	

	int idx = IDX;
	int idy = IDY;

	t_simdblock_di_tagged block = read_channel_intel(channel_activationInput);
	t_simdblock_di peBlock;
	#pragma unroll
	for (unsigned char i=0; i<SIMD_SIZE; i++) {
		peBlock.values[i] = block.values[i];
	}
	peBlock.streamingBlockIndex = block.streamingBlockIndex;
	peBlock.isLast = block.isLast;

	if (idx < (PE_COLS - 1)){
		if ( idx < block.maxTransportID ) {
			EMULATOR_PRINT ( ("[kernelWeightTransport]: Waiting to pass a weight block to the output\n") );
			write_channel_intel(channel_activationOutput, block);
		}
	}

	write_channel_intel(channel_dpActivationInput, peBlock); 
}



__attribute__((task))
__attribute__((max_global_work_dim(0)))
__kernel void kernelTestInterface (
		__global t_simdblock_host* restrict pActivationInput,
		__global t_simdblock_host* restrict pActivationOutput,
		__global t_simdblock_host* restrict pWeightInput,
		__global t_simdblock_host* restrict pWeightOutput,
		__global short * restrict pBiasIn,
		__global short * restrict pBiasOut,
		__global short * restrict pDrainIn, 
		__global short * restrict pDrainOut,
		__global t_pe_prototype_instruction_host* restrict pInstructionInput,
		__global t_pe_prototype_instruction_host* restrict pInsructionOutputHorizontal,
		__global t_pe_prototype_instruction_host* restrict pInstructionOutputVeritcal,
		unsigned short numInputActivationBlocks,
		unsigned short startIndexActivationBlocks,
		unsigned short numOutputActivationBlocks,
		unsigned short numInputWeightBlocks,
		unsigned short startIndexWeightBlocks,
		unsigned short numOutputWeightBlocks,
		unsigned short numInputBias,
		unsigned short numOutputBias,
		unsigned short numInputDrain,
		unsigned short numOutputDrain,
		unsigned short numInputInstruction,
		unsigned short numOutputInsructionHorizontal,
		unsigned short numOutputInstructionVertical,
		unsigned char maxIDX,
		unsigned char maxIDY
	)
{
	unsigned short countInputActivationBlocks = 0,
				   countOutputActivationBlocks = 0,
				   countInputWeightBlocks = 0,
				   countOutputWeightBlocks = 0,
				   countInputBias = 0,
				   countOutputBias = 0,
				   countInputDrain = 0,
				   countOutputDrain = 0,
				   countInputInstruction = 0,
				   countOutputInstructionVertical = 0,
				   countOutputInstructionHorizontal = 0;

	uint5_t		   indexActivationTracker = 0,
				   indexWeightTracker = 0;
	int idx = IDX;
	int idy = IDY;

	while (
			(countInputActivationBlocks < numInputActivationBlocks)
			||
			(countOutputActivationBlocks < numOutputActivationBlocks)
			||
			(countInputWeightBlocks < numInputWeightBlocks)
			||
			(countOutputWeightBlocks < numOutputWeightBlocks)
			||
			(countInputBias < numInputBias)
			||
			(countOutputBias < numOutputBias)
			||
			(countInputDrain < numInputDrain)
			||
			(countOutputDrain < numOutputDrain)
			||
			(countInputInstruction < numInputInstruction)
			||
			(countOutputInstructionVertical < numOutputInstructionVertical)
			||
			(countOutputInstructionHorizontal < numOutputInsructionHorizontal)
		) {

		if (countInputActivationBlocks < numInputActivationBlocks) {
			bool valid;
			t_simdblock_host block = pActivationInput[countInputActivationBlocks];
			//t_vecUnpacked unpackedValue;
			//decodeRunLength(&value, &unpackedValue, numActivationTracker);
			t_simdblock_di_tagged taggedBlock;
			#pragma unroll
			for (unsigned char i=0; i<SIMD_SIZE; i++) {
				taggedBlock.values[i] = block.values[i];
			}

			uint6_t indexInStreamingBlock = indexActivationTracker + (uint6_t) block.runLength; 
			unsigned char isLast = (countInputActivationBlocks == (numInputActivationBlocks - 1)) ?
                0x1 : 0x0;
            taggedBlock.streamingBlockIndex = indexInStreamingBlock;
            taggedBlock.isLast = isLast;;
            taggedBlock.maxTransportID = maxIDY;

			valid = write_channel_nb_intel (channel_activationInput, taggedBlock);
			if (valid) {
				countInputActivationBlocks++;
				//numActivationTracker = unpackedValue.indices[COMPRESSION_VEC_SIZE-1];
				indexActivationTracker = (indexInStreamingBlock == (SYNC_SIZE - 1)) ? 0x0 : indexInStreamingBlock + 0x1;
			}
		}

		if ( (0x1FF & idy) < (PE_NUM_Y - 1) ) {
			if (countOutputActivationBlocks < numOutputActivationBlocks) {
				bool valid;
				t_simdblock_di_tagged block = read_channel_nb_intel(channel_activationOutput, &valid);
				if (valid) {
					t_simdblock_host hostBlock;
					
					#pragma unroll
					for (unsigned char i=0; i<SIMD_SIZE; i++) {
						hostBlock.values[i] = block.values[i];
					}
					hostBlock.runLength = block.streamingBlockIndex;
					pActivationOutput[countOutputActivationBlocks++] = hostBlock;
					EMULATOR_PRINT ( ("[kernelTestInferace]: Collected %d out of %d activation blocks\n", countOutputActivationBlocks, numOutputActivationBlocks) );
				}
			}
		}

		if (countInputWeightBlocks < numInputWeightBlocks) {
			bool valid;
			t_simdblock_host block = pWeightInput[countInputWeightBlocks];
			//t_vecUnpacked unpackedValue;
			//decodeRunLength(&value, &unpackedValue, numActivationTracker);
			t_simdblock_di_tagged taggedBlock;
			#pragma unroll
			for (unsigned char i=0; i<SIMD_SIZE; i++) {
				taggedBlock.values[i] = block.values[i];
			}

			uint6_t indexInStreamingBlock = indexWeightTracker + (uint6_t) block.runLength; 
			unsigned char isLast = (countInputWeightBlocks == (numInputWeightBlocks - 1)) ?
                0x1 : 0x0;
            taggedBlock.streamingBlockIndex = indexInStreamingBlock;
            taggedBlock.isLast = isLast;;
            taggedBlock.maxTransportID = maxIDX;

			valid = write_channel_nb_intel (channel_weightInput, taggedBlock);
			if (valid) {
				countInputWeightBlocks++;
				//numActivationTracker = unpackedValue.indices[COMPRESSION_VEC_SIZE-1];
				indexWeightTracker = (indexInStreamingBlock == (SYNC_SIZE - 1)) ? 0x0 : indexInStreamingBlock + 0x1;
			}
		}

		if ( (0x1FF & idx) < (PE_NUM_X - 1) ) {
			if (countOutputWeightBlocks < numOutputWeightBlocks) {
				bool valid;
				//t_vecUnpacked value = read_channel_nb_intel(channel_weightOutput, &valid);
				t_simdblock_di_tagged block = read_channel_nb_intel(channel_weightOutput, &valid);
				if (valid) {
					t_simdblock_host hostBlock;
					
					#pragma unroll
					for (unsigned char i=0; i<SIMD_SIZE; i++) {
						hostBlock.values[i] = block.values[i];
					}
					hostBlock.runLength = block.streamingBlockIndex;
					pWeightOutput[countOutputWeightBlocks++] = hostBlock;
					EMULATOR_PRINT ( ("[kernelTestInferace]: Collected %d out of %d weight blocks\n", countOutputWeightBlocks, numOutputWeightBlocks) );
				}
			}
		}

		if (countInputBias < numInputBias) {
			bool valid;
			t_operand value = pBiasIn[countInputBias];
			valid = write_channel_nb_intel (channel_biasInput, value);
			if (valid) {
				countInputBias++;
			}
		}

		if ( (0x1FF & idx) < (PE_NUM_X - 1) ) {

			if (countOutputBias < numOutputBias) {
				bool valid;
				short value = read_channel_nb_intel(channel_biasOutput, &valid);
				if (valid) {
					pBiasOut[countOutputBias++] = value;
				}
			}
		}

		if (countInputDrain < numInputDrain) {
			bool valid;
			short value = pDrainIn [countInputDrain];
			valid = write_channel_nb_intel (channel_drainInput, value);
			if (valid) {
				countInputDrain++;
			}
		}

		if (countOutputBias < numOutputDrain) {
			bool valid;
			short value = read_channel_nb_intel(channel_drainOutput, &valid);
			if (valid) {
				pDrainOut [countOutputDrain++] = (short) value;
			}
		}

		if (countInputInstruction < numInputInstruction) {
			bool valid;
			t_pe_prototype_instruction_host valueHost = pInstructionInput [countInputInstruction];
			t_pe_prototype_instruction value;
			value.maxIDX = valueHost.maxIDX;
			value.maxIDY = valueHost.maxIDY;
			value.fracW = valueHost.fracW;
			value.fracDin = valueHost.fracDin;
			value.fracDout = valueHost.fracDout;
			valid = write_channel_nb_intel (channel_instructionInput, value);
			if (valid) {
				countInputInstruction++;
			}
		}

		if ( ( (0x1FF & idy) == 0 ) && ( (0x1FF & idx) < (PE_NUM_X - 1) ) ) {
			if (countOutputInstructionHorizontal < numOutputInsructionHorizontal) {
				bool valid;
				t_pe_prototype_instruction value = read_channel_nb_intel(channel_instructionOutputHorizontal, &valid);
				t_pe_prototype_instruction_host valueHost;
				valueHost.maxIDX = value.maxIDX;
				valueHost.maxIDY = value.maxIDY;
				valueHost.fracW = value.fracW;
				valueHost.fracDin = value.fracDin;
				valueHost.fracDout = value.fracDout;
				if (valid) {
					pInsructionOutputHorizontal [countOutputInstructionHorizontal++] = valueHost;
				}
			}
		}

		if ( (0x1FF & idy) < (PE_NUM_Y - 1)) {
			if (countOutputInstructionVertical < numOutputInstructionVertical) {
				bool valid;
				t_pe_prototype_instruction value = read_channel_nb_intel(channel_instructionOutputVertical, &valid);
				t_pe_prototype_instruction_host valueHost;
				valueHost.maxIDX = value.maxIDX;
				valueHost.maxIDY = value.maxIDY;
				valueHost.fracW = value.fracW;
				valueHost.fracDin = value.fracDin;
				valueHost.fracDout = value.fracDout;
				if (valid) {
					pInstructionOutputVeritcal [countOutputInstructionVertical++] = valueHost;
				}
			}
		}
		
	}
}


/*! \brief Dot product kernel that operates on compressed sparse weight and activation

*/

__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void kernelDotProductDispatcher (
	)
{
	/*
	Key assumption: The last uncompressed element in each streaming block, of length 63, will be preserved after 
	compression!
	*/
	
	uint6_t streamingBlockIndexActivation = 0, streamingBlockIndexWeight = 0;
	t_simdblock_di activationBlock, weightBlock;
	t_accumulator pSum=0;
	//uint1_t proceed = 0x1;
	//bool activationBlockValid = true, weightBlockValid = true, loadBlock = true;

	while (true) {
;
		if (streamingBlockIndexActivation 
					<= streamingBlockIndexWeight) {
			//EMULATOR_PRINT ( ("[kernelDotProductEngineDispatcher]: Waiting to read from the activation channel\n") );
			//activationBlock = read_channel_nb_intel(channel_dpActivationInput, &activationBlockValid);
			activationBlock = read_channel_intel(channel_dpActivationInput);

		}
		//mem_fence(CLK_CHANNEL_MEM_FENCE);
		if (streamingBlockIndexActivation 
					>= streamingBlockIndexWeight) {
			//EMULATOR_PRINT ( ("[kernelDotProductEngineDispatcher]: Waiting to read from the weight channel\n") );
			//weightBlock = read_channel_nb_intel(channel_dpWeightInput, &weightBlockValid);
			weightBlock = read_channel_intel(channel_dpWeightInput);	
		}



			streamingBlockIndexActivation = activationBlock.streamingBlockIndex;
			streamingBlockIndexWeight = weightBlock.streamingBlockIndex;


			EMULATOR_PRINT ( ("[kernelDotProductEngineDispatcher]: streamingBlockIndexActivation updated to %d\n", streamingBlockIndexActivation) );

			EMULATOR_PRINT ( ("[kernelDotProductEngineDispatcher]: streamingBlockIndexWeight updated to %d\n", streamingBlockIndexWeight) );

			if (streamingBlockIndexWeight == streamingBlockIndexActivation) {
				
				t_simdblock_mac activations, weights;
				#pragma unroll
				for (int i=0; i<SIMD_SIZE; i++) {
					activations.values[i] = activationBlock.values[i];
					weights.values[i] = weightBlock.values[i];
				}
				pSum += (t_accumulator) (mac(activations, weights));

				uint1_t weightBlockIsLast = weightBlock.isLast;
				uint1_t activationBlockIsLast = activationBlock.isLast;

				uint1_t isLast = ((weightBlockIsLast == 0x1) && (activationBlockIsLast == 0x1)) ?
				0x1 : 0x0;

				if (isLast == 0x1) {
					EMULATOR_PRINT ( ("[kernelMAC]: Waiting to commit!\n") );
					write_channel_intel(channel_peDrainOutput, pSum);
					pSum = 0;
					EMULATOR_PRINT ( ("[kernelMAC]: Committed!\n") );
				}
			}
		
	} // while
	
}
