#include "params.hpp"
#include "device_structures.hpp"
#include "ihc_apint.h"
#include "device_utils.hpp"
#include "prototypePE_structs.hpp"
#include "peComponents.hpp"


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



channel t_spValueAndZCountUnpacked channel_activationInput __attribute__((depth(1)));
channel t_spValueAndZCountUnpacked channel_activationOutput __attribute__((depth(1)));
channel t_spValueAndZCountUnpacked channel_weightInput __attribute__((depth(1)));
channel t_spValueAndZCountUnpacked channel_weightOutput __attribute__((depth(1)));
channel t_operand channel_biasInput __attribute__((depth(1)));
channel t_operand channel_biasOutput __attribute__((depth(1)));
channel t_operand channel_drainInput __attribute__((depth(1)));
channel t_operand channel_drainOutput __attribute__((depth(1)));
channel t_pe_prototype_instruction channel_instructionInput __attribute__((depth(1)));
channel t_pe_prototype_instruction channel_instructionOutputVertical __attribute__((depth(1)));
channel t_pe_prototype_instruction channel_instructionOutputHorizontal __attribute__((depth(1)));

channel short channel_peBiasInput __attribute__((depth(0)));
//channel t_pSumManagerInstruction channel_pSumManagerInstructionInput __attribute__((depth(0)));
channel t_convBlockTransportInstruction channel_activationTransportInstructionInput __attribute__((depth(0)));
channel t_convBlockTransportInstruction channel_weightTransportInstructionInput __attribute__((depth(0)));
channel t_drainTransportInstruction channel_drainTransportInstruction __attribute__((depth(0)));

//channel t_spValueAndZCountUnpacked channel_peWeightInput __attribute__((depth(0)));
//channel t_spValueAndZCountUnpacked channel_peActivationInput __attribute__((depth(0)));

channel t_spValueAndZCountUnpacked channel_dpWeightInput __attribute__((depth(PE_VEC_FIFO_SIZE)));
channel t_spValueAndZCountUnpacked channel_dpActivationInput __attribute__((depth(PE_VEC_FIFO_SIZE)));
//channel t_operand channel_pSumManagerActivationInput __attribute__((depth(0)));
channel t_macOperands channel_macOperandsInput __attribute__((depth(0)));
//channel t_accumulator channel_pSumManagerMacInput __attribute__((depth(0)));

channel short channel_peDrainOutput __attribute__((depth(0)));





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
	while (true) {
		t_pe_prototype_instruction instruction = read_channel_intel(channel_instructionInput);

		//Forward the instruction to other the activaion transport
		t_convBlockTransportInstruction activationInstruction, weightInstruction;
		activationInstruction.forwardEnable = (idy < instruction.maxIDY) ? 
			0x1 : 0x0;
		weightInstruction.forwardEnable = (idx < instruction.maxIDX) ? 
			0x1 : 0x0;

		t_drainTransportInstruction drainInstruction;
		drainInstruction.biasForwardEnable = (idx < instruction.maxIDX) ? 
			0x1 : 0x0;
		drainInstruction.numPSumToSend = (unsigned char) (instruction.maxIDY - idy + 1);
		drainInstruction.accumulatorFracWidth = instruction.fracW + instruction.fracDin;
		drainInstruction.biasFracWidth = instruction.fracW;
		drainInstruction.outputFracWidth = instruction.fracDout;

		write_channel_intel(channel_activationTransportInstructionInput, activationInstruction);
		write_channel_intel(channel_weightTransportInstructionInput, weightInstruction);
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
				t_operand result;
				bool success;
				if (drainCount == 0) {
					t_accumulator accumulator =
						read_channel_nb_intel(channel_peDrainOutput, &success);
					//Add bias, shift, and saturate
					if (success) {
						accumulator += bias;
						t_accumulator signExtensionMask;
						if(accumulator>=0)
							signExtensionMask = 0x00;
						else
							signExtensionMask = ~(0xFFFF>> accumulatorRightShift); // ">>" is logic shift, then perform sign extension manually

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
							accumulatorBiased = (t_accumulator) ((0x1FF & accumulatorWithRndBit)+ 0x01);

						// final truncation
						result = 0xFF & (accumulatorBiased>>0x01);  // remove the last rounding bit
					}

				}
				else {
					result = read_channel_nb_intel(channel_drainInput, &success);
				}
				if (success){
					EMULATOR_PRINT ( ("[kernelDrainTransport]: Waiting to write pSum %d \n", drainCount) );
					write_channel_intel(channel_drainOutput, result);
					drainCount++;
					EMULATOR_PRINT ( ("[kernelDrainTransport]: Sent %d out of %d pSum\n", drainCount, numPSumToSend) );
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
		//int idx,
		//int idy
	)
{
	

	int idx = IDX;
	int idy = IDY;

	uint1_t state = 0x0;
	t_convBlockTransportInstruction instruction;

	while (true) {
		if (state == 0x0) {
			EMULATOR_PRINT ( ("[kernelWeightTransport]: Waiting for a new instruction\n") );
			instruction =
	 			read_channel_intel(channel_weightTransportInstructionInput);
	 		state = 0x1;
		}
		else {
			t_spValueAndZCountUnpacked block = 
				read_channel_intel(channel_weightInput);
            state = (((block.metaInformation >> UNPACKED_ISLAST_BITOFFSET) & UNPACKED_ISLAST_MASK)  == 0x0) ? 0x1 : 0x0;
			//EMULATOR_PRINT ( ("[kernelWeightTransport]: Waiting to pass a weight block to the PE\n") );
			write_channel_intel(channel_dpWeightInput, block); 

			if (idy < (PE_ROWS - 1)){
				if ( instruction.forwardEnable == 0x1 ) {
					//EMULATOR_PRINT ( ("[kernelWeightTransport]: Waiting to pass a weight block to the output\n") );
					write_channel_intel(channel_weightOutput, block);
				}
			}
		}
	}
}

__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__ ((autorun))
__kernel void kernelActivationTransport (
		//int idx,
		//int idy
	)
{
	

	int idx = IDX;
	int idy = IDY;

	uint1_t state = 0x0;
	t_convBlockTransportInstruction instruction;

	while (true) {
		if (state == 0x0) {
			EMULATOR_PRINT ( ("[kernelActivationTransport]: Waiting for a new instruction\n") );
			instruction =
	 			read_channel_intel(channel_activationTransportInstructionInput);
	 		state = 0x1;
		}
		else {
			t_spValueAndZCountUnpacked block = 
				read_channel_intel(channel_activationInput);
            state = (((block.metaInformation >> UNPACKED_ISLAST_BITOFFSET) & UNPACKED_ISLAST_MASK)  == 0x0) ? 0x1 : 0x0;
            //EMULATOR_PRINT ( ("[kernelActivationTransport]: Waiting to pass a activation block to the PE\n") );
			write_channel_intel(channel_dpActivationInput,block); 

			if (idx < (PE_COLS - 1)){
				if ( instruction.forwardEnable == 0x1) {
					//EMULATOR_PRINT ( ("[kernelActivationTransport]: Waiting to pass a activation block to the output\n") );
					write_channel_intel(channel_activationOutput, block);
				}
			}
		}
	}
}



/*
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
	while (true) {
		t_pe_prototype_instruction instruction = 
			read_channel_intel(channel_instructionInput);

		//Makes sure the PE receives it first
		//mem_fence(CLK_CHANNEL_MEM_FENCE);


		if ( ( (0x1FF & idy) == 0 ) && ( (0x1FF & idx) < (PE_NUM_X - 1) ) ) {
			//pass right
			write_channel_intel(channel_instructionOutputHorizontal, instruction);
		}

		if ( (0x1FF & idy) < (PE_NUM_Y - 1)) {
			//pass down
			write_channel_intel(channel_instructionOutputVertical, instruction);
		}

		uint1_t forwardEnable = ( (idy <= instruction.maxIDY) && (idx <= instruction.maxIDX)) ?
			0x1 : 0x0;
		
		t_pSumManagerInstruction pSumManagerInstruction;
		pSumManagerInstruction.fracW = instruction.fracW;
		pSumManagerInstruction.fracDin = instruction.fracDin;
		pSumManagerInstruction.fracDout = instruction.fracDout;
		pSumManagerInstruction.enable = forwardEnable;
		pSumManagerInstruction.mode = instruction.mode;
		write_channel_intel (channel_pSumManagerInstructionInput, pSumManagerInstruction);

		if (instruction.mode == PE_MODE_DOT_PRODUCT ||
			instruction.mode == PE_MODE_MAX_POOL ||
			instruction.mode == PE_MODE_LOAD_ACTIVATION ||
			instruction.mode == PE_MODE_ELTWISE_ADD) {
			t_activationForwarderInstruction actForwarderInstruction;
			actForwarderInstruction.activationSelectStartIndex = instruction.activationSelectStartIndex;
			actForwarderInstruction.mode = instruction.mode;
			actForwarderInstruction.forwardEnable = forwardEnable;
			write_channel_intel (channel_activationForwarderInstructionInput, actForwarderInstruction);
		}

		if (instruction.mode == PE_MODE_DOT_PRODUCT) {
			t_weightForwarderInstruction wForwarderInstruction;;
			wForwarderInstruction.forwardEnable = forwardEnable;
			write_channel_intel (channel_weightForwarderInstructionInput, wForwarderInstruction);
		}
	}

}
*/

/*
__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__ ((autorun))
__kernel void kernelBiasTransport (
		//int idx,
		//int idy
	)
{
	int idx = IDX;
	int idy = IDY;
	//while (true) {
		short bias = 
			read_channel_intel(channel_biasInput);

		write_channel_intel(channel_peBiasInput, bias);
		//Makes sure the PE receives it first
		//mem_fence(CLK_CHANNEL_MEM_FENCE);

		if ( (0x1FF & idx) < (PE_NUM_X - 1) ) {
			write_channel_intel(channel_biasOutput, bias);
		}
	//}
}
*/

/*
__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__ ((autorun))
__kernel void kernelActivationTransport (
		//int idx,
		//int idy
	)
{
	int idx = IDX;
	int idy = IDY;
	t_activationForwarderInstruction instruction =
	 read_channel_intel(channel_activationForwarderInstructionInput);

	uint1_t proceed = 0x1;
	unsigned short targetIndex = 
	((unsigned short) idy) +  ( ( unsigned short ) instruction.activationSelectStartIndex);
	unsigned short activationIndexTracker = 0;
	unsigned char indexInStreamingBlock = 0;
	t_operand tempActivation = 0x0;

	while (proceed == 0x1) {
		t_spValueAndZCountUnpacked activationBlock = 
			read_channel_intel(channel_activationInput);
		proceed = (((activationBlock.metaInformation >> UNPACKED_ISLAST_BITOFFSET) & UNPACKED_ISLAST_MASK ) 
			== 0x0) ? 0x1 : 0x0;

		if (instruction.forwardEnable == 0x1) {
			if (instruction.mode == PE_MODE_DOT_PRODUCT) {
				write_channel_intel(channel_dpActivationInput, activationBlock);
			}

			unsigned char activaitonBlockIndexInStreamingBlock =
				((activationBlock.metaInformation >> UNPACKED_INDEX_BITOFFSET) & UNPACKED_INDEX_MASK);
			unsigned short tempTracker = (unsigned short)
				(activationIndexTracker + (unsigned char) activaitonBlockIndexInStreamingBlock);
			if (tempTracker == targetIndex) {
				tempActivation = activationBlock.nzValue;
			}
			if (activaitonBlockIndexInStreamingBlock == 63) {
				activationIndexTracker += 64;
			}
		} // if forwardEnable

		if ( (0x1FF & idy) < (PE_NUM_Y - 1) ) {
			write_channel_intel(channel_activationOutput, activationBlock);
		}
	} // while

	if (instruction.mode == PE_MODE_ELTWISE_ADD || instruction.mode == PE_MODE_LOAD_ACTIVATION
		|| instruction.mode == PE_MODE_MAX_POOL) {
		write_channel_intel(channel_pSumManagerActivationInput, tempActivation);
	}

}
*/

/*
__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__ ((autorun))
__kernel void kernelWeightTransport (
		//int idx,
		//int idy
	)
{
	

	int idx = IDX;
	int idy = IDY;

	uint1_t state = 0x0;
	t_weightForwarderInstruction instruction;

	while (true) {
		if (state == 0x0) {
			instruction =
	 			read_channel_intel(channel_weightForwarderInstructionInput);
	 		state = 0x1;
		}
		else {
			t_spValueAndZCountUnpacked weightBlock = 
				read_channel_intel(channel_weightInput);
            state = (((weightBlock.metaInformation >> UNPACKED_ISLAST_BITOFFSET) & UNPACKED_ISLAST_MASK)  == 0x0) ? 0x1 : 0x0;
			if (instruction.forwardEnable == 0x1) {
				write_channel_intel(channel_dpWeightInput, weightBlock);
			} 
			if ( (0x1FF & idx) < (PE_NUM_X - 1) ) {
				write_channel_intel(channel_weightOutput, weightBlock);
			}
		}
	}
}
*/

/*
__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__ ((autorun))
__kernel void kernelTransport ()
{
	int idx = IDX;
	int idy = IDY;

	uint1_t activationState = 0x0, 
			weightState = 0x0,
			drainState = 0x0,
			biasState = 0x0;

	t_accumulator bias;

	t_pe_prototype_instruction instruction;
	uint9_t drainCount;

	while (true) {
		if (activationState == 0x0 &&
			weightState == 0x0 &&
			drainState == 0x0 &&
			biasState == 0x0) {
			instruction =
				read_channel_intel(channel_instructionInput);
			if ( ( (0x1FF & idy) == 0 ) && ( (0x1FF & idx) < (PE_NUM_X - 1) ) ) {
				//pass right
				write_channel_intel(channel_instructionOutputHorizontal, instruction);
			}
			if ( (0x1FF & idy) < (PE_NUM_Y - 1)) {
				//pass down
				write_channel_intel(channel_instructionOutputVertical, instruction);
			}

			// Set up for activation read
			if ( idy <= instruction.maxIDY && idx <= instruction.maxIDX ) {
				activationState = 0x1;
				weightState = 0x1;
				drainState = 0x1;
				biasState = 0x1;
				//fracDin = instruction.fracDin;
				//fracW = instruction.fracW;
				//fracDout = instruction.fracDout;
				drainCount = 0;
			}
		} // if  read instruction
		else {
			if (activationState == 0x1) {
				bool readSuccess;
				t_spValueAndZCountUnpacked block = 
					read_channel_nb_intel(channel_activationInput, &readSuccess);
				if (readSuccess) {
					activationState = (((block.metaInformation >> UNPACKED_ISLAST_BITOFFSET) & UNPACKED_ISLAST_MASK ) 
				== 0x0) ? 0x1 : 0x0;
					if (idy < instruction.maxIDY) {
						write_channel_intel(channel_activationOutput, block);
					}
					EMULATOR_PRINT ( ("[kernelTransport]: Waiting to pass a activation block to the PE\n") );
					write_channel_intel(channel_dpActivationInput, block);
					//mem_fence(CLK_CHANNEL_MEM_FENCE);
				} // if read success
			} //activation state

			if (weightState == 0x1) {
				bool readSuccess;
				t_spValueAndZCountUnpacked block = 
					read_channel_nb_intel(channel_weightInput, &readSuccess);
				if (readSuccess) {
					weightState = (((block.metaInformation >> UNPACKED_ISLAST_BITOFFSET) & UNPACKED_ISLAST_MASK ) 
				== 0x0) ? 0x1 : 0x0;
					if (idx < instruction.maxIDX) {
						write_channel_intel(channel_weightOutput, block);
					}
					EMULATOR_PRINT ( ("[kernelTransport]: Waiting to pass a weight block to the PE\n") );
					write_channel_intel(channel_dpWeightInput, block);

				} // if read success
			} //weight state

			if (biasState == 0x1) {
				bool readSuccess;
				t_operand tempBias = read_channel_nb_intel(channel_biasInput, &readSuccess);
				if (readSuccess) {
					bias = ((t_accumulator) tempBias) >> ((unsigned char) (instruction.fracW + instruction.fracDin - instruction.fracDout-1));
					biasState = 0x0;
				}
			} // biasState

			if (drainState == 0x1) {
				t_operand result;
				bool success;
				if (drainCount == 0) {
					t_accumulator accumulator =
						read_channel_nb_intel(channel_peDrainOutput, &success);
					//Add bias, shift, and saturate
					if (success) {
						accumulator += bias;
						t_accumulator signExtensionMask;
						if(accumulator>=0)
							signExtensionMask = 0x00;
						else
							signExtensionMask = ~(0xFFFF>>(unsigned char) (instruction.fracW+instruction.fracDin-instruction.fracDout-1)); // ">>" is logic shift, then perform sign extension manually

						t_accumulator accumulatorWithRndBit = (signExtensionMask 
							| (accumulator >> ( (unsigned char) (instruction.fracW
								+ instruction.fracDin
								- instruction.fracDout- 1) ) ) );


						t_accumulator accumulatorBiased;
						if(accumulatorWithRndBit >=256)
							accumulatorBiased = 0x0FF; //=255
						else if(accumulatorWithRndBit <-256)
							accumulatorBiased = 0x0100; //=-256
						else
							accumulatorBiased = (0x1FF & accumulatorWithRndBit)+0x01;

						// final truncation
						result = 0xFF & (accumulatorBiased>>0x01);  // remove the last rounding bit
					}

				}
				else {
					result = read_channel_nb_intel(channel_drainInput, &success);
				}
				if (success){
					write_channel_intel(channel_drainOutput, result);
					drainCount++;
				}
				if (drainCount == instruction.maxIDY - idy + 1) {
					drainCount = 0;
					drainState = 0x0;
				}
			} //drainState
		}
	}
}
*/

__attribute__((task))
__attribute__((max_global_work_dim(0)))
__kernel void kernelTestInterface (
		__global t_spValueAndZCount* restrict pActivationInput,
		__global t_spValueAndZCountUnpackedHost* restrict pActivationOutput,
		__global t_spValueAndZCount* restrict pWeightInput,
		__global t_spValueAndZCountUnpackedHost* restrict pWeightOutput,
		__global short * restrict pBiasIn,
		__global short * restrict pBiasOut,
		__global short * restrict pDrainIn, 
		__global short * restrict pDrainOut,
		__global t_pe_prototype_instruction* restrict pInstructionInput,
		__global t_pe_prototype_instruction* restrict pInsructionOutputHorizontal,
		__global t_pe_prototype_instruction* restrict pInstructionOutputVeritcal,
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
		unsigned short numOutputInstructionVertical
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

	uint6_t		   indexActivationTracker = 0,
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
			t_spValueAndZCount value = pActivationInput[countInputActivationBlocks];
			//t_vecUnpacked unpackedValue;
			//decodeRunLength(&value, &unpackedValue, numActivationTracker);
			t_spValueAndZCountUnpacked unpackedValue;
			unpackedValue.nzValue = value & WEIGHT_MASK;
			unsigned char indexInStreamingBlock = indexActivationTracker + (uint6_t) ((value  >> WEIGHT_ZCOUNT_BITOFFSET) & WEIGHT_ZCOUNT_MASK); 
			unsigned char isLast = (countInputActivationBlocks == (numInputActivationBlocks - 1)) ?
                0x1 : 0x0;
            unpackedValue.metaInformation = (indexInStreamingBlock & UNPACKED_INDEX_MASK) << UNPACKED_INDEX_BITOFFSET;
            unpackedValue.metaInformation |= (isLast & UNPACKED_ISLAST_MASK) << UNPACKED_ISLAST_BITOFFSET;
			valid = write_channel_nb_intel (channel_activationInput, unpackedValue);
			if (valid) {
				countInputActivationBlocks++;
				//numActivationTracker = unpackedValue.indices[COMPRESSION_VEC_SIZE-1];
				indexActivationTracker = indexInStreamingBlock + 0x1;
			}
		}

		if ( (0x1FF & idy) < (PE_NUM_Y - 1) ) {
			if (countOutputActivationBlocks < numOutputActivationBlocks) {
				bool valid;
				t_spValueAndZCountUnpacked value = read_channel_nb_intel(channel_activationOutput, &valid);
				if (valid) {
					t_spValueAndZCountUnpackedHost hostValue;
					/*
					#pragma unroll
					for (unsigned char i=0; i<COMPRESSION_VEC_SIZE; i++) {
						hostValue.nzValues[i] = (short) value.nzValues[i];
						hostValue.validMasks[i] = (unsigned char) value.validMasks[i];
						hostValue.indices[i] = (unsigned short) value.indices[i];
					}
					*/
					hostValue.nzValue = (short) value.nzValue;
					hostValue.indexInStreamingBlock = (unsigned short) ((value.metaInformation >> UNPACKED_INDEX_BITOFFSET) & UNPACKED_INDEX_MASK);
					pActivationOutput[countOutputActivationBlocks++] = hostValue;
					//EMULATOR_PRINT ( ("[kernelTestInferace]: Collected %d out of %d activation blocks\n", countOutputActivationBlocks, numOutputActivationBlocks) );
				}
			}
		}

		if (countInputWeightBlocks < numInputWeightBlocks) {
			bool valid;
			t_spValueAndZCount value = pWeightInput[countInputWeightBlocks];
			//t_vecUnpacked unpackedValue;
			//decodeRunLength(&value, &unpackedValue, numWeightTracker);
			t_spValueAndZCountUnpacked unpackedValue;
			unpackedValue.nzValue = (t_operand) (value & WEIGHT_MASK);
			unsigned char indexInStreamingBlock = indexWeightTracker + (uint6_t) ((value  >> WEIGHT_ZCOUNT_BITOFFSET) & WEIGHT_ZCOUNT_MASK); 
			unsigned char isLast = (countInputWeightBlocks == (numInputWeightBlocks - 1)) ?
				0x1 : 0x0;
 			unpackedValue.metaInformation = (indexInStreamingBlock & UNPACKED_INDEX_MASK) << UNPACKED_INDEX_BITOFFSET;
            unpackedValue.metaInformation |= (isLast & UNPACKED_ISLAST_MASK) << UNPACKED_ISLAST_BITOFFSET;


			valid = write_channel_nb_intel (channel_weightInput, unpackedValue);

			if (valid) {
				countInputWeightBlocks++;
				//numWeightTracker = unpackedValue.indices[COMPRESSION_VEC_SIZE-1];
				indexWeightTracker = indexInStreamingBlock + 0x1; //Will wrap at index = 63
			}
		}

		if ( (0x1FF & idx) < (PE_NUM_X - 1) ) {
			if (countOutputWeightBlocks < numOutputWeightBlocks) {
				bool valid;
				//t_vecUnpacked value = read_channel_nb_intel(channel_weightOutput, &valid);
				t_spValueAndZCountUnpacked value = read_channel_nb_intel(channel_weightOutput, &valid);
				if (valid) {
					t_spValueAndZCountUnpackedHost hostValue;
//					#pragma unroll
//					for (unsigned char i=0; i<COMPRESSION_VEC_SIZE; i++) {
//						hostValue.nzValues[i] = (short) value.nzValues[i];
//						hostValue.validMasks[i] = (unsigned char) value.validMasks[i];
//						hostValue.indices[i] = (unsigned short) value.indices[i];
//					}
					hostValue.nzValue = (short) value.nzValue;
					hostValue.indexInStreamingBlock = (unsigned short) ((value.metaInformation >> UNPACKED_INDEX_BITOFFSET) & UNPACKED_INDEX_MASK);
					pWeightOutput[countOutputWeightBlocks++] = hostValue;
					//EMULATOR_PRINT ( ("[kernelTestInferace]: Collected %d out of %d weight blocks\n", countOutputWeightBlocks, numOutputWeightBlocks) );
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
			t_pe_prototype_instruction value = pInstructionInput [countInputInstruction];
			valid = write_channel_nb_intel (channel_instructionInput, value);
			if (valid) {
				countInputInstruction++;
			}
		}

		if ( ( (0x1FF & idy) == 0 ) && ( (0x1FF & idx) < (PE_NUM_X - 1) ) ) {
			if (countOutputInstructionHorizontal < numOutputInsructionHorizontal) {
				bool valid;
				t_pe_prototype_instruction value = read_channel_nb_intel(channel_instructionOutputHorizontal, &valid);
				if (valid) {
					pInsructionOutputHorizontal [countOutputInstructionHorizontal++] = value;
				}
			}
		}

		if ( (0x1FF & idy) < (PE_NUM_Y - 1)) {
			if (countOutputInstructionVertical < numOutputInstructionVertical) {
				bool valid;
				t_pe_prototype_instruction value = read_channel_nb_intel(channel_instructionOutputVertical, &valid);
				if (valid) {
					pInstructionOutputVeritcal [countOutputInstructionVertical++] = value;
				}
			}
		}
		
	}
}

/*
__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__ ((autorun))
__kernel void kernelPSumManager (
		  //int idx,
		  //int idy
	)
{
	
	
	//int idx = IDX;
	//int idy = IDY;

	//Register that holds the partial sum
	//Should be interpreted as a signed fixed-point number
	//Fraction width = REG_FF_FRAC. Total width = PE_FF_WIDTH
	t_accumulator pSumFF;
	//int biasReg;
	
	while (true) {

		//Instruction channel
		t_pSumManagerInstruction regInstruction
			= read_channel_intel(channel_pSumManagerInstructionInput);

		unsigned char instructionMode = regInstruction.mode;

		if (regInstruction.enable == 0x1){
			if (instructionMode == PE_MODE_MAX_POOL 
					|| instructionMode == PE_MODE_LOAD_ACTIVATION
					|| instructionMode == PE_MODE_ELTWISE_ADD) {


				t_operand activation = read_channel_intel(channel_pSumManagerActivationInput);

				t_accumulator wideValue = convertSignedFixedPointToAccumulator (
					activation,
					regInstruction.fracDin
				);
				switch (instructionMode) {
				case PE_MODE_ELTWISE_ADD:
					pSumFF = pSumFF + wideValue;
					break;
				case PE_MODE_LOAD_ACTIVATION:
					pSumFF = wideValue;
					break;
				case PE_MODE_MAX_POOL:
					pSumFF = pSumFF > wideValue ? pSumFF : wideValue;
					break;
				default:
					break;
					//Do nothing
				} //switch

								}
			else if (instructionMode == PE_MODE_DOT_PRODUCT) {
				// TODO Fill it in
				t_accumulator tempPSum = read_channel_intel(channel_pSumManagerMacInput);


				//Add
				pSumFF += (tempPSum << (unsigned char) (REG_FF_FRAC - regInstruction.fracDin - regInstruction.fracW));
			}

			else if (instructionMode == PE_MODE_LOAD_BIAS) {
				t_operand biasReg = (t_operand) (read_channel_intel(channel_peBiasInput) & WEIGHT_MASK);
				pSumFF = convertSignedFixedPointToAccumulator(biasReg, regInstruction.fracDin);
			}
			else if (instructionMode == PE_MODE_DRAIN_PSUM) {
				short value = (short)(convertAccumulatorToSignedFixedPoint (pSumFF, regInstruction.fracDout));
				//t_operand value = (t_operand) (pSumFF);
				write_channel_intel(channel_peDrainOutput, (t_operand) value);
			}
		} // if part of the valid PE range
	} // while
	

}
*/

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
	t_spValueAndZCountUnpacked activationBlock, weightBlock;
	//uint1_t proceed = 0x1;

	while (true) {
;
		uint1_t weightBlockIsLast;
		uint1_t activationBlockIsLast;
		//bool activationBlockValid = true, weightBlockValid = true;
		if (streamingBlockIndexActivation 
			<= streamingBlockIndexWeight) {
			EMULATOR_PRINT ( ("[kernelDotProductEngineDispatcher]: Waiting to read from the activation channel\n") );
			//activationBlock = read_channel_nb_intel(channel_pe2DpEngineDispatcherActivation, &activationBlockValid);
			activationBlock = read_channel_intel(channel_dpActivationInput);

		}
		//mem_fence(CLK_CHANNEL_MEM_FENCE);
		if (streamingBlockIndexActivation 
			>= streamingBlockIndexWeight) {
			EMULATOR_PRINT ( ("[kernelDotProductEngineDispatcher]: Waiting to read from the weight channel\n") );
			//weightBlock = read_channel_nb_intel(channel_peWeightInput, &weightBlockValid);
			weightBlock = read_channel_intel(channel_dpWeightInput);	
		}


/*
		uint1_t weightBlockIsLast = (streamingBlockIndexActivation < streamingBlockIndexWeight) ? 
			(oldBlock.metaInformation >> UNPACKED_ISLAST_BITOFFSET) & UNPACKED_ISLAST_MASK;
		uint1_t activationBlockIsLast = (activationBlock.metaInformation >> UNPACKED_ISLAST_BITOFFSET) & UNPACKED_ISLAST_MASK;
*/
		weightBlockIsLast = (weightBlock.metaInformation >> UNPACKED_ISLAST_BITOFFSET) & UNPACKED_ISLAST_MASK;
		activationBlockIsLast = (activationBlock.metaInformation >> UNPACKED_ISLAST_BITOFFSET) & UNPACKED_ISLAST_MASK;
		uint6_t streamingBlockIndexActivationTemp = ((activationBlock.metaInformation >> UNPACKED_INDEX_BITOFFSET) & UNPACKED_INDEX_MASK);
		uint6_t streamingBlockIndexWeightTemp = ((weightBlock.metaInformation >> UNPACKED_INDEX_BITOFFSET) & UNPACKED_INDEX_MASK);

		uint1_t isLast = ((weightBlockIsLast == 0x1) && (activationBlockIsLast == 0x1)) ?
			0x1 : 0x0;

		streamingBlockIndexActivation = (isLast == 0x1) ? 0 : streamingBlockIndexActivationTemp;
		streamingBlockIndexWeight = (isLast == 0x1) ? 0 : streamingBlockIndexWeightTemp;

		EMULATOR_PRINT ( ("[kernelDotProductEngineDispatcher]: streamingBlockIndexActivation updated to %d\n", streamingBlockIndexActivation) );
		EMULATOR_PRINT ( ("[kernelDotProductEngineDispatcher]: streamingBlockIndexWeight updated to %d\n", streamingBlockIndexWeight) );
		

		//if (activationBlockValid && weightBlockValid) {
		if (streamingBlockIndexWeight == streamingBlockIndexActivation) {
			t_macOperands multData;

			multData.nzWeight = weightBlock.nzValue;
			multData.nzActivation = activationBlock.nzValue;
			multData.isLast = isLast;

			write_channel_intel(channel_macOperandsInput, multData);
		}
		
	} // while
	
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void mac () 
{
	t_accumulator pSum = 0;

/*
	uint1_t state = 0x0;

	while (true) {
		if (state == 0x0) {
			t_macOperands multData = read_channel_intel(channel_macOperandsInput);

			t_operand weight = multData.nzWeight;
			t_operand activation = multData.nzActivation;
			uint1_t isLast = multData.isLast;

			pSum += weight * activation;

			if (isLast == 0x1) {
				state = 0x1;
			}
		}
		else {
			write_channel_intel(channel_pSumManagerMacInput, pSum);
			pSum = 0;
			state = 0x0;
		}
	}
*/

	while (true) {
		t_macOperands multData = read_channel_intel(channel_macOperandsInput);

		t_operand weight = multData.nzWeight;
		t_operand activation = multData.nzActivation;
		uint1_t isLast = multData.isLast;

		pSum += weight * activation;

		if (isLast == 0x1) {
			write_channel_intel(channel_peDrainOutput, pSum);
			pSum = 0;
			EMULATOR_PRINT ( ("[kernelMAC]: Committed!\n") );
		}
	}

/*
	uint1_t proceed = 0x1;
	while (proceed == 0x1) {

		//Wait for the operands to arrive
		t_macOperands multData = read_channel_intel(channel_macOperandsInput);

		t_operand weight = multData.nzWeight;
		t_operand activation = multData.nzActivation;
		uint1_t isLast = multData.isLast;

		pSum += weight * activation;
		proceed = (isLast == 0x0) ? 0x1 : 0x0;
	}
	write_channel_intel(channel_pSumManagerMacInput, pSum);
*/
}
