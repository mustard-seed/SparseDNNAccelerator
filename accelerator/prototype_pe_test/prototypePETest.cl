#include "params.hpp"
#include "device_structures.hpp"
#include "ihc_apint.h"
#include "device_utils.hpp"
#include "prototypePE_structs.hpp"
#include "peComponents.hpp"


#define PE_NUM_X PE_COLS
#define PE_NUM_Y PE_ROWS


typedef struct __attribute__((packed)) {
	uint9_t activationSelectStartIndex;

  	// Mode of the instruction
  	unsigned char mode;

  	//Whether to forward teh activation block to the compute unit / pSum manager
  	uint1_t forwardEnable;
} t_activationForwarderInstruction;

typedef struct __attribute__((packed)) {
	// PEs with id that satisfies 0<=idx<=maxIDX and -<=idy<=maxIDY will participate 
  	uint2_t forwardEnable;  //uint1_t can't compile under emulation mode
} t_weightForwarderInstruction;

typedef struct __attribute__((packed)) {
	unsigned char mode;
	// Number of bits assigned to the fraction width
  	char fracW;
  	char fracDin;
  	char fracDout;
  	uint1_t enable;
} t_pSumManagerInstruction;

typedef struct __attribute__((packed)) {
	t_operand nzWeight;
	t_operand nzActivation;
	uint1_t isLast;
} t_macOperands;



channel t_spValueAndZCountUnpacked channel_activationInput __attribute__((depth(1)));
channel t_spValueAndZCountUnpacked channel_activationOutput __attribute__((depth(1)));
channel t_spValueAndZCountUnpacked channel_weightInput __attribute__((depth(1)));
channel t_spValueAndZCountUnpacked channel_weightOutput __attribute__((depth(1)));
channel short channel_biasInput __attribute__((depth(1)));
channel short channel_biasOutput __attribute__((depth(1)));
channel short channel_drainInput __attribute__((depth(1)));
channel short channel_drainOutput __attribute__((depth(1)));
channel t_pe_prototype_instruction channel_instructionInput __attribute__((depth(1)));
channel t_pe_prototype_instruction channel_instructionOutputVertical __attribute__((depth(1)));
channel t_pe_prototype_instruction channel_instructionOutputHorizontal __attribute__((depth(1)));

channel short channel_peBiasInput __attribute__((depth(0)));
channel t_pSumManagerInstruction channel_pSumManagerInstructionInput __attribute__((depth(0)));
channel t_activationForwarderInstruction channel_activationForwarderInstructionInput __attribute__((depth(0)));
channel t_weightForwarderInstruction channel_weightForwarderInstructionInput __attribute__((depth(0)));

//channel t_spValueAndZCountUnpacked channel_peWeightInput __attribute__((depth(0)));
//channel t_spValueAndZCountUnpacked channel_peActivationInput __attribute__((depth(0)));

channel t_spValueAndZCountUnpacked channel_dpWeightInput __attribute__((depth(PE_VEC_FIFO_SIZE)));
channel t_spValueAndZCountUnpacked channel_dpActivationInput __attribute__((depth(PE_VEC_FIFO_SIZE)));
channel t_operand channel_pSumManagerActivationInput __attribute__((depth(0)));
channel t_macOperands channel_macOperandsInput __attribute__((depth(0)));
channel t_accumulator channel_pSumManagerMacInput __attribute__((depth(0)));

channel short channel_peDrainOutput __attribute__((depth(0)));





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
	uint9_t i=0;
	while (true) {
		//for (uint9_t i=0; 
		//	i <= ((uint9_t) ( (0x1FF & PE_NUM_Y) - (0x1FF & idy) - 1)); 
		//	i++) {
		short pePsum;
		if (i==0) {
			EMULATOR_PRINT ( ("[kernelDrainTransport]: Waiting to read the pe's psum.\n") );
			pePsum = read_channel_intel(channel_peDrainOutput);
		}
		else {
			EMULATOR_PRINT ( ("[kernelDrainTransport]: Waiting to read other psums.\n") );
			pePsum = read_channel_intel(channel_drainInput);
		}
		EMULATOR_PRINT ( ("[kernelDrainTransport]: Waiting to write the psum.\n") );
		write_channel_intel(channel_drainOutput, pePsum);
		if (i == ((uint9_t) ( (0x1FF & PE_NUM_Y) - (0x1FF & idy) - 1)) ) {
			i=0;
		}
		else {
			i++;
		}
		//}
	}
}

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
	//while (true) {
		t_spValueAndZCountUnpacked activationBlock = 
			read_channel_intel(channel_activationInput);

		write_channel_intel(channel_peActivationInput, activationBlock);
		//Makes sure the PE receives it first
		//mem_fence(CLK_CHANNEL_MEM_FENCE);

		if ( (0x1FF & idy) < (PE_NUM_Y - 1) ) {
			write_channel_intel(channel_activationOutput, activationBlock);
		}
	//}
}
*/

__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__ ((autorun))
__kernel void kernelActivationForwarder (
		//int idx,
		//int idy
	)
{
	int idx = IDX;
	int idy = IDY;
	/*
	uint2_t state = 0x0;
	t_activationForwarderInstruction instruction;
	unsigned short targetIndex;
	unsigned short activationIndexTracker;
	unsigned char indexInStreamingBlock;
	t_operand tempActivation;
	while (true) {
		if (state == 0x0) {
			instruction = read_channel_intel(channel_activationForwarderInstructionInput);
			targetIndex = ((unsigned short) idy) +  ( ( unsigned short ) instruction.activationSelectStartIndex);
			activationIndexTracker = 0;
			indexInStreamingBlock = 0;
			tempActivation = 0x0;
			state = 0x1;
		}
		else if (state == 0x1) {
			t_spValueAndZCountUnpacked activationBlock = read_channel_intel(channel_peActivationInput);

			if (instruction.forwardEnable == 0x1) {
				if (instruction.mode == PE_MODE_DOT_PRODUCT) {
					write_channel_intel(channel_dpActivationInput, activationBlock);
				}

				unsigned short tempTracker = (activationIndexTracker + (unsigned char) activationBlock.indexInStreamingBlock);
				if (tempTracker == targetIndex) {
					tempActivation = activationBlock.nzValue;
				}
				if (activationBlock.indexInStreamingBlock == 63) {
					activationIndexTracker += 64;
				}
			} // if forwardEnable

			if (activationBlock.isLast == 0x1) {
				state = 0x2;
			}
		}
		else {
			if (instruction.mode == PE_MODE_ELTWISE_ADD || instruction.mode == PE_MODE_LOAD_ACTIVATION
					|| instruction.mode == PE_MODE_MAX_POOL) {
						write_channel_intel(channel_pSumManagerActivationInput, tempActivation);
			}
			state = 0x0;
		}
	}
*/
//	while (true) {
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
			proceed = (activationBlock.isLast == 0x0) ?
				0x1 : 0x0;

			if (instruction.forwardEnable == 0x1) {
				if (instruction.mode == PE_MODE_DOT_PRODUCT) {
					write_channel_intel(channel_dpActivationInput, activationBlock);
				}

				unsigned short tempTracker = (activationIndexTracker + (unsigned char) activationBlock.indexInStreamingBlock);
				if (tempTracker == targetIndex) {
					tempActivation = activationBlock.nzValue;
				}
				if (activationBlock.indexInStreamingBlock == 63) {
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
//	}

}

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
	//while (true) {
		t_spValueAndZCountUnpacked weightBlock = 
			read_channel_intel(channel_weightInput);

		write_channel_intel(channel_peWeightInput, weightBlock);
		//Makes sure the PE receives it first
		//mem_fence(CLK_CHANNEL_MEM_FENCE);

		if ( (0x1FF & idx) < (PE_NUM_X - 1) ) {
			write_channel_intel(channel_weightOutput, weightBlock);
		}
	//}
}
*/

__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__ ((autorun))
__kernel void kernelWeightForwarder (
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
            state = (weightBlock.isLast == 0x0) ? 0x1 : 0x0;
			if (instruction.forwardEnable == 0x1) {
				write_channel_intel(channel_dpWeightInput, weightBlock);
			} 
			if ( (0x1FF & idx) < (PE_NUM_X - 1) ) {
				write_channel_intel(channel_weightOutput, weightBlock);
			}
		}
	}

/*	t_weightForwarderInstruction instruction =
	 read_channel_intel(channel_weightForwarderInstructionInput);

	uint1_t proceed = 0x1;

	while (proceed == 0x1) {
		t_spValueAndZCountUnpacked weightBlock = 
			read_channel_intel(channel_peWeightInput);
		proceed = (weightBlock.isLast == 0x0) ?
			0x1 : 0x0;

		if (instruction.forwardEnable == 0x1) {
			write_channel_intel(channel_dpWeightInput, weightBlock);
		} // if forwardEnable
	} // while
*/
}




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
			unpackedValue.indexInStreamingBlock = indexActivationTracker + (uint6_t) ((value & WEIGHT_ZCOUNT_MASK) >> WEIGHT_ZCOUNT_BITOFFSET); 
			unpackedValue.isLast = (countInputActivationBlocks == (numInputActivationBlocks - 1)) ?
                0x1 : 0x0;
			valid = write_channel_nb_intel (channel_activationInput, unpackedValue);
			if (valid) {
				countInputActivationBlocks++;
				//numActivationTracker = unpackedValue.indices[COMPRESSION_VEC_SIZE-1];
				indexActivationTracker = unpackedValue.indexInStreamingBlock  + 0x1;
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
					hostValue.indexInStreamingBlock = (unsigned short) value.indexInStreamingBlock;
					pActivationOutput[countOutputActivationBlocks++] = hostValue;
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
			unpackedValue.indexInStreamingBlock = indexWeightTracker + (uint6_t) ((value & WEIGHT_ZCOUNT_MASK) >> WEIGHT_ZCOUNT_BITOFFSET);
			unpackedValue.isLast = (countInputWeightBlocks == (numInputWeightBlocks - 1)) ?
				0x1 : 0x0;
			valid = write_channel_nb_intel (channel_weightInput, unpackedValue);
			if (valid) {
				countInputWeightBlocks++;
				//numWeightTracker = unpackedValue.indices[COMPRESSION_VEC_SIZE-1];
				indexWeightTracker = unpackedValue.indexInStreamingBlock  + 0x1; //Will wrap at index = 63
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
					hostValue.indexInStreamingBlock = (unsigned short) value.indexInStreamingBlock;
					pWeightOutput[countOutputWeightBlocks++] = hostValue;
				}
			}
		}

		if (countInputBias < numInputBias) {
			bool valid;
			short value = pBiasIn[countInputBias];
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

		uint1_t isLast = ((weightBlock.isLast == 0x1) && (activationBlock.isLast == 0x1)) ?
			0x1 : 0x0;

		streamingBlockIndexActivation = (isLast == 0x1) ? 0 : activationBlock.indexInStreamingBlock;
		streamingBlockIndexWeight = (isLast == 0x1) ? 0 : weightBlock.indexInStreamingBlock;

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
			write_channel_intel(channel_pSumManagerMacInput, pSum);
			pSum = 0;
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
