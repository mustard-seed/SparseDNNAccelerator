#include "params.hpp"
#include "device_structures.hpp"
#include "ihc_apint.h"
#include "device_utils.hpp"
#include "prototypePE_structs.hpp"
#include "peComponents.hpp"


#define READ_ACTIVATION_CONDITION 0X0
#define WRITE_ACTIVATION_CONDITION 0x1
#define READ_WEIGHT_CONDITION 0x2
#define WRITE_WEIGHT_CONDITION 0x3
#define READ_BIAS_CONDITION 0X0
#define WRITE_BIAS_CONDITION 0x1
#define READ_DRAIN_CONDITION 0X0
#define WRITE_DRAIN_CONDITION 0x1

#define PE_CONTROLLER_STATE_IDLE 0X0
#define PE_CONTROLLER_STATE_RUN 0X01
#define PE_CONTROLLER_STATE_END 0x02
#define PE_COTNROLLER_STATE_LOAD_BIAS 0x03
#define PE_CONTROLLER_STATE_LOAD_ACTIVATION 0X04
#define PE_CONTROLLER_STATE_ELTWISE_ADD 0x05
#define PE_CONTROLLER_STATE_DRAIN_PSUM 0x06
#define PE_CONTROLLER_STATE_MAX_POOL 0x07
#define PE_CONTROLLER_STATE_DOT_PRODUCT 0X08

#define PE_NUM_X PE_COLS
#define PE_NUM_Y PE_ROWS

channel t_vecUnpacked channel_activationInput __attribute__((depth(1)));
channel t_vecUnpacked channel_activationOutput __attribute__((depth(1)));
channel t_vecUnpacked channel_weightInput __attribute__((depth(1)));
channel t_vecUnpacked channel_weightOutput __attribute__((depth(1)));
channel short channel_biasInput __attribute__((depth(1)));
channel short channel_biasOutput __attribute__((depth(1)));
channel short channel_drainInput __attribute__((depth(1)));
channel short channel_drainOutput __attribute__((depth(1)));
channel t_pe_prototype_instruction channel_instructionInput __attribute__((depth(1)));
channel t_pe_prototype_instruction channel_instructionOutputVertical __attribute__((depth(1)));
channel t_pe_prototype_instruction channel_instructionOutputHorizontal __attribute__((depth(1)));

channel t_vecUnpacked channel_peActivationInput __attribute__((depth(1)));
channel t_vecUnpacked channel_peWeightInput __attribute__((depth(1)));
channel short channel_peBiasInput __attribute__((depth(1)));
channel short channel_peDrainOutput __attribute__((depth(1)));
channel t_pe_prototype_instruction channel_peInstruction __attribute__((depth(1)));

channel t_vecUnpacked channel_pe2DpEngineDispatcherActivation __attribute__((depth(PE_VEC_FIFO_SIZE)));
channel t_vecUnpacked channel_pe2DpEngineDispatcherWeight __attribute__((depth(PE_VEC_FIFO_SIZE)));
channel int channel_dpEngineDispatcher2PePSum __attribute__((depth(0)));
channel t_dpInstruction channel_dpInstruction __attribute__((depth(0)));

channel t_vecMultData channel_dpEngineDispatcher2MultOperands [PE_NUM_MULT] __attribute__((depth(0)));
channel int channel_mult2DpEngineDispatcherPSum [PE_NUM_MULT] __attribute__((depth(0)));



__attribute__((task))
__attribute__((max_global_work_dim(0)))
__kernel void kernelDrainTransport (
		int idx,
		int idy
	)
{
	while (true) {
		for (uint9_t i=0; 
			i <= ((uint9_t) ( (0x1FF & PE_NUM_Y) - (0x1FF & idy) - 1)); 
			i++) {
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
		}
	}
}

__attribute__((task))
__attribute__((max_global_work_dim(0)))
__kernel void kernelInstructionTransport (
		int idx,
		int idy
	)
{
	while (true) {
		t_pe_prototype_instruction instruction = 
			read_channel_intel(channel_instructionInput);

		write_channel_intel(channel_peInstruction, instruction);
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
	}
}

__attribute__((task))
__attribute__((max_global_work_dim(0)))
__kernel void kernelBiasTransport (
		int idx,
		int idy
	)
{
	while (true) {
		short bias = 
			read_channel_intel(channel_biasInput);

		write_channel_intel(channel_peBiasInput, bias);
		//Makes sure the PE receives it first
		//mem_fence(CLK_CHANNEL_MEM_FENCE);

		if ( (0x1FF & idx) < (PE_NUM_X - 1) ) {
			write_channel_intel(channel_biasOutput, bias);
		}
	}
}

__attribute__((task))
__attribute__((max_global_work_dim(0)))
__kernel void kernelActivationTransport (
		int idx,
		int idy
	)
{
	while (true) {
		t_vecUnpacked activationBlock = 
			read_channel_intel(channel_activationInput);

		write_channel_intel(channel_peActivationInput, activationBlock);
		//Makes sure the PE receives it first
		//mem_fence(CLK_CHANNEL_MEM_FENCE);

		if ( (0x1FF & idy) < (PE_NUM_Y - 1) ) {
			write_channel_intel(channel_activationOutput, activationBlock);
		}
	}
}

__attribute__((task))
__attribute__((max_global_work_dim(0)))
__kernel void kernelWeightTransport (
		int idx,
		int idy
	)
{
	while (true) {
		t_vecUnpacked weightBlock = 
			read_channel_intel(channel_weightInput);

		write_channel_intel(channel_peWeightInput, weightBlock);
		//Makes sure the PE receives it first
		//mem_fence(CLK_CHANNEL_MEM_FENCE);

		if ( (0x1FF & idx) < (PE_NUM_X - 1) ) {
			write_channel_intel(channel_weightOutput, weightBlock);
		}
	}
}




__attribute__((task))
__attribute__((max_global_work_dim(0)))
__kernel void kernelTestInterface (
		__global t_vecSpValueAndZCount* restrict pActivationInput,
		__global t_vecUnpackedHost* restrict pActivationOutput,
		__global t_vecSpValueAndZCount* restrict pWeightInput,
		__global t_vecUnpackedHost* restrict pWeightOutput,
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
				   countOutputInstructionHorizontal = 0,
				   numActivationTracker = startIndexActivationBlocks,
				   numWeightTracker = startIndexWeightBlocks;


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
			t_vecSpValueAndZCount value = pActivationInput[countInputActivationBlocks];
			t_vecUnpacked unpackedValue;
			decodeRunLength(&value, &unpackedValue, numActivationTracker);
			valid = write_channel_nb_intel (channel_activationInput, unpackedValue);
			if (valid) {
				countInputActivationBlocks++;
				numActivationTracker = unpackedValue.indices[COMPRESSION_VEC_SIZE-1];
			}
		}

		if (countOutputActivationBlocks < numOutputActivationBlocks) {
			bool valid;
			t_vecUnpacked value = read_channel_nb_intel(channel_activationOutput, &valid);
			if (valid) {
				t_vecUnpackedHost hostValue;
				#pragma unroll
				for (unsigned char i=0; i<COMPRESSION_VEC_SIZE; i++) {
					hostValue.nzValues[i] = (short) value.nzValues[i];
					hostValue.validMasks[i] = (unsigned char) value.validMasks[i];
					hostValue.indices[i] = (unsigned short) value.indices[i];
				}
				pActivationOutput[countOutputActivationBlocks++] = hostValue;
			}
		}

		if (countInputWeightBlocks < numInputWeightBlocks) {
			bool valid;
			t_vecSpValueAndZCount value = pWeightInput[countInputWeightBlocks];
			t_vecUnpacked unpackedValue;
			decodeRunLength(&value, &unpackedValue, numWeightTracker);
			valid = write_channel_nb_intel (channel_weightInput, unpackedValue);
			if (valid) {
				countInputWeightBlocks++;
				numWeightTracker = unpackedValue.indices[COMPRESSION_VEC_SIZE-1];
			}
		}

		if (countOutputWeightBlocks < numOutputWeightBlocks) {
			bool valid;
			t_vecUnpacked value = read_channel_nb_intel(channel_weightOutput, &valid);
			if (valid) {
				t_vecUnpackedHost hostValue;
				#pragma unroll
				for (unsigned char i=0; i<COMPRESSION_VEC_SIZE; i++) {
					hostValue.nzValues[i] = (short) value.nzValues[i];
					hostValue.validMasks[i] = (unsigned char) value.validMasks[i];
					hostValue.indices[i] = (unsigned short) value.indices[i];
				}
				pWeightOutput[countOutputWeightBlocks++] = hostValue;
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

		if (countOutputBias < numOutputBias) {
			bool valid;
			short value = read_channel_nb_intel(channel_biasOutput, &valid);
			if (valid) {
				pBiasOut[countOutputBias++] = value;
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
				pDrainOut [countOutputDrain++] = value;
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

		if (countOutputInstructionHorizontal < numOutputInsructionHorizontal) {
			bool valid;
			t_pe_prototype_instruction value = read_channel_nb_intel(channel_instructionOutputHorizontal, &valid);
			if (valid) {
				pInsructionOutputHorizontal [countOutputInstructionHorizontal++] = value;
			}
		}

		if (countOutputInstructionVertical < numOutputInstructionVertical) {
			bool valid;
			t_pe_prototype_instruction value = read_channel_nb_intel(channel_instructionOutputVertical, &valid);
			if (valid) {
				pInstructionOutputVeritcal [countOutputInstructionVertical++] = value;
			}
		}
	}
}

__attribute__((task))
__attribute__((max_global_work_dim(0)))
__kernel void kernelPrototypePE (
		  int idx,
		  int idy
	)
{
	



	//Register that holds the partial sum
	//Should be interpreted as a signed fixed-point number
	//Fraction width = REG_FF_FRAC. Total width = PE_FF_WIDTH
	int pSumFF = 0;
	
	while (true) {

		t_operand biasReg, scalarActivationReg;
		int dotProductReg;

		// dense index tracker
		unsigned short numActivationTracker;
		unsigned short finalActivationIndex;
		unsigned short numWeightTracker;
		unsigned short finalWeightIndex;

		unsigned short targetIndex;

		//Instruction channel
		t_pe_prototype_instruction regInstruction
			= read_channel_intel(channel_peInstruction);

		unsigned char instructionMode = regInstruction.mode;

		if (instructionMode == PE_MODE_MAX_POOL 
				|| instructionMode == PE_MODE_LOAD_ACTIVATION
				|| instructionMode == PE_MODE_ELTWISE_ADD
				|| instructionMode == PE_MODE_DOT_PRODUCT) {
			numActivationTracker = regInstruction.transmissionStartIndex;
			finalActivationIndex = regInstruction.transmissionEndIndex;
			targetIndex = 0x1FFFF & ( (0xFFFF & idy) +  (0xFFFF & regInstruction.selectStartIndex));

			if (instructionMode == PE_MODE_DOT_PRODUCT) {
				numWeightTracker = regInstruction.transmissionStartIndex;
				finalWeightIndex = regInstruction.transmissionEndIndex;

				t_dpInstruction dpInstruction;
				dpInstruction.lastIndexWeight = regInstruction.transmissionEndIndex;
				dpInstruction.lastIndexActivation = regInstruction.transmissionEndIndex;
				write_channel_intel(channel_dpInstruction, dpInstruction);
				mem_fence(CLK_CHANNEL_MEM_FENCE);
			}
			else {
				numWeightTracker = 0x01;
				finalActivationIndex = 0x0;
			}


			//Activation block update
			while (
			( (0x1FFFF & numWeightTracker) < (0x1FFFF & ((0xFFFF & finalWeightIndex) + (0xFFFF & 1) )) )
			|| ( (0x1FFFF & numActivationTracker) <  (0x1FFFF & ( (0xFFFF & finalActivationIndex) + (0xFFFF & 1) )) ) ) {
				//Activation
				if ( numActivationTracker < (0x1FFFF & ( (0xFFFF & finalActivationIndex) + 1)) ) {
					bool valid;
					t_vecUnpacked activationBlock = read_channel_nb_intel(channel_peActivationInput, &valid);
					if (valid) {
						if ( ( ((uint9_t) idx) <= ((uint9_t) regInstruction.maxIDX) ) && ( ((uint9_t) idy) <= ((uint9_t) regInstruction.maxIDY)) ) {
							if (instructionMode == PE_MODE_MAX_POOL
								|| instructionMode == PE_MODE_LOAD_ACTIVATION 
								|| instructionMode == PE_MODE_ELTWISE_ADD) {
								t_operand value;
								bool valueFound = 
									findMatchInUnpackedBlock(
											&activationBlock,
											targetIndex,
											&value
										);
								if (valueFound) {
									scalarActivationReg = value;
								} //if (valueFound)
							}
							else if (instructionMode == PE_MODE_DOT_PRODUCT) {
								// TODO Fill it in
								write_channel_intel(channel_pe2DpEngineDispatcherActivation, activationBlock);
							}
						} // if part of the valid PE range

						numActivationTracker = activationBlock.indices[COMPRESSION_VEC_SIZE-1];
					}
				} // if do something with activaiton

				//Weight
				if ( numWeightTracker < (finalWeightIndex + 1) ) {
					bool valid;
					t_vecUnpacked weightBlock = read_channel_nb_intel(channel_peWeightInput, &valid);
					if (valid) {
						if ( ( (0x1FF & idx) <= (0x1FF & regInstruction.maxIDX) ) && ( (0x1FF & idy) <= (0x1FF & regInstruction.maxIDY)) ) {
							if (instructionMode == PE_MODE_DOT_PRODUCT) {
								// TODO: Fill it in
								write_channel_intel(channel_pe2DpEngineDispatcherWeight, weightBlock);
							}
						}
						numWeightTracker = weightBlock.indices[COMPRESSION_VEC_SIZE-1];
					} // if valid
				} // if do something with weight
			} // while loop and weight and activation

			//TODO: Read value from the dot product engine during the DP mode.


		}
		
		if (instructionMode == PE_MODE_LOAD_BIAS) {
			biasReg = (t_operand) (read_channel_intel(channel_peBiasInput) & WEIGHT_MASK);
		}
		
		if (instructionMode == PE_MODE_DRAIN_PSUM) {
			short value = (short) convertAccumulatorToSignedFixedPoint (pSumFF, regInstruction.fracDout);
			write_channel_intel(channel_peDrainOutput, value);
		}

/*
		if (instructionMode == PE_MODE_DOT_PRODUCT) {
			numWeightTracker = regInstruction.transmissionStartIndex;
			finalWeightIndex = regInstruction.transmissionEndIndex;
		}
		else {
			numWeightTracker = 0x01;
			finalActivationIndex = 0x0;
		}
*/

		//Decide on whether to read from the input bias channel
/*
		bool inputBiasChannelReadEnable = 
			(instructionMode == PE_MODE_LOAD_BIAS);
		if (inputBiasChannelReadEnable) {
			biasReg = (t_operand) (read_channel_intel(channel_peBiasInput) & WEIGHT_MASK);

		}
*/

		//Decide on whether to writ eto the output drain channel
/*
		bool outputDrainChannelWriteEnable = 
			(instructionMode == PE_MODE_DRAIN_PSUM);
		if (outputDrainChannelWriteEnable) {
			short value = (short) convertAccumulatorToSignedFixedPoint (pSumFF, regInstruction.fracDout);
			write_channel_intel(channel_peDrainOutput, value);
		}
*/

		//TODO: Write the PSUM to the DP engine during the DP mode
/*
		// weight and activation channel handling
		while (
			(numWeightTracker < (finalWeightIndex + 1) )
			|| (numActivationTracker < (finalActivationIndex + 1))
			) {
			//Activation
			if ( numActivationTracker < (finalActivationIndex + 1) ) {
				bool valid;
				t_vecUnpacked activationBlock = read_channel_nb_intel(channel_peActivationInput, &valid);
				if (valid) {
					if ( ( ((uint9_t) idx) <= ((uint9_t) regInstruction.maxIDX) ) && ( ((uint9_t) idy) <= ((uint9_t) regInstruction.maxIDY)) ) {
						if (instructionMode == PE_MODE_MAX_POOL
							|| instructionMode == PE_MODE_LOAD_ACTIVATION 
							|| instructionMode == PE_MODE_ELTWISE_ADD) {
							t_operand value;
							bool valueFound = 
								findMatchInUnpackedBlock(
										&activationBlock,
										targetIndex,
										&value
									);
							if (valueFound) {
								scalarActivationReg = value;
							} //if (valueFound)
						}
						else if (instructionMode == PE_MODE_DOT_PRODUCT) {
							// Fill it in
						}
					} // if part of the valid PE range

					numActivationTracker = activationBlock.indices[COMPRESSION_VEC_SIZE-1];
				}
			} // if do something with activaiton



			//Weight
			if ( numWeightTracker < (finalWeightIndex + 1) ) {
				bool valid;
				t_vecUnpacked weightBlock = read_channel_nb_intel(channel_peWeightInput, &valid);
				if (valid) {
					if ( ( ((uint9_t) idx) <= ((uint9_t) regInstruction.maxIDX) ) && ( ((uint9_t) idy) <= ((uint9_t) regInstruction.maxIDY)) ) {
						if (instructionMode == PE_MODE_DOT_PRODUCT) {
							// Fill it in
						}
					}
					numWeightTracker = weightBlock.indices[COMPRESSION_VEC_SIZE-1];
				} // if valid
			} // if do something with weight
		} // weight and activaiton while toop
*/

		//TODO: Read value from the dot product engine during the DP mode.

		//pSum reg update
		if ( ( (0x1FF & idx) <= (0x1FF & regInstruction.maxIDX) ) && ( (0x1FF & idy) <= (0x1FF & regInstruction.maxIDY)) ) {
			if (instructionMode == PE_MODE_LOAD_BIAS) {				
				pSumFF = convertSignedFixedPointToAccumulator(biasReg, regInstruction.fracDin);
			}
			else if (instructionMode == PE_MODE_MAX_POOL
						|| instructionMode == PE_MODE_LOAD_ACTIVATION
						|| instructionMode == PE_MODE_ELTWISE_ADD) {

				int wideValue = convertSignedFixedPointToAccumulator (
									scalarActivationReg,
									regInstruction.fracDin
								);
				switch (instructionMode) {
								case PE_MODE_ELTWISE_ADD:
									pSumFF += wideValue;
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
			} // else if 
			else if (instructionMode == PE_MODE_DOT_PRODUCT) {
				//fill it in
				int tempPSum = read_channel_intel(channel_dpEngineDispatcher2PePSum);

				//Align the binary point
				int alignedPSum = tempPSum << (REG_FF_FRAC - regInstruction.fracDin - regInstruction.fracW);

				//Add
				pSumFF += alignedPSum;
			}
		} // if

	}
}

/*! \brief Dot product kernel that operates on compressed sparse weight and activation

*/

__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void kernelDotProductEngineDispatcher (
	)
{
	/*
	1. Load psum, and the final weight index and the activation index
	2. Keep loading activation and weight blocks, pick out matching pairs, and perform MAC

	*/
	unsigned short numActivationTracker = 0xFFFF & 0, numWeightTracker = 0xFFFF & 0;
	t_vecUnpacked activationBlock, weightBlock;

	//Read the number stop limit of the weight and activation
	t_dpInstruction instruction = read_channel_intel(channel_dpInstruction);
	unsigned short numActivationUpperBound = (0x1FFFF & ( (0xFFFF & instruction.lastIndexWeight) + (0XFFFF & 1) ));
	unsigned short numWeightUpperBound = (0x1FFFF & ( (0xFFFF & instruction.lastIndexActivation) + (0XFFFF & 1) ));
	uint5_t dispatchIndex = 0;
	int pSum = 0;

#pragma ii PE_NUM_MULT
	while ( (numActivationTracker < numActivationUpperBound )
		||  (numWeightTracker < numWeightUpperBound ) ) {
		if (numActivationTracker <= numWeightTracker) {
			activationBlock = read_channel_intel(channel_pe2DpEngineDispatcherActivation);
		}

		if (numWeightTracker <= numActivationTracker) {
			weightBlock = read_channel_intel(channel_pe2DpEngineDispatcherWeight);
		}

		numActivationTracker = 0xFFFF & activationBlock.indices[COMPRESSION_VEC_SIZE-1];
		numWeightTracker = 0xFFFF & weightBlock.indices[COMPRESSION_VEC_SIZE-1];

		t_vecMultData multData;

		#pragma unroll 
		for (uint5_t i=0; i<COMPRESSION_VEC_SIZE; i++) {
			multData.weightNzValues[i] = weightBlock.nzValues[i];
			multData.weightValidMasks[i] = weightBlock.validMasks[i];
			multData.weightIndices[i] = weightBlock.indices[i];
			multData.activationNzValues[i] = activationBlock.nzValues[i];
			multData.activationValidMasks[i] = activationBlock.validMasks[i];
			multData.activationIndices[i] = activationBlock.indices[i];
		}

		//Dispatch to the multiplier engine
		bool sendSuccess = false;
		while (!sendSuccess) {
			#pragma unroll
			for (uint5_t multIndex=0; multIndex < PE_NUM_MULT; multIndex++) {
				if (multIndex == dispatchIndex) {
					sendSuccess = write_channel_nb_intel(channel_dpEngineDispatcher2MultOperands[multIndex], multData);
				}
			}
		}
		dispatchIndex++;

		//Retrieve the result and perform addition
		if ( (dispatchIndex == PE_NUM_MULT)
				|| ( (numActivationTracker == numActivationUpperBound) && (numWeightTracker == numWeightUpperBound) )
			) {
			for (uint5_t i=0; i<dispatchIndex; i++) { //Retreive each sum
				bool receiveSuccess = false;
				while (!receiveSuccess) { // Keep trying to retrieve each sum
					#pragma unroll
					for (uint5_t multIndex=0; multIndex < PE_NUM_MULT; multIndex++) {
						if (multIndex == i) {
							int sum = read_channel_nb_intel(channel_mult2DpEngineDispatcherPSum[multIndex], &receiveSuccess);
							pSum += sum;
						}
					}
				} // Keep trying to retrieve each sum
			} //Retreive each sum
			dispatchIndex = 0;
		} //Retrieve the result and perform addition
	} // while

	//Send the Psum back
	write_channel_intel(channel_dpEngineDispatcher2PePSum, pSum);
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(PE_NUM_MULT)))
__kernel void peMult () 
{
	int id = get_compute_id(0);
	int pSum = 0;

	//Wait for the operands to arrive
	t_vecMultData multData = read_channel_intel(channel_dpEngineDispatcher2MultOperands[id]);

	//Unpack the data
	t_operand activationValues [COMPRESSION_VEC_SIZE];
	t_operand weightValues [COMPRESSION_VEC_SIZE];
	uint1_t activationValidMasks [COMPRESSION_VEC_SIZE];
	uint1_t weightValidMasks [COMPRESSION_VEC_SIZE];
	unsigned short activationNum [COMPRESSION_VEC_SIZE];
	unsigned short weightNum [COMPRESSION_VEC_SIZE];
	#pragma unroll
	for (uint5_t i=0; i<COMPRESSION_VEC_SIZE; i++) {
		activationValues[i] = multData.activationNzValues[i];
		activationValidMasks[i] = multData.activationValidMasks[i];
		activationNum[i] = multData.activationIndices[i];
		weightValues[i] = multData.weightNzValues[i];
		weightValidMasks[i] = multData.weightValidMasks[i];
		weightNum[i] = multData.weightIndices[i];
	}

	//Calculate the mutual masks

	uint1_t activationMutualMask[COMPRESSION_VEC_SIZE];
	uint1_t weightMutualMask[COMPRESSION_VEC_SIZE];

	#pragma unroll
	for (uint5_t k=0; k<COMPRESSION_VEC_SIZE; k++) {
		activationMutualMask[k] = 0x0;
		weightMutualMask[k] = 0x0;
	}

	#pragma unroll
	for (uint5_t i=0; i<COMPRESSION_VEC_SIZE; i++) {
		#pragma unroll
		for (uint5_t j=0; j<COMPRESSION_VEC_SIZE; j++) {
			uint1_t ijEqual = ( (activationNum[i] == weightNum[j]) 
				&& (activationValidMasks[i] == 0x1) && (weightValidMasks[j] == 0x1) ) ?
				0x1 : 0x0;
			activationMutualMask [i] = activationMutualMask [i] | ijEqual;
			weightMutualMask [j] = weightMutualMask [j] | ijEqual;
		}
	}

	unsigned char weightIndex = 0, activationIndex=0;
	while (weightIndex < COMPRESSION_VEC_SIZE && activationIndex < COMPRESSION_VEC_SIZE) {
		//Search for the position of the leading 1 in the activation
		uint1_t activationFound = 0x0;
		#pragma unroll
		for (uint5_t i=0; i<COMPRESSION_VEC_SIZE; i++) {
			unsigned char tempIndex = 0x3F & ((0x1F & activationIndex) + (0x1F & i));  //wont' exceed 5 bit
			if ( (tempIndex < COMPRESSION_VEC_SIZE) && (activationFound == 0x0) ) {
				if (activationMutualMask[0X1F & tempIndex] == 0x1) {
					activationIndex = tempIndex;
					activationFound = 0x1;
				}
			}
		} // for

		uint1_t weightFound = 0x0;
		#pragma unroll
		for (uint5_t i=0; i<COMPRESSION_VEC_SIZE; i++) {
			unsigned char tempIndex = 0x3F & ((0x1F & weightIndex) + (0x1F & i)); //won't exceed 5 bit
			if ( (tempIndex < COMPRESSION_VEC_SIZE) && (weightFound == 0x0) ) {
				if (weightMutualMask[0x1F & tempIndex] == 0x1) {
					weightIndex = tempIndex;
					weightFound = 0x1;
				}
			}
		} // for

		pSum += activationValues[activationIndex] * weightValues[weightIndex];

		activationIndex = 0x3F & ((0x1F & activationIndex) + (0x1F & 0x1));
		weightIndex += 0x1;
	}


	//Write the result back
	write_channel_intel(channel_mult2DpEngineDispatcherPSum[id], pSum);
}