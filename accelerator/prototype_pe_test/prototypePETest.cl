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

#define PE_NUM_X 3
#define PE_NUM_Y 3

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


bool accessIsPossible (int* pCount, int condition) {
	bool possible = ( (*pCount) & 0x00000003) == condition; 
	(*pCount)++;
	return possible;
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

	//Continuation flag used for this test
	uint1_t runKernel = 0x1;

	//0x0 means read, 0x1 means pass forward
	uint1_t instructionInputCount = 0x0;
	uint1_t instructionOutputVerticalCount = 0x0;
	uint1_t instructionOutputHorizontalCount = 0x0;
	uint1_t instructionPassedCount = 0x0;
	t_pe_prototype_instruction regInstruction;

	while (runKernel) {
		if ( (instructionInputCount == instructionOutputVerticalCount)
				&& (instructionInputCount == instructionOutputHorizontalCount) ) {
			bool valid = false;
			regInstruction =
				read_channel_nb_intel(channel_instructionInput, &valid);
			instructionInputCount = valid ? instructionInputCount + 0x1 : instructionInputCount;

		}
		

		bool runInstruction = false;
		if (instructionInputCount > instructionPassedCount) {
			runInstruction = true;
			instructionPassedCount++;
		}

		//Instruction passing logic
		if ( ( (0x1FF & idy) == 0 ) && ( (0x1FF & idx) < (PE_NUM_X - 1) ) ) {
			//pass right
			if (instructionInputCount > instructionOutputHorizontalCount) {
				bool valid = write_channel_nb_intel(channel_instructionOutputHorizontal, regInstruction);
				instructionOutputHorizontalCount = valid ? instructionOutputHorizontalCount + 0x1 : instructionOutputHorizontalCount;
			}
		}

		if ( (0x1FF & idy) < (PE_NUM_Y - 1)) {
			if (instructionInputCount > instructionOutputVerticalCount) {
				bool valid = write_channel_nb_intel(channel_instructionOutputVertical, regInstruction);
				instructionOutputVerticalCount = valid ? instructionOutputVerticalCount + 0x1 : instructionOutputVerticalCount;
			}
		}

		// instruction passing reset logic
		if ( (0x1FF & idy) == 0 && (0x1FF & idx) < (PE_NUM_X - 1) )  {
			if ( instructionInputCount == 0x1 
					&& instructionOutputVerticalCount == 0x1 
					&& instructionOutputHorizontalCount == 0x1) {
				instructionInputCount = 0x0;
				instructionOutputVerticalCount = 0x0;
				instructionOutputHorizontalCount = 0x0;
			}
		}
		else if ( ( (0x1FF & idy) == 0 && idx == (PE_NUM_X - 1) ) || 
					( (0x1FF & idy) > 0 && (0x1FF & idy) < (PE_NUM_Y-1) ) )  {
			if (instructionInputCount == 0x1 && instructionOutputVerticalCount == 0x1) {
				instructionInputCount = 0x0;
				instructionOutputVerticalCount = 0x0;
				instructionOutputHorizontalCount = 0x0;
			}
		}
		else {
			if (instructionInputCount == 0x1) {
				instructionInputCount = 0x0;
				instructionOutputVerticalCount = 0x0;
				instructionOutputHorizontalCount = 0x0;
			}
		}

		// registers
		t_operand scalarReg;
		t_vecUnpacked activationBlock;
		t_vecUnpacked weightBlock;

		// bias channel and registers
		unsigned char inputBiasCount = 0;
		unsigned char outputBiasCount = 0;
		unsigned char registerBiasCount = 0;

		// drain channel and registers
		unsigned char inputDrainCount=0;
		unsigned char outputDrainCount=0;

		// activaion channel count
		uint1_t inputActivationBlockCount = 0;
		uint1_t outputActivationBlockCount = 0;
		uint1_t processedActivationBlockCount = 0;

		// weight channel count
		uint1_t inputWeightBlockCount = 0;
		uint1_t outputWeightBlockCount = 0;
		uint1_t processedWeightBlockCount = 0;

		// dense index tracker
		unsigned short numActivationTracker = 0;
		unsigned short numWeightTracker=0;

		unsigned char instructionMode = regInstruction.mode;
		

		#pragma max_concurrency 1
		while (runInstruction) {
			/*Channel and memory I*/

			//Decide on whether to read from the input bias channel
			bool inputBiasChannelReadEnable = 
				(instructionMode == PE_MODE_LOAD_BIAS) 
				&& (inputBiasCount == outputBiasCount) && (inputBiasCount == registerBiasCount);
			if (inputBiasChannelReadEnable) {
				bool valid = false;
				scalarReg = (t_operand) (read_channel_nb_intel(channel_biasInput, &valid) & WEIGHT_MASK);
				if (valid) {
					inputBiasCount++;
				}

			}

			//Decide on whether to read from the input drain channel
			//Condition to read: 
			// 1. The mode is PE_MODE_DRAIN_SUM, AND
			// 2. No uncommited value AND
			// 3. Number of elements read is less than the lesser of idy or maxIDY
			bool inputDrainChannelReadEnable = 
				(instructionMode == PE_MODE_DRAIN_PSUM) 
				&& ( (0x1FF &inputDrainCount) == (0x1FF & outputDrainCount) )  && ( (0x1FF & inputDrainCount) < (0x1FF & idy) ) && ( (0x1FF & inputDrainCount) < ( 0x3FF & ( (0x1FF & regInstruction.maxIDY) + 1) ) );
			if (inputDrainChannelReadEnable) {
				bool valid;
				scalarReg = (t_operand) (read_channel_nb_intel(channel_drainInput, &valid) & WEIGHT_MASK);
				if (valid) {
					inputDrainCount++;
				}
			}

			//Decide on whether or not to read from the input activation channel
			bool inputActivationChannelReadEnable = 
				(instructionMode == PE_MODE_MAX_POOL 
					|| instructionMode == PE_MODE_LOAD_ACTIVATION
					|| instructionMode == PE_MODE_ELTWISE_ADD
					|| instructionMode == PE_MODE_DOT_PRODUCT)
				&& (inputActivationBlockCount == processedActivationBlockCount)
				&& (inputActivationBlockCount == outputActivationBlockCount);
			if (inputActivationChannelReadEnable) {
				bool valid;
				activationBlock = read_channel_nb_intel(channel_activationInput, &valid);
				if (valid) {
					inputActivationBlockCount++;
				}
			} 

			//Decide on whether or not to read from the input weight channel
			bool inputWeightChannelReadEnable = 
				(instructionMode == PE_MODE_DOT_PRODUCT )
				&& (inputWeightBlockCount == processedWeightBlockCount)
				&& (inputWeightBlockCount == outputWeightBlockCount);
			if (inputWeightChannelReadEnable) {
				bool valid;
				weightBlock = read_channel_nb_intel(channel_weightInput, &valid);
				if (valid) {
					inputWeightBlockCount++;
				}
			} 


			//Reg Updates
			
			//Processing
			if (instructionMode == PE_MODE_LOAD_BIAS) {
				if (inputBiasCount > registerBiasCount) {
					if ( ( ((uint9_t) idx) <= ((uint9_t) regInstruction.maxIDX) ) && ( ((uint9_t) idy) <= ((uint9_t) regInstruction.maxIDY)) ) {	
//					if ( ( idx) <= ( regInstruction.maxIDX) && ( idy) <= (regInstruction.maxIDY)) {					
						pSumFF = convertSignedFixedPointToAccumulator(scalarReg, regInstruction.fracDin);
					}
					registerBiasCount++;
				}
			}
			else if (instructionMode == PE_MODE_MAX_POOL
				|| instructionMode == PE_MODE_LOAD_ACTIVATION 
				|| instructionMode == PE_MODE_ELTWISE_ADD
				|| instructionMode == PE_MODE_DOT_PRODUCT)
			{
				if (inputActivationBlockCount > processedActivationBlockCount) {
					//Unpack the activation block
					if (idx <= regInstruction.maxIDX && idy <= regInstruction.maxIDY) {
						//Element-wise operation
						if (instructionMode == PE_MODE_MAX_POOL
							|| instructionMode == PE_MODE_LOAD_ACTIVATION
							|| instructionMode == PE_MODE_ELTWISE_ADD) {
							//Filter out the element
							t_operand value;
							unsigned short targetIndex = (unsigned short) ( (unsigned short) idy + (unsigned short) regInstruction.selectStartIndex);
							bool valueFound = 
								findMatchInUnpackedBlock(
										&activationBlock,
										targetIndex,
										&value
									);
							if (valueFound) {
								// convert the value to the accumulator's format
								int wideValue = convertSignedFixedPointToAccumulator (
										value,
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
							} //if (valueFound)
						} //if instructionMode == ..
						else if (instructionMode == PE_MODE_DOT_PRODUCT) {
							//....
						}
					} // if idx <= ...
					numActivationTracker = activationBlock.indices[COMPRESSION_VEC_SIZE-1];
					processedActivationBlockCount++;
				} //if inputActivationBlockCount > ...

				if (inputWeightBlockCount > processedWeightBlockCount && instructionMode == PE_MODE_DOT_PRODUCT) {
					//unpack the weight block
					if (idx <= regInstruction.maxIDX && idy <= regInstruction.maxIDY) {
					} // processing block

					numWeightTracker = weightBlock.indices[COMPRESSION_VEC_SIZE-1];
					processedWeightBlockCount++;
				} // condition for processing the weight
			} 
			// Mode for processing 

			/*Channel and memory O*/
			//Decide on whether to write to the output bias channel
			bool outputBiasChannelWriteEnable = 
				(instructionMode == PE_MODE_LOAD_BIAS) 
				&& (inputBiasCount > outputBiasCount);
			if (outputBiasChannelWriteEnable) {
				bool valid = write_channel_nb_intel(channel_biasOutput, (short) scalarReg);
				if (valid) {
					outputBiasCount++;
				}
			}

			//Decide on whether to writ eto the output drain channel
			bool outputDrainChannelWriteEnable = 
				(instructionMode == PE_MODE_DRAIN_PSUM)
				&& ( (inputDrainCount > outputDrainCount) || (inputDrainCount >= idy) || (inputDrainCount) > regInstruction.maxIDY );
			if (outputDrainChannelWriteEnable) {
				short value = inputDrainCount > outputDrainCount ?
						(short) scalarReg : (short) convertAccumulatorToSignedFixedPoint (pSumFF, regInstruction.fracDout);
				bool valid = write_channel_nb_intel(channel_drainOutput, value);
				if (valid) {
					outputDrainCount++;
				}
			}

			//Decide on whether or not to write to the output activation channel
			bool outputActivationChannelWriteEnable = 
				(instructionMode == PE_MODE_MAX_POOL 
					|| instructionMode == PE_MODE_LOAD_ACTIVATION
					|| instructionMode == PE_MODE_ELTWISE_ADD
					|| instructionMode == PE_MODE_DOT_PRODUCT)
				&& (inputActivationBlockCount > outputActivationBlockCount);
			if (outputActivationChannelWriteEnable) {
				bool success = write_channel_nb_intel(channel_activationOutput, activationBlock);
				if (success) {
					outputActivationBlockCount++;
				}
			}

			bool outputWeightChannelWriteEnable = 
				(instructionMode == PE_MODE_DOT_PRODUCT)
				&& (inputWeightBlockCount > outputWeightBlockCount);
			if (outputWeightChannelWriteEnable) {
				bool success = write_channel_nb_intel(channel_weightOutput, weightBlock);
				if (success) {
					outputWeightBlockCount++;
				}
			}

			//Continue condition
			if (instructionMode == PE_MODE_LOAD_BIAS) {
				// If value needs to be loaded in the PE, then we have to make sure the value is loaded
				if (regInstruction.maxIDX >= idx && regInstruction.maxIDY >= idy) {
					if ( inputBiasCount == 1 && outputBiasCount == 1 && registerBiasCount == 1) {
						runInstruction = false;
					}
				}
				// Otherwise just make sure one bias has passed.
				else if ( inputBiasCount == 1 && outputBiasCount == 1 ) {
						runInstruction = false;
				}
			}
			else if (
				instructionMode == PE_MODE_LOAD_ACTIVATION ||
				instructionMode == PE_MODE_ELTWISE_ADD ||
				instructionMode == PE_MODE_MAX_POOL
				) {
				if ( (numActivationTracker == (unsigned short) (regInstruction.transmissionEndIndex + 1))
					&& (inputActivationBlockCount == outputActivationBlockCount) ) {
					runInstruction = false;
				}

			}
			else if (instructionMode == PE_MODE_DRAIN_PSUM) {
				if (outputDrainCount == (inputDrainCount + 1) ) {
						runInstruction = false;
				}
			}

			//activation block counter reset logic
			if (inputActivationBlockCount == 0x1 && outputActivationBlockCount == 0x1 && processedActivationBlockCount) {
				inputActivationBlockCount = 0x0;
				outputActivationBlockCount = 0x0;
				processedActivationBlockCount = 0x0;
			}

			//weight block counter reset logic
			if (inputWeightBlockCount == 0x1 && outputWeightBlockCount == 0x1 && processedWeightBlockCount) {
				inputWeightBlockCount = 0x0;
				outputWeightBlockCount = 0x0;
				processedWeightBlockCount = 0x0;
			}


		} // while (runInstruction)


	} // while (runKernel)


}