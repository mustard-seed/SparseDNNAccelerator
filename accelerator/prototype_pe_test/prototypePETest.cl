#include "params.hpp"
#include "device_structures.hpp"
#include "channels.cl"
#include "ihc_apint.h"
#include "device_utils.hpp"
#include "prototypePE_structs.hpp"


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

bool accessIsPossible (int* pCount, int condition) {
	bool possible = ( (*pCount) & 0x00000003) == condition; 
	(*pCount)++;
	return possible;
}


__attribute__((task))
__attribute__((max_global_work_dim(0)))
__kernel void kernelPrototypePE (
		__global t_vecSpValueAndZCount* restrict pActivationInput,
		__global t_vecSpValueAndZCount* restrict pActivationOutput,
		__global t_vecSpValueAndZCount* restrict pWeightInput,
		__global t_vecSpValueAndZCount* restrict pWeightOutput,
		__global short * restrict pBiasIn,
		__global short * restrict pBiasOut,
		__global short * restrict pDrainIn,  //Convert to uint12_t channel
		__global short * restrict pDrainOut, //Convert to uint12_t channel
		__global t_pe_prototype_instruction* restrict pInstruction,
		  int numInstructions,
		  int idx,
		  int idy

	)
{
	
	//Dense index tracker used in MAC reduction
	unsigned short weightIndex = 0;
	unsigned short activationIndex = 0;

	//Memory counters used for this test
	//Will be updated whenever an instruction is read
	int counterActivationInputMemory=0;
	int counterActivationOutputMemory=0;
	int counterWeightInputMemory=0;
	int counterWeightOutputMemory=0;
	int counterBiasInputMemory=0;
	int counterBiasOutputMemory=0;
	int counterDrainInputMemory=0;
	int counterDrainOutputMemory=0;
	int counterInstructionMemory=0;

	int attemptActivationInput=0;
	int attemptActivationOutput=0;
	int attemptWeightInput=0;
	int attemptWeightOutput=0;
	int attemptBiasInput=0;
	int attemptBiasOutput=0;
	int attemptDrainInput=0;
	int attemptDrainOutput=0;


	//Register that holds the partial sum
	//Should be interpreted as a signed fixed-point number
	//Fraction width = REG_FF_FRAC. Total width = PE_FF_WIDTH
	int pSumFF = 0;

	//Continuation flag used for this test
	uint1_t runKernel = 0x1;

	//Instruction counter
	int counterInstruction = 0;

	//Controller Logic
	uint4_t regControllerState = PE_CONTROLLER_STATE_IDLE;



	while (runKernel) {
		t_pe_prototype_instruction regInstruction = pInstruction[counterInstruction];

		// registers
		t_operand scalarReg;
		t_vecSpValueAndZCount activationBlock;

		// bias channel and registers
		unsigned char inputBiasCount = 0;
		unsigned char outputBiasCount = 0;
		unsigned char registerBiasCount = 0;

		// drain channel and registers
		unsigned char inputDrainCount=0;
		unsigned char outputDrainCount=0;

		// activaion channel count
		unsigned short inputActivationBlockCount = 0;
		unsigned short outputActivationBlockCount = 0;
		unsigned short processedActivationBlockCount = 0;

		// dense index tracker
		unsigned short numActivationTracker = 0;
		unsigned short numWeightTracker=0;

		int tempCounterInstruction = counterInstruction + 1;
		unsigned char instructionMode = regInstruction.mode;
		bool runInstruction = true;

		while (runInstruction) {
			/*Channel and memory I*/

			//Decide on whether to read from the input bias channel
			bool inputBiasChannelReadEnable = 
				(instructionMode == PE_MODE_LOAD_BIAS) 
				&& (inputBiasCount == outputBiasCount) && (inputBiasCount == registerBiasCount);
			if (inputBiasChannelReadEnable) {
				bool inputBiasChannelReadSuccess = accessIsPossible(&attemptBiasInput, READ_BIAS_CONDITION);
				if (inputBiasChannelReadSuccess) {
					scalarReg = (t_operand) (pBiasIn[counterBiasInputMemory++] & WEIGHT_MASK);
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
				&& (inputDrainCount == outputDrainCount) && ( (int) inputDrainCount < idy ) && ( (int) inputDrainCount < (regInstruction.maxIDY + 1) );
			if (inputDrainChannelReadEnable) {
				bool inputDrainChannelReadSuccess = accessIsPossible(&attemptDrainInput, READ_DRAIN_CONDITION);
				if (inputDrainChannelReadSuccess) {
					scalarReg = (t_operand) (pDrainIn[counterDrainInputMemory++] & WEIGHT_MASK);
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
				bool success = accessIsPossible(&attemptActivationInput, READ_ACTIVATION_CONDITION);
				if (success) {
					activationBlock = pActivationInput[counterActivationInputMemory++];
					inputActivationBlockCount++;
				}
			} 


			//Reg Updates
			
			//Processing
			if (instructionMode == PE_MODE_LOAD_BIAS) {
				if (inputBiasCount > registerBiasCount) {
					if (idx <= regInstruction.maxIDX && idy <= regInstruction.maxIDY) {					
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
					t_vecUnpacked activationBlockUnpacked;
					decodeRunLength ( &activationBlock,
									 &activationBlockUnpacked,
									 numActivationTracker
										);
					if (idx <= regInstruction.maxIDX && idy <= regInstruction.maxIDY) {
						//Element-wise operation
						if (instructionMode == PE_MODE_MAX_POOL
							|| instructionMode == PE_MODE_LOAD_ACTIVATION
							|| instructionMode == PE_MODE_ELTWISE_ADD) {
							//Filter out the element
							t_operand value;
							unsigned short targetIndex = idy + regInstruction.selectStartIndex;
							bool valueFound = 
								findMatchInUnpackedBlock(
										&activationBlockUnpacked,
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
					} // if idx <= ...
					numActivationTracker = activationBlockUnpacked.indices[COMPRESSION_VEC_SIZE-1];
					processedActivationBlockCount++;
				} //if inputActivationBlockCount > ...
			}

			/*Channel and memory O*/
			//Decide on whether to write to the output bias channel
			bool outputBiasChannelWriteEnable = 
				(instructionMode == PE_MODE_LOAD_BIAS) 
				&& (inputBiasCount > outputBiasCount);
			if (outputBiasChannelWriteEnable) {
				bool outputBiasChannelWriteSuccess = accessIsPossible(&attemptBiasOutput, WRITE_BIAS_CONDITION);
				if (outputBiasChannelWriteSuccess) {
					pBiasOut[counterBiasOutputMemory++] = (short) scalarReg;
					outputBiasCount++;
				}
			}

			//Decide on whether to writ eto the output drain channel
			bool outputDrainChannelWriteEnable = 
				(instructionMode == PE_MODE_DRAIN_PSUM)
				&& ( (inputDrainCount > outputDrainCount) || (inputDrainCount >= idy) || (inputDrainCount) > regInstruction.maxIDY );
			if (outputDrainChannelWriteEnable) {
				bool outputDrainChannelWriteSuccess = accessIsPossible (&attemptDrainOutput, WRITE_DRAIN_CONDITION);
				if (outputDrainChannelWriteSuccess) {
					pDrainOut[counterDrainOutputMemory++] = inputDrainCount > outputDrainCount ?
						(short) scalarReg : (short) convertAccumulatorToSignedFixedPoint (pSumFF, regInstruction.fracDout);
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
				bool success = accessIsPossible(&attemptActivationOutput, WRITE_ACTIVATION_CONDITION);
				if (success) {
					pActivationOutput[counterActivationOutputMemory++] = activationBlock;
					outputActivationBlockCount++;
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
				if ( (numActivationTracker == (regInstruction.transmissionEndIndex + 1))
					&& (inputActivationBlockCount == outputActivationBlockCount) ) {
					runInstruction = false;
				}

			}
			else if (instructionMode == PE_MODE_DRAIN_PSUM) {
				if (outputDrainCount == (inputDrainCount + 1) ) {
						runInstruction = false;
				}
			}


		} // while (runInstruction)


		if (tempCounterInstruction == numInstructions) {
			runKernel = 0x0;
		}
		counterInstruction = tempCounterInstruction;
	}


}