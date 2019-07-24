#include "params.hpp"
#include "device_structures.hpp"
#include "ihc_apint.h"
#include "device_utils.hpp"
#include "prototypePE_structs.hpp"
#include "peComponents.hpp"
#include "rtl_lib.hpp"
#include "leadingZero_lib.hpp"


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


//With the final array
typedef struct __attribute__((packed)){
	char values [SIMD_SIZE];
	uint6_t streamingBlockIndex;
	uint1_t isLast;
} t_simdblock_di;

typedef struct __attribute__((packed)){
	t_simdblock_value values;
	uint1_t isLast;
} t_simdblock_bitmask;

typedef struct __attribute__((packed)){
	t_simdblock_value weights;
	t_simdblock_value activations;
	uint1_t isLast;
} t_mac_simd_operands;

//MAC Operands
typedef struct __attribute__((packed)) {
	char values [SIMD_SIZE];
} t_simdblock_mac;

t_accumulator madd (t_simdblock_value activations, t_simdblock_value weights) {
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



channel t_simdblock_bitmask_tagged channel_activationInput __attribute__((depth(1)));
channel t_simdblock_bitmask_tagged channel_activationOutput __attribute__((depth(1)));
channel t_simdblock_bitmask_tagged channel_weightInput __attribute__((depth(1)));
channel t_simdblock_bitmask_tagged channel_weightOutput __attribute__((depth(1)));
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

channel t_simdblock_bitmask channel_dpWeightInput __attribute__((depth(PE_VEC_FIFO_SIZE)));
channel t_simdblock_bitmask channel_dpActivationInput __attribute__((depth(PE_VEC_FIFO_SIZE)));
//channel t_operand channel_pSumManagerActivationInput __attribute__((depth(0)));
channel t_mac_simd_operands channel_macOperandsInput __attribute__((depth(0)));
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

	//t_simdblock_di_tagged block = read_channel_intel(channel_weightInput);
	//t_simdblock_di peBlock;
	t_simdblock_bitmask_tagged block = read_channel_intel(channel_weightInput);
	t_simdblock_bitmask peBlock;
	#pragma unroll
	for (unsigned char i=0; i<SIMD_SIZE; i++) {
		peBlock.values.values[i] = block.values[i];
	}
	//peBlock.streamingBlockIndex = block.streamingBlockIndex;
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

	//t_simdblock_di_tagged block = read_channel_intel(channel_activationInput);
	//t_simdblock_di peBlock;
	t_simdblock_bitmask_tagged block = read_channel_intel(channel_activationInput);
	t_simdblock_bitmask peBlock;
	#pragma unroll
	for (unsigned char i=0; i<SIMD_SIZE; i++) {
		peBlock.values.values[i] = block.values[i];
	}
	//peBlock.streamingBlockIndex = block.streamingBlockIndex;
	peBlock.isLast = block.isLast;

	if (idx < (PE_COLS - 1)){
		if ( idx < block.maxTransportID ) {
			EMULATOR_PRINT ( ("[kernelWeightTransport]: Waiting to pass an activation block to the output\n") );
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
		unsigned short numOutputActivationBlocks,
		unsigned short numInputWeightBlocks,
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

	//uint5_t		   indexActivationTracker = 0;
	//uint5_t		   indexWeightTracker = 0;
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
			//t_simdblock_di_tagged taggedBlock;
			t_simdblock_bitmask_tagged taggedBlock;
			#pragma unroll
			for (unsigned char i=0; i<SIMD_SIZE; i++) {
				taggedBlock.values[i] = block.values[i];
			}

			//uint6_t indexInStreamingBlock = indexActivationTracker + (uint6_t) block.runLength; 
			unsigned char isLast = (countInputActivationBlocks == (numInputActivationBlocks - 1)) ?
                0x1 : 0x0;
            //taggedBlock.streamingBlockIndex = indexInStreamingBlock;
            taggedBlock.isLast = isLast;;
            taggedBlock.maxTransportID = maxIDY;

			valid = write_channel_nb_intel (channel_activationInput, taggedBlock);
			if (valid) {
				countInputActivationBlocks++;
				//numActivationTracker = unpackedValue.indices[COMPRESSION_VEC_SIZE-1];
				//indexActivationTracker = (indexInStreamingBlock == (SYNC_SIZE - 1)) ? 0x0 : indexInStreamingBlock + 0x1;
			}
		}

		if ( (0x1FF & idy) < (PE_NUM_Y - 1) ) {
			if (countOutputActivationBlocks < numOutputActivationBlocks) {
				bool valid;
				//t_simdblock_di_tagged block = read_channel_nb_intel(channel_activationOutput, &valid);
				t_simdblock_bitmask_tagged block = read_channel_nb_intel(channel_activationOutput, &valid);
				if (valid) {
					t_simdblock_host hostBlock;
					
					#pragma unroll
					for (unsigned char i=0; i<SIMD_SIZE; i++) {
						hostBlock.values[i] = block.values[i];
					}
					//hostBlock.runLength = block.streamingBlockIndex;
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
			//t_simdblock_di_tagged taggedBlock;
			t_simdblock_bitmask_tagged taggedBlock;
			#pragma unroll
			for (unsigned char i=0; i<SIMD_SIZE; i++) {
				taggedBlock.values[i] = block.values[i];
			}

			//uint6_t indexInStreamingBlock = indexWeightTracker + (uint6_t) block.runLength; 
			unsigned char isLast = (countInputWeightBlocks == (numInputWeightBlocks - 1)) ?
                0x1 : 0x0;
            //taggedBlock.streamingBlockIndex = indexInStreamingBlock;
            taggedBlock.isLast = isLast;;
            taggedBlock.maxTransportID = maxIDX;

			valid = write_channel_nb_intel (channel_weightInput, taggedBlock);
			if (valid) {
				countInputWeightBlocks++;
				//numActivationTracker = unpackedValue.indices[COMPRESSION_VEC_SIZE-1];
				//indexWeightTracker = (indexInStreamingBlock == (SYNC_SIZE - 1)) ? 0x0 : indexInStreamingBlock + 0x1;
			}
		}

		if ( (0x1FF & idx) < (PE_NUM_X - 1) ) {
			if (countOutputWeightBlocks < numOutputWeightBlocks) {
				bool valid;
				//t_vecUnpacked value = read_channel_nb_intel(channel_weightOutput, &valid);
				//t_simdblock_di_tagged block = read_channel_nb_intel(channel_weightOutput, &valid);
				t_simdblock_bitmask_tagged block = read_channel_nb_intel(channel_weightOutput, &valid);
				if (valid) {
					t_simdblock_host hostBlock;
					
					#pragma unroll
					for (unsigned char i=0; i<SIMD_SIZE; i++) {
						hostBlock.values[i] = block.values[i];
					}
					//hostBlock.runLength = block.streamingBlockIndex;
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


#define DP_LOAD_STATE_LOAD_BITMASK 0X0
#define DP_LOAD_STATE_FILL_WINDOW 0X1
#define DP_LOAD_STATE_DONE 0x2

#define DP_MATCH_STATE_COMPUTE_MUTUAL_BITMASK 0X0
#define DP_MATCH_STATE_DRAIN_WINDOW 0X1
#define DP_MATCH_STATE_SEND_LAST 0x2
#define DP_MATCH_STATE_DONE 0x3

typedef struct {
	t_simdblock_value values[8];
} t_simdblock_window;

#define TRUE 0X1
#define FALSE 0x0
#define BASE0 0
#define BASE1 8


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
	
	uint1_t regLoadSide = 0x0; //Can be 0x0 or 0x1;

	//t_simdblock_value ramBuffer0[16];
	//t_simdblock_value ramBuffer1[16];


	//=========Registers used for the loading the activation===============
	uint2_t regStateLoadActivation = DP_LOAD_STATE_LOAD_BITMASK;
	//unsigned char regOriginalActivationBitMask0;
	//unsigned char regOriginalActivationBitMask1;
	unsigned char regOriginalActivationBitMask;
	unsigned char regRunningActivationBitMask = 0;
	//uint1_t regActivationIsLast0;
	//uint1_t regActivationIsLast1;
	uint1_t regActivationIsLast;
	unsigned char regLoadActivationPosition;
	//t_simdblock_value regActivationBuffer0[8];
	//t_simdblock_value regActivationBuffer1[8];
	//t_simdblock_window regActivationBuffer0;
	//t_simdblock_window regActivationBuffer1;
	t_simdblock_value ramActivationBuffer0[8];
	t_simdblock_value ramActivationBuffer1[8];

	//==========Registers used for loading the weight==============
	uint2_t regStateLoadWeight = DP_LOAD_STATE_LOAD_BITMASK;
	unsigned char regOriginalWeightBitMask;
	unsigned char regRunningWeightBitMask = 0;
	uint1_t regWeightIsLast;
	//t_simdblock_value regWeightBuffer0[8];
	//t_simdblock_value regWeightBuffer1[8];
	//t_simdblock_window regWeightBuffer0;
	//t_simdblock_window regWeightBuffer1;
	t_simdblock_value ramWeightBuffer0[8];
	t_simdblock_value ramWeightBuffer1[8];

	unsigned char regLoadWeightPosition;


	//========Registers used for the drain side============
	uint2_t regStateDrain = DP_MATCH_STATE_DONE;
	unsigned char regRunningDrainMutualBitMask;
	//unsigned char regOriginalDrainMutualBitMask0;
	//unsigned char regOriginalDrainMutualBitMask1;
	unsigned char regDrainPosition;
	uint1_t regDrainIsLast;
	//uint1_t regDrainIsLast0;
	//uint1_t regDrainIsLast1;

	//#pragma ivdep array(regActivationBuffer0)
	//#pragma ivdep array(regActivationBuffer1)
	//#pragma ivdep array(regWeightBuffer0)
	//#pragma ivdep array(regWeightBuffer1)
	//#pragma ivdep
	#pragma ivdep array(ramActivationBuffer0)
	#pragma ivdep array(ramActivationBuffer1)
	#pragma ivdep array(ramWeightBuffer0)
	#pragma ivdep array(ramWeightBuffer1)
	while (true) {
		bool loadActivationSuccess;
		t_simdblock_bitmask activationBlob;
		uint2_t newStateLoadActivation = regStateLoadActivation;

		if ( (regStateLoadActivation == DP_LOAD_STATE_LOAD_BITMASK)
			|| (regStateLoadActivation == DP_LOAD_STATE_FILL_WINDOW) ) {
			activationBlob = read_channel_nb_intel(channel_dpActivationInput, 
						&loadActivationSuccess);
			if (!loadActivationSuccess) {
				//EMULATOR_PRINT ( ("[kernelDotProductEngineDispatcher]: Failed to load activation\n") );	
			}
		}

		switch (regStateLoadActivation) {
			case DP_LOAD_STATE_LOAD_BITMASK: {		
				if (loadActivationSuccess) {
					regLoadActivationPosition = 0x0;
					unsigned char activationBitmask = activationBlob.values.values[0];
					/*
					if (regLoadSide == 0x0) {
						regOriginalActivationBitMask0 = activationBitmask;
					}
					else {
						regOriginalActivationBitMask1 = activationBitmask;
					}
					*/
					regOriginalActivationBitMask = activationBitmask;
					regRunningActivationBitMask = activationBitmask;
					newStateLoadActivation = 
						(activationBitmask == 0x0) ?
						DP_LOAD_STATE_DONE : DP_LOAD_STATE_FILL_WINDOW;
					if (activationBitmask == 0x0) {
						/*
						if (regLoadSide == 0x0) {
							regActivationIsLast0 = (activationBlob.isLast) ? TRUE: FALSE;
						}
						else {
							regActivationIsLast1 = (activationBlob.isLast) ? TRUE: FALSE;
						}
						*/
						regActivationIsLast = (activationBlob.isLast) ? TRUE: FALSE;
					}
				}
			} // case DP_LOAD_STATE_LOAD_BITMASK
			break;
			case DP_LOAD_STATE_FILL_WINDOW: {
				if (loadActivationSuccess) {
					unsigned char currentBitMask = regRunningActivationBitMask;
					unsigned char currentPosition =
						regLoadActivationPosition;
					unsigned char bitCountResult = 
						leadingZeroCounter (currentBitMask);
					unsigned char leadingZeroCountPlusOne =
						(bitCountResult >> 4) & 0x0F;
					unsigned char leadingZeroCount = bitCountResult & 0x0F;

					//Calculate the position of the simd block
					unsigned char index = (currentPosition + leadingZeroCount);

					//Calculate the new bitmask;
					unsigned char newBitMask = currentBitMask >> leadingZeroCountPlusOne;

					//Update the position register
					regLoadActivationPosition = currentPosition + leadingZeroCountPlusOne;

					//Update the running activation  bit mask
					regRunningActivationBitMask = newBitMask;

					//Update the activaiton buffer window
					/*
					#pragma unroll
					for (unsigned char i=0; i<SIMD_SIZE; i++) {
						if (regLoadSide == 0x0) {
							//regActivationBuffer0.values[index].values[i] =
							//	activationBlob.values[i];
							//regActivationBuffer0[index].values[i] =
							//	activationBlob.values[i];
							//ramBuffer0[index].values[i] =
							//	activationBlob.values[i];
							 ramActivationBuffer[BASE0 + index].values[i]
							 	= activationBlob.values[i];
						}
						else {
							//regActivationBuffer1.values[index].values[i] =
							//	activationBlob.values[i];
							//regActivationBuffer1[index].values[i] =
							//	activationBlob.values[i];
							//ramBuffer1[index].values[i] =
							//	activationBlob.values[i];
							ramActivationBuffer[BASE1 + index].values[i]
							 	= activationBlob.values[i];
						}
					}
					*/
					if (regLoadSide == 0x0) {
						ramActivationBuffer0[index]
							 	= activationBlob.values;
					}
					else {
						ramActivationBuffer1[index]
							 	= activationBlob.values;
					}

					//Calculate the new state
					newStateLoadActivation = (newBitMask == 0x0) ?
						DP_LOAD_STATE_DONE : DP_LOAD_STATE_FILL_WINDOW;

					//Update the last register, if necessary
					if (newBitMask == 0x0) {
						/*
						if (regLoadSide == 0x0) {
							regActivationIsLast0 = (activationBlob.isLast) ? TRUE: FALSE;
						}
						else {
							regActivationIsLast1 = (activationBlob.isLast) ? TRUE: FALSE;
						}
						*/
						regActivationIsLast = (activationBlob.isLast) ? TRUE: FALSE;
					}
				} // if loadActivationSuccess
			} // case DP_LOAD_STATE_FILL_WINDOW
			break;
			default: {
			}
		} // switch regStateLoadActivation

		bool loadWeightSuccess;
		t_simdblock_bitmask weightBlob;
		uint2_t newStateLoadWeight = regStateLoadWeight;

		if ( (regStateLoadWeight == DP_LOAD_STATE_LOAD_BITMASK)
			|| (regStateLoadWeight == DP_LOAD_STATE_FILL_WINDOW) ) {
			weightBlob = read_channel_nb_intel(channel_dpWeightInput, 
						&loadWeightSuccess);
			if (!loadWeightSuccess) {
				//EMULATOR_PRINT ( ("[kernelDotProductEngineDispatcher]: Failed to load weight\n") );	
			}
		}

		switch (regStateLoadWeight) {
			case DP_LOAD_STATE_LOAD_BITMASK: {		
				if (loadWeightSuccess) {
					regLoadWeightPosition = 0x0;
					unsigned char weightBitMask = weightBlob.values.values[0];
					/*
					if (regLoadSide == 0x0) {
						regOriginalWeightBitMask0 = weightBitMask;
					}
					else {
						regOriginalWeightBitMask1 = weightBitMask;
					}
					*/
					regOriginalWeightBitMask = weightBitMask;
					regRunningWeightBitMask = weightBitMask;
					newStateLoadWeight = 
						(weightBitMask == 0x0) ?
						DP_LOAD_STATE_DONE : DP_LOAD_STATE_FILL_WINDOW;
					if (weightBitMask == 0x0) {
						/*
						if (regLoadSide == 0x0) {
							regWeightIsLast0 = (weightBlob.isLast) ? TRUE: FALSE;
						}
						else {
							regWeightIsLast1 = (weightBlob.isLast) ? TRUE: FALSE;
						}
						*/
						regWeightIsLast = (weightBlob.isLast) ? TRUE: FALSE;
					}
				}
			} // case DP_LOAD_STATE_LOAD_BITMASK
			break;
			case DP_LOAD_STATE_FILL_WINDOW: {
				if (loadWeightSuccess) {
					unsigned char currentBitMask = regRunningWeightBitMask;
					unsigned char currentPosition =
						regLoadWeightPosition;
					unsigned char bitCountResult = 
						leadingZeroCounter (currentBitMask);
					unsigned char leadingZeroCountPlusOne =
						(bitCountResult >> 4) & 0x0F;
					unsigned char leadingZeroCount = bitCountResult & 0x0F;

					//Calculate the position of the simd block
					unsigned char index = (currentPosition + leadingZeroCount) & 0xF;

					//Calculate the new bitmask;
					unsigned char newBitMask = currentBitMask >> leadingZeroCountPlusOne;

					//Update the position register
					regLoadWeightPosition = currentPosition + leadingZeroCountPlusOne;

					//Update the bit mask
					regRunningWeightBitMask = newBitMask;

					//Update the activaiton buffer window
					/*
					#pragma unroll
					for (unsigned char i=0; i<SIMD_SIZE; i++) {
						
						if (regLoadSide == 0x0) {
							//regWeightBuffer0.values[index].values[i] =
							//	weightBlob.values[i];
							//regWeightBuffer0[index].values[i] =
							//	weightBlob.values[i];
							//ramBuffer0[index].values[i] =
							//	weightBlob.values[i];
							ramWeightBuffer[BASE0 + index].values[i]
							 	= weightBlob.values[i];
						}
						else {
							//regWeightBuffer1.values[index].values[i] =
							//	weightBlob.values[i];
							//regWeightBuffer1[index].values[i] =
							//	weightBlob.values[i];
							//ramBuffer1[index].values[i] =
							//	weightBlob.values[i];
							ramWeightBuffer[BASE1 + index].values[i]
							 	= weightBlob.values[i];
						}
						
					}
					*/
					if (regLoadSide == 0x0) {
							ramWeightBuffer0[index]
								 	= weightBlob.values;
						}
					else {
							ramWeightBuffer1[index]
								 	= weightBlob.values;
					}

					//Calculate the new state
					newStateLoadWeight = (newBitMask == 0x0) ?
						DP_LOAD_STATE_DONE : DP_LOAD_STATE_FILL_WINDOW;

					//Update the last register, if necessary
					/*
						if (regLoadSide == 0x0) {
							regWeightIsLast0 = (weightBlob.isLast) ? TRUE: FALSE;
						}
						else {
							regWeightIsLast1 = (weightBlob.isLast) ? TRUE: FALSE;
						}
						*/
						regWeightIsLast = (weightBlob.isLast) ? TRUE: FALSE;
				} // if loadActivationSuccess
			} // case DP_LOAD_STATE_FILL_WINDOW
			break;
			default: {
			}
		} // switch regStateLoadWeight


		bool writeMacSuccess;
		uint2_t newStateDrain = regStateDrain;

		unsigned char newMutualBitMask = 0;
		unsigned char newDrainPosition = 0;
		uint1_t newDrainIsLast = false;
		unsigned char drainIndex = 0;
		t_mac_simd_operands operands;

		//Local variable computation
		switch (regStateDrain) {
			case DP_MATCH_STATE_COMPUTE_MUTUAL_BITMASK : {
				/*
				if (regLoadSide == 0x1) {
					newMutualBitMask = regOriginalDrainMutualBitMask0;
					newDrainIsLast = regDrainIsLast0;
				}
				else {
					newMutualBitMask = regOriginalDrainMutualBitMask1;
					newDrainIsLast = regDrainIsLast1;
				}
				*/
				newMutualBitMask = regRunningDrainMutualBitMask;
				newDrainIsLast = regDrainIsLast;
				newDrainPosition = 0;

			}
			break;
			case DP_MATCH_STATE_DRAIN_WINDOW : {
				unsigned char bitCountResult = 
						leadingZeroCounter (regRunningDrainMutualBitMask);
				unsigned char leadingZeroCountPlusOne =
						(bitCountResult >> 4) & 0x0F;
				unsigned char leadingZeroCount = bitCountResult & 0x0F;
				newDrainPosition = regDrainPosition + leadingZeroCountPlusOne;
				drainIndex =  (regDrainPosition + leadingZeroCount) & 0xF;
				newMutualBitMask = (regRunningDrainMutualBitMask >> leadingZeroCountPlusOne);
				newDrainIsLast = regDrainIsLast;
				/*
				#pragma unroll
				for (unsigned char i=0; i<SIMD_SIZE; i++) {
					if (regLoadSide == 0x1) {
						//operands.weights[i] = regWeightBuffer0.values[drainIndex].values[i];
						//operands.activations[i] = regActivationBuffer0.values[drainIndex].values[i];
						//operands.weights[i] = regWeightBuffer0[drainIndex].values[i];
						//operands.activations[i] = regActivationBuffer0[drainIndex].values[i];
						//operands.weights[i] = ramBuffer0[drainIndex+WEIGHT_BASE].values[i];
						//operands.activations[i] = ramBuffer0[drainIndex+ACTIVATION_BASE].values[i];
						operands.weights[i] = ramWeightBuffer[drainIndex+BASE0].values[i];
						operands.activations[i] = ramActivationBuffer[drainIndex+BASE0].values[i];
					}
					else {
						//operands.weights[i] = regWeightBuffer1.values[drainIndex].values[i];
						//operands.activations[i] = regActivationBuffer1.values[drainIndex].values[i];
						//operands.weights[i] = regWeightBuffer1[drainIndex].values[i];
						//operands.activations[i] = regActivationBuffer1[drainIndex].values[i];
						//operands.weights[i] = ramBuffer1[drainIndex+WEIGHT_BASE].values[i];
						//operands.activations[i] = ramBuffer1[drainIndex+ACTIVATION_BASE].values[i];
						operands.weights[i] = ramWeightBuffer[drainIndex+BASE1].values[i];
						operands.activations[i] = ramActivationBuffer[drainIndex+BASE1].values[i];
					}
				}
				*/
				if (regLoadSide == 0x1) {
						operands.weights = ramWeightBuffer0[drainIndex];
						operands.activations = ramActivationBuffer0[drainIndex];
					}
				else {
					operands.weights = ramWeightBuffer1[drainIndex];
					operands.activations = ramActivationBuffer1[drainIndex];
				}
				operands.isLast = 0x0;

			}
			break;
			case DP_MATCH_STATE_SEND_LAST : {
				operands.isLast = 0x1;
			}
			break;
			default: {

			}
		}

		if ((regStateDrain == DP_MATCH_STATE_DRAIN_WINDOW)
			|| (regStateDrain == DP_MATCH_STATE_SEND_LAST)) {
			writeMacSuccess = write_channel_nb_intel(channel_macOperandsInput, operands);
		}


		//State variable updates for the draining side
		switch (regStateDrain) {
			case DP_MATCH_STATE_COMPUTE_MUTUAL_BITMASK : {
				regDrainPosition = newDrainPosition;
				regRunningDrainMutualBitMask = newMutualBitMask;
				regDrainIsLast = newDrainIsLast;
				if  (newMutualBitMask == 0x0) {
					if (newDrainIsLast) {
						newStateDrain = DP_MATCH_STATE_SEND_LAST;
					}
					else {
						newStateDrain = DP_MATCH_STATE_DONE;
					}
				}
				else {
					newStateDrain = DP_MATCH_STATE_DRAIN_WINDOW;
				}
			} // DP_MATCH_STATE_COMPUTE_MUTUAL_BITMASK
			break;
			case DP_MATCH_STATE_DRAIN_WINDOW : {
				if (writeMacSuccess) {
					regDrainPosition = newDrainPosition;
					regRunningDrainMutualBitMask = newMutualBitMask;
					EMULATOR_PRINT ( ("[kernelDotProductEngineDispatcher]: Sent new operands to mac. The new mutual bit mask is %d\n", newMutualBitMask) );
					if  (newMutualBitMask == 0x0) {
						if (newDrainIsLast) {
							newStateDrain = DP_MATCH_STATE_SEND_LAST;
						}
						else {
							newStateDrain = DP_MATCH_STATE_DONE;
						}
						EMULATOR_PRINT ( ("[kernelDotProductEngineDispatcher]: Finished sending the operands in one window to mac!\n") );
					}

				}
			} //DP_MATCH_STATE_DRAIN_WINDOW
			break;
			case DP_MATCH_STATE_SEND_LAST : {
				if (writeMacSuccess) {
					newStateDrain = DP_MATCH_STATE_DONE;
					EMULATOR_PRINT ( ("[kernelDotProductEngineDispatcher]: Finished sending the operands in one dot product to mac!\n") );
				}
			}
			break;
			default: {

			}
		}

		//swap and reset states if necessary, 
		if ( (newStateLoadActivation == DP_LOAD_STATE_DONE)
			&& (newStateLoadWeight == DP_LOAD_STATE_DONE)
			&& (newStateDrain == DP_MATCH_STATE_DONE) ) {

			regStateLoadActivation = DP_LOAD_STATE_LOAD_BITMASK;
			regStateLoadWeight = DP_LOAD_STATE_LOAD_BITMASK;
			regStateDrain = DP_MATCH_STATE_COMPUTE_MUTUAL_BITMASK;

			unsigned char newMutualBitMask = regOriginalActivationBitMask & regOriginalWeightBitMask;
			uint1_t newIsDrainLastLocal = regActivationIsLast & regWeightIsLast;

			//if (regLoadSide == 0x0) {
				regRunningDrainMutualBitMask = newMutualBitMask;
				regDrainIsLast = newIsDrainLastLocal;
			//}
			//else {
				//regOriginalDrainMutualBitMask1 = newMutualBitMask;
				//regDrainIsLast1 = newIsDrainLastLocal;
			//}

			regLoadSide = (regLoadSide == 0x0) ? 0x1 : 0x0;

			EMULATOR_PRINT ( ("[kernelDotProductEngineDispatcher]: Swap occured!\n") );
		}
		// otherwise just perform regular state update
		else {
			//State updates
			regStateLoadActivation = newStateLoadActivation;
			regStateLoadWeight = newStateLoadWeight;
			regStateDrain = newStateDrain;
			//EMULATOR_PRINT ( ("[kernelDotProductEngineDispatcher]: New activation state, weight state, and drain state: %d %d %d\n",
			//	regStateLoadActivation, regStateLoadWeight, regStateDrain) );	
		}
	}		
	
}

__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void mac () {
	t_accumulator pSum = 0;
	uint1_t proceed = TRUE;
	while (proceed == TRUE) {
		t_mac_simd_operands operands = read_channel_intel(channel_macOperandsInput);
		bool isLast = operands.isLast;
		if (!isLast) {
			t_simdblock_value activations, weights;
			activations = operands.activations;
			weights = operands.weights;
			t_accumulator tempPSum = madd(activations, weights);
			pSum += tempPSum;
		}
		else {
			proceed = FALSE;
		}
	}

	write_channel_intel(channel_peDrainOutput, pSum);
}