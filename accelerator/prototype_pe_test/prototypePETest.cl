#include "params.hpp"
#include "device_structures.hpp"
#include "ihc_apint.h"
#include "device_utils.hpp"
#include "prototypePE_structs.hpp"
#include "peComponents.hpp"
#include "rtl_lib.hpp"


#define PE_NUM_X PE_COLS
#define PE_NUM_Y PE_ROWS

#define TRUE 0X1
#define FALSE 0x0


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


#ifdef DIRECT_COMPRESSION_SIMD
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
#endif

#ifdef FLEXIBLE_BITMASK_COMPRESSION
typedef struct __attribute__((packed)){
    t_transfer_block values;
    uint1_t isLast;
    char maxTransportID;
} t_transferblock_tagged;

typedef struct __attribute__((packed)){
    t_transfer_block values;
    uint1_t isLast;
} t_transferblock_local;

typedef struct __attribute__((packed)) {
    char values [COMPRESSION_WINDOW_SIZE];
} t_compression_window;

typedef struct __attribute__((packed)){
    t_compression_window weightWindow;
    t_compression_window activationWindow;
    unsigned char bitmaskW;
    unsigned char bitmaskA;
    bool isLast;
} t_alignment_input;

typedef struct __attribute__((packed)){
    t_compression_window weightWindow;
    t_compression_window activationWindow;
    unsigned long alignmentData;
    bool isLast;
} t_alignment_output;
#endif //FLEXIBLE_BITMASK_COMPRESSION

//MAC Operands
typedef struct __attribute__((packed)) {
	char values [SIMD_SIZE*CLUSTER_SIZE];
} t_simd_operand;

typedef struct __attribute__((packed)){
	t_simd_operand weights;
	t_simd_operand activations;
	uint1_t isLast;
} t_mac_operands;

t_accumulator madd (t_simd_operand activations, t_simd_operand weights) {
	t_accumulator output = 0x0;

	//#ifdef DIRECT_COMPRESSION_SIMD
		#pragma unroll
		for(int i=0; i<SIMD_SIZE*CLUSTER_SIZE/4; i++){
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
	//#endif
	//#ifdef FLEXIBLE_BITMASK_COMPRESSION
	/*
		#pragma unroll
		for(int i=0; i<SIMD_SIZE/2; i++){
			//output += input.data[i]*weights.data[i];
			// use packed DSP blocks to improve efficiency
			#if defined (ARRIA10)
				output += a10_mac_8bitx2(
					activations.values[i*2],
					weights.values[i*2],
					activations.values[i*2+1],
					weights.values[i*2+1]
					);
			#elif defined (C5SOC)
				output += c5_mac_8bitx2(
						activations.values[i*2],
						weights.values[i*2],
						activations.values[i*2+1],
						weights.values[i*2+1]
					);
			#else
			#error Unsupported FPGA type!
			#endif
		}
		*/
	//#endif

	return output;
}


#ifdef DIRECT_COMPRESSION_SIMD
channel t_simdblock_bitmask_tagged channel_activationInput __attribute__((depth(1)));
channel t_simdblock_bitmask_tagged channel_activationOutput __attribute__((depth(1)));
channel t_simdblock_bitmask_tagged channel_weightInput __attribute__((depth(1)));
channel t_simdblock_bitmask_tagged channel_weightOutput __attribute__((depth(1)));
#endif
#ifdef FLEXIBLE_BITMASK_COMPRESSION
channel t_transferblock_tagged channel_activationInput __attribute__((depth(1)));
channel t_transferblock_tagged channel_activationOutput __attribute__((depth(1)));
channel t_transferblock_tagged channel_weightInput __attribute__((depth(1)));
channel t_transferblock_tagged channel_weightOutput __attribute__((depth(1)));
#endif
channel t_operand channel_biasInput __attribute__((depth(1)));
channel t_operand channel_biasOutput __attribute__((depth(1)));
channel t_accumulator channel_drainInput __attribute__((depth(1)));
channel t_operand channel_drainOutput __attribute__((depth(1)));
channel t_pe_prototype_instruction channel_instructionInput __attribute__((depth(1)));
channel t_pe_prototype_instruction channel_instructionOutputVertical __attribute__((depth(1)));
channel t_pe_prototype_instruction channel_instructionOutputHorizontal __attribute__((depth(1)));

channel t_operand channel_peBiasInput __attribute__((depth(0)));
channel t_drainTransportInstruction channel_drainTransportInstruction __attribute__((depth(0)));

#ifdef DIRECT_COMPRESSION_SIMD
channel t_simdblock_bitmask channel_dpWeightInput __attribute__((depth(PE_VEC_FIFO_SIZE)));
channel t_simdblock_bitmask channel_dpActivationInput __attribute__((depth(PE_VEC_FIFO_SIZE)));
#endif

#ifdef FLEXIBLE_BITMASK_COMPRESSION
channel t_transferblock_local channel_dpWeightInput __attribute__((depth(PE_VEC_FIFO_SIZE)));
channel t_transferblock_local channel_dpActivationInput __attribute__((depth(PE_VEC_FIFO_SIZE)));

channel t_alignment_input channel_alignmentInput __attribute__((depth(0)));
channel t_alignment_output channel_alignmentOutput __attribute__((depth(0)));

#endif
//channel t_operand channel_pSumManagerActivationInput __attribute__((depth(0)));
channel t_mac_operands channel_macOperandsInput __attribute__((depth(0)));
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
					
					//EMULATOR_PRINT ( ("[kernelDrainTransport]: Waiting to write pSum %d \n", drainCount) );
					//write_channel_intel(channel_drainOutput, result);
					write_channel_intel(channel_drainOutput, result);
					drainCount++;
				}

				if (drainCount == numPSumToSend) {
					//EMULATOR_PRINT ( ("[kernelDrainTransport]: Committed the pSum\n") );
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
	#ifdef DIRECT_COMPRESSION_SIMD
	t_simdblock_bitmask_tagged block;
	t_simdblock_bitmask peBlock;
	#endif

	#ifdef FLEXIBLE_BITMASK_COMPRESSION
	t_transferblock_tagged block;
	t_transferblock_local peBlock;
	#endif

	block = read_channel_intel(channel_weightInput);
	#pragma unroll
	for (unsigned char i=0; i<SIMD_SIZE; i++) {
		#ifdef DIRECT_COMPRESSION_SIMD
			peBlock.values.values[i] = block.values[i];
		#endif
		#ifdef FLEXIBLE_BITMASK_COMPRESSION
			peBlock.values.values[i] = block.values.values[i];
		#endif
	}
	//peBlock.streamingBlockIndex = block.streamingBlockIndex;
	peBlock.isLast = block.isLast;

	if (idy < (PE_ROWS - 1)){
		if ( idy < block.maxTransportID ) {
			//EMULATOR_PRINT ( ("[kernelWeightTransport]: Waiting to pass a weight block to the output\n") );
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

	#ifdef DIRECT_COMPRESSION_SIMD
	t_simdblock_bitmask_tagged block;
	t_simdblock_bitmask peBlock;
	#endif

	#ifdef FLEXIBLE_BITMASK_COMPRESSION
	t_transferblock_tagged block;
	t_transferblock_local peBlock;
	#endif

	block = read_channel_intel(channel_activationInput);
	#pragma unroll
	for (unsigned char i=0; i<SIMD_SIZE; i++) {
		#ifdef DIRECT_COMPRESSION_SIMD
			peBlock.values.values[i] = block.values[i];
		#endif
		#ifdef FLEXIBLE_BITMASK_COMPRESSION
			peBlock.values.values[i] = block.values.values[i];
		#endif
	}
	//peBlock.streamingBlockIndex = block.streamingBlockIndex;
	peBlock.isLast = block.isLast;

	if (idx < (PE_COLS - 1)){
		if ( idx < block.maxTransportID ) {
			//EMULATOR_PRINT ( ("[kernelWeightTransport]: Waiting to pass an activation block to the output\n") );
			write_channel_intel(channel_activationOutput, block);
		}
	}

	write_channel_intel(channel_dpActivationInput, peBlock); 
}



__attribute__((task))
__attribute__((max_global_work_dim(0)))
__kernel void kernelTestInterface (
		#ifdef DIRECT_COMPRESSION_SIMD
			__global t_simdblock_host* restrict pActivationInput,
			__global t_simdblock_host* restrict pActivationOutput,
			__global t_simdblock_host* restrict pWeightInput,
			__global t_simdblock_host* restrict pWeightOutput,
		#endif
		#ifdef FLEXIBLE_BITMASK_COMPRESSION
			__global t_transfer_block* restrict pActivationInput,
			__global t_transfer_block* restrict pActivationOutput,
			__global t_transfer_block* restrict pWeightInput,
			__global t_transfer_block* restrict pWeightOutput,
		#endif
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
			(countOutputActivationBlocks < numOutputActivationBlocks)
			||
			(countOutputWeightBlocks < numOutputWeightBlocks)
			||
			(countOutputBias < numOutputBias)
			||
			(countOutputDrain < numOutputDrain)
			||
			(countOutputInstructionVertical < numOutputInstructionVertical)
			||
			(countOutputInstructionHorizontal < numOutputInsructionHorizontal)
		) {

		if (countInputActivationBlocks < numInputActivationBlocks) {
			bool valid;
			#ifdef DIRECT_COMPRESSION_SIMD
				t_simdblock_host block;
				t_simdblock_bitmask_tagged taggedBlock;
			#endif
			#ifdef FLEXIBLE_BITMASK_COMPRESSION
				t_transfer_block block;
				t_transferblock_tagged taggedBlock;
			#endif
			
			block = pActivationInput[countInputActivationBlocks];

			#pragma unroll
			for (unsigned char i=0; i<SIMD_SIZE; i++) {
				taggedBlock.values.values[i] = block.values[i];
			}

			//uint6_t indexInStreamingBlock = indexActivationTracker + (uint6_t) block.runLength; 
			uint1_t isLast = (countInputActivationBlocks == (numInputActivationBlocks - 1)) ?
				TRUE : FALSE;
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
				#ifdef DIRECT_COMPRESSION_SIMD
					t_simdblock_host block;
					t_simdblock_bitmask_tagged taggedBlock;
				#endif
				#ifdef FLEXIBLE_BITMASK_COMPRESSION
					t_transfer_block block;
					t_transferblock_tagged taggedBlock;
				#endif
				taggedBlock = read_channel_nb_intel(channel_activationOutput, &valid);
				if (valid) {
					
					#pragma unroll
					for (unsigned char i=0; i<SIMD_SIZE; i++) {
						block.values[i] = taggedBlock.values.values[i];
					}
					//hostBlock.runLength = block.streamingBlockIndex;
					//pActivationOutput[countOutputActivationBlocks++] = block;
					countOutputActivationBlocks++;
					//EMULATOR_PRINT ( ("[kernelTestInferace]: Collected %d out of %d activation blocks\n", countOutputActivationBlocks, numOutputActivationBlocks) );
					//DEBUG_PRINT ( ("[kernelTestInferace]: Collected %d out of %d activation blocks\n", countOutputActivationBlocks, numOutputActivationBlocks) );
				}
			}
		}

		if (countInputWeightBlocks < numInputWeightBlocks) {
			bool valid;

			#ifdef DIRECT_COMPRESSION_SIMD
				t_simdblock_host block;
				t_simdblock_bitmask_tagged taggedBlock;
			#endif
			#ifdef FLEXIBLE_BITMASK_COMPRESSION
				t_transfer_block block;
				t_transferblock_tagged taggedBlock;
			#endif

			block = pWeightInput[countInputWeightBlocks];

			#pragma unroll
			for (unsigned char i=0; i<SIMD_SIZE; i++) {
				taggedBlock.values.values[i] = block.values[i];
			}

			//uint6_t indexInStreamingBlock = indexWeightTracker + (uint6_t) block.runLength; 
			uint1_t isLast = (countInputWeightBlocks == (numInputWeightBlocks - 1)) ? 
				TRUE : FALSE;
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
				#ifdef DIRECT_COMPRESSION_SIMD
					t_simdblock_host block;
					t_simdblock_bitmask_tagged taggedBlock;
				#endif
				#ifdef FLEXIBLE_BITMASK_COMPRESSION
					t_transfer_block block;
					t_transferblock_tagged taggedBlock;
				#endif

				taggedBlock = read_channel_nb_intel(channel_weightOutput, &valid);

				if (valid) {
					
					#pragma unroll
					for (unsigned char i=0; i<SIMD_SIZE; i++) {
						block.values[i] = taggedBlock.values.values[i];
					}
					//hostBlock.runLength = block.streamingBlockIndex;
					//pWeightOutput[countOutputWeightBlocks++] = block;
					countOutputWeightBlocks++;
					//EMULATOR_PRINT ( ("[kernelTestInferace]: Collected %d out of %d weight blocks\n", countOutputWeightBlocks, numOutputWeightBlocks) );
					//DEBUG_PRINT ( ("[kernelTestInferace]: Collected %d out of %d weight blocks\n", countOutputWeightBlocks, numOutputWeightBlocks) );
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

#ifdef FLEXIBLE_BITMASK_COMPRESSION
#define ASSEMBLER_STATE_LOAD_BITMASK 0X0
#define ASSEMBLER_STATE_LOAD_VALUE 0X1
//#define ASSEMBLER_STATE_ALIGN 0x2
#define ASSEMBLER_STATE_WAIT 0x2

#define BITWIDTH_COMPRESSION_WINDOW_INDEX 3
#define MASK_COMPRESSION_WINDOW_INDEX 0x7

#define MAC_STATE_WAIT 0x0
#define MAC_STATE_ALIGN 0x1
#define MAC_STATE_PROCESS_WINDOW 0x2
#define MAC_STATE_WRITE_PSUM 0x3

__attribute__((task))
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__kernel void kernelPE ()
{
	//================Ping-ponged registers========================
	//BRAM for storing the compression windows
	t_cluster activationWindow[COMPRESSION_WINDOW_SIZE+1][2]; 
	t_cluster weightWindow[COMPRESSION_WINDOW_SIZE+1][2]; 

	//Flags that indicates whether we are at the last window
	uint1_t isLast[2];
	unsigned char bitmaskA[2];
	unsigned char bitmaskW[2];

	uint1_t regLoadSide = 0x0;

	//========Assembler side registers====================
	unsigned char countActivation;
	unsigned char countWeight;
	unsigned char numActivation;
	unsigned char numWeight;
	uint2_t stateActivation = ASSEMBLER_STATE_LOAD_BITMASK;
	uint2_t stateWeight = ASSEMBLER_STATE_LOAD_BITMASK;
	//unsigned long alignmentData;


	//=========MAC side logic========================
	uint2_t stateMac = MAC_STATE_WAIT;
	t_accumulator pSum = 0;
	unsigned char countOperands;
	unsigned char numOperands;
	unsigned int indicesW;
	unsigned int indicesA;

	//================Debug====================
	//unsigned short debugCount = 0;

	#pragma ivdep array(activationWindow)
	#pragma ivdep array(weightWindow)
	//#pragma ivdep safelen(7)
	//#pragma ivdep
	while (true)
	{

		//================ACTIVATION========================
		
		uint2_t nextStateActivation = stateActivation;
		if (stateActivation == ASSEMBLER_STATE_LOAD_BITMASK
			|| stateActivation == ASSEMBLER_STATE_LOAD_VALUE)
		{
			t_transferblock_local activationTransferBlock;
			bool activationReadSuccess;

			activationTransferBlock = read_channel_nb_intel (
						channel_dpActivationInput,
						&activationReadSuccess
					);

			if (activationReadSuccess)
			{
				//isLastActivation = activationTransferBlock.isLast;
				//DEBUG_PRINT(("[Assembler] Activation read!\n"));

				if (stateActivation == ASSEMBLER_STATE_LOAD_BITMASK)
				{
					unsigned char bitmask = activationTransferBlock.values.values[0].cluster_values[0];
					bitmaskA[regLoadSide & 0x01] = bitmask;
					numActivation = popCounter(bitmask);
					countActivation = 0;
					//EMULATOR_PRINT(("[assembler] bitmaskA: %#04x \n", bitmask));
				}
				//else
				//{

					//uint3_t offset = (stateActivation == ASSEMBLER_STATE_LOAD_BITMASK) ?
					//	0X1 : 0X0; 

					#pragma unroll
					for (uint3_t i=0; i<TRANSFER_SIZE; i++)
					{
						//if (i >= offset)
						//{
							activationWindow[countActivation+i][regLoadSide & 0x01]
								= activationTransferBlock.values.values[i];
							//EMULATOR_PRINT(("[assembler] activation value: %#04x %#04x \n"
							//	, activationTransferBlock.values.values[i].cluster_values[0] & 0xFF
							//	, activationTransferBlock.values.values[i].cluster_values[1] & 0xFF));
							//EMULATOR_PRINT(("[assembler] activation offset, countActivation: %#04x %#04x\n"
							//	, offset, countActivation));
						//}
					} // for. Transfer the values in the transfer block to the compression window

					//if (debugCount < maxDebugCount)
					//{
					//	DEBUG_PRINT(("[PE] ActivationTransferBlock [0-4]: %#04x %#04x %#04x %#04x\n",
					//		activationTransferBlock.values.values[0].cluster_values[0] & 0xFF, 
					//		activationTransferBlock.values.values[0].cluster_values[1] & 0xFF,
					//		activationTransferBlock.values.values[1].cluster_values[0] & 0xFF,
					//		activationTransferBlock.values.values[1].cluster_values[1] & 0xFF));
					//}

					countActivation += (unsigned char)(TRANSFER_SIZE);
				//}

				//State update
				if (countActivation > numActivation) //countActivation needs to be strictly larger than numActivation
				{
					nextStateActivation = ASSEMBLER_STATE_WAIT;
				}
				else {
					nextStateActivation = ASSEMBLER_STATE_LOAD_VALUE;
				}

			} // if activationReadSuccess
		}
		//===================================================

		//================WEIGHT========================
		uint2_t nextStateWeight = stateWeight;
		if (stateWeight == ASSEMBLER_STATE_LOAD_BITMASK
			|| stateWeight == ASSEMBLER_STATE_LOAD_VALUE)
		{
			t_transferblock_local weightTransferBlock;
			bool weightReadSuccess;
			weightTransferBlock = read_channel_nb_intel (
						channel_dpWeightInput,
						&weightReadSuccess
					);

			if (weightReadSuccess)
			{
				isLast[regLoadSide & 0x01] = weightTransferBlock.isLast;
				//DEBUG_PRINT(("[Assembler] Weight read!\n"));

				if (stateWeight == ASSEMBLER_STATE_LOAD_BITMASK)
				{
					unsigned char bitmask =  weightTransferBlock.values.values[0].cluster_values[0];
					bitmaskW[regLoadSide & 0x01] = bitmask; 
					numWeight = popCounter(bitmask);
					countWeight = 0;
					//EMULATOR_PRINT(("[assembler] bitmaskW: %#04x \n", bitmask));
				}
				//else
				//{

					//uint3_t offset = (stateWeight == ASSEMBLER_STATE_LOAD_BITMASK) ?
					//	0X1 : 0X0; 

					#pragma unroll
					for (uint3_t i=0; i<TRANSFER_SIZE; i++)
					{
						//if (i >= offset)
						//{
							weightWindow[countWeight+i][regLoadSide & 0x01]
								= weightTransferBlock.values.values[i];
							//EMULATOR_PRINT(("[assembler] weight value: %#04x %#04x \n"
							//	, weightTransferBlock.values.values[i].cluster_values[0] & 0xFF
							//	, weightTransferBlock.values.values[i].cluster_values[1] & 0xFF));
						//}
					} // for. Transfer the values in the transfer block to the compression window

					//if (debugCount < maxDebugCount)
					//{
					//	DEBUG_PRINT(("[PE] weightTransferBlock [0-4]: %#04x %#04x %#04x %#04x\n",
					//		weightTransferBlock.values.values[0].cluster_values[0] & 0xFF, 
					//		weightTransferBlock.values.values[0].cluster_values[1] & 0xFF,
					//		weightTransferBlock.values.values[1].cluster_values[0] & 0xFF,
					//		weightTransferBlock.values.values[1].cluster_values[1] & 0xFF));
					//}

					countWeight += (unsigned char)(TRANSFER_SIZE);
				//}

				//State update
				if (countWeight > numWeight) //countWeight needs to be strictly larger than numWeight
				{
					nextStateWeight = ASSEMBLER_STATE_WAIT;
				}
				else 
				{
					nextStateWeight = ASSEMBLER_STATE_LOAD_VALUE;
				}

			} // if weightReadSuccess
		}
		//===================================================

		//==================MAC states===================
		uint2_t nextStateMac = stateMac;

		if (stateMac == MAC_STATE_ALIGN)
		{
			unsigned long alignmentData = operandMatcher8(
				bitmaskW [(~regLoadSide) & 0x1],
				bitmaskA [(~regLoadSide) & 0x1]
			);
			numOperands = (alignmentData >> 48) & 0xFF;
			indicesW = (alignmentData >> 24) & 0xFFFFFF;
			indicesA = (alignmentData) & 0xFFFFFF;
			countOperands = 0; 
			//EMULATOR_PRINT ( ("[aligner]: indicesW: %#06x indicesA: %#06x numOperands: %#04x \n"
			//		, indicesW, indicesA,  numOperands) );

			/*
			if (countOperands >= numOperands)
			{
				if (isLast[(~regLoadSide) & 0x1])
				{
					nextStateMac = MAC_STATE_WRITE_PSUM;
				}
				else
				{
					nextStateMac = MAC_STATE_WAIT;
				}
			}
			else
			{
				nextStateMac = MAC_STATE_PROCESS_WINDOW;
			}
			*/
			nextStateMac = MAC_STATE_PROCESS_WINDOW;
		}
		else if (stateMac == MAC_STATE_PROCESS_WINDOW)
		{

			t_simd_operand simdActivations;
			t_simd_operand simdWeights;
			t_cluster zeros;
			#pragma unroll
			for (int i=0; i<CLUSTER_SIZE; i++)
			{
				zeros.cluster_values[i] = 0x0;
			}


			#pragma unroll
			for (unsigned char i=0; i<SIMD_SIZE; i++)
			{
				unsigned char indexW = 
					(indicesW >> (i*BITWIDTH_COMPRESSION_WINDOW_INDEX))
					& MASK_COMPRESSION_WINDOW_INDEX;
				t_cluster w = ((countOperands + i) < numOperands) ?
					weightWindow[indexW+1][(~regLoadSide) & 0x01] : zeros;
				//char w = weightWindow[i][(~regLoadSide) & 0x1];
				//simdWeights.values[i] = w;

				unsigned char indexA = 
					(indicesA >> (i*BITWIDTH_COMPRESSION_WINDOW_INDEX))
					& MASK_COMPRESSION_WINDOW_INDEX;
				t_cluster a = ((countOperands + i) < numOperands) ?
					activationWindow[indexA+1][(~regLoadSide) & 0x01] : zeros;
				//char a = activationWindow[i][(~regLoadSide) & 0x1];

				#pragma unroll
				for (unsigned char j=0; j<CLUSTER_SIZE; j++)
				{
					simdActivations.values[CLUSTER_SIZE*i + j] = a.cluster_values[j];
					simdWeights.values[CLUSTER_SIZE*i + j] = w.cluster_values[j];
				}

				//EMULATOR_PRINT ( ("[dispatcher]: w0: %#04x w1: %#04x a0: %#04x a1: %#04x \n"
				//	, w.cluster_values[0] & 0xFF, w.cluster_values[1] & 0xFF,  a.cluster_values[0] & 0xFF, a.cluster_values[1] & 0xFF) );
				//EMULATOR_PRINT ( ("[dispatcher]: wIndex: %u aIndex :%u \n", (indexW) & 0xFF, (indexA) & 0xFF));
			}


			t_accumulator tempPSum = madd(simdActivations, simdWeights);
			pSum += tempPSum;
			//if (debugCount < maxDebugCount)
			//	{
			//		DEBUG_PRINT(("[PE Dispatcher] a0, a1, a1, a2: %#04x %#04x %#04x %#04x\n",
			//			simdActivations.values[0] & 0xFF, 
			//			simdActivations.values[1] & 0xFF,
			//			simdActivations.values[2] & 0xFF,
			//			simdActivations.values[3] & 0xFF));

			//		DEBUG_PRINT(("[PE Dispatcher] w0, w1, w2, w3: %#04x %#04x %#04x %#04x\n",
			//			simdWeights.values[0] & 0xFF, 
			//			simdWeights.values[1] & 0xFF,
			//			simdWeights.values[2] & 0xFF,
			//			simdWeights.values[3] & 0xFF));

			//		DEBUG_PRINT(("[PE Madd] Psum %#04x\n", pSum));

			//	}		
			countOperands += SIMD_SIZE;
			indicesW = indicesW >> (SIMD_SIZE*BITWIDTH_COMPRESSION_WINDOW_INDEX);
			indicesA = indicesA >> (SIMD_SIZE*BITWIDTH_COMPRESSION_WINDOW_INDEX);

			if (countOperands >= numOperands)
			{
				if (isLast[(~regLoadSide) & 0x1] == TRUE)
				{
					nextStateMac = MAC_STATE_WRITE_PSUM;
				}
				else
				{
					nextStateMac = MAC_STATE_WAIT;
				}
			}
		} // if state == MAC_STATE_PROCESS_WINDOW
		else if (stateMac == MAC_STATE_WRITE_PSUM)
		{
			bool writeSuccess;
			writeSuccess = write_channel_nb_intel(channel_peDrainOutput, pSum);

			//write_channel_intel(channel_peDrainOutput, pSum);
			if (writeSuccess)
			{
				//DEBUG_PRINT(("[MAC] Sending!\n"));
				EMULATOR_PRINT(("[MAC] Commit. pSum value: %#04x \n", pSum));
				//DEBUG_PRINT(("[PE Psum] Commit. %#04x\n", pSum));
				pSum = 0;
				nextStateMac = MAC_STATE_WAIT;
				//pSum = 0;
			}
		}


	//===================SWAP===========================
	//Take an extra iteration for swapping, otherwise Fmax is low
		if ( (stateActivation == ASSEMBLER_STATE_WAIT)
			&& (stateWeight == ASSEMBLER_STATE_WAIT)
			&& (stateMac == MAC_STATE_WAIT) )
		{
			nextStateWeight = ASSEMBLER_STATE_LOAD_BITMASK;
			nextStateActivation = ASSEMBLER_STATE_LOAD_BITMASK;
			nextStateMac = MAC_STATE_ALIGN;

			regLoadSide = ~regLoadSide;
			//countActivation = 0;
			//countWeight = 0;

		}

		//================DEBUG==============================
		//if (debugCount < maxDebugCount)
		//{
		//	DEBUG_PRINT(("[PE] countWeight, %#03x\n", countWeight));
		//	DEBUG_PRINT(("[PE] countActivation: %#03x\n", countActivation));
		//	DEBUG_PRINT(("[PE] countOperands: %#03x\n", countOperands));
		//	DEBUG_PRINT(("[PE] indicesW: %#03x\n", indicesW));
		//	DEBUG_PRINT(("[PE] indicesA: %#03x\n", indicesA));
		//	debugCount++;
		//}
		
		//===================================================

		//================Next state update==================
		stateWeight = nextStateWeight;
		stateActivation = nextStateActivation;
		stateMac = nextStateMac;
		//===================================================
	} // while true
} // end of kernel
#endif